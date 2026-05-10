from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .errors import ExportError
from .palette import PaletteCodec
from .png_codec import encode_png_rgba
from .records import dense_record_from_sparse
from .shard_store import SparseTileRef, SparseTileStore


def _export_one(cfg: Config, palette: PaletteCodec, store_root: Path, ref: SparseTileRef) -> tuple[int, int, int]:
    store = SparseTileStore(store_root, cfg)
    positions, values = store.load_sparse(ref)
    idx = dense_record_from_sparse(positions, values, cfg.tile_pixels).reshape((cfg.tile_size, cfg.tile_size))
    rgba = palette.index_to_rgba(idx)
    png = encode_png_rgba(rgba)
    out_path = cfg.xyz_output_dir / str(cfg.xyz_z) / str(ref.x) / f"{ref.y}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(png)
    tmp.replace(out_path)
    return ref.x, ref.y, len(png)


def export_state_to_xyz(cfg: Config, palette: PaletteCodec) -> dict:
    if not (cfg.state_root / "manifest.json").exists():
        raise ExportError(f"state store missing: {cfg.state_root}")
    store = SparseTileStore(cfg.state_root, cfg)
    refs = list(store.iter_refs())
    stats = {"tiles": 0, "png_bytes": 0, "out": str(cfg.xyz_output_dir / str(cfg.xyz_z))}
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex, tqdm(total=len(refs), desc="export XYZ", unit="tile") as pbar:
        futures = [ex.submit(_export_one, cfg, palette, cfg.state_root, ref) for ref in refs]
        for fut in as_completed(futures):
            _, _, n = fut.result()
            stats["tiles"] += 1
            stats["png_bytes"] += n
            pbar.update(1)
    from .utils import atomic_write_json
    cfg.stats_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.stats_root / "export.json", stats)
    return stats
