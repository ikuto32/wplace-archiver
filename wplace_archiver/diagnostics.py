from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .config import Config
from .decompress import StreamingTar
from .palette import PaletteCodec
import numpy as np

from .png_codec import decode_png_array, encode_png_rgba
from .utils import atomic_write_json, parse_tile_path


def diagnose_rgb_transparency(parts: list[Path], cfg: Config, sample: int, out: Path | None = None) -> dict:
    out = out or (cfg.diagnostics_root / "rgb_transparency_samples")
    out.mkdir(parents=True, exist_ok=True)
    palette = PaletteCodec(cfg)
    rows = []
    scanned = 0
    with StreamingTar(parts, cfg) as tar, tqdm(total=sample, desc="diagnose RGB transparency", unit="tile") as pbar:
        for member in tar:
            if len(rows) >= sample:
                break
            if not member.isfile():
                continue
            xy = parse_tile_path(member.name)
            if xy is None:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            arr = decode_png_array(data, cfg)
            scanned += 1
            if arr.ndim == 3 and arr.shape[-1] == 3:
                payload, count, encoding, diag = palette.image_to_record(arr)
                rows.append({"x": xy[0], "y": xy[1], "count": count, "encoding": encoding, **diag})
                from PIL import Image
                Image.fromarray(arr, mode="RGB").save(out / f"{xy[0]}_{xy[1]}_original.png")
                # Save inferred transparency preview.
                if encoding == "sparse-u32-u8-v1":
                    pos = np.frombuffer(payload[: count * 4], dtype=np.uint32, count=count).copy()
                    val = np.frombuffer(payload[count * 4 :], dtype=np.uint8, count=count).copy()
                    dense = np.zeros(cfg.tile_pixels, dtype=np.uint8)
                    if count:
                        dense[pos.astype(np.intp, copy=False)] = val
                    rgba = palette.index_to_rgba(dense.reshape((cfg.tile_size, cfg.tile_size)))
                else:
                    dense = np.frombuffer(payload, dtype=np.uint8).copy().reshape((cfg.tile_size, cfg.tile_size))
                    rgba = palette.index_to_rgba(dense)
                (out / f"{xy[0]}_{xy[1]}_inferred.png").write_bytes(encode_png_rgba(rgba))
                pbar.update(1)
    result = {"scanned_tiles": scanned, "rgb_samples": len(rows), "samples": rows}
    atomic_write_json(out / "diagnostics.json", result)
    return result
