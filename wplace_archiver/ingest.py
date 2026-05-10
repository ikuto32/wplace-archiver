from __future__ import annotations

import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .decompress import StreamingTar
from .errors import IngestError, TarScanError
from .palette import PaletteCodec
from .png_codec import decode_png_array
from .shard_store import SparseShardWriter
from .utils import atomic_write_json, parse_tile_path


def _decode_to_payload(x: int, y: int, png_bytes: bytes, cfg: Config, palette: PaletteCodec):
    arr, decode_meta = decode_png_array(png_bytes, cfg, include_meta=True)
    # Decode stays fully parallel, but palette mutation is serialized internally
    # by PaletteCodec when unknown colors are discovered/mapped.
    payload, count, encoding, diag = palette.image_to_record(arr)
    return x, y, payload, count, encoding, diag, decode_meta


def count_png_members(part_files: list[Path], cfg: Config) -> int:
    count = 0
    try:
        with StreamingTar(part_files, cfg) as tar:
            for member in tar:
                if member.isfile() and parse_tile_path(member.name) is not None:
                    count += 1
    except Exception as exc:
        raise TarScanError(f"pre-scan failed: {exc}") from exc
    return count


def ingest_tag_from_parts(tag: str, part_files: list[Path], cfg: Config, palette: PaletteCodec) -> dict:
    """Ingest one tag into temporary tag store. Palette is saved by caller after apply success."""
    tag_root = cfg.tags_root / tag
    if tag_root.exists():
        shutil.rmtree(tag_root)
    tag_root.mkdir(parents=True, exist_ok=True)
    total = count_png_members(part_files, cfg) if cfg.ingest_prescan else None
    rgb_diagnostics_sample: list[dict] = []
    rgba_black_warning_sample: list[dict] = []
    diagnostics_limit = 250
    rgb_tiles_seen = 0
    rgba_warning_tiles = 0
    p_tiles_seen = 0
    p_has_trns = 0
    p_no_trns = 0
    errors: list[str] = []
    png_seen = 0
    records_written = 0
    visible_pixels = 0
    tdesc = f"ingest {tag}"
    max_pending = max(1, cfg.decode_workers * cfg.prefetch_factor)
    with SparseShardWriter(tag_root, cfg, label=f"tag:{tag}") as writer, ThreadPoolExecutor(max_workers=max(1, cfg.decode_workers)) as pool:
        pending = set()

        def drain(done_set):
            nonlocal records_written, visible_pixels, rgb_tiles_seen, rgba_warning_tiles, p_tiles_seen, p_has_trns, p_no_trns
            for fut in done_set:
                try:
                    x, y, payload, count, encoding, diag, decode_meta = fut.result()
                    if count:
                        writer.write_tile_payload(x, y, payload, count, encoding)
                        records_written += 1
                        visible_pixels += int(count)
                    if decode_meta.get("source_mode") == "P":
                        p_tiles_seen += 1
                        if decode_meta.get("p_has_trns"):
                            p_has_trns += 1
                        else:
                            p_no_trns += 1
                    if diag.get("input") == "rgb":
                        rgb_tiles_seen += 1
                        if len(rgb_diagnostics_sample) < diagnostics_limit:
                            rgb_diagnostics_sample.append({"x": x, "y": y, **diag})
                    elif diag.get("input") == "rgba" and diag.get("warning"):
                        rgba_warning_tiles += 1
                        if len(rgba_black_warning_sample) < diagnostics_limit:
                            rgba_black_warning_sample.append({"x": x, "y": y, **diag})
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        try:
            with StreamingTar(part_files, cfg) as tar, tqdm(total=total, desc=tdesc, unit="tile") as pbar:
                for member in tar:
                    if not member.isfile():
                        continue
                    xy = parse_tile_path(member.name)
                    if xy is None:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    png_seen += 1
                    pending.add(pool.submit(_decode_to_payload, xy[0], xy[1], data, cfg, palette))
                    pbar.update(1)
                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        drain(done)
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)
        except Exception as exc:
            raise IngestError(f"ingest stream failed for {tag}: {exc}") from exc
    if errors:
        raise IngestError(f"{len(errors)} tile errors during ingest for {tag}; first errors: {errors[:5]}")
    rgb_black_dominant_tiles = sum(1 for d in rgb_diagnostics_sample if d.get("black_dominant"))
    stats = {
        "tag": tag,
        "png_tiles_seen": png_seen,
        "records_written": records_written,
        "visible_pixels": visible_pixels,
        "rgb_tiles_seen": rgb_tiles_seen,
        "p_tiles_seen": p_tiles_seen,
        "p_has_trns": p_has_trns,
        "p_no_trns": p_no_trns,
        "rgba_black_warning_tiles": rgba_warning_tiles,
        "rgb_black_dominant_tiles_in_sample": rgb_black_dominant_tiles,
        "rgb_transparency_diagnostics_sample": rgb_diagnostics_sample,
        "rgba_black_warning_sample": rgba_black_warning_sample,
    }
    cfg.stats_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.stats_root / f"ingest_{tag}.json", stats)
    return stats
