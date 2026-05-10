from __future__ import annotations

from pathlib import Path

from .config import Config
from .errors import StateConsistencyError
from .utils import load_json, shard_bin_path, shard_index_path, store_manifest_path


def validate_store(cfg: Config) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    state = load_json(cfg.pipeline_state_path, None)
    if state is None:
        errors.append(f"missing pipeline state: {cfg.pipeline_state_path}")
    elif state.get("format") != "wplace-pipeline-state-v2":
        errors.append("invalid pipeline_state.json format")
    palette = load_json(cfg.palette_path, None)
    if palette is None:
        warnings.append(f"missing palette: {cfg.palette_path}")
    else:
        expected = 1
        for item in palette.get("colors", []):
            if int(item.get("index", -1)) != expected:
                errors.append(f"palette index is not contiguous at expected={expected}")
                break
            expected += 1
    manifest = load_json(store_manifest_path(cfg.state_root), None)
    checked_shards = 0
    checked_tiles = 0
    if manifest is None:
        warnings.append(f"missing state manifest: {store_manifest_path(cfg.state_root)}")
    else:
        for shard in manifest.get("shards", []):
            sx, sy = int(shard["sx"]), int(shard["sy"])
            idx_path = shard_index_path(cfg.state_root, sx, sy)
            bin_path = shard_bin_path(cfg.state_root, sx, sy)
            if not idx_path.exists():
                errors.append(f"missing shard index: {idx_path}")
                continue
            if not bin_path.exists():
                errors.append(f"missing shard bin: {bin_path}")
                continue
            idx = load_json(idx_path, {})
            size = bin_path.stat().st_size
            for e in idx.get("tiles", []):
                off = int(e["offset"]); n = int(e["size"])
                if off < 0 or n < 0 or off + n > size:
                    errors.append(f"bad offset/size in {idx_path}: {e}")
                if int(e.get("count", 0)) < 0 or int(e.get("count", 0)) > cfg.tile_pixels:
                    errors.append(f"bad count in {idx_path}: {e}")
                compression = str(e.get("compression", "none"))
                if compression not in ("none", "zstd"):
                    errors.append(f"bad compression in {idx_path}: {e}")
                uncomp = int(e.get("uncompressed_size", e.get("size", 0)))
                if uncomp < 0:
                    errors.append(f"bad uncompressed_size in {idx_path}: {e}")
                checked_tiles += 1
            checked_shards += 1
    result = {"ok": not errors, "errors": errors, "warnings": warnings, "checked_shards": checked_shards, "checked_tiles": checked_tiles}
    if errors:
        raise StateConsistencyError(result)
    return result
