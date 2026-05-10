from __future__ import annotations

from pathlib import Path

from .errors import AssetValidationError
from .utils import asset_compression, asset_prefix, filter_split_assets, is_strict_split_asset_name, sort_split_part_paths


def select_release_split_assets(assets: list[dict], compression_preference: str = "auto") -> list[dict]:
    """Return strict byte-split tar assets.

    New releases may publish .tar.zst.* / .tar.zstd.* parts. Existing .tar.gz.*
    parts remain supported for backward compatibility. With preference=auto,
    zstd is selected when both formats are present.
    """
    selected = filter_split_assets(assets, compression_preference=compression_preference)
    if not selected:
        raise AssetValidationError("release has no strict split assets matching *.tar.zst.<suffix>, *.tar.zstd.<suffix>, or *.tar.gz.<suffix>")
    return selected


def validate_local_part_names(paths: list[Path]) -> list[Path]:
    bad = [str(p) for p in paths if not is_strict_split_asset_name(p.name)]
    if bad:
        raise AssetValidationError(f"invalid split asset filenames: {bad}")
    if paths:
        groups = {(asset_compression(p.name), asset_prefix(p.name)) for p in paths}
        if len(groups) > 1:
            raise AssetValidationError(f"local split parts must have one compression/prefix group, got: {sorted(groups)}")
    return sort_split_part_paths(paths)
