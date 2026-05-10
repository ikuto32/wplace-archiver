from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .errors import AssetValidationError

SPLIT_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?\.tar\.(?P<compression>gz|zst|zstd)\.)(?P<suffix>[A-Za-z]+|\d+)$", re.IGNORECASE)
STRICT_ASSET_RE = re.compile(r"^(?P<prefix>.+?\.tar\.(?P<compression>gz|zst|zstd)\.)(?P<suffix>[A-Za-z]+|\d+)$", re.IGNORECASE)
NATURAL_TOKEN_RE = re.compile(r"\d+|\D+")
TILE_PATH_RE = re.compile(r"(?P<x>\d+)/(?P<y>\d+)\.png$", re.IGNORECASE)


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_hex_from_github(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if value.lower().startswith("sha256:"):
        return value.split(":", 1)[1].lower()
    return None


def tag_datetime(tag: str) -> datetime:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})[-_:](\d{2})[-_:](\d{2})", tag)
    if not m:
        return datetime.max
    return datetime(*map(int, m.groups()))


def sort_tags_chronologically(tags: Iterable[str]) -> list[str]:
    return sorted(tags, key=tag_datetime)


def natural_key_text(s: str):
    return tuple((0, int(t)) if t.isdigit() else (1, t.lower()) for t in NATURAL_TOKEN_RE.findall(s))


def alpha_suffix_number(s: str) -> int:
    # split-style: aa=0, ab=1, ..., az=25, ba=26.
    n = 0
    for ch in s.lower():
        if not ("a" <= ch <= "z"):
            return -1
        n = n * 26 + (ord(ch) - ord("a"))
    return n - sum(26**i for i in range(len(s) - 1))


def split_part_sort_key(path_or_name) -> tuple:
    name = Path(path_or_name).name
    m = SPLIT_SUFFIX_RE.match(name)
    if m:
        prefix = m.group("prefix").lower()
        suffix = m.group("suffix")
        if suffix.isdigit():
            suffix_key = (0, int(suffix))
        else:
            suffix_key = (1, len(suffix), alpha_suffix_number(suffix), suffix.lower())
        return (prefix, suffix_key, name.lower())
    return ("", (9, natural_key_text(name)), name.lower())


def sort_split_part_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted([Path(p) for p in paths], key=split_part_sort_key)


def is_strict_split_asset_name(name: str) -> bool:
    return STRICT_ASSET_RE.match(Path(name).name) is not None


def asset_prefix(name: str) -> str:
    m = STRICT_ASSET_RE.match(Path(name).name)
    if not m:
        raise AssetValidationError(f"not a strict split tar asset: {name}")
    return m.group("prefix")


def asset_compression(name: str) -> str:
    m = STRICT_ASSET_RE.match(Path(name).name)
    if not m:
        raise AssetValidationError(f"not a strict split tar asset: {name}")
    comp = m.group("compression").lower()
    return "gzip" if comp == "gz" else "zstd"


def filter_split_assets(assets: list[dict], compression_preference: str = "auto") -> list[dict]:
    """Return strict split tar assets, preferring zstd over gzip by default."""
    preference = (compression_preference or "auto").strip().lower()
    if preference not in {"auto", "zstd", "gzip"}:
        raise AssetValidationError("compression preference must be one of: auto, zstd, gzip")

    filtered = [a for a in assets if is_strict_split_asset_name(str(a.get("name", "")))]
    if not filtered:
        return []

    groups: dict[tuple[str, str], list[dict]] = {}
    for a in filtered:
        name = str(a["name"])
        groups.setdefault((asset_compression(name), asset_prefix(name)), []).append(a)

    if preference == "zstd":
        groups = {k: v for k, v in groups.items() if k[0] == "zstd"}
    elif preference == "gzip":
        groups = {k: v for k, v in groups.items() if k[0] == "gzip"}
    elif any(k[0] == "zstd" for k in groups):
        groups = {k: v for k, v in groups.items() if k[0] == "zstd"}
    else:
        groups = {k: v for k, v in groups.items() if k[0] == "gzip"}

    if not groups:
        return []
    if len(groups) > 1:
        sizes = {f"{comp}:{prefix}": len(v) for (comp, prefix), v in groups.items()}
        raise AssetValidationError(f"multiple split asset groups in release: {sizes}")

    selected = next(iter(groups.values()))
    return sorted(selected, key=lambda a: split_part_sort_key(str(a.get("name", ""))))


def parse_tile_path(path: str) -> tuple[int, int] | None:
    m = TILE_PATH_RE.search(path.replace("\\", "/"))
    if not m:
        return None
    return int(m.group("x")), int(m.group("y"))


def validate_tile_xy(x: int, y: int, grid_tiles: int) -> None:
    if not (0 <= x < grid_tiles and 0 <= y < grid_tiles):
        raise ValueError(f"tile out of range: x={x}, y={y}, expected 0..{grid_tiles - 1}")


def shard_id_from_xy(x: int, y: int, shard_tiles: int) -> tuple[int, int]:
    return x // shard_tiles, y // shard_tiles


def shard_name(sx: int, sy: int) -> str:
    return f"s{sx:04d}_{sy:04d}"


def parse_shard_name(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"s(\d{4})_(\d{4})", name)
    if not m:
        raise ValueError(f"invalid shard name: {name}")
    return int(m.group(1)), int(m.group(2))


def shard_bin_path(root: Path, sx: int, sy: int) -> Path:
    return root / f"{shard_name(sx, sy)}.bin"


def shard_index_path(root: Path, sx: int, sy: int) -> Path:
    return root / f"{shard_name(sx, sy)}.index.json"


def store_manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
