from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    repo: str = "murolem/wplace-archives"
    download_dir: Path = Path("./wplace_downloads")
    store_root: Path = Path("./wplace_sparse_store")
    xyz_output_dir: Path = Path("./wplace_xyz")

    interval_days: int = 10
    xyz_z: int = 11
    grid_tiles: int = 2048
    tile_size: int = 1000
    shard_tiles: int = 32
    max_colors: int = 64

    max_concurrent_downloads: int = 4
    workers: int = os.cpu_count() or 4
    prefetch_factor: int = 4
    apply_workers: int = os.cpu_count() or 4
    decode_workers: int = max(1, (os.cpu_count() or 4))
    io_buffer_bytes: int = 16 * 1024 * 1024

    compression_backend: str = "auto"  # archive input: auto | zstd | gzip
    store_compression: str = "zstd"  # intermediate store payload: zstd | none
    store_zstd_level: int = 3
    gzip_backend: str = "auto"  # gzip-compatible input: auto | pigz | isal | python
    pigz_path: str = "pigz"
    pigz_threads: int = max(1, os.cpu_count() or 1)
    use_isal_gzip: bool = True

    validate_download_digest: bool = False
    keep_archives: bool = False
    keep_tag_stores: bool = False
    strict_rgba: bool = False
    strict_binary_alpha: bool = False

    rgb_transparency_mode: str = "corners"  # none | corners | colors | auto
    rgb_transparent_colors: str = ""
    rgb_transparent_dominant_min: float = 0.90
    black_warning_ratio: float = 0.70

    fixed_palette_path: Path | None = None
    ingest_prescan: bool = False

    apply_executor: str = "thread"  # thread | process | sequential | isolated-process
    apply_max_tasks_per_child: int = 0

    @classmethod
    def from_toml(cls, path: Path | None = None) -> "Config":
        if path is None or not path.exists():
            return cls()
        with path.open("rb") as f:
            loaded = tomllib.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid config TOML: expected table root: {path}")

        data = dict(loaded)
        for key in ["download_dir", "store_root", "xyz_output_dir", "fixed_palette_path"]:
            if data.get(key) is not None:
                data[key] = Path(data[key])
        return cls().with_overrides(**data)

    def with_overrides(self, **kwargs) -> "Config":
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **cleaned)

    @property
    def tags_root(self) -> Path:
        return self.store_root / "tags"

    @property
    def state_root(self) -> Path:
        return self.store_root / "state"

    @property
    def palette_path(self) -> Path:
        return self.store_root / "palette.json"

    @property
    def pipeline_state_path(self) -> Path:
        return self.store_root / "pipeline_state.json"

    @property
    def diagnostics_root(self) -> Path:
        return self.store_root / "diagnostics"

    @property
    def stats_root(self) -> Path:
        return self.diagnostics_root / "stats"

    @property
    def shard_count_axis(self) -> int:
        return (self.grid_tiles + self.shard_tiles - 1) // self.shard_tiles

    @property
    def tile_pixels(self) -> int:
        return self.tile_size * self.tile_size

    @property
    def dense_tile_bytes(self) -> int:
        return self.tile_pixels

    @property
    def dense_fallback_bytes(self) -> int:
        return self.tile_pixels

    def stable_dict(self) -> dict:
        data = asdict(self)
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
        return data

    def config_hash(self) -> str:
        relevant = self.stable_dict()
        # Paths are operational, not data-format identity.
        for key in ["download_dir", "store_root", "xyz_output_dir", "max_concurrent_downloads", "workers", "apply_workers", "decode_workers"]:
            relevant.pop(key, None)
        raw = json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
