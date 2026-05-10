from __future__ import annotations

import hashlib
import json
import os
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
    def from_env(cls) -> "Config":
        cpu = os.cpu_count() or 4
        return cls(
            repo=os.environ.get("WPLACE_REPO", "murolem/wplace-archives"),
            download_dir=Path(os.environ.get("WPLACE_DOWNLOAD_DIR", "./wplace_downloads")),
            store_root=Path(os.environ.get("WPLACE_STORE_ROOT", "./wplace_sparse_store")),
            xyz_output_dir=Path(os.environ.get("WPLACE_XYZ_OUTPUT_DIR", "./wplace_xyz")),
            interval_days=int(os.environ.get("WPLACE_INTERVAL_DAYS", "10")),
            xyz_z=int(os.environ.get("WPLACE_XYZ_Z", "11")),
            grid_tiles=int(os.environ.get("WPLACE_GRID_TILES", "2048")),
            tile_size=int(os.environ.get("WPLACE_TILE_SIZE", "1000")),
            shard_tiles=int(os.environ.get("WPLACE_SHARD_TILES", "32")),
            max_colors=int(os.environ.get("WPLACE_MAX_COLORS", "64")),
            max_concurrent_downloads=int(os.environ.get("WPLACE_MAX_DOWNLOADS", "4")),
            workers=int(os.environ.get("WPLACE_WORKERS", str(cpu))),
            prefetch_factor=int(os.environ.get("WPLACE_PREFETCH", "4")),
            apply_workers=int(os.environ.get("WPLACE_APPLY_WORKERS", os.environ.get("WPLACE_WORKERS", str(cpu)))),
            decode_workers=int(os.environ.get("WPLACE_DECODE_WORKERS", os.environ.get("WPLACE_WORKERS", str(cpu)))),
            io_buffer_bytes=int(os.environ.get("WPLACE_IO_BUFFER_MB", "16")) * 1024 * 1024,
            compression_backend=os.environ.get("WPLACE_COMPRESSION_BACKEND", "auto").strip().lower(),
            store_compression=os.environ.get("WPLACE_STORE_COMPRESSION", "zstd").strip().lower(),
            store_zstd_level=int(os.environ.get("WPLACE_STORE_ZSTD_LEVEL", "3")),
            gzip_backend=os.environ.get("WPLACE_GZIP_BACKEND", "auto").strip().lower(),
            pigz_path=os.environ.get("WPLACE_PIGZ_PATH", "pigz-2.3-bin-win32/pigz.exe" if os.name == "nt" else "pigz"),
            pigz_threads=int(os.environ.get("WPLACE_PIGZ_THREADS", str(max(1, cpu)))),
            use_isal_gzip=os.environ.get("WPLACE_USE_ISAL_GZIP", "1") == "1",
            validate_download_digest=os.environ.get("WPLACE_VALIDATE_DOWNLOAD_DIGEST", "0") == "1",
            keep_archives=os.environ.get("WPLACE_KEEP_ARCHIVES", "0") == "1",
            keep_tag_stores=os.environ.get("WPLACE_KEEP_TAG_STORES", "0") == "1",
            strict_rgba=os.environ.get("WPLACE_STRICT_RGBA", "0") == "1",
            strict_binary_alpha=os.environ.get("WPLACE_STRICT_BINARY_ALPHA", "0") == "1",
            rgb_transparency_mode=os.environ.get("WPLACE_RGB_TRANSPARENCY_MODE", "corners").strip().lower(),
            rgb_transparent_colors=os.environ.get("WPLACE_RGB_TRANSPARENT_COLORS", ""),
            rgb_transparent_dominant_min=float(os.environ.get("WPLACE_RGB_TRANSPARENT_DOMINANT_MIN", "0.90")),
            black_warning_ratio=float(os.environ.get("WPLACE_BLACK_WARNING_RATIO", "0.70")),
            fixed_palette_path=Path(os.environ["WPLACE_FIXED_PALETTE"]) if os.environ.get("WPLACE_FIXED_PALETTE") else None,
            ingest_prescan=os.environ.get("WPLACE_INGEST_PRESCAN", "0") == "1",
            apply_executor=os.environ.get("WPLACE_APPLY_EXECUTOR", "thread").strip().lower(),
            apply_max_tasks_per_child=int(os.environ.get("WPLACE_APPLY_MAX_TASKS_PER_CHILD", "0")),
        )

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
        return int(os.environ.get("WPLACE_DENSE_FALLBACK_BYTES", str(self.tile_pixels)))

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
