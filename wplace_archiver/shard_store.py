from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config
from .records import (
    DENSE_ENCODING,
    SPARSE_ENCODING,
    dense_record_from_sparse,
    ensure_u32_positions,
    ensure_u8_values,
    sparse_from_payload,
    sparse_record_size,
)
from .store_codec import compress_store_payload, decompress_store_payload
from .utils import (
    atomic_write_json,
    load_json,
    parse_shard_name,
    shard_bin_path,
    shard_id_from_xy,
    shard_index_path,
    shard_name,
    store_manifest_path,
    validate_tile_xy,
)


@dataclass(frozen=True)
class SparseTileRef:
    x: int
    y: int
    sx: int
    sy: int
    offset: int
    size: int
    count: int
    encoding: str
    data_file: str
    compression: str = "none"
    uncompressed_size: int | None = None


class SparseShardWriter:
    """Append-only writer for one sparse sharded tile store.

    New stores compress each tile record payload with cfg.store_compression.
    Existing uncompressed records remain readable because missing compression
    metadata is interpreted as compression="none".
    """

    def __init__(self, root: Path, cfg: Config, *, label: str):
        self.root = root
        self.cfg = cfg
        self.label = label
        self.root.mkdir(parents=True, exist_ok=True)
        self._files: dict[tuple[int, int], io.BufferedWriter] = {}
        self._indexes: dict[tuple[int, int], dict[str, dict]] = {}
        self._tile_counts: dict[tuple[int, int], int] = {}
        self._visible_counts: dict[tuple[int, int], int] = {}
        self._byte_counts: dict[tuple[int, int], int] = {}
        self._uncompressed_byte_counts: dict[tuple[int, int], int] = {}
        self._total_tiles = 0
        self._total_visible = 0
        self._total_bytes = 0
        self._total_uncompressed_bytes = 0
        self._closed = False

    def _open_shard(self, sx: int, sy: int) -> io.BufferedWriter:
        key = (sx, sy)
        f = self._files.get(key)
        if f is not None:
            return f
        path = shard_bin_path(self.root, sx, sy)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = path.open("ab", buffering=self.cfg.io_buffer_bytes)
        self._files[key] = f
        self._indexes.setdefault(key, {})
        return f

    def write_tile_payload(self, x: int, y: int, payload: bytes, count: int, encoding: str) -> None:
        if self._closed:
            raise RuntimeError("writer is already closed")
        validate_tile_xy(x, y, self.cfg.grid_tiles)
        count = int(count)
        if count == 0:
            return
        if encoding not in (SPARSE_ENCODING, DENSE_ENCODING):
            raise ValueError(f"unknown tile encoding: {encoding}")
        expected_uncompressed = sparse_record_size(count) if encoding == SPARSE_ENCODING else self.cfg.dense_tile_bytes
        if len(payload) != expected_uncompressed:
            raise ValueError(f"bad {encoding} payload size for {x}/{y}: expected={expected_uncompressed}, got={len(payload)}")

        stored_payload, compression, uncompressed_size = compress_store_payload(payload, self.cfg)
        sx, sy = shard_id_from_xy(x, y, self.cfg.shard_tiles)
        f = self._open_shard(sx, sy)
        offset = f.tell()
        f.write(stored_payload)
        size = len(stored_payload)
        key = f"{x},{y}"
        prev = self._indexes[(sx, sy)].get(key)
        self._indexes[(sx, sy)][key] = {
            "x": x,
            "y": y,
            "offset": offset,
            "size": size,
            "uncompressed_size": uncompressed_size,
            "count": count,
            "encoding": encoding,
            "compression": compression,
        }
        if prev is None:
            self._tile_counts[(sx, sy)] = self._tile_counts.get((sx, sy), 0) + 1
            self._total_tiles += 1
        else:
            self._visible_counts[(sx, sy)] -= int(prev.get("count", 0))
            self._byte_counts[(sx, sy)] -= int(prev.get("size", 0))
            self._uncompressed_byte_counts[(sx, sy)] -= int(prev.get("uncompressed_size", prev.get("size", 0)))
            self._total_visible -= int(prev.get("count", 0))
            self._total_bytes -= int(prev.get("size", 0))
            self._total_uncompressed_bytes -= int(prev.get("uncompressed_size", prev.get("size", 0)))
        self._visible_counts[(sx, sy)] = self._visible_counts.get((sx, sy), 0) + count
        self._byte_counts[(sx, sy)] = self._byte_counts.get((sx, sy), 0) + size
        self._uncompressed_byte_counts[(sx, sy)] = self._uncompressed_byte_counts.get((sx, sy), 0) + uncompressed_size
        self._total_visible += count
        self._total_bytes += size
        self._total_uncompressed_bytes += uncompressed_size

    def close(self) -> None:
        if self._closed:
            return
        for f in self._files.values():
            f.flush()
            f.close()
        self._files.clear()
        for (sx, sy), tiles in self._indexes.items():
            entries = sorted(tiles.values(), key=lambda e: (e["x"], e["y"]))
            index = {
                "format": "wplace-sparse-shard-index-v1",
                "label": self.label,
                "tile_size": self.cfg.tile_size,
                "tile_pixels": self.cfg.tile_pixels,
                "grid_tiles": self.cfg.grid_tiles,
                "shard_tiles": self.cfg.shard_tiles,
                "shard": {"sx": sx, "sy": sy, "name": shard_name(sx, sy)},
                "data_file": shard_bin_path(self.root, sx, sy).name,
                "store_compression": self.cfg.store_compression,
                "tile_count": len(entries),
                "visible_pixels": int(sum(int(e.get("count", 0)) for e in entries)),
                "stored_bytes": int(sum(int(e.get("size", 0)) for e in entries)),
                "uncompressed_bytes": int(sum(int(e.get("uncompressed_size", e.get("size", 0))) for e in entries)),
                "tiles": entries,
            }
            atomic_write_json(shard_index_path(self.root, sx, sy), index)
        self.write_manifest()
        self._closed = True

    def write_manifest(self) -> None:
        manifest = {
            "format": "wplace-sparse-tile-store-v1",
            "label": self.label,
            "tile_size": self.cfg.tile_size,
            "tile_pixels": self.cfg.tile_pixels,
            "grid_tiles": self.cfg.grid_tiles,
            "shard_tiles": self.cfg.shard_tiles,
            "shard_count_axis": self.cfg.shard_count_axis,
            "transparent_index": 0,
            "palette_value_dtype": "uint8",
            "position_dtype": "uint32",
            "store_compression": self.cfg.store_compression,
            "tile_count": self._total_tiles,
            "visible_pixels": self._total_visible,
            "stored_bytes": self._total_bytes,
            "uncompressed_bytes": self._total_uncompressed_bytes,
            "dense_fallback_bytes": self.cfg.dense_fallback_bytes,
            "shards": [
                {
                    "sx": sx,
                    "sy": sy,
                    "name": shard_name(sx, sy),
                    "tile_count": self._tile_counts.get((sx, sy), 0),
                    "visible_pixels": self._visible_counts.get((sx, sy), 0),
                    "stored_bytes": self._byte_counts.get((sx, sy), 0),
                    "uncompressed_bytes": self._uncompressed_byte_counts.get((sx, sy), 0),
                    "data_file": shard_bin_path(self.root, sx, sy).name,
                    "index_file": shard_index_path(self.root, sx, sy).name,
                }
                for sx, sy in sorted(self._tile_counts)
            ],
        }
        atomic_write_json(store_manifest_path(self.root), manifest)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SparseTileStore:
    def __init__(self, root: Path, cfg: Config):
        self.root = root
        self.cfg = cfg
        manifest = load_json(store_manifest_path(root), None)
        if manifest is None:
            shards = []
            for p in sorted(root.glob("s????_????.index.json")):
                sx, sy = parse_shard_name(p.name.removesuffix(".index.json"))
                idx = load_json(p, {})
                shards.append({
                    "sx": sx,
                    "sy": sy,
                    "name": shard_name(sx, sy),
                    "tile_count": int(idx.get("tile_count", len(idx.get("tiles", [])))),
                    "visible_pixels": int(idx.get("visible_pixels", 0)),
                    "stored_bytes": int(idx.get("stored_bytes", 0)),
                    "uncompressed_bytes": int(idx.get("uncompressed_bytes", idx.get("stored_bytes", 0))),
                    "data_file": shard_bin_path(root, sx, sy).name,
                    "index_file": p.name,
                })
            manifest = {
                "format": "wplace-sparse-tile-store-v1",
                "tile_size": cfg.tile_size,
                "tile_pixels": cfg.tile_pixels,
                "grid_tiles": cfg.grid_tiles,
                "shard_tiles": cfg.shard_tiles,
                "tile_count": sum(s["tile_count"] for s in shards),
                "visible_pixels": sum(s["visible_pixels"] for s in shards),
                "stored_bytes": sum(s["stored_bytes"] for s in shards),
                "uncompressed_bytes": sum(s["uncompressed_bytes"] for s in shards),
                "shards": shards,
            }
        self.manifest = manifest
        self._index_cache: dict[tuple[int, int], dict] = {}
        self._shard_set = {(int(s["sx"]), int(s["sy"])) for s in self.manifest.get("shards", [])}

    def shard_ids(self) -> list[tuple[int, int]]:
        return sorted(self._shard_set)

    def tile_count(self) -> int:
        return int(self.manifest.get("tile_count", 0))

    def visible_pixels(self) -> int:
        return int(self.manifest.get("visible_pixels", 0))

    def stored_bytes(self) -> int:
        return int(self.manifest.get("stored_bytes", 0))

    def _load_index(self, sx: int, sy: int) -> dict:
        key = (sx, sy)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        path = shard_index_path(self.root, sx, sy)
        idx = load_json(path, None)
        if idx is None:
            raise FileNotFoundError(path)
        self._index_cache[key] = idx
        return idx

    def iter_refs(self, sx: int | None = None, sy: int | None = None) -> Iterator[SparseTileRef]:
        shard_ids = [(sx, sy)] if sx is not None and sy is not None else self.shard_ids()
        for ssx, ssy in shard_ids:
            idx = self._load_index(int(ssx), int(ssy))
            data_file = idx["data_file"]
            for e in idx.get("tiles", []):
                compression = str(e.get("compression", "none"))
                uncompressed_size = int(e.get("uncompressed_size", e.get("size", 0)))
                yield SparseTileRef(
                    x=int(e["x"]),
                    y=int(e["y"]),
                    sx=int(ssx),
                    sy=int(ssy),
                    offset=int(e["offset"]),
                    size=int(e["size"]),
                    count=int(e.get("count", 0)),
                    encoding=str(e.get("encoding", SPARSE_ENCODING)),
                    data_file=data_file,
                    compression=compression,
                    uncompressed_size=uncompressed_size,
                )

    def load_sparse(self, ref: SparseTileRef) -> tuple[np.ndarray, np.ndarray]:
        return load_sparse_record_from_root(self.root, ref, self.cfg)


def load_sparse_record_from_root(root: Path, ref: SparseTileRef, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    path = root / ref.data_file
    with path.open("rb") as f:
        f.seek(ref.offset)
        data = f.read(ref.size)
    if len(data) != ref.size:
        raise IOError(f"short read: {path} offset={ref.offset} expected={ref.size} got={len(data)}")
    payload = decompress_store_payload(data, ref.compression, ref.uncompressed_size)
    return sparse_from_payload(payload, ref.count, ref.encoding, cfg.tile_pixels, cfg.dense_tile_bytes)


class AtomicSparseShardWriter:
    def __init__(self, root: Path, cfg: Config, sx: int, sy: int, *, label: str):
        self.root = root
        self.cfg = cfg
        self.sx = sx
        self.sy = sy
        self.label = label
        self.root.mkdir(parents=True, exist_ok=True)
        self.bin_final = shard_bin_path(root, sx, sy)
        self.idx_final = shard_index_path(root, sx, sy)
        self.bin_tmp = self.bin_final.with_name(self.bin_final.name + ".tmp")
        self.idx_tmp = self.idx_final.with_name(self.idx_final.name + ".tmp")

    def write(self, tiles: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> dict | None:
        entries = []
        offset = 0
        stored_bytes = 0
        uncompressed_bytes = 0
        visible_pixels = 0
        with self.bin_tmp.open("wb") as f:
            for (x, y), (positions, values) in sorted(tiles.items()):
                positions = ensure_u32_positions(positions)
                values = ensure_u8_values(values)
                if positions.size != values.size:
                    raise ValueError(f"position/value count mismatch: {positions.size} != {values.size}")
                count = int(positions.size)
                if count == 0:
                    continue
                if sparse_record_size(count) < self.cfg.dense_fallback_bytes:
                    encoding = SPARSE_ENCODING
                    payload = positions.tobytes(order="C") + values.tobytes(order="C")
                else:
                    encoding = DENSE_ENCODING
                    dense = dense_record_from_sparse(positions, values, self.cfg.tile_pixels)
                    payload = dense.tobytes(order="C")
                stored_payload, compression, uncompressed_size = compress_store_payload(payload, self.cfg)
                f.write(stored_payload)
                size = len(stored_payload)
                entries.append({
                    "x": int(x),
                    "y": int(y),
                    "offset": offset,
                    "size": size,
                    "uncompressed_size": uncompressed_size,
                    "count": count,
                    "encoding": encoding,
                    "compression": compression,
                })
                offset += size
                stored_bytes += size
                uncompressed_bytes += uncompressed_size
                visible_pixels += count
        if not entries:
            for p in [self.bin_tmp, self.idx_tmp, self.bin_final, self.idx_final]:
                if p.exists():
                    p.unlink()
            return None
        index = {
            "format": "wplace-sparse-shard-index-v1",
            "label": self.label,
            "tile_size": self.cfg.tile_size,
            "tile_pixels": self.cfg.tile_pixels,
            "grid_tiles": self.cfg.grid_tiles,
            "shard_tiles": self.cfg.shard_tiles,
            "shard": {"sx": self.sx, "sy": self.sy, "name": shard_name(self.sx, self.sy)},
            "data_file": self.bin_final.name,
            "store_compression": self.cfg.store_compression,
            "tile_count": len(entries),
            "visible_pixels": visible_pixels,
            "stored_bytes": stored_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "tiles": entries,
        }
        atomic_write_json(self.idx_tmp, index)
        os.replace(self.bin_tmp, self.bin_final)
        os.replace(self.idx_tmp, self.idx_final)
        return {
            "sx": self.sx,
            "sy": self.sy,
            "name": shard_name(self.sx, self.sy),
            "tile_count": len(entries),
            "visible_pixels": visible_pixels,
            "stored_bytes": stored_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "data_file": self.bin_final.name,
            "index_file": self.idx_final.name,
        }


def write_store_manifest(root: Path, cfg: Config, label: str, shard_metas: list[dict]) -> dict:
    shard_metas = sorted([m for m in shard_metas if m], key=lambda s: (s["sx"], s["sy"]))
    manifest = {
        "format": "wplace-sparse-tile-store-v1",
        "label": label,
        "tile_size": cfg.tile_size,
        "tile_pixels": cfg.tile_pixels,
        "grid_tiles": cfg.grid_tiles,
        "shard_tiles": cfg.shard_tiles,
        "shard_count_axis": cfg.shard_count_axis,
        "transparent_index": 0,
        "palette_value_dtype": "uint8",
        "position_dtype": "uint32",
        "store_compression": cfg.store_compression,
        "tile_count": int(sum(int(s.get("tile_count", 0)) for s in shard_metas)),
        "visible_pixels": int(sum(int(s.get("visible_pixels", 0)) for s in shard_metas)),
        "stored_bytes": int(sum(int(s.get("stored_bytes", 0)) for s in shard_metas)),
        "uncompressed_bytes": int(sum(int(s.get("uncompressed_bytes", s.get("stored_bytes", 0))) for s in shard_metas)),
        "dense_fallback_bytes": cfg.dense_fallback_bytes,
        "shards": shard_metas,
    }
    atomic_write_json(store_manifest_path(root), manifest)
    return manifest
