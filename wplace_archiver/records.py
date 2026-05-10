from __future__ import annotations

import numpy as np

SPARSE_ENCODING = "sparse-u32-u8-v1"
DENSE_ENCODING = "dense-u8-v1"


def sparse_record_size(count: int) -> int:
    return int(count) * 5


def ensure_u32_positions(pos: np.ndarray) -> np.ndarray:
    if pos.dtype != np.uint32:
        pos = pos.astype(np.uint32, copy=False)
    if not pos.flags.c_contiguous:
        pos = np.ascontiguousarray(pos)
    return pos


def ensure_u8_values(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8:
        values = values.astype(np.uint8, copy=False)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    return values


def dense_record_from_sparse(positions: np.ndarray, values: np.ndarray, tile_pixels: int) -> np.ndarray:
    arr = np.zeros(tile_pixels, dtype=np.uint8)
    if positions.size:
        arr[positions.astype(np.intp, copy=False)] = values
    return arr


def sparse_from_payload(data: bytes, count: int, encoding: str, tile_pixels: int, dense_tile_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    if encoding == SPARSE_ENCODING:
        pos_bytes = count * 4
        expected = pos_bytes + count
        if expected != len(data):
            raise ValueError(f"bad sparse record size: expected={expected}, got={len(data)}")
        positions = np.frombuffer(data, dtype=np.uint32, count=count, offset=0)
        values = np.frombuffer(data, dtype=np.uint8, count=count, offset=pos_bytes)
        return positions.copy(), values.copy()
    if encoding == DENSE_ENCODING:
        if len(data) != dense_tile_bytes:
            raise ValueError(f"bad dense record size: expected={dense_tile_bytes}, got={len(data)}")
        idx = np.frombuffer(data, dtype=np.uint8, count=tile_pixels)
        positions = np.flatnonzero(idx != 0).astype(np.uint32, copy=False)
        values = idx[positions.astype(np.intp, copy=False)].astype(np.uint8, copy=False)
        return positions, values
    raise ValueError(f"unknown encoding: {encoding}")


def merge_sparse_overlay(old_pos: np.ndarray, old_val: np.ndarray, new_pos: np.ndarray, new_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge visible overlay pixels into visible state pixels."""
    if old_pos.size == 0:
        order = np.argsort(new_pos, kind="stable")
        return new_pos[order].astype(np.uint32, copy=False), new_val[order].astype(np.uint8, copy=False)
    if new_pos.size == 0:
        order = np.argsort(old_pos, kind="stable")
        return old_pos[order].astype(np.uint32, copy=False), old_val[order].astype(np.uint8, copy=False)
    # Dict is memory efficient enough per tile and avoids allocating dense 1M arrays for sparse tiles.
    merged = {int(p): int(v) for p, v in zip(old_pos, old_val)}
    merged.update({int(p): int(v) for p, v in zip(new_pos, new_val)})
    positions = np.fromiter(merged.keys(), dtype=np.uint32, count=len(merged))
    values = np.fromiter(merged.values(), dtype=np.uint8, count=len(merged))
    order = np.argsort(positions, kind="stable")
    return positions[order], values[order]
