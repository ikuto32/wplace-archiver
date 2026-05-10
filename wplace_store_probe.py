#!/usr/bin/env python3
"""
wplace_store_probe.py

Standalone validator/prober for wplace_archiver intermediate shard stores.

Purpose
-------
When apply fails with BrokenProcessPool, the real error is often hidden inside a
worker process. This script isolates shard processing in a fresh subprocess per
shard, records the last processed shard, and reports the exact shard that fails.

It does not modify the store.

Supported record formats
------------------------
- sparse-u32-u8-v1
- dense-u8-v1

Supported payload compression
-----------------------------
- zstd
- none
- missing compression field is treated as none for backward compatibility.

Python
------
- Python 3.14+: uses stdlib compression.zstd
- Python <=3.13: uses backports.zstd if installed
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


SPARSE_ENCODING = "sparse-u32-u8-v1"
DENSE_ENCODING = "dense-u8-v1"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _import_zstd():
    try:
        from compression import zstd  # type: ignore
        return zstd, "compression.zstd"
    except Exception:
        try:
            import backports.zstd as zstd  # type: ignore
            return zstd, "backports.zstd"
        except Exception as exc:
            raise RuntimeError(
                "zstd support is unavailable. Use Python 3.14+ or install backports.zstd:\n"
                "  uv add backports.zstd\n"
            ) from exc


def _zstd_decompress(data: bytes) -> bytes:
    zstd, _name = _import_zstd()
    if hasattr(zstd, "decompress"):
        return zstd.decompress(data)
    if hasattr(zstd, "open"):
        with zstd.open(io.BytesIO(data), "rb") as f:
            return f.read()
    if hasattr(zstd, "ZstdFile"):
        with zstd.ZstdFile(io.BytesIO(data), "rb") as f:
            return f.read()
    raise RuntimeError("zstd module does not expose decompress/open/ZstdFile")


def _zstd_compress(data: bytes, level: int = 3) -> bytes:
    zstd, _name = _import_zstd()
    if hasattr(zstd, "compress"):
        try:
            return zstd.compress(data, level=level)
        except TypeError:
            return zstd.compress(data)
    if hasattr(zstd, "open"):
        bio = io.BytesIO()
        try:
            with zstd.open(bio, "wb", level=level) as f:
                f.write(data)
        except TypeError:
            with zstd.open(bio, "wb") as f:
                f.write(data)
        return bio.getvalue()
    if hasattr(zstd, "ZstdFile"):
        bio = io.BytesIO()
        with zstd.ZstdFile(bio, "wb", level=level) as f:
            f.write(data)
        return bio.getvalue()
    raise RuntimeError("zstd module does not expose compress/open/ZstdFile")


def _shard_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".index.json"):
        return name[: -len(".index.json")]
    return path.stem


def _index_path(root: Path, shard: str) -> Path:
    return root / f"{shard}.index.json"


def _bin_path(root: Path, idx: dict[str, Any]) -> Path:
    return root / str(idx["data_file"])


def _read_record_payload(root: Path, idx: dict[str, Any], entry: dict[str, Any]) -> bytes:
    data_path = _bin_path(root, idx)
    offset = int(entry["offset"])
    size = int(entry["size"])
    if not data_path.exists():
        raise FileNotFoundError(f"data_file not found: {data_path}")

    file_size = data_path.stat().st_size
    if offset < 0 or size < 0 or offset + size > file_size:
        raise ValueError(
            f"record range out of file: file={data_path} file_size={file_size} "
            f"offset={offset} size={size}"
        )

    with data_path.open("rb") as f:
        f.seek(offset)
        raw = f.read(size)

    if len(raw) != size:
        raise IOError(f"short read: {data_path} offset={offset} expected={size} got={len(raw)}")

    compression = str(entry.get("compression", "none")).lower()
    if compression in ("", "none", "raw"):
        payload = raw
    elif compression == "zstd":
        payload = _zstd_decompress(raw)
    else:
        raise ValueError(f"unsupported record compression: {compression!r}")

    uncomp = entry.get("uncompressed_size")
    if uncomp is not None and int(uncomp) != len(payload):
        raise ValueError(
            f"uncompressed_size mismatch for tile {entry.get('x')},{entry.get('y')}: "
            f"index={uncomp} actual={len(payload)} compression={compression}"
        )
    return payload


def _expected_payload_size(entry: dict[str, Any], tile_pixels: int) -> int:
    encoding = str(entry.get("encoding", SPARSE_ENCODING))
    count = int(entry.get("count", 0))
    if encoding == SPARSE_ENCODING:
        return count * 5
    if encoding == DENSE_ENCODING:
        return tile_pixels
    raise ValueError(f"unknown encoding: {encoding!r}")


def _validate_record_payload(payload: bytes, entry: dict[str, Any], tile_pixels: int, max_palette_index: int) -> dict[str, Any]:
    encoding = str(entry.get("encoding", SPARSE_ENCODING))
    count = int(entry.get("count", 0))
    expected = _expected_payload_size(entry, tile_pixels)

    if len(payload) != expected:
        raise ValueError(
            f"payload size mismatch for tile {entry.get('x')},{entry.get('y')}: "
            f"encoding={encoding} count={count} expected={expected} got={len(payload)}"
        )

    result: dict[str, Any] = {
        "encoding": encoding,
        "count": count,
        "payload_bytes": len(payload),
        "max_position": None,
        "max_value": None,
        "nonzero": None,
    }

    if encoding == SPARSE_ENCODING:
        pos_bytes = count * 4
        positions = np.frombuffer(payload[:pos_bytes], dtype=np.uint32, count=count)
        values = np.frombuffer(payload[pos_bytes:], dtype=np.uint8, count=count)

        if len(positions) != count or len(values) != count:
            raise ValueError("sparse array length mismatch")

        if count:
            max_pos = int(positions.max())
            min_pos = int(positions.min())
            max_val = int(values.max())
            min_val = int(values.min())
            if min_pos < 0 or max_pos >= tile_pixels:
                raise ValueError(
                    f"sparse position out of range: tile={entry.get('x')},{entry.get('y')} "
                    f"min={min_pos} max={max_pos} tile_pixels={tile_pixels}"
                )
            if min_val == 0:
                raise ValueError(f"sparse record contains transparent value 0: tile={entry.get('x')},{entry.get('y')}")
            if max_val > max_palette_index:
                raise ValueError(
                    f"palette value out of range: tile={entry.get('x')},{entry.get('y')} "
                    f"max_value={max_val} max_palette_index={max_palette_index}"
                )
            result.update(max_position=max_pos, max_value=max_val)
        return result

    if encoding == DENSE_ENCODING:
        arr = np.frombuffer(payload, dtype=np.uint8, count=tile_pixels)
        if len(arr) != tile_pixels:
            raise ValueError("dense array length mismatch")
        max_val = int(arr.max(initial=0))
        nonzero = int(np.count_nonzero(arr))
        indexed_count = int(entry.get("count", nonzero))
        if max_val > max_palette_index:
            raise ValueError(
                f"dense palette value out of range: tile={entry.get('x')},{entry.get('y')} "
                f"max_value={max_val} max_palette_index={max_palette_index}"
            )
        if indexed_count != nonzero:
            raise ValueError(
                f"dense count mismatch: tile={entry.get('x')},{entry.get('y')} "
                f"index_count={indexed_count} actual_nonzero={nonzero}"
            )
        result.update(max_value=max_val, nonzero=nonzero)
        return result

    raise AssertionError("unreachable")


@dataclass
class LoadedTile:
    x: int
    y: int
    positions: np.ndarray
    values: np.ndarray


def _payload_to_sparse(payload: bytes, entry: dict[str, Any], tile_pixels: int) -> LoadedTile:
    encoding = str(entry.get("encoding", SPARSE_ENCODING))
    count = int(entry.get("count", 0))
    x = int(entry["x"])
    y = int(entry["y"])

    if encoding == SPARSE_ENCODING:
        pos_bytes = count * 4
        positions = np.frombuffer(payload[:pos_bytes], dtype=np.uint32, count=count).copy()
        values = np.frombuffer(payload[pos_bytes:], dtype=np.uint8, count=count).copy()
        return LoadedTile(x=x, y=y, positions=positions, values=values)

    if encoding == DENSE_ENCODING:
        arr = np.frombuffer(payload, dtype=np.uint8, count=tile_pixels)
        positions = np.flatnonzero(arr != 0).astype(np.uint32, copy=False)
        values = arr[positions.astype(np.intp, copy=False)].astype(np.uint8, copy=False)
        return LoadedTile(x=x, y=y, positions=positions, values=values)

    raise ValueError(f"unknown encoding: {encoding!r}")


def _merge_sparse(old: LoadedTile | None, overlay: LoadedTile, tile_pixels: int) -> LoadedTile:
    if old is None or old.positions.size == 0:
        return overlay

    dense = np.zeros(tile_pixels, dtype=np.uint8)
    dense[old.positions.astype(np.intp, copy=False)] = old.values
    dense[overlay.positions.astype(np.intp, copy=False)] = overlay.values
    positions = np.flatnonzero(dense != 0).astype(np.uint32, copy=False)
    values = dense[positions.astype(np.intp, copy=False)].astype(np.uint8, copy=False)
    return LoadedTile(x=overlay.x, y=overlay.y, positions=positions, values=values)


def _load_shard_tiles(root: Path, shard: str, tile_pixels: int, max_palette_index: int, validate_only: bool) -> dict[tuple[int, int], LoadedTile]:
    idx_path = _index_path(root, shard)
    if not idx_path.exists():
        return {}
    idx = _load_json(idx_path)
    tiles: dict[tuple[int, int], LoadedTile] = {}

    for entry in idx.get("tiles", []):
        payload = _read_record_payload(root, idx, entry)
        _validate_record_payload(payload, entry, tile_pixels, max_palette_index)
        if not validate_only:
            tile = _payload_to_sparse(payload, entry, tile_pixels)
            tiles[(tile.x, tile.y)] = tile

    return tiles


def _validate_shard_root(root: Path, shard: str, tile_pixels: int, max_palette_index: int, exercise_compress: bool, zstd_level: int) -> dict[str, Any]:
    idx_path = _index_path(root, shard)
    if not idx_path.exists():
        return {"root": str(root), "shard": shard, "exists": False, "tiles": 0}

    idx = _load_json(idx_path)
    entries = idx.get("tiles", [])
    data_path = _bin_path(root, idx)

    compression_counts: dict[str, int] = {}
    enc_counts: dict[str, int] = {}
    total_payload = 0
    total_stored = 0
    max_count = 0

    for entry in entries:
        compression = str(entry.get("compression", "none")).lower()
        encoding = str(entry.get("encoding", SPARSE_ENCODING))
        compression_counts[compression] = compression_counts.get(compression, 0) + 1
        enc_counts[encoding] = enc_counts.get(encoding, 0) + 1
        total_stored += int(entry.get("size", 0))
        max_count = max(max_count, int(entry.get("count", 0)))

        payload = _read_record_payload(root, idx, entry)
        _validate_record_payload(payload, entry, tile_pixels, max_palette_index)
        total_payload += len(payload)

        if exercise_compress:
            _ = _zstd_compress(payload, level=zstd_level)

    return {
        "root": str(root),
        "shard": shard,
        "exists": True,
        "data_file": str(data_path),
        "tiles": len(entries),
        "compression_counts": compression_counts,
        "encoding_counts": enc_counts,
        "stored_bytes": total_stored,
        "uncompressed_payload_bytes": total_payload,
        "max_record_count": max_count,
    }


def child_main(args: argparse.Namespace) -> int:
    store_root = Path(args.store_root)
    tag = args.tag
    shard = args.worker_shard
    tile_pixels = args.tile_size * args.tile_size

    overlay_root = store_root / "tags" / tag
    state_root = store_root / "state"

    started = time.time()
    result: dict[str, Any] = {
        "ok": False,
        "stage": "start",
        "shard": shard,
        "started_at": _now(),
        "pid": os.getpid(),
    }

    try:
        result["stage"] = "validate_overlay"
        overlay_stats = _validate_shard_root(
            overlay_root, shard, tile_pixels, args.max_palette_index,
            exercise_compress=args.exercise_compress,
            zstd_level=args.zstd_level,
        )
        result["overlay"] = overlay_stats

        result["stage"] = "validate_state"
        state_stats = _validate_shard_root(
            state_root, shard, tile_pixels, args.max_palette_index,
            exercise_compress=args.exercise_compress,
            zstd_level=args.zstd_level,
        )
        result["state"] = state_stats

        if args.mode in ("dry-run-apply", "both"):
            result["stage"] = "load_state_tiles"
            state_tiles = _load_shard_tiles(state_root, shard, tile_pixels, args.max_palette_index, validate_only=False)

            result["stage"] = "load_overlay_tiles"
            overlay_tiles = _load_shard_tiles(overlay_root, shard, tile_pixels, args.max_palette_index, validate_only=False)

            result["stage"] = "merge"
            merged: dict[tuple[int, int], LoadedTile] = dict(state_tiles)
            for key, overlay_tile in overlay_tiles.items():
                merged[key] = _merge_sparse(merged.get(key), overlay_tile, tile_pixels)

            merged_tiles = len(merged)
            merged_visible = int(sum(t.positions.size for t in merged.values()))
            result["dry_run_apply"] = {
                "state_tiles": len(state_tiles),
                "overlay_tiles": len(overlay_tiles),
                "merged_tiles": merged_tiles,
                "merged_visible_pixels": merged_visible,
            }

            if args.exercise_compress:
                result["stage"] = "compress_merged_payloads"
                compressed_bytes = 0
                for tile in merged.values():
                    count = int(tile.positions.size)
                    if count * 5 < tile_pixels:
                        payload = tile.positions.astype(np.uint32, copy=False).tobytes(order="C") + tile.values.astype(np.uint8, copy=False).tobytes(order="C")
                    else:
                        dense = np.zeros(tile_pixels, dtype=np.uint8)
                        dense[tile.positions.astype(np.intp, copy=False)] = tile.values.astype(np.uint8, copy=False)
                        payload = dense.tobytes(order="C")
                    compressed_bytes += len(_zstd_compress(payload, level=args.zstd_level))
                result["dry_run_apply"]["compressed_merged_bytes"] = compressed_bytes

        result["stage"] = "done"
        result["ok"] = True
        result["elapsed_sec"] = round(time.time() - started, 3)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as exc:
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["elapsed_sec"] = round(time.time() - started, 3)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


def _discover_shards(store_root: Path, tag: str) -> list[str]:
    roots = [store_root / "tags" / tag, store_root / "state"]
    shards: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("s????_????.index.json"):
            shards.add(_shard_name_from_path(p))
    return sorted(shards)


def _parse_shard_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    out = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


def _read_report_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("ok") is True and row.get("shard"):
                done.add(str(row["shard"]))
    return done


def parent_main(args: argparse.Namespace) -> int:
    store_root = Path(args.store_root)
    report = Path(args.out)
    report.parent.mkdir(parents=True, exist_ok=True)

    shards = _parse_shard_list(args.shards)
    if shards is None:
        shards = _discover_shards(store_root, args.tag)

    if args.only_missing_from_report:
        done = _read_report_done(report)
        shards = [s for s in shards if s not in done]

    if args.limit is not None:
        shards = shards[: args.limit]

    if not shards:
        print("No shards to process.")
        return 0

    print(f"store_root={store_root}")
    print(f"tag={args.tag}")
    print(f"shards={len(shards)}")
    print(f"mode={args.mode}")
    print(f"report={report}")
    print("Processing one shard per subprocess to isolate native crashes.")

    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--store-root", str(store_root),
        "--tag", args.tag,
        "--mode", args.mode,
        "--tile-size", str(args.tile_size),
        "--max-palette-index", str(args.max_palette_index),
        "--zstd-level", str(args.zstd_level),
    ]
    if args.exercise_compress:
        base_cmd.append("--exercise-compress")

    failures = 0
    for shard in tqdm(shards, desc="probe shards", unit="shard"):
        cmd = base_cmd + ["--worker-shard", shard]
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec if args.timeout_sec > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            row = {
                "ok": False,
                "shard": shard,
                "stage": "subprocess_timeout",
                "error_type": "TimeoutExpired",
                "error": str(exc),
                "elapsed_sec": round(time.time() - started, 3),
                "timestamp": _now(),
            }
            _append_jsonl(report, row)
            failures += 1
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            if args.stop_on_fail:
                return 1
            continue

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        parsed: dict[str, Any] | None = None
        for candidate in [stdout.splitlines()[-1] if stdout else "", stderr.splitlines()[-1] if stderr else ""]:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                break
            except Exception:
                pass

        if parsed is None:
            parsed = {
                "ok": False,
                "shard": shard,
                "stage": "subprocess_no_json",
                "returncode": proc.returncode,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }

        parsed["returncode"] = proc.returncode
        parsed["timestamp"] = _now()
        _append_jsonl(report, parsed)

        if proc.returncode != 0 or not parsed.get("ok"):
            failures += 1
            print("\nFAILED SHARD:")
            print(json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True))
            if args.stop_on_fail:
                print(f"\nStopped on failing shard: {shard}")
                return 1

    summary = {
        "ok": failures == 0,
        "shards_processed": len(shards),
        "failures": failures,
        "report": str(report),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if failures == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Validate/probe wplace intermediate shard stores and isolate BrokenProcessPool-causing shards."
    )
    ap.add_argument("--store-root", default="wplace_sparse_store", help="Path to wplace_sparse_store")
    ap.add_argument("--tag", required=True, help="Tag to probe, e.g. world-2026-01-20T10-37-37.596Z")
    ap.add_argument("--mode", choices=["validate", "dry-run-apply", "both"], default="both")
    ap.add_argument("--tile-size", type=int, default=1000)
    ap.add_argument("--max-palette-index", type=int, default=64)
    ap.add_argument("--out", default="wplace_probe_report.jsonl")
    ap.add_argument("--shards", help="Comma-separated shard names, e.g. s0063_0061,s0063_0062")
    ap.add_argument("--limit", type=int, help="Limit number of shards processed")
    ap.add_argument("--timeout-sec", type=int, default=0, help="Per-shard subprocess timeout; 0 disables")
    ap.add_argument("--stop-on-fail", action="store_true", default=True, help="Stop at first failing shard")
    ap.add_argument("--no-stop-on-fail", dest="stop_on_fail", action="store_false")
    ap.add_argument("--only-missing-from-report", action="store_true", help="Skip shards already marked ok in report")
    ap.add_argument("--exercise-compress", action="store_true", help="Also exercise zstd compression for loaded payloads")
    ap.add_argument("--zstd-level", type=int, default=3)

    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-shard", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.worker:
        if not args.worker_shard:
            ap.error("--worker requires --worker-shard")
        return child_main(args)

    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
