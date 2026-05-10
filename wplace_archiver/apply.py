from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import Config
from .errors import ApplyError
from .records import merge_sparse_overlay
from .shard_store import AtomicSparseShardWriter, SparseTileRef, SparseTileStore, load_sparse_record_from_root, write_store_manifest
from .utils import atomic_write_json, load_json, shard_index_path, shard_name, store_manifest_path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _apply_checkpoint_path(cfg: Config, tag: str) -> Path:
    return cfg.store_root / ".apply_shards" / f"{tag}.json"


def _load_apply_checkpoint(cfg: Config, tag: str) -> dict[str, Any]:
    return load_json(
        _apply_checkpoint_path(cfg, tag),
        {
            "format": "wplace-apply-shard-checkpoint-v1",
            "tag": tag,
            "completed_shards": [],
            "failed_shards": {},
        },
    )


def _save_apply_checkpoint(cfg: Config, tag: str, checkpoint: dict[str, Any]) -> None:
    checkpoint["format"] = "wplace-apply-shard-checkpoint-v1"
    checkpoint["tag"] = tag
    checkpoint["updated_at"] = _now()
    atomic_write_json(_apply_checkpoint_path(cfg, tag), checkpoint)


def _mark_checkpoint_completed(cfg: Config, tag: str, checkpoint: dict[str, Any], shard: str) -> None:
    completed = set(str(s) for s in checkpoint.get("completed_shards", []))
    completed.add(shard)
    checkpoint["completed_shards"] = sorted(completed)
    failed = dict(checkpoint.get("failed_shards", {}))
    failed.pop(shard, None)
    checkpoint["failed_shards"] = failed
    _save_apply_checkpoint(cfg, tag, checkpoint)


def _mark_checkpoint_failed(cfg: Config, tag: str, checkpoint: dict[str, Any], shard: str, stage: str, exc: BaseException) -> None:
    failed = dict(checkpoint.get("failed_shards", {}))
    failed[shard] = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "failed_at": _now(),
    }
    checkpoint["failed_shards"] = failed
    _save_apply_checkpoint(cfg, tag, checkpoint)


def _shard_meta_from_index(root: Path, cfg: Config, sx: int, sy: int) -> dict | None:
    idx_path = shard_index_path(root, sx, sy)
    if not idx_path.exists():
        return None
    idx = load_json(idx_path, None)
    if idx is None:
        return None
    name = shard_name(sx, sy)
    return {
        "sx": sx,
        "sy": sy,
        "name": name,
        "tile_count": int(idx.get("tile_count", len(idx.get("tiles", [])))),
        "visible_pixels": int(idx.get("visible_pixels", 0)),
        "stored_bytes": int(idx.get("stored_bytes", 0)),
        "uncompressed_bytes": int(idx.get("uncompressed_bytes", 0)),
        "data_file": idx.get("data_file", f"{name}.bin"),
        "index_file": idx_path.name,
    }


def _shard_weight_from_index(root: Path, cfg: Config, sx: int, sy: int) -> int:
    idx_path = shard_index_path(root, sx, sy)
    if not idx_path.exists():
        return 0
    idx = load_json(idx_path, None)
    if idx is None:
        return 0
    stored_bytes = idx.get("stored_bytes")
    if stored_bytes is not None:
        return int(stored_bytes)
    tile_count = idx.get("tile_count")
    if tile_count is not None:
        return int(tile_count)
    return 0


def load_shard_tiles_as_dict(root: Path, cfg: Config, sx: int, sy: int) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    idx_path = shard_index_path(root, sx, sy)
    if not idx_path.exists():
        return {}
    store = SparseTileStore(root, cfg)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ref in store.iter_refs(sx, sy):
        out[(ref.x, ref.y)] = load_sparse_record_from_root(root, ref, cfg)
    return out


def _chunk_tile_refs(refs: list[SparseTileRef], target_bytes: int) -> list[list[SparseTileRef]]:
    if not refs:
        return []
    if target_bytes <= 0:
        return [refs]
    chunks: list[list[SparseTileRef]] = []
    current: list[SparseTileRef] = []
    current_bytes = 0
    for ref in refs:
        # Prefer uncompressed size estimate because decoding is the memory-heavy step.
        ref_bytes = int(ref.uncompressed_size or ref.size)
        if current and current_bytes + ref_bytes > target_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(ref)
        current_bytes += ref_bytes
    if current:
        chunks.append(current)
    return chunks


def apply_one_overlay_shard(state_root: Path, overlay_root: Path, cfg: Config, sx: int, sy: int) -> dict | None:
    state_store = SparseTileStore(state_root, cfg)
    overlay_store = SparseTileStore(overlay_root, cfg)

    state_refs = list(state_store.iter_refs(sx, sy))
    overlay_refs = list(overlay_store.iter_refs(sx, sy))

    # Keep full shard state in one dictionary for final atomic write, but avoid
    # expanding all overlay records at once.
    state_tiles: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ref in state_refs:
        state_tiles[(ref.x, ref.y)] = load_sparse_record_from_root(state_root, ref, cfg)

    chunk_target_bytes = max(1, int(getattr(cfg, "io_buffer_bytes", 32 * 1024 * 1024)))
    overlay_chunks = _chunk_tile_refs(overlay_refs, chunk_target_bytes)
    for chunk in overlay_chunks:
        for ref in chunk:
            key = (ref.x, ref.y)
            new_pos, new_val = load_sparse_record_from_root(overlay_root, ref, cfg)
            old_pos, old_val = state_tiles.get(key, (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint8)))
            state_tiles[key] = merge_sparse_overlay(old_pos, old_val, new_pos, new_val)

    writer = AtomicSparseShardWriter(state_root, cfg, sx, sy, label="rolling-state")
    meta = writer.write(state_tiles)
    if meta is None:
        return None
    # Keep worker return intentionally tiny. Full tile entries stay in the index
    # JSON written by AtomicSparseShardWriter.
    return {
        "sx": int(meta["sx"]),
        "sy": int(meta["sy"]),
        "name": str(meta["name"]),
        "tile_count": int(meta.get("tile_count", 0)),
        "visible_pixels": int(meta.get("visible_pixels", 0)),
        "stored_bytes": int(meta.get("stored_bytes", 0)),
        "uncompressed_bytes": int(meta.get("uncompressed_bytes", 0)),
        "data_file": str(meta.get("data_file", "")),
        "index_file": str(meta.get("index_file", "")),
    }


def _apply_worker(args):
    state_root, overlay_root, cfg, sx, sy = args
    started = time.time()
    meta = apply_one_overlay_shard(state_root, overlay_root, cfg, sx, sy)
    if meta:
        meta["elapsed_sec"] = round(time.time() - started, 3)
    return meta


def _cfg_to_jsonable(cfg: Config) -> dict[str, Any]:
    data = asdict(cfg)
    for key in ["download_dir", "store_root", "xyz_output_dir", "fixed_palette_path"]:
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data


def _cfg_from_jsonable(data: dict[str, Any]) -> Config:
    for key in ["download_dir", "store_root", "xyz_output_dir", "fixed_palette_path"]:
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return Config(**data)


def _isolated_apply_worker(cfg_path: Path, state_root: Path, overlay_root: Path, sx: int, sy: int) -> dict | None:
    cmd = [
        sys.executable,
        "-m",
        "wplace_archiver.apply",
        "--worker",
        "--cfg-json",
        str(cfg_path),
        "--state-root",
        str(state_root),
        "--overlay-root",
        str(overlay_root),
        "--sx",
        str(sx),
        "--sy",
        str(sy),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise ApplyError(
            f"isolated apply worker failed for {shard_name(sx, sy)}: "
            f"returncode={proc.returncode}; stdout_tail={stdout[-1000:]!r}; stderr_tail={stderr[-2000:]!r}"
        )
    if not stdout:
        return None
    try:
        return json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        raise ApplyError(f"isolated worker returned non-JSON for {shard_name(sx, sy)}: {stdout[-2000:]!r}") from exc


def _run_isolated_worker_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cfg-json", required=True)
    ap.add_argument("--state-root", required=True)
    ap.add_argument("--overlay-root", required=True)
    ap.add_argument("--sx", type=int, required=True)
    ap.add_argument("--sy", type=int, required=True)
    args = ap.parse_args(argv)
    cfg = _cfg_from_jsonable(load_json(Path(args.cfg_json), {}))
    meta = apply_one_overlay_shard(Path(args.state_root), Path(args.overlay_root), cfg, args.sx, args.sy)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_progress_mode(tasks) -> tuple[int | float, bool]:
    weights = [t[5] for t in tasks]
    if any(w is None for w in weights):
        return len(tasks), False
    return sum(float(w) for w in weights), True


def _apply_tasks_sequential(tag: str, cfg: Config, tasks, checkpoint, shard_metas, stats):
    total, weighted = _resolve_progress_mode(tasks)
    with tqdm(total=total, desc=f"apply {tag}", unit="shard") as pbar:
        for t in tasks:
            _state_root, _overlay_root, _cfg, sx, sy, shard_weight = t
            sname = shard_name(sx, sy)
            try:
                meta = _apply_worker((_state_root, _overlay_root, _cfg, sx, sy))
                if meta:
                    shard_metas[meta["name"]] = meta
                _mark_checkpoint_completed(cfg, tag, checkpoint, sname)
                stats["applied_shards"] += 1
                delta = float(shard_weight) if weighted else 1
                pbar.update(delta)
                pbar.set_postfix(shard=sname, weight=delta)
            except Exception as exc:
                _mark_checkpoint_failed(cfg, tag, checkpoint, sname, "apply_shard", exc)
                raise


def _apply_tasks_isolated(tag: str, cfg: Config, tasks, checkpoint, shard_metas, stats):
    cfg_path = _apply_checkpoint_path(cfg, tag).with_suffix(".config.json")
    atomic_write_json(cfg_path, _cfg_to_jsonable(cfg))
    try:
        total, weighted = _resolve_progress_mode(tasks)
        with tqdm(total=total, desc=f"apply {tag}", unit="shard") as pbar:
            for t in tasks:
                state_root, overlay_root, _cfg, sx, sy, shard_weight = t
                sname = shard_name(sx, sy)
                try:
                    meta = _isolated_apply_worker(cfg_path, state_root, overlay_root, sx, sy)
                    if meta:
                        shard_metas[meta["name"]] = meta
                    _mark_checkpoint_completed(cfg, tag, checkpoint, sname)
                    stats["applied_shards"] += 1
                    delta = float(shard_weight) if weighted else 1
                    pbar.update(delta)
                    pbar.set_postfix(shard=sname, weight=delta)
                except Exception as exc:
                    _mark_checkpoint_failed(cfg, tag, checkpoint, sname, "isolated_apply_shard", exc)
                    raise
    finally:
        cfg_path.unlink(missing_ok=True)


def _apply_tasks_executor(tag: str, cfg: Config, tasks, checkpoint, shard_metas, stats):
    Executor = ProcessPoolExecutor if cfg.apply_executor == "process" else ThreadPoolExecutor
    kwargs: dict[str, Any] = {"max_workers": max(1, cfg.apply_workers)}
    if cfg.apply_executor == "process" and cfg.apply_max_tasks_per_child and cfg.apply_max_tasks_per_child > 0:
        kwargs["max_tasks_per_child"] = int(cfg.apply_max_tasks_per_child)

    total, weighted = _resolve_progress_mode(tasks)
    with Executor(**kwargs) as ex, tqdm(total=total, desc=f"apply {tag}", unit="shard") as pbar:
        future_to_shard = {}
        for t in tasks:
            _state_root, _overlay_root, _cfg, sx, sy, shard_weight = t
            fut = ex.submit(_apply_worker, (_state_root, _overlay_root, _cfg, sx, sy))
            future_to_shard[fut] = (sx, sy, shard_name(sx, sy), shard_weight)

        for fut in as_completed(future_to_shard):
            sx, sy, sname, shard_weight = future_to_shard[fut]
            try:
                meta = fut.result()
                if meta:
                    shard_metas[meta["name"]] = meta
                _mark_checkpoint_completed(cfg, tag, checkpoint, sname)
                stats["applied_shards"] += 1
                delta = float(shard_weight) if weighted else 1
                pbar.update(delta)
                pbar.set_postfix(shard=sname, weight=delta)
            except Exception as exc:
                _mark_checkpoint_failed(cfg, tag, checkpoint, sname, f"{cfg.apply_executor}_apply_shard", exc)
                raise


def apply_tag_store_to_state(tag: str, cfg: Config) -> dict:
    overlay_root = cfg.tags_root / tag
    if not (overlay_root / "manifest.json").exists():
        raise ApplyError(f"tag store missing or incomplete: {overlay_root}")

    if cfg.apply_executor not in ("thread", "process", "sequential", "isolated-process"):
        raise ApplyError("WPLACE_APPLY_EXECUTOR must be one of: thread, process, sequential, isolated-process")

    overlay = SparseTileStore(overlay_root, cfg)
    overlay_shards = overlay.shard_ids()
    existing_manifest = load_json(store_manifest_path(cfg.state_root), {"shards": []})
    shard_metas = {s["name"]: s for s in existing_manifest.get("shards", [])}

    cfg.state_root.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_apply_checkpoint(cfg, tag)
    completed = set(str(s) for s in checkpoint.get("completed_shards", []))

    tasks = []
    skipped = 0
    for sx, sy in overlay_shards:
        sname = shard_name(sx, sy)
        if sname in completed:
            meta = _shard_meta_from_index(cfg.state_root, cfg, sx, sy)
            if meta is not None:
                shard_metas[sname] = meta
                skipped += 1
                continue
            # checkpoint is stale; redo shard
            completed.discard(sname)
        overlay_meta = _shard_meta_from_index(overlay_root, cfg, sx, sy)
        shard_weight = None
        if overlay_meta is not None:
            shard_weight = overlay_meta.get("visible_pixels") or overlay_meta.get("tile_count")
        tasks.append((cfg.state_root, overlay_root, cfg, sx, sy, shard_weight))

    shard_weights = {
        shard_name(sx, sy): _shard_weight_from_index(overlay_root, cfg, sx, sy)
        for sx, sy in overlay_shards
    }
    tasks.sort(key=lambda t: shard_weights.get(shard_name(t[3], t[4]), 0), reverse=True)

    checkpoint["completed_shards"] = sorted(completed)
    _save_apply_checkpoint(cfg, tag, checkpoint)

    stats = {
        "tag": tag,
        "executor": cfg.apply_executor,
        "apply_workers": cfg.apply_workers,
        "apply_max_tasks_per_child": cfg.apply_max_tasks_per_child,
        "overlay_shards": len(overlay_shards),
        "skipped_checkpoint_shards": skipped,
        "pending_shards": len(tasks),
        "applied_shards": 0,
    }

    if tasks:
        if cfg.apply_executor == "sequential":
            _apply_tasks_sequential(tag, cfg, tasks, checkpoint, shard_metas, stats)
        elif cfg.apply_executor == "isolated-process":
            _apply_tasks_isolated(tag, cfg, tasks, checkpoint, shard_metas, stats)
        elif len(tasks) <= 1:
            _apply_tasks_sequential(tag, cfg, tasks, checkpoint, shard_metas, stats)
        else:
            _apply_tasks_executor(tag, cfg, tasks, checkpoint, shard_metas, stats)

    manifest = write_store_manifest(cfg.state_root, cfg, "rolling-state", list(shard_metas.values()))
    stats.update({
        "state_tiles": manifest["tile_count"],
        "state_visible_pixels": manifest["visible_pixels"],
        "state_stored_bytes": manifest["stored_bytes"],
        "state_uncompressed_bytes": manifest.get("uncompressed_bytes", 0),
        "completed_shards_total": len(set(checkpoint.get("completed_shards", []))),
    })
    cfg.stats_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.stats_root / f"apply_{tag}.json", stats)

    # Tag-level pipeline state is the source of truth for "applied". The shard
    # checkpoint is retained as successful diagnostic evidence rather than
    # deleted immediately.
    checkpoint["completed"] = True
    checkpoint["completed_at"] = _now()
    _save_apply_checkpoint(cfg, tag, checkpoint)
    return stats


def delete_tag_store_if_needed(tag: str, cfg: Config) -> None:
    if cfg.keep_tag_stores:
        return
    shutil.rmtree(cfg.tags_root / tag, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(_run_isolated_worker_cli())
