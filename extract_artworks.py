"""
ダウンロード済みタイルから、個々のドット絵を切り出して保存する。

ロジック:
  1. 各タイルで「塗られたピクセル(alpha>0)」をマスクとして抽出
  2. タイル境界をまたぐ絵を救うため、対象タイル + 周囲8タイルの3x3を結合
  3. 近接ピクセルを同じ絵としてまとめるため、マスクを少し膨張(dilate)
  4. 連結成分(scipy.ndimage.label)を抽出、バウンディングボックスで切り出し
  5. 重複を避けるため、絵の中心が中央タイル内にある成分だけ保存

使い方:
    python extract_artworks.py --tiles tiles --out artworks
    python extract_artworks.py --dilate 3 --min-pixels 16
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


STRUCT_CACHE: dict[int, np.ndarray] = {}
WORKER_AVAILABLE_TILES = None


class InterProcessLock:
    """簡易的なプロセス間ロック。lockファイルを排他的に作成して制御する。"""

    def __init__(self, lock_path: Path, poll_interval: float = 0.05) -> None:
        self.lock_path = lock_path
        self.poll_interval = poll_interval
        self.fd = None

    def __enter__(self):
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                sleep(self.poll_interval)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


def allocate_output_path(out_dir: Path, out_name: str, max_files_per_dir: int, skip_existing: bool = False) -> Path | None:
    """
    保存先ディレクトリをmax_files_per_dir件ごとに分割して返す。
    skip_existing=True かつ out_name が既存なら None を返す。
    ルート配下に artworks_part_00001 のような連番ディレクトリを作る。
    """
    meta_path = out_dir / ".dir_split_state.json"
    lock_path = out_dir / ".dir_split_state.lock"
    with InterProcessLock(lock_path):
        if skip_existing and any(out_dir.glob(f"**/{out_name}")):
            return None

        if max_files_per_dir <= 0:
            return out_dir / out_name
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fp:
                state = json.load(fp)
        else:
            state = {"shard": 0, "count": 0}

        shard = int(state.get("shard", 0))
        count = int(state.get("count", 0))
        if count >= max_files_per_dir:
            shard += 1
            count = 0

        shard_dir = out_dir if shard == 0 else out_dir / f"artworks_part_{shard:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        state["shard"] = shard
        state["count"] = count + 1
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=False)

    return shard_dir / out_name


def get_struct(dilate):
    struct = STRUCT_CACHE.get(dilate)
    if struct is None:
        struct = np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=bool)
        STRUCT_CACHE[dilate] = struct
    return struct


@dataclass
class TileResult:
    x: int = 0
    y: int = 0
    saved: int = 0
    skipped_existing: int = 0
    skipped_small: int = 0
    skipped_center: int = 0
    skipped_empty: int = 0
    skipped_monochrome: int = 0
    errors: int = 0
    elapsed_ms: int = 0
    components_total: int = 0
    components_saved: int = 0
    pixels_painted: int = 0
    phase_ms: dict = field(default_factory=dict)
    save_queue_depth: int = 0
    queued_outputs: int = 0


def _load_tile_impl(path):
    """RGBA配列とアルファ>0のboolマスクを返す。読めなければNone。"""
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    arr.setflags(write=False)
    mask = arr[..., 3] > 0
    mask.setflags(write=False)
    return arr, mask


@lru_cache(maxsize=8192)
def _load_tile_cached(path_str):
    return _load_tile_impl(Path(path_str))


def load_tile(path, use_shared_cache=False):
    if use_shared_cache:
        return _load_tile_cached(str(path))
    return _load_tile_impl(path)


def neighbor_path(tiles_dir, x, y):
    return tiles_dir / str(x) / f"{y}.png"


def build_mosaic(tiles_dir, x, y, tile_size, tile_cache, available_tiles, use_shared_cache=False):
    """中央タイル(x,y) + 周囲8タイルを3x3に結合して返す。"""
    big = np.zeros((tile_size * 3, tile_size * 3, 4), dtype=np.uint8)
    big_mask = np.zeros((tile_size * 3, tile_size * 3), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor = (x + dx, y + dy)
            if neighbor not in available_tiles:
                continue
            p = neighbor_path(tiles_dir, neighbor[0], neighbor[1])
            t = tile_cache.get(p)
            if t is None:
                t = load_tile(p, use_shared_cache=use_shared_cache)
                tile_cache[p] = t
            if t is None:
                continue
            arr, m = t
            # サイズが想定外ならスキップ
            if arr.shape[0] != tile_size or arr.shape[1] != tile_size:
                continue
            yo = (dy + 1) * tile_size
            xo = (dx + 1) * tile_size
            big[yo:yo + tile_size, xo:xo + tile_size] = arr
            big_mask[yo:yo + tile_size, xo:xo + tile_size] = m
    return big, big_mask


def process_tile(
    tiles_dir, x, y, dilate, min_pixels, out_dir, tile_size,
    use_cache=True, use_shared_cache=False, skip_existing=False, available_tiles=None,
    collect_outputs=False, max_files_per_dir=100000, existing_names=None,
):
    start = perf_counter()
    t_load = 0.0
    t_mosaic = 0.0
    t_label = 0.0
    t_save = 0.0
    result = TileResult(x=x, y=y)
    central_path = neighbor_path(tiles_dir, x, y)
    if available_tiles is None:
        available_tiles = {(x, y)}
    tile_cache = {} if use_cache else None

    t0 = perf_counter()
    if tile_cache is not None:
        central = tile_cache.get(central_path)
        if central is None:
            central = load_tile(central_path, use_shared_cache=use_shared_cache)
            tile_cache[central_path] = central
    else:
        central = load_tile(central_path, use_shared_cache=False)
    t_load += perf_counter() - t0
    if central is None or not central[1].any():
        result.skipped_empty = 1
        result.elapsed_ms = int((perf_counter() - start) * 1000)
        return result
    # central tileのサイズ確認（正方形タイル前提）
    h, w = central[0].shape[:2]
    if h != w:
        print(f"[ERROR] non-square tile at ({x}, {y}): actual size={w}x{h}")
        result.errors = 1
        result.elapsed_ms = int((perf_counter() - start) * 1000)
        return result
    if h != tile_size:
        print(f"[WARN] tile size mismatch at ({x}, {y}): expected={tile_size}, actual={w}x{h}")

    t0 = perf_counter()
    big, big_mask = build_mosaic(
        tiles_dir, x, y, tile_size,
        tile_cache if tile_cache is not None else {},
        available_tiles,
        use_shared_cache=use_shared_cache,
    )
    t_mosaic += perf_counter() - t0
    if not big_mask.any():
        result.skipped_empty = 1
        result.elapsed_ms = int((perf_counter() - start) * 1000)
        return result

    # 近接ピクセルを束ねる
    t0 = perf_counter()
    if dilate > 0:
        struct = get_struct(dilate)
        merged = ndimage.binary_dilation(big_mask, structure=struct)
    else:
        merged = big_mask

    labeled, n = ndimage.label(merged)
    t_label += perf_counter() - t0
    result.components_total = int(n)
    if n == 0:
        result.skipped_empty = 1
        result.elapsed_ms = int((perf_counter() - start) * 1000)
        return result

    cy0, cx0 = tile_size, tile_size
    cy1, cx1 = tile_size * 2, tile_size * 2
    
    pending_outputs = []

    objects = ndimage.find_objects(labeled)
    # 各成分ごとに処理
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        local = (labeled[slc] == label_id)
        # 元の塗られたピクセルのみで囲む
        original_local = local & big_mask[slc]
        if not original_local.any():
            continue
        ys, xs = np.where(original_local)
        y0 = slc[0].start + int(ys.min())
        y1 = slc[0].start + int(ys.max() + 1)
        x0 = slc[1].start + int(xs.min())
        x1 = slc[1].start + int(xs.max() + 1)

        # 中心が中央タイル内にあるものだけ保存(タイル境界での重複防止)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        if not (cy0 <= cy < cy1 and cx0 <= cx < cx1):
            result.skipped_center += 1
            continue

        # ノイズ除去
        pixels = int(original_local.sum())
        if pixels < min_pixels:
            result.skipped_small += 1
            continue

        # 切り出し: バウンディングボックス内、成分外は透明にする
        crop = big[y0:y1, x0:x1].copy()
        crop_mask = original_local[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop[~crop_mask] = (0, 0, 0, 0)

        # 単色(塗られたピクセルが1色のみ)は保存しない
        painted_pixels = crop[crop_mask]
        if painted_pixels.shape[0] > 0 and np.unique(painted_pixels, axis=0).shape[0] <= 1:
            result.skipped_monochrome += 1
            continue

        # ワールド座標(全体マップ上のピクセル位置)
        world_x = x * tile_size + (x0 - tile_size)
        world_y = y * tile_size + (y0 - tile_size)
        out_name = f"art_x{world_x}_y{world_y}_w{x1 - x0}_h{y1 - y0}.png"

        if skip_existing and existing_names is not None and out_name in existing_names:
          out_path = allocate_output_path(
              out_dir, out_name, max_files_per_dir, skip_existing=skip_existing
          )
          if out_path is None:
              result.skipped_existing += 1
              continue
        if collect_outputs:
            pending_outputs.append((out_path, crop))
            result.queued_outputs += 1
        else:
            t1 = perf_counter()
            Image.fromarray(crop, "RGBA").save(out_path)
            t_save += perf_counter() - t1
      
        if existing_names is not None:
            existing_names.add(out_name)
            result.saved += 1
            result.components_saved += 1
        result.pixels_painted += pixels
    result.elapsed_ms = int((perf_counter() - start) * 1000)
    result.phase_ms = {
        "load": int(t_load * 1000),
        "mosaic": int(t_mosaic * 1000),
        "label": int(t_label * 1000),
        "save": int(t_save * 1000),
        "total": result.elapsed_ms,
    }
    if collect_outputs:
        return result, pending_outputs
    return result


def save_artwork(out_path, crop):
    t0 = perf_counter()
    Image.fromarray(crop, "RGBA").save(out_path)
    return int((perf_counter() - t0) * 1000)


def save_artwork_for_tile(tile_key, out_path, crop):
    return tile_key, save_artwork(out_path, crop)


def init_worker(available_tiles):
    global WORKER_AVAILABLE_TILES
    WORKER_AVAILABLE_TILES = available_tiles


def process_tile_worker(task):
    """ProcessPoolExecutor向け: シリアライズしやすい引数を受け取る。"""
    tiles_dir_s, x, y, dilate, min_pixels, out_dir_s, tile_size, use_cache, skip_existing, max_files_per_dir, existing_names = task
    try:
        return process_tile(
            Path(tiles_dir_s), x, y, dilate, min_pixels, Path(out_dir_s), tile_size,
            use_cache=use_cache,
            use_shared_cache=False,
            skip_existing=skip_existing,
            available_tiles=WORKER_AVAILABLE_TILES,
            max_files_per_dir=max_files_per_dir,
            existing_names=existing_names,
        )
    except Exception:
        return TileResult(x=x, y=y, errors=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="wplace_xyz/11", help="downloaded tiles directory")
    p.add_argument("--out", default="artworks", help="output directory for cropped artworks")
    p.add_argument("--dilate", type=int, default=3,
                   help="px dilation radius for grouping nearby pixels (default 3)")
    p.add_argument("--min-pixels", type=int, default=64,
                   help="minimum painted pixels to count as an artwork (filters noise)")
    p.add_argument("--tile-size", type=int, default=1000, help="tile pixel size (square tiles only, default 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="number of worker processes (default: os.cpu_count(), 1 = serial)")
    p.add_argument("--save-workers", type=int, default=1,
                   help="number of save worker threads (0 or 1 = synchronous save)")
    p.add_argument("--save-queue-max", type=int, default=64,
                   help="maximum number of queued save tasks for backpressure")
    p.add_argument("--no-cache", action="store_true",
                   help="disable tile cache (local and shared)")
    p.add_argument("--verbose", action="store_true",
                   help="print failed tile coordinates")
    p.add_argument("--stats-json", default=None,
                   help="write aggregated stats as JSON to this path")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip saving when the output file already exists")
    p.add_argument("--runlog-jsonl", default="artworks_runlog.jsonl",
                   help="append per-tile result records to JSONL log")
    p.add_argument("--resume-from-log", default=None,
                   help="path to JSONL runlog used to skip successful (x,y) tasks")
    p.add_argument("--progress-interval", type=float, default=1.0,
                   help="progress reporting interval in seconds (default 1.0)")
    p.add_argument("--max-files-per-dir", type=int, default=100000,
                   help="maximum files per output directory before creating a split directory (default 100000)")
    args = p.parse_args()

    tiles_dir = Path(args.tiles)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_names = None
    if args.skip_existing:
        existing_names = {path.name for path in out_dir.rglob("*.png")}
        print(f"Indexed {len(existing_names)} existing PNG files for --skip-existing")

    tile_files = sorted(tiles_dir.glob("*/*.png"))
    print(f"Found {len(tile_files)} tile files in {tiles_dir}")
    if not tile_files:
        print("No tiles found. Run download_tiles.py first.")
        return

    done_tasks = set()
    if args.resume_from_log:
        resume_path = Path(args.resume_from_log)
        if resume_path.exists():
            with resume_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(rec.get("errors", 1)) == 0 and "x" in rec and "y" in rec:
                        done_tasks.add((int(rec["x"]), int(rec["y"])))
        print(f"Loaded {len(done_tasks)} successful tasks from {args.resume_from_log}")

    available_tiles = set()
    task_coords = []
    for tf in tqdm(tile_files, desc="Processing tiles"):
        try:
            x = int(tf.parent.name)
            y = int(tf.stem)
        except ValueError:
            continue
        available_tiles.add((x, y))
        if (x, y) in done_tasks:
            continue
        task_coords.append((x, y))
    tasks = [
        (
            str(tiles_dir), x, y, args.dilate, args.min_pixels, str(out_dir),
            args.tile_size, not args.no_cache, args.skip_existing, args.max_files_per_dir, existing_names,
        )
        for x, y in task_coords
    ]
    print(f"Tasks to process: {len(tasks)}")

    runlog_path = Path(args.runlog_jsonl)
    runlog_path.parent.mkdir(parents=True, exist_ok=True)

    def append_runlog(tile_result):
        rec = {
            "x": tile_result.x,
            "y": tile_result.y,
            "saved": tile_result.saved,
            "errors": tile_result.errors,
            "skipped_existing": tile_result.skipped_existing,
            "skipped_monochrome": tile_result.skipped_monochrome,
            "elapsed_ms": tile_result.elapsed_ms,
            "components_total": tile_result.components_total,
            "components_saved": tile_result.components_saved,
            "pixels_painted": tile_result.pixels_painted,
            "save_queue_depth": tile_result.save_queue_depth,
            "queued_outputs": tile_result.queued_outputs,
            "phase_ms": tile_result.phase_ms,
        }
        with runlog_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    start_ts = perf_counter()
    last_report_ts = start_ts

    def report_progress(done, total_tasks, total_saved, total_errors, total_elapsed_ms):
        elapsed_s = max(perf_counter() - start_ts, 1e-9)
        speed = done / elapsed_s
        rate_pct = (done / total_tasks * 100.0) if total_tasks else 100.0
        eta_s = ((total_tasks - done) / speed) if speed > 0 else float("inf")
        eta_text = f"{eta_s:.1f}s" if eta_s != float("inf") else "N/A"
        print(
            f"[{rate_pct:6.2f}%] {done}/{total_tasks} | "
            f"speed={speed:.2f} tiles/s | ETA={eta_text} | "
            f"saved={total_saved} errors={total_errors} elapsed_ms={total_elapsed_ms}"
        )

    def maybe_report_progress(done, total_tasks, force=False):
        nonlocal last_report_ts
        now = perf_counter()
        if force or (now - last_report_ts) >= args.progress_interval:
            report_progress(done, total_tasks, total.saved, total.errors, total.elapsed_ms)
            refresh_progress(done)
            last_report_ts = now

    total = TileResult()
    failed_tiles = []
    slow_tiles = []
    total_save_completed_ms = 0
    save_workers = max(0, args.save_workers)
    async_save_enabled = save_workers > 1 and args.workers == 1
    if args.save_workers > 1 and args.workers != 1:
        print("[WARN] --save-workers > 1 is supported only with --workers 1; using synchronous save.")

    save_executor = ThreadPoolExecutor(max_workers=save_workers) if async_save_enabled else None
    inflight_saves = set()
    save_future_to_tile = {}

    progress_bar = tqdm(total=len(tasks), desc="Extracting artworks", unit="tile")

    def refresh_progress(done):
        progress_bar.n = done
        progress_bar.set_postfix(saved=total.saved, errors=total.errors, refresh=False)
        progress_bar.refresh()

    def drain_save_queue(block_until_room=False):
        nonlocal total_save_completed_ms
        while inflight_saves and (block_until_room or len(inflight_saves) >= args.save_queue_max):
            done, _ = wait(inflight_saves, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight_saves.remove(fut)
                tile_key = save_future_to_tile.pop(fut, None)
                try:
                    completed_tile_key, save_ms = fut.result()
                    total_save_completed_ms += save_ms
                    total.saved += 1
                    total.components_saved += 1
                    if completed_tile_key in async_tile_states:
                        async_tile_states[completed_tile_key]["saved"] += 1
                except Exception as e:
                    total.errors += 1
                    if tile_key in async_tile_states:
                        async_tile_states[tile_key]["errors"] += 1
                    print(f"[ERROR] async save failed: {e}")
                if tile_key in async_tile_states:
                    maybe_flush_async_runlog(tile_key)
            if not block_until_room:
                break

    def accumulate_tile_result(tile_result):
        total.saved += tile_result.saved
        total.skipped_existing += tile_result.skipped_existing
        total.skipped_small += tile_result.skipped_small
        total.skipped_center += tile_result.skipped_center
        total.skipped_empty += tile_result.skipped_empty
        total.skipped_monochrome += tile_result.skipped_monochrome
        total.errors += tile_result.errors
        total.elapsed_ms += tile_result.elapsed_ms
        total.components_total += tile_result.components_total
        total.components_saved += tile_result.components_saved
        total.pixels_painted += tile_result.pixels_painted

    async_tile_states = {}

    def maybe_flush_async_runlog(tile_key):
        state = async_tile_states.get(tile_key)
        if state is None or state["logged"]:
            return
        if (state["saved"] + state["errors"]) < state["expected"]:
            return
        tr = state["result"]
        tr.saved = state["saved"]
        tr.components_saved = state["saved"]
        tr.errors += state["errors"]
        append_runlog(tr)
        state["logged"] = True

    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            _, x, y, *_ = task
            try:

                if async_save_enabled:
                    tile_result, pending_outputs = process_tile(
                        Path(task[0]), task[1], task[2], task[3], task[4], Path(task[5]), task[6],
                        use_cache=task[7],
                        use_shared_cache=(not args.no_cache),
                        skip_existing=task[8],
                        available_tiles=available_tiles,
                        collect_outputs=True,
                        max_files_per_dir=task[9],
                        existing_names=task[10],
                    )
                    tile_key = (tile_result.x, tile_result.y)
                    async_tile_states[tile_key] = {
                        "result": tile_result,
                        "expected": tile_result.queued_outputs,
                        "saved": 0,
                        "errors": 0,
                        "logged": False,
                    }
                    for out_path, crop in pending_outputs:
                        drain_save_queue(block_until_room=False)
                        fut = save_executor.submit(save_artwork_for_tile, tile_key, out_path, crop)
                        inflight_saves.add(fut)
                        save_future_to_tile[fut] = tile_key
                    tile_result.save_queue_depth = len(inflight_saves)
                    accumulate_tile_result(tile_result)
                    maybe_flush_async_runlog(tile_key)
                else:
                    tile_result = process_tile(
                        Path(task[0]), task[1], task[2], task[3], task[4], Path(task[5]), task[6],
                        use_cache=task[7],
                        use_shared_cache=(not args.no_cache),
                        skip_existing=task[8],
                        available_tiles=available_tiles,
                        max_files_per_dir=task[9],
                        existing_names=task[10],
                    )
                    accumulate_tile_result(tile_result)
                    append_runlog(tile_result)

                slow_tiles.append(tile_result)
            except Exception as e:
                print(f"[ERROR] tile ({x}, {y}) failed: {e}")
                total.errors += 1
                failed_tiles.append((x, y))
                append_runlog(TileResult(x=x, y=y, errors=1))
            maybe_report_progress(i, len(tasks))
    else:
        workers = max(1, args.workers or 1)
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(frozenset(available_tiles),)) as executor:
            futures = {
                executor.submit(process_tile_worker, task): (task[1], task[2])
                for task in tasks
            }
            for i, fut in enumerate(as_completed(futures), 1):
                x, y = futures[fut]
                try:
                    tile_result = fut.result()
                    accumulate_tile_result(tile_result)
                    append_runlog(tile_result)
                    slow_tiles.append(tile_result)
                    if tile_result.errors > 0:
                        failed_tiles.append((x, y))
                except Exception as e:
                    print(f"[ERROR] tile ({x}, {y}) failed: {e}")
                    total.errors += 1
                    failed_tiles.append((x, y))
                    append_runlog(TileResult(x=x, y=y, errors=1))
                maybe_report_progress(i, len(tasks))

    if save_executor is not None:
        drain_save_queue(block_until_room=True)
        save_executor.shutdown(wait=True)
        for tile_key in list(async_tile_states):
            maybe_flush_async_runlog(tile_key)

    if tasks:
        maybe_report_progress(len(tasks), len(tasks), force=True)
    progress_bar.close()

    wall_elapsed_s = max(perf_counter() - start_ts, 1e-9)
    throughput = len(tasks) / wall_elapsed_s

    print(f"\nDone. Saved {total.saved} artworks to {out_dir}/")
    print(
        "Breakdown: "
        f"skipped_existing={total.skipped_existing}, "
        f"skipped_small={total.skipped_small}, "
        f"skipped_center={total.skipped_center}, "
        f"skipped_empty={total.skipped_empty}, "
        f"skipped_monochrome={total.skipped_monochrome}, "
        f"errors={total.errors}, "
        f"elapsed_ms={total.elapsed_ms}, "
        f"components_total={total.components_total}, "
        f"components_saved={total.components_saved}, "
        f"pixels_painted={total.pixels_painted}"
        f", save_completed_ms={total_save_completed_ms} "
        f"throughput_tiles_per_sec={throughput:.2f}"
    )
    if args.verbose and slow_tiles:
        top_n = min(10, len(slow_tiles))
        print(f"Top {top_n} slowest tiles by elapsed_ms:")
        for tr in sorted(slow_tiles, key=lambda r: r.elapsed_ms, reverse=True)[:top_n]:
            print(f"  ({tr.x}, {tr.y}) elapsed_ms={tr.elapsed_ms} saved={tr.saved} errors={tr.errors}")
    if args.verbose and failed_tiles:
        print("Failed tiles:")
        for x, y in failed_tiles:
            print(f"  ({x}, {y})")

    if args.stats_json:
        stats_payload = {
            "tiles_total": len(tasks),
            "saved": total.saved,
            "skipped_existing": total.skipped_existing,
            "skipped_small": total.skipped_small,
            "skipped_center": total.skipped_center,
            "skipped_empty": total.skipped_empty,
            "skipped_monochrome": total.skipped_monochrome,
            "errors": total.errors,
            "elapsed_ms": total.elapsed_ms,
            "components_total": total.components_total,
            "components_saved": total.components_saved,
            "pixels_painted": total.pixels_painted,
            "save_completed_ms": total_save_completed_ms,
            "failed_tiles": failed_tiles,
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as fp:
            json.dump(stats_payload, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
