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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage


STRUCT_CACHE: dict[int, np.ndarray] = {}


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
    errors: int = 0
    elapsed_ms: int = 0
    components_total: int = 0
    components_saved: int = 0
    pixels_painted: int = 0
    phase_ms: dict = field(default_factory=dict)


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


def build_mosaic(tiles_dir, x, y, tile_size, tile_cache, use_shared_cache=False):
    """中央タイル(x,y) + 周囲8タイルを3x3に結合して返す。"""
    big = np.zeros((tile_size * 3, tile_size * 3, 4), dtype=np.uint8)
    big_mask = np.zeros((tile_size * 3, tile_size * 3), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            p = neighbor_path(tiles_dir, x + dx, y + dy)
            if not p.exists():
                continue
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
    use_cache=True, use_shared_cache=False, skip_existing=False,
):
    start = perf_counter()
    t_load = 0.0
    t_mosaic = 0.0
    t_label = 0.0
    t_save = 0.0
    result = TileResult(x=x, y=y)
    central_path = neighbor_path(tiles_dir, x, y)
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
    # central tileのサイズ確定
    h, w = central[0].shape[:2]
    if h != tile_size or w != tile_size:
        # 想定外サイズの場合は使う
        tile_size = h

    t0 = perf_counter()
    big, big_mask = build_mosaic(
        tiles_dir, x, y, tile_size,
        tile_cache if tile_cache is not None else {},
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
    # 各成分ごとに処理
    for label_id in range(1, n + 1):
        comp = labeled == label_id
        # 元の塗られたピクセルのみで囲む
        original = comp & big_mask
        if not original.any():
            continue
        ys, xs = np.where(original)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        x0, x1 = int(xs.min()), int(xs.max() + 1)

        # 中心が中央タイル内にあるものだけ保存(タイル境界での重複防止)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        if not (cy0 <= cy < cy1 and cx0 <= cx < cx1):
            result.skipped_center += 1
            continue

        # ノイズ除去
        pixels = int(original.sum())
        if pixels < min_pixels:
            result.skipped_small += 1
            continue

        # 切り出し: バウンディングボックス内、成分外は透明にする
        crop = big[y0:y1, x0:x1].copy()
        crop_mask = original[y0:y1, x0:x1]
        crop[~crop_mask] = (0, 0, 0, 0)

        # ワールド座標(全体マップ上のピクセル位置)
        world_x = x * tile_size + (x0 - tile_size)
        world_y = y * tile_size + (y0 - tile_size)
        out_name = f"art_x{world_x}_y{world_y}_w{x1 - x0}_h{y1 - y0}.png"
        out_path = out_dir / out_name
        if skip_existing and out_path.exists():
            result.skipped_existing += 1
            continue
        t1 = perf_counter()
        Image.fromarray(crop, "RGBA").save(out_path)
        t_save += perf_counter() - t1
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
    return result


def process_tile_worker(task):
    """ProcessPoolExecutor向け: シリアライズしやすい引数を受け取る。"""
    tiles_dir_s, x, y, dilate, min_pixels, out_dir_s, tile_size, use_cache, skip_existing = task
    try:
        return process_tile(
            Path(tiles_dir_s), x, y, dilate, min_pixels, Path(out_dir_s), tile_size,
            use_cache=use_cache,
            use_shared_cache=False,
            skip_existing=skip_existing,
        )
    except Exception:
        return TileResult(x=x, y=y, errors=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="wplace_xyz/11", help="downloaded tiles directory")
    p.add_argument("--out", default="artworks", help="output directory for cropped artworks")
    p.add_argument("--dilate", type=int, default=3,
                   help="px dilation radius for grouping nearby pixels (default 3)")
    p.add_argument("--min-pixels", type=int, default=16,
                   help="minimum painted pixels to count as an artwork (filters noise)")
    p.add_argument("--tile-size", type=int, default=1000, help="tile pixel size (default 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="number of worker processes (default: os.cpu_count(), 1 = serial)")
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
    args = p.parse_args()

    tiles_dir = Path(args.tiles)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    tasks = []
    for tf in tile_files:
        try:
            x = int(tf.parent.name)
            y = int(tf.stem)
        except ValueError:
            continue
        if (x, y) in done_tasks:
            continue
        tasks.append(
            (str(tiles_dir), x, y, args.dilate, args.min_pixels, str(out_dir), args.tile_size, not args.no_cache, args.skip_existing)
        )
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
            "elapsed_ms": tile_result.elapsed_ms,
            "components_total": tile_result.components_total,
            "components_saved": tile_result.components_saved,
            "pixels_painted": tile_result.pixels_painted,
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
            last_report_ts = now

    total = TileResult()
    failed_tiles = []
    slow_tiles = []
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            _, x, y, *_ = task
            try:
                tile_result = process_tile(
                    Path(task[0]), task[1], task[2], task[3], task[4], Path(task[5]), task[6],
                    use_cache=task[7],
                    use_shared_cache=(not args.no_cache),
                    skip_existing=task[8],
                )
                total.saved += tile_result.saved
                total.skipped_existing += tile_result.skipped_existing
                total.skipped_small += tile_result.skipped_small
                total.skipped_center += tile_result.skipped_center
                total.skipped_empty += tile_result.skipped_empty
                total.errors += tile_result.errors
                total.elapsed_ms += tile_result.elapsed_ms
                total.components_total += tile_result.components_total
                total.components_saved += tile_result.components_saved
                total.pixels_painted += tile_result.pixels_painted
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
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_tile_worker, task): (task[1], task[2])
                for task in tasks
            }
            for i, fut in enumerate(as_completed(futures), 1):
                x, y = futures[fut]
                try:
                    tile_result = fut.result()
                    total.saved += tile_result.saved
                    total.skipped_existing += tile_result.skipped_existing
                    total.skipped_small += tile_result.skipped_small
                    total.skipped_center += tile_result.skipped_center
                    total.skipped_empty += tile_result.skipped_empty
                    total.errors += tile_result.errors
                    total.elapsed_ms += tile_result.elapsed_ms
                    total.components_total += tile_result.components_total
                    total.components_saved += tile_result.components_saved
                    total.pixels_painted += tile_result.pixels_painted
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

    if tasks:
        maybe_report_progress(len(tasks), len(tasks), force=True)

    wall_elapsed_s = max(perf_counter() - start_ts, 1e-9)
    throughput = len(tasks) / wall_elapsed_s

    print(f"\nDone. Saved {total.saved} artworks to {out_dir}/")
    print(
        "Breakdown: "
        f"skipped_existing={total.skipped_existing}, "
        f"skipped_small={total.skipped_small}, "
        f"skipped_center={total.skipped_center}, "
        f"skipped_empty={total.skipped_empty}, "
        f"errors={total.errors}, "
        f"elapsed_ms={total.elapsed_ms}, "
        f"components_total={total.components_total}, "
        f"components_saved={total.components_saved}, "
        f"pixels_painted={total.pixels_painted}"
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
            "errors": total.errors,
            "elapsed_ms": total.elapsed_ms,
            "components_total": total.components_total,
            "components_saved": total.components_saved,
            "pixels_painted": total.pixels_painted,
            "failed_tiles": failed_tiles,
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as fp:
            json.dump(stats_payload, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
