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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_tile(path):
    """RGBA配列とアルファ>0のboolマスクを返す。読めなければNone。"""
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    return arr, arr[..., 3] > 0


def neighbor_path(tiles_dir, x, y):
    return tiles_dir / str(x) / f"{y}.png"


def build_mosaic(tiles_dir, x, y, tile_size):
    """中央タイル(x,y) + 周囲8タイルを3x3に結合して返す。"""
    big = np.zeros((tile_size * 3, tile_size * 3, 4), dtype=np.uint8)
    big_mask = np.zeros((tile_size * 3, tile_size * 3), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            p = neighbor_path(tiles_dir, x + dx, y + dy)
            if not p.exists():
                continue
            t = load_tile(p)
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


def process_tile(tiles_dir, x, y, dilate, min_pixels, out_dir, tile_size):
    central_path = neighbor_path(tiles_dir, x, y)
    central = load_tile(central_path)
    if central is None or not central[1].any():
        return 0
    # central tileのサイズ確定
    h, w = central[0].shape[:2]
    if h != tile_size or w != tile_size:
        # 想定外サイズの場合は使う
        tile_size = h

    big, big_mask = build_mosaic(tiles_dir, x, y, tile_size)
    if not big_mask.any():
        return 0

    # 近接ピクセルを束ねる
    if dilate > 0:
        struct = np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=bool)
        merged = ndimage.binary_dilation(big_mask, structure=struct)
    else:
        merged = big_mask

    labeled, n = ndimage.label(merged)
    if n == 0:
        return 0

    cy0, cx0 = tile_size, tile_size
    cy1, cx1 = tile_size * 2, tile_size * 2
    saved = 0

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
            continue

        # ノイズ除去
        if int(original.sum()) < min_pixels:
            continue

        # 切り出し: バウンディングボックス内、成分外は透明にする
        crop = big[y0:y1, x0:x1].copy()
        crop_mask = original[y0:y1, x0:x1]
        crop[~crop_mask] = (0, 0, 0, 0)

        # ワールド座標(全体マップ上のピクセル位置)
        world_x = x * tile_size + (x0 - tile_size)
        world_y = y * tile_size + (y0 - tile_size)
        out_name = f"art_x{world_x}_y{world_y}_w{x1 - x0}_h{y1 - y0}.png"
        Image.fromarray(crop, "RGBA").save(out_dir / out_name)
        saved += 1
    return saved


def process_tile_worker(task):
    """ProcessPoolExecutor向け: シリアライズしやすい引数を受け取る。"""
    tiles_dir_s, x, y, dilate, min_pixels, out_dir_s, tile_size = task
    return process_tile(Path(tiles_dir_s), x, y, dilate, min_pixels, Path(out_dir_s), tile_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="tiles", help="downloaded tiles directory")
    p.add_argument("--out", default="artworks", help="output directory for cropped artworks")
    p.add_argument("--dilate", type=int, default=3,
                   help="px dilation radius for grouping nearby pixels (default 3)")
    p.add_argument("--min-pixels", type=int, default=16,
                   help="minimum painted pixels to count as an artwork (filters noise)")
    p.add_argument("--tile-size", type=int, default=1000, help="tile pixel size (default 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="number of worker processes (default: os.cpu_count(), 1 = serial)")
    args = p.parse_args()

    tiles_dir = Path(args.tiles)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(tiles_dir.glob("*/*.png"))
    print(f"Found {len(tile_files)} tile files in {tiles_dir}")
    if not tile_files:
        print("No tiles found. Run download_tiles.py first.")
        return

    tasks = []
    for tf in tile_files:
        try:
            x = int(tf.parent.name)
            y = int(tf.stem)
        except ValueError:
            continue
        tasks.append((str(tiles_dir), x, y, args.dilate, args.min_pixels, str(out_dir), args.tile_size))

    total = 0
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            _, x, y, *_ = task
            try:
                total += process_tile_worker(task)
            except Exception as e:
                print(f"[ERROR] tile ({x}, {y}) failed: {e}")
            if i % 200 == 0:
                print(f"[{i}/{len(tasks)}] artworks saved so far: {total}")
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
                    total += fut.result()
                except Exception as e:
                    print(f"[ERROR] tile ({x}, {y}) failed: {e}")
                if i % 200 == 0:
                    print(f"[{i}/{len(tasks)}] artworks saved so far: {total}")

    print(f"\nDone. Saved {total} artworks to {out_dir}/")


if __name__ == "__main__":
    main()
