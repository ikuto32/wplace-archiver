"""
wplace.live の全タイルをダウンロードする。

機能:
  - tqdm によるプログレスバー(成功/失敗/レート制限の内訳付き)
  - HTTP 429 で Retry-After を尊重して再試行(全ワーカーをグローバルにポーズ)
  - 5xx/ネットワークエラーは指数バックオフで再試行
  - 既存ファイルを自動スキップ → 中断後の再開はコマンド再実行だけでOK
  - Ctrl+C で安全に中断可能

使い方:
    python download_tiles.py                              # 全範囲
    python download_tiles.py --x 1000,1100 --y 600,700    # 特定範囲だけ
    python download_tiles.py --rate 5 --concurrency 10
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

BASE_URL = "https://backend.wplace.live/files/s0/tiles/{x}/{y}.png"
USER_AGENT = "wplace-archive-script/1.0 (personal archive use)"


class RateController:
    """秒あたりN個のトークンで全体ペースを制御し、429時は全員ポーズさせる。"""

    def __init__(self, rate: float):
        self.delay = 1.0 / rate
        self.lock = asyncio.Lock()
        self.next_time = 0.0
        self.pause_until = 0.0  # monotonic 時刻

    async def acquire(self):
        # グローバルポーズ中なら待機(必要なら複数回ループ)
        while True:
            now = time.monotonic()
            if now < self.pause_until:
                await asyncio.sleep(self.pause_until - now)
                continue
            async with self.lock:
                now = time.monotonic()
                if now < self.pause_until:
                    continue
                wait = max(0.0, self.next_time - now)
                self.next_time = max(now, self.next_time) + self.delay
            if wait > 0:
                await asyncio.sleep(wait)
            return

    def pause_for(self, seconds: float):
        """全ワーカーをseconds秒間ポーズ(累積ではなく最大値)。"""
        until = time.monotonic() + seconds
        if until > self.pause_until:
            self.pause_until = until


def parse_retry_after(value: str) -> float:
    """Retry-After ヘッダーを秒数に変換。秒数 or HTTP-date のどちらにも対応。"""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            dt = parsedate_to_datetime(value)
            now = datetime.now(timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return 30.0


async def download_tile(session, x, y, output_dir, stats, rate_ctl,
                        max_retries, failed_log, pbar):
    out_path = output_dir / str(x) / f"{y}.png"
    if out_path.exists():
        stats["skipped"] += 1
        pbar.update(1)
        return

    url = BASE_URL.format(x=x, y=y)
    last_error = None

    for attempt in range(max_retries + 1):
        await rate_ctl.acquire()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    # アトミック書き込み: tmp に書いて rename。中断で半端なファイルが残らない
                    tmp = out_path.with_suffix(".png.part")
                    tmp.write_bytes(data)
                    tmp.replace(out_path)
                    stats["saved"] += 1
                    stats["bytes"] += len(data)
                    pbar.update(1)
                    return

                if resp.status == 404:
                    stats["empty"] += 1
                    pbar.update(1)
                    return

                if resp.status == 429:
                    stats["rate_limited"] += 1
                    wait = parse_retry_after(resp.headers.get("Retry-After", ""))
                    if wait <= 0:
                        wait = min(120.0, 5.0 * (2 ** attempt))  # 5,10,20,40,80,120
                    rate_ctl.pause_for(wait)
                    pbar.write(f"[429] {x},{y}: pausing all workers for {wait:.1f}s "
                               f"(attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait)
                    last_error = "http_429"
                    continue

                if 500 <= resp.status < 600:
                    last_error = f"http_{resp.status}"
                    await asyncio.sleep(min(30.0, 2 ** attempt))
                    continue

                # 4xx の他のもの: リトライしない
                stats["error"] += 1
                failed_log.write(f"{x},{y},http_{resp.status}\n")
                failed_log.flush()
                pbar.update(1)
                return

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = type(e).__name__
            if attempt == max_retries:
                break
            await asyncio.sleep(min(30.0, 2 ** attempt))

    # ここまで来たらリトライ上限
    stats["error"] += 1
    failed_log.write(f"{x},{y},{last_error or 'max_retries'}\n")
    failed_log.flush()
    pbar.update(1)


async def main(x_range, y_range, output_dir, rate, max_retries, concurrency):
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log_path = output_dir / "failed_tiles.log"

    stats = {"saved": 0, "skipped": 0, "empty": 0,
             "error": 0, "rate_limited": 0, "bytes": 0}

    rate_ctl = RateController(rate)
    headers = {"User-Agent": USER_AGENT}
    connector = aiohttp.TCPConnector(limit=concurrency)
    sem = asyncio.Semaphore(concurrency)

    total = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])

    # 既存ファイル数を事前にカウント(参考用)
    print(f"Scanning existing files in {output_dir} ...")
    existing = 0
    for x in range(*x_range):
        d = output_dir / str(x)
        if d.is_dir():
            for y in range(*y_range):
                if (d / f"{y}.png").exists():
                    existing += 1
    remaining = total - existing
    print(f"Total tiles : {total}")
    print(f"Already have: {existing}  (will be skipped instantly)")
    print(f"Remaining   : {remaining}")
    print(f"Rate: {rate} req/s, concurrency: {concurrency}, max retries: {max_retries}")
    print()

    async def worker(x, y, session, pbar, failed_log):
        async with sem:
            await download_tile(session, x, y, output_dir, stats,
                                rate_ctl, max_retries, failed_log, pbar)

    bar_fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} "
               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    with open(failed_log_path, "a") as failed_log:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            with tqdm(total=total, desc="tiles", unit="tile",
                      bar_format=bar_fmt, smoothing=0.05) as pbar:

                # 統計を定期的に postfix に流す
                async def update_postfix():
                    while True:
                        await asyncio.sleep(1)
                        pbar.set_postfix(
                            saved=stats["saved"],
                            empty=stats["empty"],
                            skip=stats["skipped"],
                            err=stats["error"],
                            rl=stats["rate_limited"],
                            refresh=False,
                        )

                postfix_task = asyncio.create_task(update_postfix())
                tasks = [
                    asyncio.create_task(worker(x, y, session, pbar, failed_log))
                    for x in range(*x_range)
                    for y in range(*y_range)
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    postfix_task.cancel()
                    try:
                        await postfix_task
                    except asyncio.CancelledError:
                        pass

    print("\n=== Done ===")
    print(f"Saved        : {stats['saved']}")
    print(f"Empty (404)  : {stats['empty']}")
    print(f"Skipped      : {stats['skipped']}")
    print(f"Rate limited : {stats['rate_limited']}  (429 responses, including retried)")
    print(f"Errors       : {stats['error']}")
    print(f"Total bytes  : {stats['bytes'] / 1e9:.2f} GB")
    if stats["error"]:
        print(f"\nFailed tiles logged to: {failed_log_path}")
        print("再実行すれば成功済みは自動スキップされ、失敗分のみ再取得されます。")


def parse_range(s, default_max):
    if "," in s:
        lo, hi = s.split(",")
        return int(lo), int(hi)
    return 0, default_max


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download wplace.live tiles with progress bar, 429 handling, and resume"
    )
    p.add_argument("--x", default="0,2048", help="x range, e.g. 0,2048")
    p.add_argument("--y", default="0,2048", help="y range, e.g. 0,2048")
    p.add_argument("--out", default="tiles", help="output directory")
    p.add_argument("--rate", type=float, default=5.0, help="requests per second (be polite)")
    p.add_argument("--concurrency", type=int, default=10,
                   help="max concurrent in-flight requests")
    p.add_argument("--max-retries", type=int, default=5,
                   help="retry budget per tile for 429/5xx/network errors")
    args = p.parse_args()

    x_range = parse_range(args.x, 2048)
    y_range = parse_range(args.y, 2048)

    try:
        asyncio.run(main(x_range, y_range, Path(args.out),
                         args.rate, args.max_retries, args.concurrency))
    except KeyboardInterrupt:
        print("\n\n中断しました。同じコマンドを再実行すれば続きから再開します。")
        sys.exit(130)