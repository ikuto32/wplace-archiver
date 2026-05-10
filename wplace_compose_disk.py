"""
wplace アーカイブ取得・合成スクリプト (GPU 補助 + 高速 codec 版)。

このバージョンの追加最適化 (前版 RAM-cache 版からの差分):
  ■ PNG コーデックの差し替え (CPU 側、最大効果)
      - encode: fpng (libpng 比 ~20x) があれば自動使用。無ければ PIL。
      - decode: pyspng (PIL 比 ~3x) があれば自動使用。無ければ PIL。
  ■ GPU 補助 (任意)
      - VRAM 上に L1 タイルキャッシュ (既定 18 GB) を保持
      - alpha_composite と (増/消) ピクセル統計を GPU で実行
      - L2 = 既存 RAM キャッシュ (32 GB)、L3 = ディスク

GPU での PNG 直接 codec は 2026 年現在まともなライブラリが無いため (nvImageCodec も
PNG は内部で OpenCV/CPU フォールバック)、本コードでは GPU をコーデックには使わず
合成段とキャッシュにだけ用いている。本質的な codec 高速化は fpng/pyspng で行う。

依存はすべて optional (どれが無くてもフォールバック動作する):
  pip install fpng_py        # encode 高速化 (最大の効果)
  pip install py-fpng-nb     # 上の代替実装 (どちらか片方でよい)
  pip install pyspng         # decode 高速化
  pip install torch          # GPU 利用 (CUDA build を入れること)

環境変数:
  WPLACE_CACHE_GB     RAM キャッシュ予算 GB (既定 32)
  WPLACE_VRAM_GB      VRAM キャッシュ予算 GB (既定 18 / GPU 無効化は 0)
  WPLACE_WORKERS      合成スレッド数 (既定 = CPU 数)
  WPLACE_PREFETCH     ペンディング上限の倍率 (既定 4)
  WPLACE_DISABLE_GPU  1 を指定すると torch があっても GPU を使わない
"""
import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

import aiohttp
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# 任意の高速化バックエンド検出
# ============================================================
# --- decode: pyspng ---
try:
    import pyspng  # type: ignore
    _HAS_PYSPNG = True
except ImportError:
    _HAS_PYSPNG = False

# --- encode: fpng (2 系統対応) ---
_FPNG_KIND = None
try:
    import fpng_py  # type: ignore
    _FPNG_KIND = "fpng_py"
except ImportError:
    try:
        import fpng  # type: ignore
        if hasattr(fpng, "from_ndarray"):
            _FPNG_KIND = "py_fpng_nb"
    except ImportError:
        pass

# --- GPU: torch + CUDA ---
_HAS_TORCH_CUDA = False
torch = None  # type: ignore
if os.environ.get("WPLACE_DISABLE_GPU", "0") != "1":
    try:
        import torch  # type: ignore
        _HAS_TORCH_CUDA = bool(torch.cuda.is_available())
    except ImportError:
        torch = None  # type: ignore


# ============================================================
# 設定
# ============================================================
REPO = "murolem/wplace-archives"
DOWNLOAD_DIR = "./wplace_downloads"
OUTPUT_DIR = "./wplace_output"
INTERVAL_DAYS = 10
MAX_CONCURRENT_DOWNLOADS = 4
COMPOSITE_WORKERS = int(os.environ.get("WPLACE_WORKERS", str(os.cpu_count() or 4)))
TILE_CACHE_BYTES = int(os.environ.get("WPLACE_CACHE_GB", "64")) * 1024 ** 3
VRAM_CACHE_BYTES = int(os.environ.get("WPLACE_VRAM_GB", "18")) * 1024 ** 3
PREFETCH_FACTOR = int(os.environ.get("WPLACE_PREFETCH", "4"))
KEEP_ARCHIVES = False

PROCESSED_STATE = os.path.join(OUTPUT_DIR, ".processed_tags.json")
STATS_STATE = os.path.join(OUTPUT_DIR, ".tile_stats.json")


# ============================================================
# 永続状態
# ============================================================
def load_processed() -> set:
    if os.path.exists(PROCESSED_STATE):
        try:
            with open(PROCESSED_STATE, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_processed(processed: set) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = PROCESSED_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)
    os.replace(tmp, PROCESSED_STATE)


def load_stats() -> dict:
    if os.path.exists(STATS_STATE):
        try:
            with open(STATS_STATE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_stats(all_stats: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = STATS_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, STATS_STATE)


# ============================================================
# タグ列挙・HTTP・連結ストリーム (元コードと同じ)
# ============================================================
def get_all_tags():
    print("リポジトリから全タグ情報を取得しています (API制限回避)...")
    url = f"https://github.com/{REPO}.git"
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", url],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        print("エラー: 'git' コマンドが見つかりません。")
        raise SystemExit(1)
    tags = []
    for line in result.stdout.splitlines():
        if "refs/tags/world-" in line:
            tag = line.split("refs/tags/")[-1].replace("^{}", "")
            m = re.search(
                r"(\d{4})-(\d{2})-(\d{2})T(\d{2})[-_:](\d{2})[-_:](\d{2})", tag
            )
            if m:
                pub_date = datetime(*map(int, m.groups()))
                tags.append({"tag_name": tag, "published_at": pub_date})
    unique_tags = {t["tag_name"]: t for t in tags}.values()
    return sorted(unique_tags, key=lambda x: x["published_at"])


async def fetch_api_with_retry(session, url):
    while True:
        async with session.get(url) as resp:
            if resp.status in (403, 429):
                reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                now = int(time.time())
                wait_sec = max(reset_time - now, 60) if reset_time else 60
                tqdm.write(f"\n[API制限] {wait_sec}秒待機して再試行...")
                await asyncio.sleep(wait_sec)
                continue
            resp.raise_for_status()
            return await resp.json()


async def download_file(session, url, file_path, sem, position):
    async with sem:
        filename = os.path.basename(file_path)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                headers = {}
                file_size = 0
                mode = "wb"
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    headers["Range"] = f"bytes={file_size}-"
                    mode = "ab"
                async with session.get(url, headers=headers) as resp:
                    if resp.status in (403, 429):
                        wait_sec = int(resp.headers.get("Retry-After", 30))
                        tqdm.write(f"[{filename}] 429制限。{wait_sec}秒待機")
                        await asyncio.sleep(wait_sec)
                        continue
                    if resp.status == 416:
                        return file_path
                    if resp.status not in (200, 206):
                        resp.raise_for_status()
                    if resp.status == 200:
                        mode = "wb"
                        file_size = 0
                    total_size = int(resp.headers.get("content-length", 0)) + file_size
                    with tqdm(
                        total=total_size, initial=file_size, unit="iB",
                        unit_scale=True, desc=filename[:24],
                        position=position, leave=False,
                    ) as pbar:
                        with open(file_path, mode) as f:
                            async for chunk in resp.content.iter_chunked(1024 * 64):
                                f.write(chunk)
                                pbar.update(len(chunk))
                return file_path
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    tqdm.write(f"[{filename}] ダウンロード失敗: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)


class _ConcatReader(io.RawIOBase):
    def __init__(self, file_paths):
        self._paths = list(file_paths)
        self._idx = 0
        self._cur = None
        self._open_next()

    def _open_next(self):
        if self._cur is not None:
            self._cur.close()
            self._cur = None
        if self._idx < len(self._paths):
            self._cur = open(self._paths[self._idx], "rb")
            self._idx += 1

    def readable(self):
        return True

    def readinto(self, b):
        if self._cur is None:
            return 0
        n = self._cur.readinto(b)
        while n == 0:
            self._open_next()
            if self._cur is None:
                return 0
            n = self._cur.readinto(b)
        return n

    def close(self):
        if self._cur is not None:
            self._cur.close()
            self._cur = None
        super().close()


# ============================================================
# PNG コーデック層 (CPU 側、可能な限り高速ライブラリへ差替)
# ============================================================
def decode_rgba(png_bytes: bytes) -> np.ndarray:
    """PNG bytes → RGBA uint8 numpy array。"""
    if _HAS_PYSPNG:
        try:
            arr = pyspng.load(png_bytes)
            if arr.ndim == 2:
                rgb = np.stack([arr] * 3, axis=-1)
                alpha = np.full(arr.shape + (1,), 255, dtype=np.uint8)
                return np.concatenate([rgb, alpha], axis=-1)
            if arr.shape[-1] == 3:
                alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
                return np.concatenate([arr, alpha], axis=-1)
            return arr
        except Exception:
            pass

    with Image.open(io.BytesIO(png_bytes)) as im:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        return np.array(im, dtype=np.uint8)


def decode_rgba_from_path(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_rgba(f.read())


def encode_rgba(arr: np.ndarray) -> bytes:
    """RGBA uint8 numpy array → PNG bytes。
    fpng → PIL の優先順位。fpng は libpng 比 encode が ~20x、ファイルは 5-10% 大。
    """
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    if _FPNG_KIND == "py_fpng_nb":
        try:
            return fpng.from_ndarray(arr)  # type: ignore
        except Exception:
            pass
    elif _FPNG_KIND == "fpng_py":
        try:
            h, w = arr.shape[:2]
            return fpng_py.fpng_encode_image_to_memory(  # type: ignore
                arr.tobytes(), w, h, num_chans=4
            )
        except Exception:
            pass

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(
        buf, format="PNG", optimize=False, compress_level=1
    )
    return buf.getvalue()


# ============================================================
# RAM 上の LRU タイルキャッシュ (L2 として機能)
# ============================================================
class TileLRUCache:
    """合成済みタイル (numpy uint8 RGBA) を RAM に保持する LRU。"""
    def __init__(self, max_bytes: int):
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cur_bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        with self._lock:
            return len(self._d)

    @property
    def cur_bytes(self) -> int:
        with self._lock:
            return self._cur_bytes

    def get(self, key: str):
        with self._lock:
            arr = self._d.get(key)
            if arr is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return arr

    def put(self, key: str, arr: np.ndarray):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        nb = arr.nbytes
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._cur_bytes -= old.nbytes
            self._d[key] = arr
            self._cur_bytes += nb
            while self._cur_bytes > self._max_bytes and len(self._d) > 1:
                _, evicted = self._d.popitem(last=False)
                self._cur_bytes -= evicted.nbytes

    def reset_counters(self):
        with self._lock:
            self.hits = 0
            self.misses = 0


# ============================================================
# CPU 側の合成 + 統計 (PIL の alpha_composite を使う互換実装)
# ============================================================
def _cpu_composite_with_stats(base_arr, new_arr):
    """戻り値: (result_arr, added, disappeared, dim_mismatch)。"""
    new_alpha = new_arr[..., 3]

    if base_arr is None:
        return new_arr, int(np.count_nonzero(new_alpha)), 0, False

    if base_arr.shape != new_arr.shape:
        added = int(np.count_nonzero(new_alpha))
        disappeared = int(np.count_nonzero(base_arr[..., 3]))
        return new_arr, added, disappeared, True

    base_alpha = base_arr[..., 3]
    new_visible = new_alpha > 0
    base_visible = base_alpha > 0
    added = int(np.count_nonzero(new_visible & ~base_visible))
    disappeared = int(np.count_nonzero(~new_visible & base_visible))

    base_im = Image.fromarray(base_arr, mode="RGBA")
    new_im = Image.fromarray(new_arr, mode="RGBA")
    base_im.alpha_composite(new_im)
    return np.array(base_im, dtype=np.uint8), added, disappeared, False


# ============================================================
# GPU 補助コンポジタ (任意。VRAM L1 + alpha_composite + 統計を CUDA で)
# ============================================================
class GpuCompositor:
    """torch CUDA を用いた合成器。VRAM 上に L1 LRU を保持する。

    スレッド安全性は内部の Lock で保証 (CUDA を 1 度に 1 ワーカが使う形)。
    PIL の alpha_composite と数値的にはほぼ等価だが、float→uint8 丸めにより
    最下位ビットが 1 ずれるピクセルが理論上ありうる (実用上は無視できる)。
    """
    def __init__(self, vram_bytes: int):
        assert _HAS_TORCH_CUDA, "GPU not available"
        self._device = torch.device("cuda")  # type: ignore
        self._lock = threading.Lock()
        self._d: "OrderedDict[str, torch.Tensor]" = OrderedDict()  # type: ignore
        self._cur_bytes = 0
        self._max_bytes = vram_bytes
        self.hits = 0
        self.misses = 0

    def __len__(self):
        with self._lock:
            return len(self._d)

    @property
    def cur_bytes(self) -> int:
        with self._lock:
            return self._cur_bytes

    def reset_counters(self):
        with self._lock:
            self.hits = 0
            self.misses = 0

    def _vram_get(self, key):
        t = self._d.get(key)
        if t is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return t

    def _vram_put(self, key, t):
        nb = t.numel() * t.element_size()
        old = self._d.pop(key, None)
        if old is not None:
            self._cur_bytes -= old.numel() * old.element_size()
        self._d[key] = t
        self._cur_bytes += nb
        while self._cur_bytes > self._max_bytes and len(self._d) > 1:
            _, ev = self._d.popitem(last=False)
            self._cur_bytes -= ev.numel() * ev.element_size()

    @staticmethod
    def _alpha_composite_cuda(base_t, new_t):
        """base_t, new_t: (H,W,4) uint8 cuda。`over` 演算 uint8 を返す。"""
        bf = base_t.to(torch.float32) / 255.0  # type: ignore
        nf = new_t.to(torch.float32) / 255.0   # type: ignore
        a_n = nf[..., 3:4]
        a_b = bf[..., 3:4]
        out_a = a_n + a_b * (1.0 - a_n)
        denom = out_a.clamp(min=1e-6)
        out_rgb = (nf[..., :3] * a_n + bf[..., :3] * a_b * (1.0 - a_n)) / denom
        out_rgb = torch.where(out_a > 0, out_rgb, torch.zeros_like(out_rgb))  # type: ignore
        out = torch.cat([out_rgb, out_a], dim=-1)  # type: ignore
        return (out * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)  # type: ignore

    def composite(self, path, new_arr, base_arr_hint):
        """戻り値: (result_arr_cpu_numpy, added, disappeared, dim_mismatch)。"""
        with self._lock:
            base_t = self._vram_get(path)
            if base_t is None and base_arr_hint is not None:
                base_t = torch.from_numpy(base_arr_hint).to(  # type: ignore
                    self._device, non_blocking=True
                )

            new_t = torch.from_numpy(new_arr).to(self._device, non_blocking=True)  # type: ignore

            new_alpha = new_t[..., 3]
            if base_t is None:
                added = int((new_alpha > 0).sum().item())
                disappeared = 0
                result_t = new_t
                dim_mismatch = False
            elif base_t.shape != new_t.shape:
                added = int((new_alpha > 0).sum().item())
                disappeared = int((base_t[..., 3] > 0).sum().item())
                result_t = new_t
                dim_mismatch = True
            else:
                base_alpha = base_t[..., 3]
                new_vis = new_alpha > 0
                base_vis = base_alpha > 0
                added = int((new_vis & ~base_vis).sum().item())
                disappeared = int((~new_vis & base_vis).sum().item())
                result_t = self._alpha_composite_cuda(base_t, new_t)
                dim_mismatch = False

            self._vram_put(path, result_t)
            result_arr = result_t.detach().cpu().numpy()

        return result_arr, added, disappeared, dim_mismatch


# ============================================================
# 1 タイル処理 (CPU/GPU 両対応のディスパッチ)
# ============================================================
def _process_tile(out_path, png_bytes, ram_cache, gpu):
    """戻り値: (error_or_None, added, disappeared)。"""

    base_arr = ram_cache.get(out_path)
    if base_arr is None and os.path.exists(out_path):
        try:
            base_arr = decode_rgba_from_path(out_path)
        except Exception as e:
            return f"[base read failed] {out_path}: {e}", 0, 0

    try:
        new_arr = decode_rgba(png_bytes)
    except Exception as e:
        return f"[new decode failed] {out_path}: {e}", 0, 0

    try:
        if gpu is not None:
            result_arr, added, disappeared, dim_mismatch = gpu.composite(
                out_path, new_arr, base_arr
            )
        else:
            result_arr, added, disappeared, dim_mismatch = _cpu_composite_with_stats(
                base_arr, new_arr
            )
    except Exception as e:
        return f"[composite failed] {out_path}: {e}", 0, 0

    # 完全新規 / 寸法不一致のときは元バイトをそのまま使うと最速
    use_raw_bytes = (base_arr is None) or dim_mismatch
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".part"
        if use_raw_bytes:
            with open(tmp, "wb") as f:
                f.write(png_bytes)
        else:
            with open(tmp, "wb") as f:
                f.write(encode_rgba(result_arr))
        os.replace(tmp, out_path)
    except Exception as e:
        return f"[write failed] {out_path}: {e}", added, disappeared

    ram_cache.put(out_path, result_arr)
    return None, added, disappeared


# ============================================================
# パイプライン: tar ストリーム → スレッドプール並列合成
# ============================================================
def stream_extract_and_composite(part_files, output_dir, tag_name, ram_cache, gpu):
    raw = _ConcatReader(part_files)
    buffered = io.BufferedReader(raw, buffer_size=16 * 1024 * 1024)

    ram_cache.reset_counters()
    if gpu is not None:
        gpu.reset_counters()

    added_total = 0
    disappeared_total = 0
    errors = 0
    tiles_processed = 0
    max_pending = max(8, COMPOSITE_WORKERS * PREFETCH_FACTOR)

    def _consume(fut):
        nonlocal added_total, disappeared_total, errors
        err, added, disappeared = fut.result()
        if err:
            errors += 1
            tqdm.write(err)
        added_total += added
        disappeared_total += disappeared

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=COMPOSITE_WORKERS) as ex:
        with tarfile.open(fileobj=buffered, mode="r|gz") as tar:
            with tqdm(desc=f"[{tag_name}] 合成", unit="tile", leave=False) as pbar:
                pending = set()
                for member in tar:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(".png"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    out_path = os.path.join(output_dir, member.name)

                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            _consume(fut)
                            tiles_processed += 1
                            pbar.update(1)

                    pending.add(ex.submit(_process_tile, out_path, data, ram_cache, gpu))

                for fut in pending:
                    _consume(fut)
                    tiles_processed += 1
                    pbar.update(1)

    buffered.close()
    raw.close()

    duration = time.time() - t0
    ram_total = ram_cache.hits + ram_cache.misses
    ram_hit_rate = ram_cache.hits / ram_total if ram_total else 0.0

    stats = {
        "tag_name": tag_name,
        "tiles_processed": tiles_processed,
        "added_pixels": added_total,
        "disappeared_pixels": disappeared_total,
        "net_pixel_change": added_total - disappeared_total,
        "errors": errors,
        "duration_seconds": round(duration, 2),
        "ram_cache_hits": ram_cache.hits,
        "ram_cache_misses": ram_cache.misses,
        "ram_cache_hit_rate": round(ram_hit_rate, 4),
        "ram_cache_bytes_used": ram_cache.cur_bytes,
        "ram_cache_entries": len(ram_cache),
    }
    if gpu is not None:
        gpu_total = gpu.hits + gpu.misses
        stats.update({
            "vram_cache_hits": gpu.hits,
            "vram_cache_misses": gpu.misses,
            "vram_cache_hit_rate": round(gpu.hits / gpu_total, 4) if gpu_total else 0.0,
            "vram_cache_bytes_used": gpu.cur_bytes,
            "vram_cache_entries": len(gpu),
        })
    return stats


# ============================================================
# リリース1件の処理
# ============================================================
async def process_release_async(session, tag_name, ram_cache, gpu, all_stats):
    release_dir = os.path.join(DOWNLOAD_DIR, tag_name)
    os.makedirs(release_dir, exist_ok=True)

    api_url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag_name}"
    release_data = await fetch_api_with_retry(session, api_url)
    assets = [a for a in release_data.get("assets", []) if ".tar.gz" in a["name"]]
    if not assets:
        return False

    print(f"\n=== [{tag_name}] DL開始 ({len(assets)}ファイル) ===")
    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    tasks = [
        download_file(
            session, asset["browser_download_url"],
            os.path.join(release_dir, asset["name"]), sem, idx,
        )
        for idx, asset in enumerate(assets)
    ]
    await asyncio.gather(*tasks)
    print(f"[{tag_name}] DL完了。ストリーム解凍+並列合成へ")

    part_files = sorted(os.path.join(release_dir, a["name"]) for a in assets)
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(
        None, stream_extract_and_composite,
        part_files, OUTPUT_DIR, tag_name, ram_cache, gpu,
    )

    msg = (
        f"[{tag_name}] 合成完了 | tiles={stats['tiles_processed']} "
        f"+{stats['added_pixels']:,}px / -{stats['disappeared_pixels']:,}px "
        f"| RAM hit {stats['ram_cache_hit_rate']:.1%}"
    )
    if "vram_cache_hit_rate" in stats:
        msg += f" / VRAM hit {stats['vram_cache_hit_rate']:.1%}"
    msg += f" | {stats['duration_seconds']:.1f}s"
    if stats["errors"]:
        msg += f" | errors={stats['errors']}"
    print(msg)

    all_stats[tag_name] = stats
    save_stats(all_stats)

    if not KEEP_ARCHIVES:
        shutil.rmtree(release_dir, ignore_errors=True)
        print(f"[{tag_name}] DLパーツをクリーンアップ")
    return True


# ============================================================
# メイン
# ============================================================
async def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed = load_processed()
    all_stats = load_stats()
    tags_info = get_all_tags()
    print(f"合計 {len(tags_info)} 件のリリースを発見")

    target_tags = []
    last_date = None
    for info in tags_info:
        pub = info["published_at"]
        if last_date is None or (pub - last_date).days >= INTERVAL_DAYS:
            target_tags.append(info["tag_name"])
            last_date = pub

    skip_n = len(set(target_tags) & processed)
    print(f"処理対象: {len(target_tags)} 件 (うち処理済 {skip_n} 件はスキップ)")

    enc = "fpng" if _FPNG_KIND else "PIL"
    dec = "pyspng" if _HAS_PYSPNG else "PIL"
    if _HAS_TORCH_CUDA and VRAM_CACHE_BYTES > 0:
        gpu_name = torch.cuda.get_device_name(0)  # type: ignore
        gpu_label = f"{gpu_name} (VRAM {VRAM_CACHE_BYTES // (1024**3)} GB)"
    else:
        gpu_label = "off"
    print(
        f"設定: WORKERS={COMPOSITE_WORKERS} "
        f"RAM={TILE_CACHE_BYTES // (1024**3)}GB "
        f"PREFETCH={PREFETCH_FACTOR}x "
        f"DECODE={dec} ENCODE={enc} GPU={gpu_label}"
    )
    if not _FPNG_KIND:
        print("  ※ encode が最大ボトルネックです。`pip install fpng_py` または "
              "`pip install py-fpng-nb` の導入を強く推奨します (体感 2-3 倍速)。")

    ram_cache = TileLRUCache(TILE_CACHE_BYTES)
    gpu = (
        GpuCompositor(VRAM_CACHE_BYTES)
        if (_HAS_TORCH_CUDA and VRAM_CACHE_BYTES > 0)
        else None
    )

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for tag in target_tags:
            if tag in processed:
                continue
            try:
                ok = await process_release_async(session, tag, ram_cache, gpu, all_stats)
                if ok:
                    processed.add(tag)
                    save_processed(processed)
            except Exception as e:
                print(f"[{tag}] エラー: {e}  (次回再試行されます)")

    print(f"\nすべての処理が完了しました。最終データは '{OUTPUT_DIR}' に保存されています。")
    print(f"統計情報: '{STATS_STATE}'")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n中断しました。同じコマンドを再実行すれば続きから再開します。")