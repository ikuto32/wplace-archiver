from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .decompress import StreamingTar, active_decompress_backend
from .split_assets import validate_local_part_names


def benchmark_decompress(parts: list[Path], cfg: Config, mode: str = "inflate") -> dict:
    parts = validate_local_part_names(parts)
    backend = active_decompress_backend(parts, cfg)
    total_bytes = sum(p.stat().st_size for p in parts)
    t0 = time.time()
    members = 0
    if mode == "tar-scan":
        with StreamingTar(parts, cfg) as tar, tqdm(total=None, desc=f"tar-scan {backend}", unit="member") as pbar:
            for member in tar:
                members += 1
                pbar.update(1)
    elif mode == "inflate":
        # Count decompressed stream by reading tar members and draining file payloads.
        with StreamingTar(parts, cfg) as tar, tqdm(total=None, desc=f"inflate {backend}", unit="member") as pbar:
            for member in tar:
                members += 1
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        for _ in iter(lambda: f.read(8 * 1024 * 1024), b""):
                            pass
                pbar.update(1)
    else:
        raise ValueError("mode must be inflate or tar-scan")
    elapsed = time.time() - t0
    return {"backend": backend, "mode": mode, "input_bytes": total_bytes, "members": members, "elapsed_sec": elapsed, "input_mib_per_sec": total_bytes / 1024 / 1024 / elapsed if elapsed else None}
