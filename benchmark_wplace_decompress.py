#!/usr/bin/env python
"""
Benchmark gzip backends for wplace split tar.gz assets.

対象:
  wplace_downloads/world-2026-02-09T14-20-06.949Z/*.tar.gz.aa ... *.tar.gz.af

このスクリプトは展開結果をディスクに書きません。
split された gzip stream を正しい順序で各 backend の stdin / reader へ流し、
展開された tar stream を読み捨てて速度を測定します。

測定モード:
  inflate  : gzip -> tar stream を読み捨てる。gzip inflate速度の比較向け。
  tar-scan : gzip -> tar stream を tarfile で走査し、各member payloadも読み捨てる。
             実パイプラインの tar 逐次処理負荷に近い。

依存:
  uv add tqdm isal

外部コマンド:
  pigz.exe が PATH にあること
  7z.exe / 7zz.exe / 7za.exe のいずれかが PATH にあること
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from tqdm import tqdm


DEFAULT_PART_GLOB = r"wplace_downloads/world-2026-02-09T14-20-06.949Z/*.tar.gz.*"
DEFAULT_BACKENDS = ("pigz", "isal", "python", "7zip")


# -----------------------------
# split part handling
# -----------------------------
_SPLIT_SUFFIX_RE = re.compile(r"\.tar\.gz\.([A-Za-z]+|\d+)$")


def _split_suffix_key(path: Path) -> tuple[int, int | str]:
    """
    Sort GNU split-like suffixes:
      aa, ab, ..., az, ba ...
    Numeric suffixes are also accepted.
    """
    m = _SPLIT_SUFFIX_RE.search(path.name)
    if not m:
        return (2, path.name)

    s = m.group(1)
    if s.isdigit():
        return (0, int(s))

    # base-26 order. aa < ab < ... < az < ba.
    value = 0
    for ch in s.lower():
        if not ("a" <= ch <= "z"):
            return (2, path.name)
        value = value * 26 + (ord(ch) - ord("a"))
    return (1, value)


def find_parts(pattern: str) -> list[Path]:
    paths = sorted((Path(p) for p in glob_paths(pattern)), key=_split_suffix_key)
    if not paths:
        raise FileNotFoundError(f"No split parts matched: {pattern}")

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing part files: {missing[:5]}")

    zero = [p for p in paths if p.stat().st_size == 0]
    if zero:
        raise RuntimeError(f"Zero-byte part files detected: {zero[:5]}")

    return paths


def glob_paths(pattern: str) -> list[str]:
    # pathlib.Path.glob does not handle Windows absolute drive patterns well.
    import glob

    return glob.glob(pattern)


def total_size(paths: Iterable[Path]) -> int:
    return sum(p.stat().st_size for p in paths)


def human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if abs(x) < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}TiB"


# -----------------------------
# readers / counters
# -----------------------------
class ConcatReader(io.RawIOBase):
    """Read split files as one sequential binary stream."""

    def __init__(self, paths: list[Path], pbar: tqdm | None = None):
        self.paths = list(paths)
        self.pbar = pbar
        self.idx = 0
        self.cur: BinaryIO | None = None
        self._open_next()

    def _open_next(self) -> None:
        if self.cur is not None:
            self.cur.close()
            self.cur = None
        if self.idx < len(self.paths):
            self.cur = self.paths[self.idx].open("rb")
            self.idx += 1

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.cur is None:
            return 0

        n = self.cur.readinto(b)
        while n == 0:
            self._open_next()
            if self.cur is None:
                return 0
            n = self.cur.readinto(b)

        if self.pbar is not None:
            self.pbar.update(n)
        return n

    def close(self) -> None:
        if self.cur is not None:
            self.cur.close()
            self.cur = None
        super().close()


class CountingReader(io.RawIOBase):
    """Wrap a readable binary object and count bytes read."""

    def __init__(self, src: BinaryIO):
        self.src = src
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self.src.read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self.src.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n


def drain_stream(src: BinaryIO, chunk_size: int) -> int:
    n = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        n += len(chunk)
    return n


def tar_scan_stream(src: BinaryIO, chunk_size: int) -> tuple[int, int]:
    """
    Parse a tar stream and read every regular file payload.
    Returns: (members, payload_bytes)
    """
    members = 0
    payload_bytes = 0
    with tarfile.open(fileobj=src, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            members += 1
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                payload_bytes += len(chunk)
    return members, payload_bytes


# -----------------------------
# backend runners
# -----------------------------
@dataclass
class BenchResult:
    backend: str
    mode: str
    repeat_index: int
    ok: bool
    seconds: float
    compressed_bytes: int
    decompressed_stream_bytes: int
    tar_members: int
    tar_payload_bytes: int
    compressed_mib_s: float
    decompressed_mib_s: float
    command: str
    error: str = ""


def run_python_backend(
    *,
    backend: str,
    paths: list[Path],
    compressed_bytes: int,
    mode: str,
    chunk_size: int,
    repeat_index: int,
) -> BenchResult:
    desc = f"{backend} feed r{repeat_index}"
    command = "python gzip.GzipFile" if backend == "python" else "isal.igzip.GzipFile"

    t0 = time.perf_counter()
    decompressed_stream_bytes = 0
    tar_members = 0
    tar_payload_bytes = 0
    error = ""
    ok = True

    with tqdm(
        total=compressed_bytes,
        desc=desc,
        unit="B",
        unit_scale=True,
        leave=False,
    ) as pbar:
        raw = ConcatReader(paths, pbar=pbar)
        buffered = io.BufferedReader(raw, buffer_size=max(chunk_size, 1024 * 1024))
        try:
            if backend == "python":
                gz: BinaryIO = gzip.GzipFile(fileobj=buffered, mode="rb")
            elif backend == "isal":
                try:
                    from isal import igzip
                except Exception as e:  # pragma: no cover - depends on local env
                    raise RuntimeError("isal is not installed. Run: uv add isal") from e
                gz = igzip.GzipFile(fileobj=buffered, mode="rb")
            else:
                raise ValueError(backend)

            counting = CountingReader(gz)
            if mode == "inflate":
                decompressed_stream_bytes = drain_stream(counting, chunk_size)
            elif mode == "tar-scan":
                tar_members, tar_payload_bytes = tar_scan_stream(counting, chunk_size)
                decompressed_stream_bytes = counting.bytes_read
            else:
                raise ValueError(mode)
        except Exception as e:
            ok = False
            error = repr(e)
        finally:
            try:
                buffered.close()
            finally:
                raw.close()

    seconds = time.perf_counter() - t0
    return make_result(
        backend=backend,
        mode=mode,
        repeat_index=repeat_index,
        ok=ok,
        seconds=seconds,
        compressed_bytes=compressed_bytes,
        decompressed_stream_bytes=decompressed_stream_bytes,
        tar_members=tar_members,
        tar_payload_bytes=tar_payload_bytes,
        command=command,
        error=error,
    )


def _feed_parts_to_stdin(
    proc: subprocess.Popen,
    paths: list[Path],
    compressed_bytes: int,
    chunk_size: int,
    desc: str,
    errors: list[str],
) -> None:
    assert proc.stdin is not None
    try:
        with tqdm(
            total=compressed_bytes,
            desc=desc,
            unit="B",
            unit_scale=True,
            leave=False,
        ) as pbar:
            for path in paths:
                with path.open("rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
                        pbar.update(len(chunk))
        proc.stdin.close()
    except Exception as e:
        errors.append(f"stdin feeder failed: {e!r}")
        try:
            proc.stdin.close()
        except Exception:
            pass


def _read_stderr(proc: subprocess.Popen, sink: list[str]) -> None:
    assert proc.stderr is not None
    try:
        data = proc.stderr.read()
        if data:
            sink.append(data.decode("utf-8", errors="replace"))
    except Exception as e:
        sink.append(f"stderr reader failed: {e!r}")


def run_external_backend(
    *,
    backend: str,
    cmd: list[str],
    paths: list[Path],
    compressed_bytes: int,
    mode: str,
    chunk_size: int,
    repeat_index: int,
) -> BenchResult:
    t0 = time.perf_counter()
    decompressed_stream_bytes = 0
    tar_members = 0
    tar_payload_bytes = 0
    error_parts: list[str] = []
    stderr_parts: list[str] = []
    ok = True

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError as e:
        return make_result(
            backend=backend,
            mode=mode,
            repeat_index=repeat_index,
            ok=False,
            seconds=0.0,
            compressed_bytes=compressed_bytes,
            decompressed_stream_bytes=0,
            tar_members=0,
            tar_payload_bytes=0,
            command=" ".join(cmd),
            error=f"executable not found: {e}",
        )

    feeder = threading.Thread(
        target=_feed_parts_to_stdin,
        args=(
            proc,
            paths,
            compressed_bytes,
            chunk_size,
            f"{backend} feed r{repeat_index}",
            error_parts,
        ),
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=_read_stderr,
        args=(proc, stderr_parts),
        daemon=True,
    )
    feeder.start()
    stderr_reader.start()

    assert proc.stdout is not None
    try:
        counting = CountingReader(proc.stdout)
        if mode == "inflate":
            decompressed_stream_bytes = drain_stream(counting, chunk_size)
        elif mode == "tar-scan":
            tar_members, tar_payload_bytes = tar_scan_stream(counting, chunk_size)
            decompressed_stream_bytes = counting.bytes_read
        else:
            raise ValueError(mode)
    except Exception as e:
        ok = False
        error_parts.append(f"stdout/tar read failed: {e!r}")

    feeder.join()
    rc = proc.wait()
    stderr_reader.join(timeout=2.0)

    if rc != 0:
        ok = False
        stderr_text = "".join(stderr_parts).strip()
        error_parts.append(f"exit code {rc}: {stderr_text}")

    seconds = time.perf_counter() - t0
    return make_result(
        backend=backend,
        mode=mode,
        repeat_index=repeat_index,
        ok=ok,
        seconds=seconds,
        compressed_bytes=compressed_bytes,
        decompressed_stream_bytes=decompressed_stream_bytes,
        tar_members=tar_members,
        tar_payload_bytes=tar_payload_bytes,
        command=" ".join(cmd),
        error=" | ".join(x for x in error_parts if x),
    )


def make_result(
    *,
    backend: str,
    mode: str,
    repeat_index: int,
    ok: bool,
    seconds: float,
    compressed_bytes: int,
    decompressed_stream_bytes: int,
    tar_members: int,
    tar_payload_bytes: int,
    command: str,
    error: str = "",
) -> BenchResult:
    compressed_mib_s = (
        compressed_bytes / 1024 / 1024 / seconds if seconds > 0 and ok else 0.0
    )
    decompressed_mib_s = (
        decompressed_stream_bytes / 1024 / 1024 / seconds
        if seconds > 0 and ok and decompressed_stream_bytes
        else 0.0
    )
    return BenchResult(
        backend=backend,
        mode=mode,
        repeat_index=repeat_index,
        ok=ok,
        seconds=round(seconds, 4),
        compressed_bytes=compressed_bytes,
        decompressed_stream_bytes=decompressed_stream_bytes,
        tar_members=tar_members,
        tar_payload_bytes=tar_payload_bytes,
        compressed_mib_s=round(compressed_mib_s, 2),
        decompressed_mib_s=round(decompressed_mib_s, 2),
        command=command,
        error=error,
    )


# -----------------------------
# command discovery
# -----------------------------
def resolve_pigz(path: str | None) -> str | None:
    if path:
        return path
    return shutil.which("pigz") or shutil.which("pigz.exe")


def resolve_7zip(path: str | None) -> str | None:
    if path:
        return path
    for name in ("7z", "7z.exe", "7zz", "7zz.exe", "7za", "7za.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def backend_command(
    backend: str,
    *,
    pigz_path: str | None,
    pigz_threads: int,
    sevenzip_path: str | None,
) -> list[str] | None:
    if backend == "pigz":
        exe = resolve_pigz(pigz_path)
        if not exe:
            return None
        return [exe, "-dc", "-p", str(pigz_threads)]
    if backend == "7zip":
        exe = resolve_7zip(sevenzip_path)
        if not exe:
            return None
        # Read gzip stream from stdin and emit decompressed tar stream to stdout.
        return [exe, "x", "-so", "-tgzip", "-si"]
    return None


# -----------------------------
# output
# -----------------------------
def write_results(results: list[BenchResult], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    jsonl = out_prefix.with_suffix(".jsonl")
    csv_path = out_prefix.with_suffix(".csv")

    with jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(results[0]).keys()) if results else list(BenchResult.__annotations__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nresults: {jsonl}")
    print(f"results: {csv_path}")


def print_summary(results: list[BenchResult]) -> None:
    print("\n=== Summary ===")
    ok_results = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    if ok_results:
        rows = sorted(ok_results, key=lambda r: r.seconds)
        print(
            f"{'backend':<10} {'mode':<8} {'rep':>3} {'sec':>10} "
            f"{'comp MiB/s':>12} {'decomp MiB/s':>14} {'members':>10}"
        )
        for r in rows:
            print(
                f"{r.backend:<10} {r.mode:<8} {r.repeat_index:>3} "
                f"{r.seconds:>10.2f} {r.compressed_mib_s:>12.2f} "
                f"{r.decompressed_mib_s:>14.2f} {r.tar_members:>10}"
            )

    if failed:
        print("\n=== Failed ===")
        for r in failed:
            print(f"[{r.backend} r{r.repeat_index}] {r.error}")


# -----------------------------
# main
# -----------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pigz / isal / python gzip / 7zip on wplace split tar.gz parts.",
    )
    parser.add_argument("--parts", default=DEFAULT_PART_GLOB, help="split part glob")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=list(DEFAULT_BACKENDS),
        help="backends to benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["inflate", "tar-scan"],
        default="inflate",
        help="inflate only, or parse tar and read all member payloads",
    )
    parser.add_argument("--repeat", type=int, default=1, help="repeat count per backend")
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=8,
        help="read/write chunk size in MiB",
    )
    parser.add_argument(
        "--pigz-path",
        default=os.environ.get("PIGZ_PATH") or os.environ.get("WPLACE_PIGZ_PATH"),
        help="./pigz-2.3-bin-win32/pigz.exe",
    )
    parser.add_argument(
        "--pigz-threads",
        type=int,
        default=int(os.environ.get("PIGZ_THREADS", os.cpu_count() or 4)),
        help="pigz -p threads",
    )
    parser.add_argument(
        "--sevenzip-path",
        default=os.environ.get("SEVENZIP_PATH") or os.environ.get("WPLACE_7ZIP_PATH"),
        help="C:/Program Files/7-Zip/7z.exe",
    )
    parser.add_argument(
        "--out-prefix",
        default="wplace_decompress_bench",
        help="output prefix for .jsonl and .csv",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on first backend failure",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    chunk_size = args.chunk_mib * 1024 * 1024
    parts = find_parts(args.parts)
    compressed_bytes = total_size(parts)

    print("=== Input ===")
    print(f"parts glob : {args.parts}")
    print(f"parts      : {len(parts)}")
    print(f"first      : {parts[0]}")
    print(f"last       : {parts[-1]}")
    print(f"compressed : {human_bytes(compressed_bytes)}")
    print(f"mode       : {args.mode}")
    print(f"chunk      : {args.chunk_mib} MiB")

    results: list[BenchResult] = []

    for repeat_index in range(1, args.repeat + 1):
        for backend in args.backends:
            print(f"\n--- backend={backend} repeat={repeat_index}/{args.repeat} ---")
            if backend in ("python", "isal"):
                r = run_python_backend(
                    backend=backend,
                    paths=parts,
                    compressed_bytes=compressed_bytes,
                    mode=args.mode,
                    chunk_size=chunk_size,
                    repeat_index=repeat_index,
                )
            elif backend in ("pigz", "7zip"):
                cmd = backend_command(
                    backend,
                    pigz_path=args.pigz_path,
                    pigz_threads=args.pigz_threads,
                    sevenzip_path=args.sevenzip_path,
                )
                if cmd is None:
                    r = make_result(
                        backend=backend,
                        mode=args.mode,
                        repeat_index=repeat_index,
                        ok=False,
                        seconds=0.0,
                        compressed_bytes=compressed_bytes,
                        decompressed_stream_bytes=0,
                        tar_members=0,
                        tar_payload_bytes=0,
                        command="",
                        error=f"{backend} executable not found",
                    )
                else:
                    print(f"cmd: {' '.join(cmd)}")
                    r = run_external_backend(
                        backend=backend,
                        cmd=cmd,
                        paths=parts,
                        compressed_bytes=compressed_bytes,
                        mode=args.mode,
                        chunk_size=chunk_size,
                        repeat_index=repeat_index,
                    )
            else:
                raise ValueError(backend)

            results.append(r)
            status = "OK" if r.ok else "FAIL"
            print(
                f"{status}: {backend} {r.seconds:.2f}s "
                f"compressed={r.compressed_mib_s:.2f}MiB/s "
                f"decompressed={r.decompressed_mib_s:.2f}MiB/s "
                f"members={r.tar_members}"
            )
            if not r.ok:
                print(f"error: {r.error}")
                if args.fail_fast:
                    write_results(results, Path(args.out_prefix))
                    print_summary(results)
                    return 1

    write_results(results, Path(args.out_prefix))
    print_summary(results)

    return 0 if all(r.ok for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
