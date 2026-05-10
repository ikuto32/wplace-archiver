from __future__ import annotations

import gzip
import io
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path
from types import ModuleType

try:
    import isal.igzip as igzip  # type: ignore
except Exception:  # pragma: no cover
    igzip = None

from .config import Config
from .errors import DecompressError
from .split_assets import validate_local_part_names
from .utils import asset_compression


def get_zstd_module() -> ModuleType | None:
    """Return Python 3.14 stdlib compression.zstd, or backports.zstd on older Pythons."""
    try:
        from compression import zstd  # type: ignore
        return zstd
    except Exception:
        try:
            import backports.zstd as zstd  # type: ignore
            return zstd
        except Exception:
            return None


class ConcatReader(io.RawIOBase):
    def __init__(self, paths: list[Path], buffer_size: int = 8 * 1024 * 1024):
        super().__init__()
        self.paths = list(paths)
        self.buffer_size = buffer_size
        self._idx = 0
        self._fh = None
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._idx >= len(self.paths):
            return False
        self._fh = self.paths[self._idx].open("rb", buffering=self.buffer_size)
        self._idx += 1
        return True

    def readinto(self, b) -> int:
        view = memoryview(b)
        while True:
            if self._fh is None and not self._open_next():
                return 0
            n = self._fh.readinto(view)
            if n is None:
                return 0
            if n > 0:
                self.bytes_read += n
                return n
            self._fh.close()
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


class PigzConcatReader(io.RawIOBase):
    """Raw reader over `pigz -dc` stdout with strict exit-code propagation."""

    def __init__(self, paths: list[Path], cfg: Config):
        super().__init__()
        self.paths = list(paths)
        self.cfg = cfg
        cmd = [cfg.pigz_path, "-dc", "-p", str(cfg.pigz_threads), "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise DecompressError(f"pigz not found: {cfg.pigz_path}") from exc
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._stdout = self.proc.stdout
        self._stderr_chunks: list[bytes] = []
        self._feed_error: BaseException | None = None
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="pigz-stderr", daemon=True)
        self._feed_thread = threading.Thread(target=self._feed_stdin, name="pigz-stdin", daemon=True)
        self._stderr_thread.start()
        self._feed_thread.start()
        self._checked = False

    def readable(self) -> bool:
        return True

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        try:
            for chunk in iter(lambda: self.proc.stderr.read(64 * 1024), b""):
                if chunk:
                    self._stderr_chunks.append(chunk)
        except BaseException as exc:  # pragma: no cover
            self._stderr_chunks.append(f"stderr drain error: {exc}".encode())

    def _feed_stdin(self) -> None:
        assert self.proc.stdin is not None
        try:
            with self.proc.stdin as stdin:
                for path in self.paths:
                    with path.open("rb") as f:
                        shutil.copyfileobj(f, stdin, length=8 * 1024 * 1024)
        except BrokenPipeError as exc:
            self._feed_error = exc
        except BaseException as exc:
            self._feed_error = exc
            try:
                self.proc.kill()
            except Exception:
                pass

    def readinto(self, b) -> int:
        try:
            data = self._stdout.read(len(b))
        except BaseException as exc:
            raise DecompressError(f"pigz stdout read failed: {exc}") from exc
        if not data:
            return 0
        n = len(data)
        memoryview(b)[:n] = data
        return n

    def _check_returncode(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            self._feed_thread.join(timeout=5)
            self._stdout.close()
        except Exception:
            pass
        ret = self.proc.wait(timeout=30)
        self._stderr_thread.join(timeout=5)
        stderr = b"".join(self._stderr_chunks).decode("utf-8", errors="replace").strip()
        if self._feed_error is not None:
            raise DecompressError(f"pigz stdin feed failed: {self._feed_error}; stderr={stderr}")
        if ret != 0:
            raise DecompressError(f"pigz failed with exit code {ret}; stderr={stderr}")
        fatal_words = ("crc", "corrupt", "invalid", "unexpected end", "truncated", "deflate")
        if stderr and any(w in stderr.lower() for w in fatal_words):
            raise DecompressError(f"pigz reported decompression error: {stderr}")

    def close(self) -> None:
        try:
            self._check_returncode()
        finally:
            super().close()


def compression_from_parts(part_files: list[Path]) -> str:
    if not part_files:
        raise DecompressError("no split parts provided")
    comps = {asset_compression(p.name) for p in part_files}
    if len(comps) != 1:
        raise DecompressError(f"mixed compression formats in split parts: {sorted(comps)}")
    return next(iter(comps))


def active_gzip_backend(cfg: Config) -> str:
    backend = cfg.gzip_backend.lower()
    if backend == "auto":
        if shutil.which(cfg.pigz_path) or Path(cfg.pigz_path).exists():
            return "pigz"
        if cfg.use_isal_gzip and igzip is not None:
            return "isal"
        return "python"
    if backend == "pigz":
        if not (shutil.which(cfg.pigz_path) or Path(cfg.pigz_path).exists()):
            raise DecompressError(f"WPLACE_GZIP_BACKEND=pigz but pigz not found: {cfg.pigz_path}")
        return "pigz"
    if backend == "isal":
        if igzip is None:
            raise DecompressError("WPLACE_GZIP_BACKEND=isal but python-isal is not installed")
        return "isal"
    if backend == "python":
        return "python"
    raise DecompressError("WPLACE_GZIP_BACKEND must be one of: auto, pigz, isal, python")


def active_decompress_backend(part_files: list[Path], cfg: Config) -> str:
    comp = compression_from_parts(part_files)
    if comp == "zstd":
        if get_zstd_module() is None:
            raise DecompressError("zstd input requires Python 3.14 compression.zstd or backports.zstd on Python <=3.13")
        return "zstd"
    return f"gzip:{active_gzip_backend(cfg)}"


class StreamingTar:
    """Open byte-split tar.zst/tar.zstd or tar.gz parts as one tar stream."""

    def __init__(self, part_files: list[Path], cfg: Config):
        self.part_files = validate_local_part_names(list(part_files))
        self.cfg = cfg
        self.compression = compression_from_parts(self.part_files)
        self.gzip_backend = active_gzip_backend(cfg) if self.compression == "gzip" else None
        self.raw = None
        self.buffered = None
        self.codec_file = None
        self.tar = None

    def __enter__(self):
        if self.compression == "zstd":
            zstd = get_zstd_module()
            if zstd is None:
                raise DecompressError("zstd input requires Python 3.14 compression.zstd or backports.zstd on Python <=3.13")
            self.raw = ConcatReader(self.part_files, buffer_size=self.cfg.io_buffer_bytes)
            self.buffered = io.BufferedReader(self.raw, buffer_size=self.cfg.io_buffer_bytes)
            self.codec_file = zstd.ZstdFile(self.buffered, mode="rb")
            self.tar = tarfile.open(fileobj=self.codec_file, mode="r|")
        elif self.gzip_backend == "pigz":
            self.raw = PigzConcatReader(self.part_files, self.cfg)
            self.buffered = io.BufferedReader(self.raw, buffer_size=self.cfg.io_buffer_bytes)
            self.tar = tarfile.open(fileobj=self.buffered, mode="r|")
        else:
            self.raw = ConcatReader(self.part_files, buffer_size=self.cfg.io_buffer_bytes)
            self.buffered = io.BufferedReader(self.raw, buffer_size=self.cfg.io_buffer_bytes)
            if self.gzip_backend == "isal":
                self.codec_file = igzip.GzipFile(fileobj=self.buffered, mode="rb")
            else:
                self.codec_file = gzip.GzipFile(fileobj=self.buffered, mode="rb")
            self.tar = tarfile.open(fileobj=self.codec_file, mode="r|")
        return self.tar

    def __exit__(self, exc_type, exc, tb):
        first_error = None
        for obj in [self.tar, self.codec_file, self.buffered, self.raw]:
            if obj is None:
                continue
            try:
                obj.close()
            except BaseException as e:
                if first_error is None:
                    first_error = e
        if first_error is not None and exc_type is None:
            raise first_error
        return False
