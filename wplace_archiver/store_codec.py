from __future__ import annotations

import io
from types import ModuleType

from .config import Config


def get_zstd_module() -> ModuleType | None:
    """Return Python 3.14 stdlib compression.zstd, or backports.zstd."""
    try:
        from compression import zstd  # type: ignore
        return zstd
    except Exception:
        try:
            import backports.zstd as zstd  # type: ignore
            return zstd
        except Exception:
            return None


def zstd_available() -> bool:
    return get_zstd_module() is not None


def _zstd_compress(data: bytes, level: int) -> bytes:
    zstd = get_zstd_module()
    if zstd is None:
        raise RuntimeError("zstd store compression requires Python 3.14 compression.zstd or backports.zstd")

    if hasattr(zstd, "compress"):
        compress = getattr(zstd, "compress")
        for kwargs in ({"level": level}, {"compression_level": level}, {}):
            try:
                return compress(data, **kwargs)
            except TypeError:
                continue

    if hasattr(zstd, "ZstdFile"):
        bio = io.BytesIO()
        try:
            zf = zstd.ZstdFile(bio, mode="wb", level=level)
        except TypeError:
            try:
                zf = zstd.ZstdFile(bio, mode="wb", compression_level=level)
            except TypeError:
                zf = zstd.ZstdFile(bio, mode="wb")
        with zf:
            zf.write(data)
        return bio.getvalue()

    raise RuntimeError("zstd module does not provide compress() or ZstdFile")


def _zstd_decompress(data: bytes) -> bytes:
    zstd = get_zstd_module()
    if zstd is None:
        raise RuntimeError("zstd store decompression requires Python 3.14 compression.zstd or backports.zstd")

    if hasattr(zstd, "decompress"):
        try:
            return zstd.decompress(data)
        except TypeError:
            pass

    if hasattr(zstd, "ZstdFile"):
        with zstd.ZstdFile(io.BytesIO(data), mode="rb") as zf:
            return zf.read()

    raise RuntimeError("zstd module does not provide decompress() or ZstdFile")


def compress_store_payload(payload: bytes, cfg: Config) -> tuple[bytes, str, int]:
    """Return (stored_payload, compression, uncompressed_size)."""
    compression = cfg.store_compression.lower()
    if compression in ("none", "off", "0"):
        return payload, "none", len(payload)
    if compression == "zstd":
        return _zstd_compress(payload, cfg.store_zstd_level), "zstd", len(payload)
    raise ValueError("WPLACE_STORE_COMPRESSION must be one of: zstd, none")


def decompress_store_payload(data: bytes, compression: str | None, uncompressed_size: int | None = None) -> bytes:
    compression = (compression or "none").lower()
    if compression in ("none", "off", "0"):
        out = data
    elif compression == "zstd":
        out = _zstd_decompress(data)
    else:
        raise ValueError(f"unknown store payload compression: {compression}")
    if uncompressed_size is not None and int(uncompressed_size) >= 0 and len(out) != int(uncompressed_size):
        raise ValueError(f"bad decompressed payload size: expected={uncompressed_size}, got={len(out)}")
    return out
