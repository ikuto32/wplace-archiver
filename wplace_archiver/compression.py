from __future__ import annotations

from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=1)
def get_zstd_module() -> ModuleType | None:
    """Return stdlib compression.zstd (Py3.14+) or backports.zstd."""
    try:
        from compression import zstd  # type: ignore

        return zstd
    except Exception:
        try:
            import backports.zstd as zstd  # type: ignore

            return zstd
        except Exception:
            return None
