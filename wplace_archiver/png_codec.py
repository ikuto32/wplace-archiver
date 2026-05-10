from __future__ import annotations

import io

import numpy as np
from PIL import Image

try:
    import pyspng  # type: ignore
except Exception:  # pragma: no cover
    pyspng = None

try:
    import fpng_py  # type: ignore
except Exception:  # pragma: no cover
    fpng_py = None

from .config import Config
from .errors import PngDecodeError


def decode_png_array(png_bytes: bytes, cfg: Config, *, include_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Decode PNG into uint8 RGB or RGBA.

    Indexed-color PNGs (P mode) must go through Pillow so that PNG tRNS
    transparency is preserved. Some fast decoders can return P/tRNS inputs as
    RGB and thereby lose alpha, which would persist background pixels into the
    rolling state. RGB inputs still remain RGB so the RGB-only black-background
    rule can be applied later.
    """
    try:
        im = Image.open(io.BytesIO(png_bytes))
        mode = im.mode
        meta: dict[str, object] = {"source_mode": mode, "p_has_trns": False}
        if mode == "P":
            meta["p_has_trns"] = bool(im.info.get("transparency") is not None)

        if cfg.strict_rgba and mode != "RGBA":
            raise PngDecodeError(f"expected RGBA PNG, got mode={mode}")

        # Critical correctness path: P mode + tRNS must preserve alpha.
        # Do not pass these through pyspng, because a decoder returning RGB here
        # would silently discard PNG transparency metadata.
        if mode == "P":
            arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            return (arr, meta) if include_meta else arr

        # Preserve actual RGB inputs as RGB. Black-background handling is applied
        # only to this source mode in PaletteCodec.
        if mode == "RGB":
            arr = np.asarray(im, dtype=np.uint8)
            return (arr, meta) if include_meta else arr

        # For true RGBA PNGs, use the fast path when available.
        if mode == "RGBA" and pyspng is not None:
            arr = pyspng.load(png_bytes)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                out = np.ascontiguousarray(arr)
                return (out, meta) if include_meta else out

        # Grayscale, LA, and other modes are normalized to RGBA so their alpha,
        # if present, is represented explicitly.
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        return (arr, meta) if include_meta else arr
    except PngDecodeError:
        raise
    except Exception as exc:
        raise PngDecodeError(str(exc)) from exc


def encode_png_rgba(rgba: np.ndarray) -> bytes:
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"expected uint8 RGBA image, got shape={rgba.shape}, dtype={rgba.dtype}")
    if not rgba.flags.c_contiguous:
        rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    if fpng_py is not None:
        return fpng_py.fpng_encode_image_to_memory(rgba.tobytes(), w, h, 4)
    bio = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(bio, format="PNG", optimize=False)
    return bio.getvalue()
