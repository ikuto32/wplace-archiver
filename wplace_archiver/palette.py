from __future__ import annotations

import copy
import re
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config
from .errors import PaletteError
from .utils import atomic_write_json, load_json


def rgb24_from_triplet(r: int, g: int, b: int) -> int:
    return ((int(r) & 255) << 16) | ((int(g) & 255) << 8) | (int(b) & 255)


def parse_rgb24_color_list(spec: str) -> set[int]:
    out: set[int] = set()
    spec = (spec or "").strip()
    if not spec:
        return out
    for item in re.split(r"[;|]\s*", spec):
        item = item.strip()
        if not item:
            continue
        if item.startswith("#") and len(item) == 7:
            out.add(int(item[1:], 16))
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise PaletteError(f"bad RGB color spec: {item!r}; expected r,g,b or #rrggbb")
        out.add(rgb24_from_triplet(int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def rgb24_flat_from_rgb(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.uint32, copy=False)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def summarize_black_pixels(rgb24: np.ndarray, total_pixels: int, threshold: float) -> dict:
    black = np.uint32(0)
    black_pixels = int(np.count_nonzero(rgb24 == black))
    black_ratio = float(black_pixels) / float(total_pixels) if total_pixels else 0.0
    return {
        "black_pixels": black_pixels,
        "black_ratio": black_ratio,
        "black_dominant": bool(black_ratio >= threshold),
    }


class PaletteCodec:
    """Thread-safe RGB24 -> uint8 palette index codec.

    If a fixed palette is provided, unknown visible colors are fatal.
    Otherwise unknown colors are added lazily up to max_colors. Snapshot/restore
    is used by the pipeline so failed ingest attempts cannot persist partial palettes.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.palette_path
        self.max_colors = cfg.max_colors
        self.fixed = cfg.fixed_palette_path is not None
        self._lock = threading.Lock()
        self._rgb_to_index: dict[int, int] = {}
        self._colors: list[int] = []
        self._lut = np.zeros(1 << 24, dtype=np.uint8)
        self._rgba_lut_cache: np.ndarray | None = None
        if cfg.fixed_palette_path is not None:
            self._load_palette_file(cfg.fixed_palette_path, fixed=True)
        else:
            self.load()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "rgb_to_index": copy.deepcopy(self._rgb_to_index),
                "colors": list(self._colors),
                "fixed": self.fixed,
            }

    def restore(self, snapshot: dict) -> None:
        with self._lock:
            self._rgb_to_index = copy.deepcopy(snapshot["rgb_to_index"])
            self._colors = list(snapshot["colors"])
            self.fixed = bool(snapshot.get("fixed", self.fixed))
            self._lut.fill(0)
            for i, rgb24 in enumerate(self._colors, start=1):
                self._lut[int(rgb24)] = i
            self._rgba_lut_cache = None

    def _load_palette_file(self, path: Path, fixed: bool) -> None:
        data = load_json(path, {})
        colors = data.get("colors", [])
        with self._lock:
            self._rgb_to_index.clear()
            self._colors.clear()
            self._lut.fill(0)
            for item in colors:
                idx = int(item.get("index", len(self._colors) + 1))
                rgb = item["rgb"]
                rgb24 = rgb24_from_triplet(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                while len(self._colors) < idx:
                    self._colors.append(-1)
                self._colors[idx - 1] = rgb24
                self._rgb_to_index[rgb24] = idx
                self._lut[rgb24] = idx
            self._colors = [c for c in self._colors if c >= 0]
            self._rgba_lut_cache = None
            self.fixed = fixed

    def load(self) -> None:
        if self.path.exists():
            self._load_palette_file(self.path, fixed=False)

    def save(self) -> None:
        with self._lock:
            colors = []
            for i, rgb24 in enumerate(self._colors, start=1):
                r = (rgb24 >> 16) & 255
                g = (rgb24 >> 8) & 255
                b = rgb24 & 255
                colors.append({"index": i, "rgb": [r, g, b], "hex": f"#{r:02x}{g:02x}{b:02x}"})
            data = {
                "format": "wplace-palette-index-v1",
                "transparent_index": 0,
                "max_visible_colors": self.max_colors,
                "fixed": self.fixed,
                "colors": colors,
            }
        atomic_write_json(self.path, data)

    def _add_unknown_colors_locked(self, unknown_rgb24: Iterable[int]) -> None:
        for raw in unknown_rgb24:
            rgb24 = int(raw)
            if rgb24 in self._rgb_to_index:
                continue
            if self.fixed:
                r = (rgb24 >> 16) & 255
                g = (rgb24 >> 8) & 255
                b = rgb24 & 255
                raise PaletteError(f"unknown visible color for fixed palette: #{r:02x}{g:02x}{b:02x}")
            if len(self._colors) >= self.max_colors:
                r = (rgb24 >> 16) & 255
                g = (rgb24 >> 8) & 255
                b = rgb24 & 255
                raise PaletteError(f"palette exceeded {self.max_colors} visible colors; unknown=#{r:02x}{g:02x}{b:02x}")
            idx = len(self._colors) + 1
            self._colors.append(rgb24)
            self._rgb_to_index[rgb24] = idx
            self._lut[rgb24] = idx
            self._rgba_lut_cache = None

    def _map_rgb24_to_index(self, rgb24: np.ndarray) -> np.ndarray:
        rgb24 = rgb24.astype(np.uint32, copy=False)
        mapped = self._lut[rgb24]
        if np.any(mapped == 0):
            unknown = np.unique(rgb24[mapped == 0])
            with self._lock:
                self._add_unknown_colors_locked(unknown)
            mapped = self._lut[rgb24]
        if np.any(mapped == 0):
            raise PaletteError("visible pixels remained unmapped after palette update")
        return mapped.astype(np.uint8, copy=False)

    def infer_rgb_transparent_mask(self, rgb: np.ndarray, rgb24: np.ndarray) -> tuple[np.ndarray, dict]:
        """Infer transparent pixels for RGB-only PNG tiles.

        Black (#000000) is always treated as transparent for RGB inputs only.
        Additional configured colors are also honored. Corner-based inference is
        applied on top of that when enabled.
        """
        mode = self.cfg.rgb_transparency_mode
        mask = np.zeros(self.cfg.tile_pixels, dtype=bool)
        black_summary = summarize_black_pixels(rgb24, self.cfg.tile_pixels, self.cfg.black_warning_ratio)
        diag = {
            "mode": mode,
            "input": "rgb",
            "explicit_colors": 0,
            "corner_color": None,
            "corner_ratio": 0.0,
            "transparent_pixels": 0,
            "forced_black_transparent": True,
            **black_summary,
            "warning": None,
        }

        # RGB-only files: always treat black as transparent background.
        black_mask = rgb24 == np.uint32(0)
        mask |= black_mask

        explicit = parse_rgb24_color_list(self.cfg.rgb_transparent_colors)
        if explicit:
            colors = np.fromiter(explicit, dtype=np.uint32)
            explicit_mask = np.isin(rgb24, colors)
            mask |= explicit_mask
            diag["explicit_colors"] = len(explicit)
            diag["explicit_transparent_pixels"] = int(np.count_nonzero(explicit_mask))

        if mode in ("corners", "corner", "auto", "auto-corners"):
            c0 = rgb24_from_triplet(*rgb[0, 0])
            c1 = rgb24_from_triplet(*rgb[0, -1])
            c2 = rgb24_from_triplet(*rgb[-1, 0])
            c3 = rgb24_from_triplet(*rgb[-1, -1])
            if c0 == c1 == c2 == c3:
                bg = np.uint32(c0)
                bg_mask = rgb24 == bg
                ratio = float(np.count_nonzero(bg_mask)) / float(self.cfg.tile_pixels)
                diag["corner_color"] = f"#{int(c0):06x}"
                diag["corner_ratio"] = ratio
                if ratio >= self.cfg.rgb_transparent_dominant_min:
                    mask |= bg_mask
        elif mode in ("none", "off", "opaque", "0", "colors", "color", "explicit"):
            pass
        else:
            raise PaletteError("RGB_TRANSPARENCY_MODE must be one of: none, corners, colors, auto")

        diag["transparent_pixels"] = int(np.count_nonzero(mask))
        if diag["black_dominant"] and diag["transparent_pixels"] == 0:
            diag["warning"] = "rgb_black_dominant_but_not_transparent"
        elif diag["black_dominant"]:
            diag["warning"] = "rgb_black_dominant"
        return mask, diag

    def _rgba_warning_diag(self, arr: np.ndarray) -> dict:
        alpha = arr[..., 3]
        rgb24 = rgb24_flat_from_rgb(arr[..., :3])
        black_summary = summarize_black_pixels(rgb24, self.cfg.tile_pixels, self.cfg.black_warning_ratio)
        all_opaque = bool(np.all(alpha == 255))
        warning = None
        if all_opaque and black_summary["black_dominant"]:
            warning = "rgba_all_opaque_black_dominant"
        return {
            "mode": "rgba",
            "input": "rgba",
            "transparent_pixels": int(np.count_nonzero(alpha == 0)),
            "all_alpha_opaque": all_opaque,
            **black_summary,
            "warning": warning,
        }

    def image_to_record(self, arr: np.ndarray) -> tuple[bytes, int, str, dict]:
        if arr.ndim != 3 or arr.dtype != np.uint8 or arr.shape[-1] not in (3, 4):
            raise PaletteError(f"expected uint8 RGB/RGBA array, got shape={arr.shape}, dtype={arr.dtype}")
        if arr.shape[0] != self.cfg.tile_size or arr.shape[1] != self.cfg.tile_size:
            raise PaletteError(f"expected tile shape ({self.cfg.tile_size}, {self.cfg.tile_size}), got {arr.shape}")
        if arr.shape[-1] == 3:
            rgb24 = rgb24_flat_from_rgb(arr)
            transparent, diag = self.infer_rgb_transparent_mask(arr, rgb24)
            positions = np.flatnonzero(~transparent) if np.any(transparent) else np.arange(self.cfg.tile_pixels, dtype=np.intp)
            visible = int(positions.size)
            if visible == 0:
                return b"", 0, "sparse-u32-u8-v1", diag
            mapped = self._map_rgb24_to_index(rgb24[positions])
            return self._record_payload(positions.astype(np.uint32, copy=False), mapped, diag)

        alpha = arr[..., 3]
        if self.cfg.strict_binary_alpha:
            bad_alpha = (alpha != 0) & (alpha != 255)
            if np.any(bad_alpha):
                raise PaletteError("non-binary alpha detected; expected alpha 0 or 255")
        positions = np.flatnonzero(alpha.ravel() != 0)
        visible = int(positions.size)
        diag = self._rgba_warning_diag(arr)
        if visible == 0:
            return b"", 0, "sparse-u32-u8-v1", diag
        flat = arr.reshape(-1, 4)
        rgb = flat[positions, :3].astype(np.uint32, copy=False)
        rgb24 = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        mapped = self._map_rgb24_to_index(rgb24)
        return self._record_payload(positions.astype(np.uint32, copy=False), mapped, diag)

    def _record_payload(self, positions: np.ndarray, values: np.ndarray, diag: dict) -> tuple[bytes, int, str, dict]:
        count = int(positions.size)
        if count * 5 < self.cfg.dense_fallback_bytes:
            return positions.tobytes(order="C") + values.astype(np.uint8, copy=False).tobytes(order="C"), count, "sparse-u32-u8-v1", {**diag, "dense_fallback": False}
        dense = np.zeros(self.cfg.tile_pixels, dtype=np.uint8)
        dense[positions.astype(np.intp, copy=False)] = values.astype(np.uint8, copy=False)
        return dense.tobytes(order="C"), count, "dense-u8-v1", {**diag, "dense_fallback": True}

    def rgba_lut(self) -> np.ndarray:
        with self._lock:
            if self._rgba_lut_cache is not None:
                return self._rgba_lut_cache
            lut = np.zeros((len(self._colors) + 1, 4), dtype=np.uint8)
            for i, rgb24 in enumerate(self._colors, start=1):
                lut[i] = [(rgb24 >> 16) & 255, (rgb24 >> 8) & 255, rgb24 & 255, 255]
            self._rgba_lut_cache = lut
            return lut

    def index_to_rgba(self, idx: np.ndarray) -> np.ndarray:
        if idx.dtype != np.uint8:
            idx = idx.astype(np.uint8, copy=False)
        lut = self.rgba_lut()
        max_idx = int(idx.max(initial=0))
        if max_idx >= len(lut):
            raise PaletteError(f"tile contains palette index {max_idx}, but palette has {len(lut) - 1} colors")
        return lut[idx]

    @property
    def color_count(self) -> int:
        with self._lock:
            return len(self._colors)
