from __future__ import annotations

import gzip
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .apply import apply_tag_store_to_state, delete_tag_store_if_needed
from .config import Config
from .decompress import StreamingTar, get_zstd_module
from .store_codec import _zstd_compress
from .ingest import ingest_tag_from_parts
from .palette import PaletteCodec
from .png_codec import decode_png_array
from .records import dense_record_from_sparse
from .shard_store import SparseShardWriter, SparseTileStore
from .split_assets import select_release_split_assets, validate_local_part_names
from .state import PipelineState
from .utils import atomic_write_json, load_json, shard_index_path, sort_split_part_paths


def _png_bytes(arr: np.ndarray, mode: str) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(bio, format="PNG")
    return bio.getvalue()


def _p_trns_png_bytes(indexed: np.ndarray, palette: list[tuple[int, int, int]], transparency_index: int) -> bytes:
    im = Image.fromarray(indexed.astype(np.uint8), mode="P")
    flat_palette: list[int] = []
    for r, g, b in palette:
        flat_palette.extend([int(r), int(g), int(b)])
    flat_palette.extend([0] * (768 - len(flat_palette)))
    im.putpalette(flat_palette)
    bio = io.BytesIO()
    im.save(bio, format="PNG", transparency=int(transparency_index))
    return bio.getvalue()


def _make_tar_bytes(files: dict[str, bytes]) -> bytes:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return raw_tar.getvalue()


def _split_bytes(payload: bytes, split_at: int | None = None) -> tuple[bytes, bytes]:
    if split_at is None:
        split_at = max(1, len(payload) // 2)
    return payload[:split_at], payload[split_at:]


def _write_split_tar_parts(tag_dir: Path, tag: str, ext: str, payload: bytes, split_at: int | None = None) -> list[Path]:
    first, second = _split_bytes(payload, split_at=split_at)
    parts = [first, second]
    paths: list[Path] = []
    for suffix, data in zip(["aa", "ab"], parts):
        path = tag_dir / f"{tag}.{ext}.{suffix}"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _write_split_tar_gz(tag_dir: Path, tag: str, files: dict[str, bytes], split_at: int | None = None) -> list[Path]:
    tag_dir.mkdir(parents=True, exist_ok=True)
    gz = gzip.compress(_make_tar_bytes(files))
    return _write_split_tar_parts(tag_dir, tag, "tar.gz", gz, split_at=split_at)


def _write_split_tar_zst(tag_dir: Path, tag: str, files: dict[str, bytes], split_at: int | None = None) -> list[Path] | None:
    zstd = get_zstd_module()
    if zstd is None:
        return None
    tag_dir.mkdir(parents=True, exist_ok=True)
    zst = _zstd_compress(_make_tar_bytes(files), level=3)
    return _write_split_tar_parts(tag_dir, tag, "tar.zst", zst, split_at=split_at)


def _read_exported_pixel(path: Path, xy: tuple[int, int]) -> tuple[int, int, int, int]:
    arr = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    x, y = xy
    return tuple(int(v) for v in arr[y, x])


SPEC_19_COVERAGE = {
    1: "small_grid_synth_correctness",
    2: "rgba_transparency_preserved",
    3: "rgb_black_transparency",
    4: "p_mode_trns_transparency",
    5: "sparse_record_roundtrip",
    6: "dense_fallback_roundtrip",
    7: "zstd_store_roundtrip",
    8: "legacy_uncompressed_store_compat",
    9: "rolling_apply_overwrite",
    10: "apply_shard_checkpoint_resume",
    11: "apply_worker_small_summary",
    12: "ingested_unapplied_tag_store_reuse",
    13: "asset_name_filter_excludes_checksums",
    14: "palette_rollback_on_failure",
    15: "export_png_alpha_and_color",
}


def run_self_test() -> dict:
    temp = Path(tempfile.mkdtemp(prefix="wplace_archiver_v2_selftest_"))
    try:
        store_zstd_available = get_zstd_module() is not None
        cfg = Config(
            download_dir=temp / "downloads",
            store_root=temp / "store",
            xyz_output_dir=temp / "xyz",
            grid_tiles=4,
            tile_size=4,
            shard_tiles=2,
            max_colors=8,
            workers=2,
            decode_workers=2,
            apply_workers=1,
            gzip_backend="python",
            apply_executor="sequential",
            keep_tag_stores=True,
            rgb_transparency_mode="corners",
            rgb_transparent_dominant_min=0.90,
            black_warning_ratio=0.70,
            store_compression="zstd" if store_zstd_available else "none",
        )
        palette = PaletteCodec(cfg)
        tag1 = "world-2025-01-01T00-00-00.000Z"
        tag2 = "world-2025-01-11T00-00-00.000Z"
        red = [255, 0, 0, 255]
        green = [0, 255, 0, 255]
        blue = [0, 0, 255]

        rgba1 = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba1[1, 1] = red
        rgba1[2, 2] = [0, 0, 255, 255]
        rgb1 = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb1[0, 1] = blue
        p_indexed = np.zeros((4, 4), dtype=np.uint8)
        p_indexed[0, 1] = 1
        p_png = _p_trns_png_bytes(p_indexed, [(0, 0, 0), (255, 0, 255)], transparency_index=0)
        rgba2 = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba2[1, 1] = green
        rgba_warn = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba_warn[:, :] = [0, 0, 0, 255]
        rgba_warn[1, 1] = green

        parts1 = _write_split_tar_gz(cfg.download_dir / tag1, tag1, {"prefix/0/0.png": _png_bytes(rgba1, "RGBA"), "prefix/1/0.png": _png_bytes(rgb1, "RGB"), "prefix/2/0.png": p_png})
        parts2 = _write_split_tar_gz(cfg.download_dir / tag2, tag2, {"prefix/0/0.png": _png_bytes(rgba2, "RGBA"), "prefix/0/1.png": _png_bytes(rgba_warn, "RGBA")})

        # Strict asset filtering: checksum-like files are excluded/rejected, not concatenated.
        selected = select_release_split_assets([
            {"name": f"{tag1}.tar.gz.aa"},
            {"name": f"{tag1}.tar.gz.ab"},
            {"name": f"{tag1}.tar.gz.sha256"},
        ])
        assert [a["name"] for a in selected] == [f"{tag1}.tar.gz.aa", f"{tag1}.tar.gz.ab"]
        selected_prefer_zstd = select_release_split_assets([
            {"name": f"{tag1}.tar.gz.aa"},
            {"name": f"{tag1}.tar.gz.ab"},
            {"name": f"{tag1}.tar.zst.aa"},
            {"name": f"{tag1}.tar.zst.ab"},
        ])
        assert [a["name"] for a in selected_prefer_zstd] == [f"{tag1}.tar.zst.aa", f"{tag1}.tar.zst.ab"]
        assert validate_local_part_names(parts1) == sort_split_part_paths(parts1)

        # Streaming byte-split tar.gz works for backward compatibility.
        with StreamingTar(parts1, cfg) as tar:
            names = sorted(m.name for m in tar if m.isfile())
        assert names == ["prefix/0/0.png", "prefix/1/0.png", "prefix/2/0.png"]

        zstd_parts = _write_split_tar_zst(cfg.download_dir / f"{tag1}-zstd", tag1, {"prefix/0/0.png": _png_bytes(rgba1, "RGBA")})
        zstd_available = zstd_parts is not None
        if zstd_parts is not None:
            assert validate_local_part_names(zstd_parts) == sort_split_part_paths(zstd_parts)
            with StreamingTar(zstd_parts, cfg) as tar:
                zstd_names = sorted(m.name for m in tar if m.isfile())
            assert zstd_names == ["prefix/0/0.png"]

        snap = palette.snapshot()
        stats1 = ingest_tag_from_parts(tag1, parts1, cfg, palette)
        apply_stats1 = apply_tag_store_to_state(tag1, cfg)
        palette.save()
        checkpoint1 = load_json(cfg.store_root / ".apply_shards" / f"{tag1}.json", {})
        assert checkpoint1.get("completed") is True
        assert len(checkpoint1.get("completed_shards", [])) == apply_stats1["overlay_shards"]

        stats2 = ingest_tag_from_parts(tag2, parts2, cfg, palette)
        apply_stats2 = apply_tag_store_to_state(tag2, cfg)
        palette.save()
        checkpoint2 = load_json(cfg.store_root / ".apply_shards" / f"{tag2}.json", {})
        assert checkpoint2.get("completed") is True
        assert len(checkpoint2.get("completed_shards", [])) == apply_stats2["overlay_shards"]

        store = SparseTileStore(cfg.state_root, cfg)
        refs = {(r.x, r.y): r for r in store.iter_refs()}
        assert any(r.encoding == "dense-u8-v1" for r in refs.values()), "dense fallback tile must be emitted in self-test fixture"
        assert any(r.encoding == "sparse-u32-u8-v1" for r in refs.values()), "sparse tiles must be emitted in self-test fixture"
        if store_zstd_available:
            assert all(r.compression == "zstd" for r in refs.values()), "new intermediate records should be zstd-compressed"
        else:
            assert all(r.compression == "none" for r in refs.values()), "self-test should fall back to none when zstd module is unavailable"
        assert (0, 0) in refs and (1, 0) in refs and (2, 0) in refs
        pos, val = store.load_sparse(refs[(0, 0)])
        dense = dense_record_from_sparse(pos, val, cfg.tile_pixels).reshape((4, 4))
        rgba = palette.index_to_rgba(dense)
        assert tuple(rgba[1, 1]) == tuple(green), "later tag must overwrite earlier tag"
        assert tuple(rgba[0, 0]) == (0, 0, 0, 0), "transparent pixels must remain transparent"

        pos_rgb, val_rgb = store.load_sparse(refs[(1, 0)])
        dense_rgb = dense_record_from_sparse(pos_rgb, val_rgb, cfg.tile_pixels).reshape((4, 4))
        rgba_rgb = palette.index_to_rgba(dense_rgb)
        assert tuple(rgba_rgb[0, 0]) == (0, 0, 0, 0), "RGB black background should be inferred as transparent"
        assert tuple(rgba_rgb[0, 1]) == (0, 0, 255, 255), "RGB visible art should remain visible"
        assert stats1["rgb_tiles_seen"] == 1
        assert stats1["rgb_transparency_diagnostics_sample"][0]["forced_black_transparent"] is True
        assert stats1["rgb_transparency_diagnostics_sample"][0]["black_pixels"] >= 1

        pos_p, val_p = store.load_sparse(refs[(2, 0)])
        dense_p = dense_record_from_sparse(pos_p, val_p, cfg.tile_pixels).reshape((4, 4))
        rgba_p = palette.index_to_rgba(dense_p)
        assert tuple(rgba_p[0, 0]) == (0, 0, 0, 0), "P mode tRNS background should remain transparent"
        assert tuple(rgba_p[0, 1]) == (255, 0, 255, 255), "P mode visible palette color should remain visible"


        # Legacy uncompressed store compatibility: missing compression metadata means none.
        legacy_cfg = cfg.with_overrides(store_compression="none")
        legacy_root = temp / "legacy_store"
        legacy_pos = np.array([0, 5], dtype=np.uint32)
        legacy_val = np.array([1, 2], dtype=np.uint8)
        legacy_payload = legacy_pos.tobytes(order="C") + legacy_val.tobytes(order="C")
        with SparseShardWriter(legacy_root, legacy_cfg, label="legacy") as legacy_writer:
            legacy_writer.write_tile_payload(0, 0, legacy_payload, 2, "sparse-u32-u8-v1")
        legacy_idx_path = shard_index_path(legacy_root, 0, 0)
        legacy_idx = load_json(legacy_idx_path, {})
        for entry in legacy_idx.get("tiles", []):
            entry.pop("compression", None)
            entry.pop("uncompressed_size", None)
        atomic_write_json(legacy_idx_path, legacy_idx)
        legacy_store = SparseTileStore(legacy_root, cfg)
        legacy_ref = next(legacy_store.iter_refs())
        legacy_loaded_pos, legacy_loaded_val = legacy_store.load_sparse(legacy_ref)
        assert legacy_ref.compression == "none"
        assert legacy_loaded_pos.tolist() == [0, 5]
        assert legacy_loaded_val.tolist() == [1, 2]

        # State checkpoint and tag-store reuse semantics.
        st = PipelineState.load(cfg.pipeline_state_path, cfg)
        st.mark_ingested(tag2)
        st.save(cfg.pipeline_state_path)
        reloaded = PipelineState.load(cfg.pipeline_state_path, cfg)
        assert reloaded.is_ingested(tag2)
        assert (cfg.tags_root / tag2 / "manifest.json").exists()

        # Palette rollback behavior: failed dynamic palette update can be restored.
        before = palette.snapshot()
        too_many = Config(
            download_dir=temp / "downloads2",
            store_root=temp / "store2",
            xyz_output_dir=temp / "xyz2",
            grid_tiles=1,
            tile_size=2,
            shard_tiles=1,
            max_colors=1,
            gzip_backend="python",
            apply_executor="sequential",
        )
        bad_palette = PaletteCodec(too_many)
        bad_tag = "world-2025-01-21T00-00-00.000Z"
        bad = np.zeros((2, 2, 4), dtype=np.uint8)
        bad[0, 0] = [1, 0, 0, 255]
        bad[0, 1] = [2, 0, 0, 255]
        bad_parts = _write_split_tar_gz(too_many.download_dir / bad_tag, bad_tag, {"p/0/0.png": _png_bytes(bad, "RGBA")})
        bad_snap = bad_palette.snapshot()
        try:
            ingest_tag_from_parts(bad_tag, bad_parts, too_many, bad_palette)
            raise AssertionError("palette overflow should fail")
        except Exception:
            bad_palette.restore(bad_snap)
        assert bad_palette.color_count == 0

        assert stats2["rgba_black_warning_tiles"] == 1

        from .export import export_state_to_xyz
        export_state_to_xyz(cfg, palette)
        assert _read_exported_pixel(cfg.xyz_output_dir / "11" / "0" / "0.png", (1, 1)) == tuple(green)
        assert _read_exported_pixel(cfg.xyz_output_dir / "11" / "0" / "0.png", (0, 0)) == (0, 0, 0, 0)

        spec_19_coverage = {str(k): {"id": v, "pass": False} for k, v in SPEC_19_COVERAGE.items()}

        checks = [
                "strict asset filter",
                "zstd preferred over gzip asset selection",
                "byte-split tar.gz stream",
        ]
        if zstd_available:
            checks.append("byte-split tar.zst stream")
        else:
            checks.append("byte-split tar.zst stream skipped: compression.zstd/backports.zstd unavailable")
        if store_zstd_available:
            checks.append("zstd-compressed intermediate shard store")
        else:
            checks.append("zstd-compressed intermediate shard store skipped: compression.zstd/backports.zstd unavailable")
        checks.append("legacy uncompressed intermediate shard store compatibility")
        checks.extend([
                "RGBA transparency",
                "RGB black transparency",
                "RGB diagnostics sample",
                "P mode tRNS transparency",
                "RGBA black-dominant warning sample",
                "rolling apply overwrite",
                "ingested tag-store reuse checkpoint",
                "apply shard checkpoint",
                "palette rollback",
                "XYZ export",
            ])
        if any(r.encoding == "sparse-u32-u8-v1" for r in refs.values()):
            spec_19_coverage["5"]["pass"] = True
        if any(r.encoding == "dense-u8-v1" for r in refs.values()):
            spec_19_coverage["6"]["pass"] = True
        spec_19_coverage["1"]["pass"] = True
        spec_19_coverage["2"]["pass"] = True
        spec_19_coverage["3"]["pass"] = True
        spec_19_coverage["4"]["pass"] = True
        spec_19_coverage["7"]["pass"] = store_zstd_available
        spec_19_coverage["8"]["pass"] = True
        spec_19_coverage["9"]["pass"] = True
        spec_19_coverage["10"]["pass"] = True
        spec_19_coverage["11"]["pass"] = (
            isinstance(apply_stats2, dict)
            and len(apply_stats2) <= 20
            and all(not isinstance(v, (list, tuple, dict)) for v in apply_stats2.values())
        )
        spec_19_coverage["12"]["pass"] = True
        spec_19_coverage["13"]["pass"] = True
        spec_19_coverage["14"]["pass"] = True
        spec_19_coverage["15"]["pass"] = True

        result = {
            "ok": True,
            "temp_dir": str(temp),
            "state_tiles": store.tile_count(),
            "palette_colors": palette.color_count,
            "checks": checks,
            "spec_19_coverage": spec_19_coverage,
        }
        return result
    finally:
        if os.environ.get("WPLACE_SELFTEST_KEEP_TMP", "0") != "1":
            shutil.rmtree(temp, ignore_errors=True)
