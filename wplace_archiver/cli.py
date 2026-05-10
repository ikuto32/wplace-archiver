from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .benchmark import benchmark_decompress
from .config import Config
from .diagnostics import diagnose_rgb_transparency
from .export import export_state_to_xyz
from .palette import PaletteCodec
from .pipeline import run_pipeline
from .self_test import run_self_test
from .split_assets import validate_local_part_names
from .validation import validate_store


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="wplace Archiver v2 rolling sparse pipeline")
    p.add_argument("--config", type=Path, default=Path("wplace_archiver.toml"), help="TOML config path")
    p.add_argument("--store-root", type=Path)
    p.add_argument("--download-dir", type=Path)
    p.add_argument("--xyz-output-dir", type=Path)
    p.add_argument("--compression-backend", choices=["auto", "zstd", "gzip"], help="archive input compression selection")
    p.add_argument("--store-compression", choices=["zstd", "none"], help="intermediate shard payload compression")
    p.add_argument("--store-zstd-level", type=int, help="Zstandard compression level for intermediate shard payloads")
    p.add_argument("--gzip-backend", choices=["auto", "pigz", "isal", "python"])
    p.add_argument("--workers", type=int)
    p.add_argument("--apply-workers", type=int)
    p.add_argument("--apply-executor", choices=["thread", "process", "sequential", "isolated-process"])
    p.add_argument("--apply-max-tasks-per-child", type=int)
    p.add_argument("--decode-workers", type=int)
    p.add_argument("--keep-tag-stores", action="store_true")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--ingest-prescan", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("self-test", help="run offline regression tests")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("run", help="download, ingest/reuse, apply to rolling state, optionally export")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--from-tag")
    sp.add_argument("--to-tag")
    sp.add_argument("--no-export", action="store_true")

    sub.add_parser("export", help="export existing rolling state to XYZ PNG")

    sp = sub.add_parser("benchmark-decompress", help="benchmark zstd/gzip tar stream over local split parts")
    sp.add_argument("--parts", required=True, help="glob, e.g. wplace_downloads/<tag>/*.tar.zst.* or *.tar.gz.*")
    sp.add_argument("--mode", choices=["inflate", "tar-scan"], default="inflate")
    sp.add_argument("--backends", nargs="+", choices=["zstd", "pigz", "isal", "python"], default=None)

    sp = sub.add_parser("diagnose-rgb-transparency", help="sample RGB tiles and report inferred transparency")
    sp.add_argument("--parts", required=True)
    sp.add_argument("--sample", type=int, default=200)
    sp.add_argument("--out", type=Path)

    sub.add_parser("validate-store", help="validate pipeline state and rolling state store")
    sub.add_parser("clean-temp", help="remove transient *.tmp and *.part files under store/download roots")
    return p


def cfg_from_args(args) -> Config:
    cfg = Config.from_toml(args.config)
    overrides = {
        "store_root": args.store_root,
        "download_dir": args.download_dir,
        "xyz_output_dir": args.xyz_output_dir,
        "compression_backend": args.compression_backend,
        "store_compression": args.store_compression,
        "store_zstd_level": args.store_zstd_level,
        "gzip_backend": args.gzip_backend,
        "workers": args.workers,
        "apply_workers": args.apply_workers,
        "apply_executor": args.apply_executor,
        "apply_max_tasks_per_child": args.apply_max_tasks_per_child,
        "decode_workers": args.decode_workers,
    }
    if getattr(args, "keep_tag_stores", False):
        overrides["keep_tag_stores"] = True
    if getattr(args, "keep_archives", False):
        overrides["keep_archives"] = True
    if getattr(args, "ingest_prescan", False):
        overrides["ingest_prescan"] = True
    return cfg.with_overrides(**overrides)


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _expand_parts(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return validate_local_part_names(paths)


def clean_temp(cfg: Config) -> dict:
    removed = []
    for root in [cfg.store_root, cfg.download_dir, cfg.xyz_output_dir]:
        if not root.exists():
            continue
        for pat in ["**/*.tmp", "**/*.part"]:
            for p in root.glob(pat):
                try:
                    p.unlink()
                    removed.append(str(p))
                except FileNotFoundError:
                    pass
    return {"removed": len(removed), "files": sorted(removed)[:1000]}


def _cmd_self_test(args, _cfg: Config) -> dict:
    result = run_self_test()
    if not args.json:
        print("self-test OK")
    return result


def _cmd_run(args, cfg: Config) -> dict:
    return run_pipeline(cfg, limit=args.limit, from_tag=args.from_tag, to_tag=args.to_tag, no_export=args.no_export)


def _cmd_export(_args, cfg: Config) -> dict:
    return export_state_to_xyz(cfg, PaletteCodec(cfg))


def _cmd_benchmark_decompress(args, cfg: Config) -> dict:
    parts = _expand_parts(args.parts)
    results = []
    backends = args.backends or [cfg.compression_backend]
    for backend in backends:
        if backend == "zstd":
            run_cfg = cfg.with_overrides(compression_backend="zstd")
        else:
            run_cfg = cfg.with_overrides(compression_backend="gzip", gzip_backend=backend)
        results.append(benchmark_decompress(parts, run_cfg, mode=args.mode))
    return {"results": results}


def _cmd_diagnose_rgb_transparency(args, cfg: Config) -> dict:
    return diagnose_rgb_transparency(_expand_parts(args.parts), cfg, sample=args.sample, out=args.out)


def _cmd_validate_store(_args, cfg: Config) -> dict:
    return validate_store(cfg)


def _cmd_clean_temp(_args, cfg: Config) -> dict:
    return clean_temp(cfg)


COMMAND_HANDLERS = {
    "self-test": _cmd_self_test,
    "run": _cmd_run,
    "export": _cmd_export,
    "benchmark-decompress": _cmd_benchmark_decompress,
    "diagnose-rgb-transparency": _cmd_diagnose_rgb_transparency,
    "validate-store": _cmd_validate_store,
    "clean-temp": _cmd_clean_temp,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = cfg_from_args(args)
    try:
        handler = COMMAND_HANDLERS.get(args.cmd)
        if handler is None:
            parser.error(f"unknown command: {args.cmd}")
            return 2
        _print_json(handler(args, cfg))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
