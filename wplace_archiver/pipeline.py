from __future__ import annotations

import shutil

from .apply import apply_tag_store_to_state, delete_tag_store_if_needed
from .config import Config
from .download import download_assets
from .export import export_state_to_xyz
from .github import get_all_tags, get_release_assets, select_target_tags
from .ingest import ingest_tag_from_parts
from .palette import PaletteCodec
from .split_assets import select_release_split_assets
from .state import PipelineState
from .utils import sort_split_part_paths


def run_pipeline(cfg: Config, *, limit: int | None = None, from_tag: str | None = None, to_tag: str | None = None, no_export: bool = False) -> dict:
    cfg.store_root.mkdir(parents=True, exist_ok=True)
    state = PipelineState.load(cfg.pipeline_state_path, cfg)
    palette = PaletteCodec(cfg)

    tags_info = get_all_tags(cfg)
    target_tags = select_target_tags(tags_info, cfg, limit=limit, from_tag=from_tag, to_tag=to_tag)
    state.set_selected(target_tags)
    state.save(cfg.pipeline_state_path)

    summary = {"selected_tags": len(target_tags), "ingested": 0, "applied": 0, "skipped": 0}
    for tag in target_tags:
        if state.is_applied(tag):
            summary["skipped"] += 1
            continue
        tag_root = cfg.tags_root / tag
        try:
            if state.is_ingested(tag) and (tag_root / "manifest.json").exists():
                pass
            else:
                if state.is_ingested(tag) and not (tag_root / "manifest.json").exists():
                    # Inconsistent checkpoint. Recreate tag store.
                    shutil.rmtree(tag_root, ignore_errors=True)
                state.begin(tag, "download")
                state.save(cfg.pipeline_state_path)
                assets = select_release_split_assets(get_release_assets(cfg, tag), compression_preference=cfg.compression_backend)
                part_files = sort_split_part_paths(download_assets(cfg, tag, assets))
                snapshot = palette.snapshot()
                try:
                    state.begin(tag, "ingest")
                    state.save(cfg.pipeline_state_path)
                    ingest_tag_from_parts(tag, part_files, cfg, palette)
                except Exception:
                    palette.restore(snapshot)
                    raise
                state.mark_ingested(tag)
                state.save(cfg.pipeline_state_path)
                summary["ingested"] += 1
            state.begin(tag, "apply")
            state.save(cfg.pipeline_state_path)
            apply_tag_store_to_state(tag, cfg)
            palette.save()
            state.mark_applied(tag)
            state.clear_progress()
            state.save(cfg.pipeline_state_path)
            delete_tag_store_if_needed(tag, cfg)
            if not cfg.keep_archives:
                shutil.rmtree(cfg.download_dir / tag, ignore_errors=True)
            summary["applied"] += 1
        except Exception as exc:
            state.mark_failed(tag, state.in_progress.get("stage") or "unknown", exc)
            state.save(cfg.pipeline_state_path)
            raise
    if not no_export:
        summary["export"] = export_state_to_xyz(cfg, palette)
    return summary
