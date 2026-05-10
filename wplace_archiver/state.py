from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .utils import atomic_write_json, load_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PipelineState:
    format: str = "wplace-pipeline-state-v2"
    schema_version: int = 2
    config_hash: str = ""
    selected_tags: list[str] = field(default_factory=list)
    ingested_tags: list[str] = field(default_factory=list)
    applied_tags: list[str] = field(default_factory=list)
    failed_tags: dict[str, dict[str, Any]] = field(default_factory=dict)
    in_progress: dict[str, str | None] = field(default_factory=lambda: {"tag": None, "stage": None})
    state_manifest_hash: str | None = None
    palette_hash: str | None = None

    @classmethod
    def load(cls, path: Path, cfg: Config) -> "PipelineState":
        data = load_json(path, None)
        if data is None:
            return cls(config_hash=cfg.config_hash())
        st = cls(**{**cls(config_hash=cfg.config_hash()).to_dict(), **data})
        if st.config_hash != cfg.config_hash():
            # Do not fail hard; validation command reports this. Runtime may be path-only changes.
            pass
        return st

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "selected_tags": sorted(set(self.selected_tags)),
            "ingested_tags": sorted(set(self.ingested_tags)),
            "applied_tags": sorted(set(self.applied_tags)),
            "failed_tags": self.failed_tags,
            "in_progress": self.in_progress,
            "state_manifest_hash": self.state_manifest_hash,
            "palette_hash": self.palette_hash,
        }

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())

    def set_selected(self, tags: list[str]) -> None:
        self.selected_tags = list(tags)

    def begin(self, tag: str, stage: str) -> None:
        self.in_progress = {"tag": tag, "stage": stage}

    def clear_progress(self) -> None:
        self.in_progress = {"tag": None, "stage": None}

    def mark_ingested(self, tag: str) -> None:
        self.ingested_tags = sorted(set(self.ingested_tags) | {tag})
        self.failed_tags.pop(tag, None)

    def mark_applied(self, tag: str) -> None:
        self.applied_tags = sorted(set(self.applied_tags) | {tag})
        self.failed_tags.pop(tag, None)

    def mark_failed(self, tag: str, stage: str, exc: BaseException) -> None:
        self.failed_tags[tag] = {
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at": utc_now_iso(),
            "traceback_tail": traceback.format_exc(limit=6),
        }
        self.clear_progress()

    def is_ingested(self, tag: str) -> bool:
        return tag in set(self.ingested_tags)

    def is_applied(self, tag: str) -> bool:
        return tag in set(self.applied_tags)
