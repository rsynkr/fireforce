"""Runtime settings for the local wildfire research API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_root: Path
    manifest_path: Path
    checkpoint_dir: Path
    device: str | None = None
    evaluation_summary_path: Path | None = None

    def __post_init__(self) -> None:
        if self.evaluation_summary_path is None:
            object.__setattr__(
                self,
                "evaluation_summary_path",
                self.repo_root / "artifacts" / "evaluation" / "100-test-v1" / "summary.json",
            )

    @classmethod
    def default(cls) -> "Settings":
        return cls(
            repo_root=REPO_ROOT,
            data_root=_path_from_env("WILDFIRE_DATA_ROOT", REPO_ROOT / "archive"),
            manifest_path=_path_from_env(
                "WILDFIRE_MANIFEST", REPO_ROOT / "artifacts" / "sample_manifest.json"
            ),
            checkpoint_dir=_path_from_env("WILDFIRE_CHECKPOINT_DIR", REPO_ROOT),
            device=os.environ.get("WILDFIRE_DEVICE") or None,
            evaluation_summary_path=_path_from_env(
                "WILDFIRE_EVALUATION_SUMMARY",
                REPO_ROOT / "artifacts" / "evaluation" / "100-test-v1" / "summary.json",
            ),
        )
