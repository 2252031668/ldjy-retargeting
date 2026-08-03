"""Persist one tuning session's virtual-tip settings before exporting assets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ldjy_retargeting.retarget_tip_frames import normalize_tip_offsets

if TYPE_CHECKING:
    from .session import TuningSession


def persist_tip_offsets(session: "TuningSession", output_path: str | Path) -> dict[str, dict[str, float]]:
    """Save the active YAML, then mirror its tip offsets to the asset source."""
    session.save()
    offsets = normalize_tip_offsets(session.config.get("tip_offsets"))
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"version": 1, "units": "mm", "fingers": offsets},
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return offsets
