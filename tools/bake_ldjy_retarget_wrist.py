#!/usr/bin/env python3
"""Place LDJY retarget_wrist at MANO joint 0 from the saved zero-pose fit.

Usage:
    uv run --extra wilor python tools/bake_ldjy_retarget_wrist.py --apply
    uv run python tools/build_ldjy_urdf.py
    uv run python tools/build_ldjy_mjcf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

from ldjy_retargeting.mano_ldjy_overlay import apply_registration

try:
    from tools.ldjy_asset_frames import (
        DEFAULT_ROOT_PALM_TRANSLATION,
        RETARGET_WRIST_POINT_IN_ORIGINAL_ROOT,
    )
except ModuleNotFoundError:
    from ldjy_asset_frames import DEFAULT_ROOT_PALM_TRANSLATION, RETARGET_WRIST_POINT_IN_ORIGINAL_ROOT


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "ldjy_retargeting" / "assets" / "robots" / "ldjy_hand"
REFERENCE_PATH = ASSET_DIR / "mano_ldjy_reference.yaml"
CALIBRATION_PATH = ASSET_DIR / "retarget_wrist_calibration.yaml"
MANO_LEFT_FROM_RIGHT = np.diag((1.0, -1.0, 1.0))


def _mano_joint0(reference: dict) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "example"))
    from mano_viewer import MANO_MODEL_PATH, MANOModel

    mano = reference["mano"]
    registration = reference["mano_to_ldjy_registration"]
    model = MANOModel(str(MANO_MODEL_PATH))
    _, joints, _ = model.compute(
        np.asarray(mano["betas"], dtype=float),
        np.asarray(mano["hand_pose"], dtype=float),
        np.asarray(mano.get("global_orient", (0.0, 0.0, 0.0)), dtype=float),
        np.asarray(mano.get("translation", (0.0, 0.0, 0.0)), dtype=float),
        float(mano.get("scale", 1.0)),
    )
    return apply_registration(
        joints[0:1],
        np.asarray(registration["rotation_axis_angle"], dtype=float),
        np.asarray(registration["translation"], dtype=float),
        float(registration["scale"]),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write calibration and update the reference")
    args = parser.parse_args()

    reference = yaml.safe_load(REFERENCE_PATH.read_text(encoding="utf-8")) or {}
    current_point = _mano_joint0(reference)
    point_right = RETARGET_WRIST_POINT_IN_ORIGINAL_ROOT["right"] + current_point
    point_left = point_right @ MANO_LEFT_FROM_RIGHT
    points = {"right": point_right, "left": point_left}
    calibration = {
        "version": 1,
        **{
            side: {
                "mano_joint0_in_original_root": point.tolist(),
                "root_to_palm_translation": (DEFAULT_ROOT_PALM_TRANSLATION - point).tolist(),
            }
            for side, point in points.items()
        },
    }
    updated_translation = (
        np.asarray(reference["mano_to_ldjy_registration"]["translation"], dtype=float) - current_point
    )

    print("right MANO joint 0 in current LDJY root:", current_point.tolist())
    print("right MANO joint 0 from original LDJY root:", calibration["right"]["mano_joint0_in_original_root"])
    print("right root -> palm:", calibration["right"]["root_to_palm_translation"])
    print("reference registration translation:", updated_translation.tolist())
    if not args.apply:
        print("preview only; rerun with --apply to write files")
        return

    CALIBRATION_PATH.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")
    reference["mano_to_ldjy_registration"]["translation"] = updated_translation.tolist()
    REFERENCE_PATH.write_text(yaml.safe_dump(reference, sort_keys=False), encoding="utf-8")
    print(f"wrote {CALIBRATION_PATH}")
    print(f"updated {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
