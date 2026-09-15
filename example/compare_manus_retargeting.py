"""Compare the Raw and Ergonomics Hybrid MANUS retargeters on one recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ldjy_retargeting.tuning.replay import ManusReplay
from ldjy_retargeting.tuning.runtime import TuningRuntime
from ldjy_retargeting.tuning.session import TuningSession


def _runtime(name: str, side: str) -> TuningRuntime:
    path = PROJECT_ROOT / "example" / "config" / name
    session = TuningSession(path)
    return TuningRuntime(session.config, side, path.parent)


def _summary(name: str, diagnostics: list[dict], qpos: list[np.ndarray], limits: np.ndarray) -> str:
    if not diagnostics:
        return f"{name}: no compatible frames in this recording"
    solve_ms = np.asarray([item.get("solve_ms", np.nan) for item in diagnostics], dtype=np.float64)
    failed = sum(item.get("failure") is not None for item in diagnostics)
    values = np.stack(qpos) if qpos else np.empty((0, len(limits)))
    saturated = (np.isclose(values, limits[:, 0], atol=1e-4) | np.isclose(values, limits[:, 1], atol=1e-4))
    jitter = 0.0 if len(values) < 2 else float(np.median(np.linalg.norm(np.diff(values, axis=0), axis=1)))
    pad_errors = [np.mean(np.linalg.norm(item["pad_actual_m"] - item["pad_targets_m"], axis=1))
                  for item in diagnostics if "pad_targets_m" in item]
    return (
        f"{name}: frames={len(diagnostics)} failure={failed / max(len(diagnostics), 1):.1%} "
        f"solve_p95={np.nanpercentile(solve_ms, 95):.2f}ms "
        f"limit_hit={saturated.mean():.1%} jitter_median={jitter:.4f}rad "
        + (f"pad_error_median={np.median(pad_errors) * 1000:.2f}mm" if pad_errors else "pad_error=n/a")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较 MANUS Raw 与 Ergonomics Hybrid 重定向")
    parser.add_argument("record", type=Path, help="outputs/tuning_records/manus/<timestamp>")
    args = parser.parse_args(argv)
    replay = ManusReplay(args.record)
    raw = _runtime("manus_full_skeleton.yaml", replay.hand_side)
    hybrid = _runtime("manus_ergonomics_hybrid.yaml", replay.hand_side)
    calibration = replay.calibration()
    if calibration is None:
        raise RuntimeError("该记录没有 Raw 中立标定快照，无法运行 Raw 基线")
    raw.optimizer.set_calibration(calibration)
    hybrid_calibration = replay.ergonomics_hybrid_calibration()
    if hybrid_calibration is None:
        path = hybrid.manus_ergonomics_calibration_path(replay.record.glove_id, replay.hand_side)
        if path.is_file():
            from ldjy_retargeting.manus import ManusErgonomicsCalibration
            hybrid_calibration = ManusErgonomicsCalibration.load(path)
    if hybrid_calibration is not None:
        hybrid.optimizer.set_calibration(hybrid_calibration)
    results: dict[str, tuple[list[dict], list[np.ndarray]]] = {"raw": ([], []), "hybrid": ([], [])}
    for index in range(replay.record.frame_count):
        frame = replay.input_at(index)
        for name, runner in (("raw", raw), ("hybrid", hybrid)):
            if name == "hybrid" and hybrid_calibration is None:
                continue
            try:
                qpos, diagnostics = runner.process_manus(frame)
            except Exception as exc:
                diagnostics, qpos = {"failure": str(exc), "solve_ms": np.nan}, np.zeros(runner.optimizer.num_joints)
            results[name][0].append(diagnostics)
            results[name][1].append(qpos)
    print(_summary("Raw Full Skeleton", *results["raw"], raw.optimizer.limits))
    print(_summary("Ergonomics Hybrid", *results["hybrid"], hybrid.optimizer.limits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
