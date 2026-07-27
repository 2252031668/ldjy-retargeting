"""Path-resolution helpers shared across example scripts.

resolve_mjcf_path(config_path):
    If the retarget config YAML sets optimizer.mjcf_path, return the MJCF xml
    file path verbatim (relative paths resolved against the config dir).
    Return None when mjcf_path is absent — caller falls back to its default.

mjcf_joint_order(mjcf_path) / qpos_reorder_perm(src_names, dst_names):
    Bridge the optimizer's URDF/Pinocchio qpos joint order to a consumer's
    expected order (MJCF) by joint name, so a URDF that declares joints in a
    different order does not land values on the wrong fingers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ldjy_retargeting.joint_mapping import qpos_reorder_perm


def resolve_mjcf_path(config_path: Path) -> Optional[str]:
    """Return the full MJCF xml path from ``optimizer.mjcf_path``.

    Returns the configured path verbatim (relative paths resolved against the
    config file's directory, the same convention ``optimizer.urdf_path`` uses).
    Returns ``None`` when ``optimizer.mjcf_path`` is absent so callers keep their
    bundled default.
    """
    import yaml
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    mjcf_rel = (cfg.get("optimizer") or {}).get("mjcf_path")
    if not mjcf_rel:
        return None
    return str((config_path.parent / mjcf_rel).resolve())


def mjcf_joint_order(mjcf_path) -> Optional[list]:
    """Return joint names in MuJoCo qpos order for an MJCF (None if path falsy)."""
    if not mjcf_path:
        return None
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(m.njnt)
    ]
