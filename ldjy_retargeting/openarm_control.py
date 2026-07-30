"""Compose fixed-arm OpenArm controls around one LDJY hand command."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .joint_mapping import qpos_reorder_perm


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "robots" / "openarm_hand"
OPENARM_MJCF_PATH = ASSET_DIR / "mjcf" / "openarm_bimanual_mano.xml"
ARM_HOME_QPOS = {
    "left": np.array((-0.510, -0.541, -0.0628, 1.380, 0.958, 0.251, 0.989)),
    "right": np.array((0.510, 0.541, 0.0628, 1.380, -0.958, -0.251, -0.989)),
}
FINGERS = ("thumb", "finger1", "finger2", "finger3", "finger4")


class OpenArmTeleopControl:
    """Hold both arms while replacing one hand's targets with retargeted qpos."""

    def __init__(self, model: mujoco.MjModel, hand_side: str):
        if hand_side not in ARM_HOME_QPOS:
            raise ValueError(f"Unknown hand side {hand_side!r}")
        self.model = model
        self.hand_side = hand_side
        self._actuator_ids = self._build_actuator_ids()
        self._hand_actuator_ids = [
            self.actuator_id(f"{hand_side}_{finger}_joint{index}")
            for finger in FINGERS
            for index in range(1, 5)
        ]
        self._arm_targets = self._zero_targets()
        for index, target in enumerate(ARM_HOME_QPOS[hand_side], start=1):
            actuator_id = self.actuator_id(f"openarm_{hand_side}_joint{index}")
            lower, upper = model.actuator_ctrlrange[actuator_id]
            if not lower <= target <= upper:
                raise ValueError(
                    f"OpenArm {hand_side} arm home joint{index}={target} is outside "
                    f"its actuator range [{lower}, {upper}]"
                )
            self._arm_targets[actuator_id] = target

    def actuator_id(self, joint_name: str) -> int:
        try:
            return self._actuator_ids[joint_name]
        except KeyError as error:
            raise ValueError(f"OpenArm model has no position actuator for {joint_name}") from error

    def targets(self, hand_joint_names, hand_qpos: np.ndarray) -> np.ndarray:
        """Return all 54 actuator targets for the selected hand command."""
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        if hand_qpos.shape != (20,):
            raise ValueError(f"Expected 20 selected-hand joint values, got {hand_qpos.shape}")
        actuator_names = [
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                self.model.actuator_trnid[actuator_id, 0],
            )
            for actuator_id in self._hand_actuator_ids
        ]
        permutation = qpos_reorder_perm(hand_joint_names, actuator_names)
        if permutation is None:
            raise ValueError("Retargeter joint names do not match the selected OpenArm hand")
        targets = self._arm_targets.copy()
        targets[self._hand_actuator_ids] = hand_qpos[permutation]
        return targets

    def set_initial_pose(self, data: mujoco.MjData) -> None:
        """Place the selected arm at its home pose before the first simulation step."""
        for index, target in enumerate(ARM_HOME_QPOS[self.hand_side], start=1):
            actuator_id = self.actuator_id(f"openarm_{self.hand_side}_joint{index}")
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            data.qpos[self.model.jnt_qposadr[joint_id]] = target

    def _build_actuator_ids(self) -> dict[str, int]:
        actuator_ids = {}
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None or joint_name in actuator_ids:
                raise ValueError("OpenArm actuators must have unique joint transmissions")
            actuator_ids[joint_name] = actuator_id
        if len(actuator_ids) != 54:
            raise ValueError(f"Expected 54 OpenArm position actuators, got {len(actuator_ids)}")
        return actuator_ids

    def _zero_targets(self) -> np.ndarray:
        targets = np.zeros(self.model.nu, dtype=np.float64)
        limited = self.model.actuator_ctrllimited.astype(bool)
        ranges = self.model.actuator_ctrlrange
        targets[limited] = np.clip(0.0, ranges[limited, 0], ranges[limited, 1])
        return targets


__all__ = ["ARM_HOME_QPOS", "OPENARM_MJCF_PATH", "OpenArmTeleopControl"]
