from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MJCF = (
    ROOT
    / "ldjy_retargeting"
    / "assets"
    / "robots"
    / "openarm_hand"
    / "mjcf"
    / "openarm_bimanual_mano.xml"
)


class OpenArmTeleopControlTests(unittest.TestCase):
    def test_initial_pose_raises_only_the_selected_arm(self):
        import mujoco
        from ldjy_retargeting.openarm_control import ARM_HOME_QPOS, OpenArmTeleopControl

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        for side in ("left", "right"):
            data = mujoco.MjData(model)
            control = OpenArmTeleopControl(model, side)
            control.set_initial_pose(data)
            for index in range(1, 8):
                joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{index}"
                )
                self.assertAlmostEqual(
                    data.qpos[model.jnt_qposadr[joint]], ARM_HOME_QPOS[side][index - 1]
                )
                other_side = "right" if side == "left" else "left"
                other_joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{other_side}_joint{index}"
                )
                self.assertAlmostEqual(data.qpos[model.jnt_qposadr[other_joint]], 0.0)

    def test_selected_hand_overrides_only_its_twenty_actuators(self):
        import mujoco
        from ldjy_retargeting import Retargeter
        from ldjy_retargeting.openarm_control import (
            ARM_HOME_QPOS,
            OpenArmTeleopControl,
        )

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        for side in ("left", "right"):
            retargeter = Retargeter.from_config(
                {"optimizer": {"type": "AdaptiveOptimizerAnalytical"}}, side
            )
            command = np.linspace(0.05, 0.25, 20)
            control = OpenArmTeleopControl(model, side)
            targets = control.targets(retargeter.optimizer.robot.dof_joint_names, command)

            self.assertEqual(targets.shape, (model.nu,))
            for index in range(1, 8):
                selected_id = control.actuator_id(f"openarm_{side}_joint{index}")
                self.assertAlmostEqual(targets[selected_id], ARM_HOME_QPOS[side][index - 1])

                other_side = "right" if side == "left" else "left"
                other_id = control.actuator_id(f"openarm_{other_side}_joint{index}")
                self.assertAlmostEqual(targets[other_id], 0.0)

            expected = dict(zip(retargeter.optimizer.robot.dof_joint_names, command))
            for joint_name, value in expected.items():
                self.assertAlmostEqual(targets[control.actuator_id(joint_name)], value)

            other_hand = "right" if side == "left" else "left"
            for finger in ("thumb", "finger1", "finger2", "finger3", "finger4"):
                for index in range(1, 5):
                    joint_name = f"{other_hand}_{finger}_joint{index}"
                    self.assertAlmostEqual(targets[control.actuator_id(joint_name)], 0.0)


if __name__ == "__main__":
    unittest.main()
