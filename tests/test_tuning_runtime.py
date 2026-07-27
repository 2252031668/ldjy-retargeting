import copy
import unittest

import numpy as np

from ldjy_retargeting.tuning.runtime import TuningRuntime


class TuningRuntimeTests(unittest.TestCase):
    def test_apply_config_replaces_retargeter_and_resets_state(self):
        config = {
            "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
            "retarget": {"scaling": 1.0},
        }
        runtime = TuningRuntime(config, hand_side="right")
        previous = runtime.retargeter
        runtime.retargeter.optimizer.last_qpos = np.ones(20)
        changed = copy.deepcopy(config)
        changed["retarget"]["scaling"] = 1.1

        runtime.apply_config(changed)

        self.assertIsNot(runtime.retargeter, previous)
        self.assertIsNone(runtime.retargeter.optimizer.last_qpos)
        self.assertEqual(runtime.config["retarget"]["scaling"], 1.1)


if __name__ == "__main__":
    unittest.main()
