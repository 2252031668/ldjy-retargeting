import unittest


class SimulationTimingTests(unittest.TestCase):
    def test_120_hz_control_schedule_preserves_500_hz_physics(self):
        from ldjy_retargeting.simulation_timing import physics_steps_for_tick

        steps = [physics_steps_for_tick(tick, 0.002, 120) for tick in range(120)]

        self.assertEqual(sum(steps), 500)
        self.assertEqual(set(steps), {4, 5})

    def test_schedule_rejects_invalid_rates(self):
        from ldjy_retargeting.simulation_timing import physics_steps_for_tick

        with self.assertRaises(ValueError):
            physics_steps_for_tick(0, 0.0, 120)
        with self.assertRaises(ValueError):
            physics_steps_for_tick(0, 0.002, 0)


if __name__ == "__main__":
    unittest.main()
