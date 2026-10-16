"""Chapter 3 tests. Run: uv run python -m unittest discover -s tests -v"""
import unittest

from chapters.ch02_velocity.chapter02 import State
from chapters.ch03_feedback_control.chapter03 import (
    p_control, pd_control, run_closed_loop,
)


class TestChapter03(unittest.TestCase):
    def test_p_controller_uses_position_error(self):
        self.assertAlmostEqual(p_control(State(8, 25), target=10, kp=2), 4)

    def test_p_at_target_cannot_brake(self):
        self.assertAlmostEqual(p_control(State(10, 5), target=10, kp=1), 0)

    def test_pd_at_target_brakes_positive_velocity(self):
        self.assertAlmostEqual(pd_control(State(10, 5), target=10, kp=1, kd=2), -10)

    def test_pd_opposes_negative_velocity(self):
        self.assertAlmostEqual(pd_control(State(10, -5), target=10, kp=1, kd=2), +10)

    def test_action_chosen_before_world_step(self):
        ep = run_closed_loop(State(0, 0), 10, lambda s, goal: p_control(s, goal, 1),
                             dt=1, steps=1)
        self.assertEqual(ep.requested, (10.0,))
        self.assertEqual(ep.executed, (10.0,))
        self.assertAlmostEqual(ep.states[-1].position, 5)
        self.assertAlmostEqual(ep.states[-1].velocity, 10)

    def test_requested_and_executed_separated_by_actuator(self):
        ep = run_closed_loop(State(0, 0), 10, lambda s, g: p_control(s, g, 1),
                             dt=1, steps=1, max_acceleration=2)
        self.assertEqual(ep.requested, (10.0,))
        self.assertEqual(ep.executed, (2.0,))
        self.assertAlmostEqual(ep.states[-1].position, 1)
        self.assertAlmostEqual(ep.states[-1].velocity, 2)

    def test_actuator_can_brake_both_directions(self):
        ep = run_closed_loop(State(12, 3), 10,
                             lambda s, g: pd_control(s, g, kp=1, kd=2),
                             dt=1, steps=1, max_acceleration=2)
        self.assertEqual(ep.requested, (-8.0,))
        self.assertEqual(ep.executed, (-2.0,))

    def test_trajectory_dimensions(self):
        ep = run_closed_loop(State(0, 0), 10, lambda s, g: 0.0, .1, 7)
        self.assertEqual((len(ep.states), len(ep.requested), len(ep.executed)), (8, 7, 7))

    def test_p_controller_overshoots_in_ideal_unlimited_world(self):
        ep = run_closed_loop(State(0, 0), 10, lambda s, g: p_control(s, g, 1),
                             dt=.05, steps=150)
        self.assertGreater(max(s.position for s in ep.states), 10.5)

    def test_pd_controller_settles(self):
        ep = run_closed_loop(State(0, 0), 10,
                             lambda s, g: pd_control(s, g, 1, 2),
                             dt=.05, steps=400)
        self.assertLess(abs(ep.states[-1].position - 10), 0.02)
        self.assertLess(abs(ep.states[-1].velocity), 0.02)

    def test_validation(self):
        controller = lambda s, g: 0.0
        with self.assertRaises(ValueError):
            run_closed_loop(State(0, 0), 10, controller, 0, 1)
        with self.assertRaises(ValueError):
            run_closed_loop(State(0, 0), 10, controller, .1, -1)
        with self.assertRaises(ValueError):
            run_closed_loop(State(0, 0), 10, controller, .1, 1, max_acceleration=0)


if __name__ == '__main__':
    unittest.main()
