"""Small, explicit tests for our handwritten constant-acceleration world."""
import unittest

from chapters.ch02_velocity.chapter02 import State, clip_acceleration, rollout, transition


class TestChapter02(unittest.TestCase):
    def test_known_transition(self):
        self.assertEqual(transition(State(0.0, 1.0), 2.0, 1.0), State(2.0, 3.0))

    def test_coasting_does_not_stop(self):
        self.assertEqual(transition(State(0.0, 2.0), 0.0, 2.0), State(4.0, 2.0))

    def test_same_position_different_velocity(self):
        right = rollout(State(5.0, 2.0), [0.0, 0.0], 1.0)
        left = rollout(State(5.0, -2.0), [0.0, 0.0], 1.0)
        self.assertEqual(right[-1], State(9.0, 2.0))
        self.assertEqual(left[-1], State(1.0, -2.0))

    def test_constant_acceleration_for_two_seconds(self):
        # x = 0 + 0*2 + 0.5*1*2^2 = 2; v = 0 + 1*2 = 2
        self.assertEqual(transition(State(0.0, 0.0), 1.0, 2.0), State(2.0, 2.0))

    def test_initial_state_is_not_modified(self):
        initial = State(1.0, 3.0)
        states = rollout(initial, [1.0, -1.0], 1.0)
        self.assertIs(states[0], initial)
        self.assertEqual(initial, State(1.0, 3.0))
        self.assertEqual(len(states), 3)

    def test_reject_invalid_timestep(self):
        with self.assertRaises(ValueError):
            transition(State(0.0, 0.0), 1.0, 0.0)
        with self.assertRaises(ValueError):
            rollout(State(0.0, 0.0), [], -1.0)

    def test_acceleration_limit(self):
        self.assertEqual(clip_acceleration(100, 2), 2)
        self.assertEqual(clip_acceleration(-100, 2), -2)
        self.assertEqual(clip_acceleration(1.5, 2), 1.5)


if __name__ == "__main__":
    unittest.main()
