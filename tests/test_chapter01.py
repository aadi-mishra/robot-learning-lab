"""Physics stays simple, but the result should always be correct."""
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "chapters/ch01_stupid_universe/chapter01.py"
spec = importlib.util.spec_from_file_location("chapter01", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestChapter01(unittest.TestCase):
    def test_transition(self):
        self.assertEqual(module.transition(4.0, -3.0), 1.0)

    def test_trajectory(self):
        self.assertEqual(
            module.rollout(0.0, [1, 1, 2, -3, 4]),
            [0, 1, 2, 4, 1, 5],
        )

    def test_large_actions_are_legal_in_this_world(self):
        self.assertEqual(module.transition(1.0, 50.0), 51.0)

    def test_zero_actions_do_not_move_dot(self):
        self.assertEqual(module.rollout(5.0, [0, 0]), [5, 5, 5])


if __name__ == "__main__":
    unittest.main()
