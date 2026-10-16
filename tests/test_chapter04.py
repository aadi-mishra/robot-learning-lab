"""Boss Fight checks: repeatability, physical constraints, and honest metrics."""
import unittest

from chapters.ch02_velocity.chapter02 import State
from chapters.ch03_feedback_control.chapter03 import p_control, pd_control
from chapters.ch04_reliability.chapter04 import (
    DT, TrialSpec, make_trial_specs, simulate_trial, score_trial,
    evaluate, summarize,
)


class TestChapter04(unittest.TestCase):
    def test_seed_is_reproducible(self):
        self.assertEqual(make_trial_specs(seed=3), make_trial_specs(seed=3))

    def test_different_seeds_change_trials(self):
        self.assertNotEqual(make_trial_specs(seed=3), make_trial_specs(seed=4))

    def test_four_scenarios_and_count(self):
        specs = make_trial_specs(seed=1, per_scenario=2)
        self.assertEqual(len(specs), 8)
        self.assertEqual([x.scenario for x in specs], ["easy"]*2 + ["hard"]*2 +
                         ["early_push"]*2 + ["late_push"]*2)

    def test_no_push_has_zero_kick(self):
        for spec in make_trial_specs(per_scenario=2):
            if spec.scenario in ("easy", "hard"):
                self.assertEqual((spec.kick_step, spec.velocity_kick), (None, 0.0))

    def test_zero_controller_coasts(self):
        spec = TrialSpec(State(1, 2), target=99, scenario="test")
        trace = simulate_trial(spec, lambda s, g: 0, dt=0.1, horizon=1)
        self.assertAlmostEqual(trace.states[-1].position, 3)
        self.assertAlmostEqual(trace.states[-1].velocity, 2)

    def test_push_changes_velocity_not_position_instantly(self):
        spec = TrialSpec(State(0, 0), target=0, scenario="test",
                         kick_step=1, velocity_kick=3)
        trace = simulate_trial(spec, lambda s, g: 0, dt=.1, horizon=.2)
        self.assertAlmostEqual(trace.states[1].position, 0)
        self.assertAlmostEqual(trace.states[1].velocity, 3)
        self.assertAlmostEqual(trace.states[2].position, .3)

    def test_saturation_is_enforced_after_controller(self):
        spec = TrialSpec(State(0, 0), target=10, scenario="test")
        trace = simulate_trial(spec, lambda s, g: 50, dt=.1, horizon=.1,
                               acceleration_limit=2)
        self.assertEqual(trace.requested, (50.0,))
        self.assertEqual(trace.executed, (2.0,))
        self.assertAlmostEqual(trace.states[-1].velocity, .2)
        self.assertAlmostEqual(score_trial(trace, hold_seconds=.05).saturation_fraction, 1)

    def test_at_goal_but_fast_is_not_success(self):
        spec = TrialSpec(State(10, 4), target=10, scenario="test")
        trace = simulate_trial(spec, lambda s, g: 0, dt=.05, horizon=1)
        self.assertFalse(score_trial(trace, hold_seconds=.5).success)

    def test_standing_at_goal_counts_as_success(self):
        spec = TrialSpec(State(2, 0), target=2, scenario="test")
        trace = simulate_trial(spec, lambda s, g: 0, dt=.05, horizon=1)
        score = score_trial(trace, hold_seconds=.5)
        self.assertTrue(score.success)
        self.assertEqual(score.settling_seconds, 0)

    def test_late_push_resets_settling(self):
        spec = TrialSpec(State(0, 0), target=0, scenario="test",
                         kick_step=100, velocity_kick=1)
        trace = simulate_trial(spec, lambda s, g: pd_control(s, g, 5, 5),
                               dt=.05, horizon=12)
        score = score_trial(trace)
        self.assertTrue(score.success)
        self.assertGreater(score.settling_seconds, 5)

    def test_identical_trials_for_controller_comparison(self):
        specs = make_trial_specs(seed=12, per_scenario=2)
        a = evaluate(specs, lambda s, g: p_control(s, g, 1))
        b = evaluate(specs, lambda s, g: pd_control(s, g, 1, 2))
        self.assertEqual([r.spec for r in a], [r.spec for r in b])
        self.assertNotEqual([r.trace.states for r in a], [r.trace.states for r in b])

    def test_summary_range(self):
        specs = make_trial_specs(seed=1, per_scenario=2)
        report = summarize(evaluate(specs, lambda s, g: pd_control(s, g, 1, 2)))
        self.assertGreaterEqual(report["success_rate"], 0)
        self.assertLessEqual(report["success_rate"], 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            make_trial_specs(per_scenario=0)
        with self.assertRaises(ValueError):
            simulate_trial(TrialSpec(State(0,0), 0, "test"), lambda s,g: 0, dt=0)
        with self.assertRaises(ValueError):
            simulate_trial(TrialSpec(State(0,0), 0, "test"), lambda s,g: 0, acceleration_limit=-1)
        with self.assertRaises(ValueError):
            simulate_trial(TrialSpec(State(0,0), 0, "test"), lambda s,g: float('nan'))


if __name__ == "__main__":
    unittest.main()
