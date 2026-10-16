# Chapter 4 — Boss Fight: Make the Dot Behave

**The experiment:** A PD controller succeeded in Chapter 3 under a convenient initial state. Can it reliably reach a target when the starting state, goal, actuator commands, and external disturbances change?

This chapter **reuses Chapter 2's dynamics** (`State`, `transition`, `clip_acceleration`) and **Chapter 3's controllers** (`p_control`, `pd_control`). No new physics engine, no machine learning yet.

## Run

From the repository root:

```bash
uv run python -m chapters.ch04_reliability.chapter04
uv run python -m unittest discover -s tests -v
```

The program evaluates **100 seeded trials** in four scenarios (25 each):

| Scenario | Starting conditions | Push |
| --- | --- | --- |
| Easy | Moderate initial positions and velocities | None |
| Hard | Wider initial positions, velocities, and goals | None |
| Early push | Moderate starts | Sudden velocity change at 3 s |
| Late push | Moderate starts | Sudden velocity change at 9 s |

Controllers see the **same trials**, and both face a maximum acceleration of **±2 m/s²**. Initial states, goals, and push magnitudes are randomized using a **fixed seed**. A push is modeled as an *instantaneous external change in velocity*, not as a command sent through the actuator and not as a realistic force-contact model.

**Success** means |x − goal| ≤ **0.35 m** and |velocity| ≤ **0.18 m/s** throughout the final **0.75 seconds** of a **12-second** episode. These are demonstration thresholds, not universal robotics standards.

The report also prints:
- Mean absolute final position error (all trials)
- Mean maximum overshoot, measured relative to the starting side of the goal
- Median settling time **among successful trials only** (not comparable alone when success rates differ)
- Fraction of commands clipped by the actuator

## Artifacts

- `artifacts/four_scenarios.png`: one identical initial state per scenario for P and PD.
- `artifacts/reliability_scorecard.png`: success percentages for each scenario.
- `artifacts/failure_gallery.png`: four worst PD failures (or four hardest cases, if none fail).
- `artifacts/push_recovery.gif`: one example of a sudden velocity kick and controller recovery.

## Boss Fight: challenge yourself

**Predict:** Which scenario will hurt the success rate most: wide initial conditions, early pushes, or late pushes? Why?

**Run:** First use the default PD gains (`Kp=1`, `Kd=2`). Don't assume 100% success because the Chapter 3 demo worked.

**Inspect:** Which trajectories fail? Is the controller inherently unstable, or does it simply run out of time under limited acceleration?

**Tune:** Change **only** the controller gains, not the success criteria or cases:

```bash
uv run python -m chapters.ch04_reliability.chapter04 --kp 1.6 --kd 2.5
```

**Challenge:** Find gains that increase the number of successful trials. Watch for overshoot, saturation, and settling time. Compare on the SAME seed and same scenarios. You may also vary `--seed` as a *new* unseen test after tuning—report both results, don't cherry-pick.

**Reflect:** Why is a single pretty demo not sufficient evidence of reliability? What would have to change when scaling this test from a dot to a 7-DOF arm, two coordinated arms, or a balancing humanoid?

## From tiny dot to real robots

- **Single arm:** random joint configurations, tool targets, joint limits, payloads, varying joint friction, position/velocity estimation errors.
- **Dual arm:** coordinated task success, object pose, closed-chain constraints, robot-robot collision safety, timing/synchronization failures.
- **Humanoid:** floating-base dynamics, balance, contacts, support surfaces, changing contact modes and disturbances.
- **Learned policy:** same seeded episode suite, separate held-out scenarios, command-vs-execution telemetry, safety supervision and physical acceptance criteria.

Our 1D suite demonstrates the *evaluation discipline*, not real hardware safety or a validated sim-to-real performance guarantee.

**Next: Chapter 5 — Build Our First Robot Arm.**
