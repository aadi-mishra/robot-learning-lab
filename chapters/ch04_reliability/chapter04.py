"""Chapter 4 — Boss Fight: can our controller handle a bad day?

Run from the repository root:
    uv run python -m chapters.ch04_reliability.chapter04

We reuse Chapter 2's physics and Chapter 3's controller. This chapter adds:
  * deterministic randomized trial generation (the same trials for both policies)
  * discrete external velocity impulses (pushes), outside actuator acceleration
  * explicit pass/fail criteria and meaningful evaluation metrics
  * failure analysis and reproducible visual artifacts

The disturbances here are *instantaneous velocity kicks*, not force pulses.
This is a toy robustness test, NOT a guarantee of real-robot safety.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from statistics import median
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from chapters.ch02_velocity.chapter02 import (
    State, transition, clip_acceleration, TEAL, ORANGE, INK, BACKGROUND,
)
from chapters.ch03_feedback_control.chapter03 import p_control, pd_control

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PURPLE = "#7655A6"
RED = "#BE4957"

# An evaluation is meaningless without defining *what counts as success*.
DT = 0.05                     # seconds per physics/controller step
HORIZON = 12.0               # seconds, fixed time budget
MAX_ACCELERATION = 2.0       # m/s^2, same actuator for everyone
POSITION_TOLERANCE = 0.35    # metres
VELOCITY_TOLERANCE = 0.18    # m/s
HOLD_SECONDS = 0.75          # final uninterrupted time in tolerance band
DEFAULT_SEED = 2026

Controller = Callable[[State, float], float]


@dataclass(frozen=True)
class TrialSpec:
    initial: State
    target: float
    scenario: str
    kick_step: int | None = None
    velocity_kick: float = 0.0


@dataclass(frozen=True)
class TrialTrace:
    spec: TrialSpec
    states: tuple[State, ...]  # length = steps + 1
    requested: tuple[float, ...]
    executed: tuple[float, ...]
    dt: float


@dataclass(frozen=True)
class TrialScore:
    success: bool
    settling_seconds: float | None
    final_position_error: float
    final_speed: float
    peak_overshoot: float
    saturation_fraction: float
    control_effort: float


@dataclass(frozen=True)
class EvaluatedTrial:
    spec: TrialSpec
    trace: TrialTrace
    score: TrialScore


def make_trial_specs(seed: int = DEFAULT_SEED, per_scenario: int = 25,
                     dt: float = DT) -> tuple[TrialSpec, ...]:
    """Draw seeded scenarios. Policy A and B get exactly these same trials."""
    if per_scenario <= 0 or dt <= 0:
        raise ValueError("per_scenario and dt must be positive")
    rng = random.Random(seed)
    trials: list[TrialSpec] = []

    for scenario in ("easy", "hard", "early_push", "late_push"):
        for _ in range(per_scenario):
            if scenario == "easy":
                start = State(rng.uniform(-6, 6), rng.uniform(-1, 1))
                target = rng.uniform(-6, 6)
            elif scenario == "hard":
                start = State(rng.uniform(-15, 15), rng.uniform(-4, 4))
                target = rng.uniform(-12, 12)
            else:
                start = State(rng.uniform(-10, 10), rng.uniform(-2, 2))
                target = rng.uniform(-8, 8)

            # A push changes velocity *after* an ordinary world step.
            # Its magnitude is large enough to expose poor recovery.
            kick_time = {"early_push": 3.0, "late_push": 9.0}.get(scenario)
            step = round(kick_time / dt) if kick_time is not None else None
            if step is None:
                velocity_kick = 0.0
            else:
                velocity_kick = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 4.0)
            trials.append(TrialSpec(start, target, scenario, step, velocity_kick))
    return tuple(trials)


def simulate_trial(spec: TrialSpec, controller: Controller,
                   *, dt: float = DT, horizon: float = HORIZON,
                   acceleration_limit: float = MAX_ACCELERATION) -> TrialTrace:
    """Observe -> request -> clip -> step physics -> possibly apply push."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(acceleration_limit) or acceleration_limit <= 0:
        raise ValueError("acceleration_limit must be finite and positive")
    if spec.kick_step is not None and spec.kick_step < 1:
        raise ValueError("kick_step must refer to a future state")
    steps = round(horizon / dt)
    if steps < 1:
        raise ValueError("simulation must include at least one step")
    if spec.kick_step is not None and spec.kick_step > steps:
        raise ValueError("kick_step outside simulation")

    state = spec.initial
    states = [state]
    requested: list[float] = []
    executed: list[float] = []
    for i in range(steps):
        command = float(controller(state, spec.target))
        if not math.isfinite(command):
            raise ValueError("controller returned non-finite command")
        physical_action = clip_acceleration(command, acceleration_limit)
        state = transition(state, physical_action, dt)
        if spec.kick_step == i + 1:
            state = State(position=state.position,
                          velocity=state.velocity + spec.velocity_kick)
        requested.append(command)
        executed.append(physical_action)
        states.append(state)
    return TrialTrace(spec, tuple(states), tuple(requested), tuple(executed), dt)


def score_trial(trace: TrialTrace,
                position_tolerance: float = POSITION_TOLERANCE,
                velocity_tolerance: float = VELOCITY_TOLERANCE,
                hold_seconds: float = HOLD_SECONDS,
                acceleration_limit: float = MAX_ACCELERATION) -> TrialScore:
    """Success = close to the goal AND nearly stopped throughout final window.

    Settling time = earliest time after which it stays inside both tolerances
    until the end, provided the final hold requirement is satisfied.
    """
    if position_tolerance <= 0 or velocity_tolerance <= 0 or hold_seconds <= 0:
        raise ValueError("all tolerances and hold_seconds must be positive")
    if acceleration_limit <= 0:
        raise ValueError("acceleration_limit must be positive")
    states = trace.states
    steps_in_hold = math.ceil(hold_seconds / trace.dt)
    if steps_in_hold >= len(states):
        raise ValueError("hold_seconds exceeds simulation")
    in_bounds = [abs(s.position - trace.spec.target) <= position_tolerance
                 and abs(s.velocity) <= velocity_tolerance for s in states]
    success = all(in_bounds[-(steps_in_hold + 1):])

    settling: float | None = None
    if success:
        # Work backwards to find the first state in the final uninterrupted
        # run within bounds. A push that knocks us out resets the clock.
        i = len(in_bounds) - 1
        while i > 0 and in_bounds[i - 1]:
            i -= 1
        settling = i * trace.dt

    x0 = trace.spec.initial.position
    goal = trace.spec.target
    if x0 < goal:
        overshoot = max(0.0, max(s.position for s in states) - goal)
    elif x0 > goal:
        overshoot = max(0.0, goal - min(s.position for s in states))
    else:
        overshoot = max(abs(s.position - goal) for s in states)

    sat = sum(abs(r) > acceleration_limit + 1e-10 for r in trace.requested)
    n = len(trace.requested)
    return TrialScore(
        success=success,
        settling_seconds=settling,
        final_position_error=abs(states[-1].position - goal),
        final_speed=abs(states[-1].velocity),
        peak_overshoot=overshoot,
        saturation_fraction=sat / n,
        control_effort=sum(abs(a) * trace.dt for a in trace.executed),
    )


def evaluate(specs: tuple[TrialSpec, ...], controller: Controller,
             *, dt: float = DT, horizon: float = HORIZON,
             acceleration_limit: float = MAX_ACCELERATION) -> tuple[EvaluatedTrial, ...]:
    return tuple(EvaluatedTrial(spec, trace, score_trial(trace, acceleration_limit=acceleration_limit))
                 for spec in specs
                 for trace in (simulate_trial(spec, controller, dt=dt, horizon=horizon,
                                              acceleration_limit=acceleration_limit),))


def summarize(results: tuple[EvaluatedTrial, ...]) -> dict[str, float | None]:
    if not results:
        raise ValueError("no trials to summarize")
    scores = [r.score for r in results]
    settled = [s.settling_seconds for s in scores if s.settling_seconds is not None]
    return {
        "success_rate": sum(s.success for s in scores) / len(scores),
        "median_settling_successes": median(settled) if settled else None,
        "mean_final_error": sum(s.final_position_error for s in scores) / len(scores),
        "mean_overshoot": sum(s.peak_overshoot for s in scores) / len(scores),
        "mean_saturation_fraction": sum(s.saturation_fraction for s in scores) / len(scores),
    }


def _style(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(alpha=0.19)
    ax.spines[["top", "right"]].set_visible(False)


def save_trajectories(specs: tuple[TrialSpec, ...], p_results: tuple[EvaluatedTrial, ...],
                      pd_results: tuple[EvaluatedTrial, ...], path: Path) -> None:
    scenarios = ("easy", "hard", "early_push", "late_push")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for ax, name in zip(axs.flat, scenarios):
        _style(ax)
        matches = [i for i, s in enumerate(specs) if s.scenario == name]
        i = matches[0]
        p, pd = p_results[i], pd_results[i]
        times = [j * DT for j in range(len(p.trace.states))]
        ax.plot(times, [s.position for s in p.trace.states], color=ORANGE,
                lw=1.8, label="P")
        ax.plot(times, [s.position for s in pd.trace.states], color=TEAL,
                lw=2.3, label="PD")
        ax.axhline(specs[i].target, color=INK, ls="--", alpha=.7, label="target")
        if specs[i].kick_step is not None:
            ax.axvline(specs[i].kick_step * DT, color=RED, ls=":", alpha=.9,
                       label="push")
        ax.set_title(f"{name.replace('_',' ').title()} · target={specs[i].target:.1f} m")
        ax.set_ylabel("Position [m]")
        ax.legend(frameon=False, fontsize=9)
    for ax in axs[-1, :]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Same trial, same actuator, two controllers", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=165)
    plt.close(fig)


def save_scorecard(p_results: tuple[EvaluatedTrial, ...],
                   pd_results: tuple[EvaluatedTrial, ...], path: Path) -> None:
    scenarios = ("easy", "hard", "early_push", "late_push")
    p_rates = []
    pd_rates = []
    for name in scenarios:
        p_rates.append(100 * summarize(tuple(r for r in p_results if r.spec.scenario == name))["success_rate"])
        pd_rates.append(100 * summarize(tuple(r for r in pd_results if r.spec.scenario == name))["success_rate"])
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    _style(ax)
    y = list(range(4))
    ax.barh([z - .19 for z in y], p_rates, height=.35, color=ORANGE, label="P")
    ax.barh([z + .19 for z in y], pd_rates, height=.35, color=TEAL, label="PD")
    for z, val in zip([a-.19 for a in y], p_rates):
        ax.text(min(val+1.3, 101), z, f"{val:.0f}%", va="center", fontsize=9, color=INK)
    for z, val in zip([a+.19 for a in y], pd_rates):
        ax.text(min(val+1.3, 101), z, f"{val:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [s.replace('_',' ').title() for s in scenarios])
    ax.set_xlim(0, 109)
    ax.invert_yaxis()
    ax.set_xlabel("Success rate [%]")
    ax.set_title("Reliability is a distribution, not a demo", color=INK, weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=165)
    plt.close(fig)


def save_failure_gallery(pd_results: tuple[EvaluatedTrial, ...], path: Path) -> None:
    failed = [r for r in pd_results if not r.score.success]
    failed.sort(key=lambda r: r.score.final_position_error + 1.5 * r.score.final_speed,
                reverse=True)
    candidates = failed[:4] if failed else sorted(
        pd_results, key=lambda r: r.score.final_position_error, reverse=True)[:4]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor(BACKGROUND)
    for ax, result in zip(axs.flat, candidates):
        _style(ax)
        trace = result.trace
        times = [j * trace.dt for j in range(len(trace.states))]
        ax.plot(times, [s.position for s in trace.states], color=TEAL, lw=2.3)
        ax.axhline(result.spec.target, color=INK, ls="--", alpha=.7)
        ax.axhspan(result.spec.target-POSITION_TOLERANCE,
                   result.spec.target+POSITION_TOLERANCE, color=TEAL, alpha=.12)
        if result.spec.kick_step is not None:
            ax.axvline(result.spec.kick_step * trace.dt, color=RED, ls=":")
        verdict = "PASS" if result.score.success else "FAIL"
        ax.set_title(f"{verdict} · {result.spec.scenario.replace('_', ' ')} · "
                     f"end error={result.score.final_position_error:.2f} m")
        ax.set(xlabel="Time [s]", ylabel="Position [m]")
    fig.suptitle("Investigate the failures instead of hiding them", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=165)
    plt.close(fig)


def demo_trial() -> TrialSpec:
    return TrialSpec(State(-6.0, 0.0), target=6.0, scenario="demo_push",
                     kick_step=round(7.0 / DT), velocity_kick=-4.0)


def save_push_recovery_gif(path: Path, kp: float, kd: float) -> None:
    spec = demo_trial()
    trace = simulate_trial(spec, lambda s, g: pd_control(s, g, kp, kd))
    # The GIF follows the same x-axis for all frames, and shows velocity too.
    xs = [s.position for s in trace.states] + [spec.target]
    xmin, xmax = min(xs)-2.0, max(xs)+2.0
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set(xlim=(xmin, xmax), ylim=(-.9, .95), xlabel="Position [m]")
    ax.set_yticks([])
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.axhline(0, color=INK, alpha=.27)
    ax.axvline(spec.target, color=INK, ls="--", alpha=.6)
    ax.text(spec.target, .68, "GOAL", ha="center", color=INK, weight="bold")
    dot, = ax.plot([], [], "o", color=TEAL, ms=16)
    velocity, = ax.plot([], [], color=ORANGE, lw=3)
    status = ax.text(.02, .88, "", transform=ax.transAxes, color=INK)
    push = ax.text(.98, .88, "", transform=ax.transAxes, color=RED,
                   ha="right", weight="bold")
    ax.set_title("Boss Fight: a sudden push tests the feedback loop", weight="bold", color=INK)
    frames = list(range(0, len(trace.states), 4))
    if frames[-1] != len(trace.states)-1:
        frames.append(len(trace.states)-1)

    def update(k):
        i = frames[k]
        state = trace.states[i]
        dot.set_data([state.position], [0])
        # Velocity line: capped visually, not physically.
        length = max(-2.0, min(2.0, state.velocity * .35))
        velocity.set_data([state.position, state.position + length], [0, 0])
        status.set_text(f"t={i*trace.dt:.1f}s    x={state.position:.2f} m    v={state.velocity:+.2f} m/s")
        push.set_text("EXTERNAL PUSH!" if spec.kick_step <= i < spec.kick_step+16 else "")
        return dot, velocity, status, push

    animation = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-scenario", type=int, default=25,
                        help="Trials in each of four categories, default=25")
    parser.add_argument("--kp", type=float, default=1.0,
                        help="PD proportional gain, default=1.0")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="PD velocity gain, default=2.0")
    args = parser.parse_args()
    if not math.isfinite(args.kp) or not math.isfinite(args.kd) or args.kp < 0 or args.kd < 0:
        parser.error("kp and kd must be finite and nonnegative")
    specs = make_trial_specs(seed=args.seed, per_scenario=args.per_scenario)
    p_results = evaluate(specs, lambda s, g: p_control(s, g, kp=1.0))
    pd_results = evaluate(specs, lambda s, g: pd_control(s, g, args.kp, args.kd))

    print(f"BOSS FIGHT | seed={args.seed}, trials={len(specs)}, dt={DT}s, horizon={HORIZON}s")
    print(f"Actuator |a| <= {MAX_ACCELERATION} m/s² | success: |x-goal| <= "
          f"{POSITION_TOLERANCE}m and |v| <= {VELOCITY_TOLERANCE}m/s "
          f"for final {HOLD_SECONDS}s")
    print(f"Comparing P(1.0) and PD(kp={args.kp}, kd={args.kd}) on IDENTICAL trials")
    for name, result in [("P", p_results), ("PD", pd_results)]:
        summary = summarize(result)
        settled = summary["median_settling_successes"]
        label = f"{settled:.2f}s" if settled is not None else "N/A (no successes)"
        print(f"  {name}: {summary['success_rate']:.1%} success | "
              f"mean final error={summary['mean_final_error']:.2f}m | "
              f"mean overshoot={summary['mean_overshoot']:.2f}m | "
              f"median settling [successes only]={label} | "
              f"saturated={summary['mean_saturation_fraction']:.1%}")
        for scenario in ("easy", "hard", "early_push", "late_push"):
            subset = tuple(r for r in result if r.spec.scenario == scenario)
            rate = summarize(subset)["success_rate"]
            print(f"       {scenario:10}: {rate:.0%} ({sum(r.score.success for r in subset)}/{len(subset)})")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_trajectories(specs, p_results, pd_results, ARTIFACTS / "four_scenarios.png")
    save_scorecard(p_results, pd_results, ARTIFACTS / "reliability_scorecard.png")
    save_failure_gallery(pd_results, ARTIFACTS / "failure_gallery.png")
    save_push_recovery_gif(ARTIFACTS / "push_recovery.gif", args.kp, args.kd)
    print(f"Saved 4 artifacts to {ARTIFACTS}")
    print("Challenge: change --kp and --kd. Can you improve the pass rate without")
    print("           merely hiding hard cases or relaxing success criteria?")


if __name__ == "__main__":
    main()
