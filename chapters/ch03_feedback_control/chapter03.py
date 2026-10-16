"""Chapter 3 — feedback control: P, PD, and actuator saturation.

From the repository root:
    uv run python -m chapters.ch03_feedback_control.chapter03

This chapter REUSES Chapter 2's physical world. It changes only who chooses
an action, closing the loop between observed state and future action.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from chapters.ch02_velocity.chapter02 import (
    State, transition, clip_acceleration, TEAL, ORANGE, INK, BACKGROUND,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PURPLE = "#7655A6"
RED = "#BE4957"
Controller = Callable[[State, float], float]


@dataclass(frozen=True)
class Episode:
    """The closed-loop data contract: states, commands, and actual actions."""
    states: tuple[State, ...]
    requested: tuple[float, ...]
    executed: tuple[float, ...]
    dt: float
    target: float


def p_control(state: State, target: float, kp: float) -> float:
    """Return an acceleration command using position error only."""
    return kp * (target - state.position)


def pd_control(state: State, target: float, kp: float, kd: float) -> float:
    """Add a velocity-dependent braking/damping term."""
    return kp * (target - state.position) - kd * state.velocity


def run_closed_loop(
    initial_state: State,
    target: float,
    controller: Controller,
    dt: float,
    steps: int,
    max_acceleration: float | None = None,
) -> Episode:
    """Observe -> choose action -> enforce actuator limit -> advance world.

    The actuator limit is applied *outside* the controller. A policy can ask
    for an impossible acceleration, but the world executes a legal one.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if max_acceleration is not None and max_acceleration <= 0:
        raise ValueError("max_acceleration must be positive")

    states = [initial_state]
    requested_actions = []
    executed_actions = []
    state = initial_state

    for _ in range(steps):
        requested = float(controller(state, target))
        executed = (clip_acceleration(requested, max_acceleration)
                    if max_acceleration is not None else requested)
        next_state = transition(state, acceleration=executed, dt=dt)
        requested_actions.append(requested)
        executed_actions.append(executed)
        states.append(next_state)
        state = next_state

    return Episode(tuple(states), tuple(requested_actions),
                   tuple(executed_actions), dt, target)


def _axes_style(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(alpha=0.19)
    ax.spines[["top", "right"]].set_visible(False)


def save_signals(episode: Episode, path: Path, title: str, color: str) -> None:
    times = [i * episode.dt for i in range(len(episode.states))]
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axs:
        _axes_style(ax)
    axs[0].plot(times, [s.position for s in episode.states], color=color, lw=2.3)
    axs[0].axhline(episode.target, color=ORANGE, ls="--", lw=1.5, label="target")
    axs[0].set_ylabel("Position [m]")
    axs[0].legend(frameon=False, loc="upper right")
    axs[1].plot(times, [s.velocity for s in episode.states], color=color, lw=2.3)
    axs[1].axhline(0, color=INK, alpha=.35, lw=1)
    axs[1].set_ylabel("Velocity [m/s]")
    axs[2].plot(times[:-1], episode.executed, color=color, lw=2.3)
    axs[2].set_ylabel("Acceleration [m/s²]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle(title, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=170)
    plt.close(fig)


def save_comparison(p: Episode, pd: Episode, path: Path) -> None:
    times = [i * p.dt for i in range(len(p.states))]
    fig, ax = plt.subplots(figsize=(9.4, 5))
    fig.patch.set_facecolor(BACKGROUND)
    _axes_style(ax)
    ax.plot(times, [s.position for s in p.states], color=ORANGE, lw=2.5, label="P: only position error")
    ax.plot(times, [s.position for s in pd.states], color=TEAL, lw=2.5, label="PD: position + velocity")
    ax.axhline(p.target, color=INK, ls="--", alpha=.6, label="goal = 10 m")
    ax.set(xlabel="Time [s]", ylabel="Position [m]",
           title="Same robot. Same target. Different feedback.")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=170)
    plt.close(fig)


def save_comparison_gif(p: Episode, pd: Episode, path: Path) -> None:
    """Short side-by-side animation; dotted goal shared by two robot dots."""
    all_x = [s.position for s in p.states] + [s.position for s in pd.states] + [p.target]
    x_min = min(all_x) - 2
    x_max = max(all_x) + 2
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set(xlim=(x_min, x_max), ylim=(-1.1, 1.3), xlabel="Position [m]")
    ax.set_yticks([])
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.axvline(p.target, ls="--", lw=2, color=INK, alpha=.65)
    ax.text(p.target, 1.06, "GOAL", ha="center", color=INK, weight="bold")
    ax.axhline(.4, color=INK, alpha=.2)
    ax.axhline(-.4, color=INK, alpha=.2)
    p_dot, = ax.plot([], [], "o", color=ORANGE, ms=14)
    pd_dot, = ax.plot([], [], "o", color=TEAL, ms=14)
    ax.text(.02, .86, "P controller", color=ORANGE, transform=ax.transAxes, weight="bold")
    ax.text(.02, .22, "PD controller", color=TEAL, transform=ax.transAxes, weight="bold")
    status = ax.text(.5, .04, "", ha="center", transform=ax.transAxes, color=INK)
    ax.set_title("Why position feedback alone doesn't stop the dot", color=INK, weight="bold")

    indices = list(range(0, len(p.states), 4))
    if indices[-1] != len(p.states) - 1:
        indices.append(len(p.states) - 1)

    def update(frame: int):
        i = indices[frame]
        ps, pds = p.states[i], pd.states[i]
        p_dot.set_data([ps.position], [.4])
        pd_dot.set_data([pds.position], [-.4])
        status.set_text(f"t={i*p.dt:.1f} s    P: x={ps.position:.1f}, v={ps.velocity:+.1f}    |    PD: x={pds.position:.1f}, v={pds.velocity:+.1f}")
        return p_dot, pd_dot, status

    animation = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def save_saturation_plot(unlimited: Episode, limited: Episode, path: Path) -> None:
    """Show the commanded vs physically executed action, not just positions."""
    end = min(len(limited.requested), round(8 / limited.dt))
    times = [i * limited.dt for i in range(end)]
    times_state = [i * limited.dt for i in range(end+1)]
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axs:
        _axes_style(ax)
    axs[0].plot(times_state, [s.position for s in unlimited.states[:end+1]],
                color=PURPLE, lw=2.2, label="No actuator limit")
    axs[0].plot(times_state, [s.position for s in limited.states[:end+1]],
                color=TEAL, lw=2.2, label="Limit: |a| ≤ 2")
    axs[0].axhline(limited.target, color=INK, ls="--", alpha=.5)
    axs[0].set_ylabel("Position [m]")
    axs[0].legend(frameon=False)
    axs[1].plot(times, limited.requested[:end], color=ORANGE, lw=2, label="Requested")
    axs[1].plot(times, limited.executed[:end], color=TEAL, lw=2, label="Executed")
    axs[1].axhline(2, color=RED, ls="--", lw=1)
    axs[1].axhline(-2, color=RED, ls="--", lw=1)
    axs[1].set(xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    axs[1].legend(frameon=False, loc="upper right")
    fig.suptitle("A policy requests motion; the actuator enforces its limits", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=170)
    plt.close(fig)


def save_gain_sweep(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BACKGROUND)
    _axes_style(ax)
    for kd, color in [(0.0, ORANGE), (.5, RED), (2.0, TEAL), (8.0, PURPLE)]:
        episode = run_closed_loop(State(0, 0), 10.0,
            lambda s, goal, damping=kd: pd_control(s, goal, kp=1.0, kd=damping),
            dt=.05, steps=400)
        times = [i * episode.dt for i in range(len(episode.states))]
        ax.plot(times, [s.position for s in episode.states], color=color,
                lw=2, label=f"Kd = {kd:g}")
    ax.axhline(10, color=INK, ls="--", alpha=.5)
    ax.set(xlabel="Time [s]", ylabel="Position [m]",
           title="P gain fixed at 1; what does increasing damping change?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=BACKGROUND, dpi=170)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    initial = State(position=0.0, velocity=0.0)
    target, dt, steps = 10.0, .05, 400
    p = run_closed_loop(initial, target, lambda s, goal: p_control(s, goal, kp=1), dt, steps)
    pd = run_closed_loop(initial, target, lambda s, goal: pd_control(s, goal, kp=1, kd=2), dt, steps)
    save_signals(p, ARTIFACTS / "p_signals.png", "P control: oscillation instead of settling", ORANGE)
    save_signals(pd, ARTIFACTS / "pd_signals.png", "PD control: velocity feedback adds damping", TEAL)
    save_comparison(p, pd, ARTIFACTS / "p_vs_pd.png")
    save_comparison_gif(p, pd, ARTIFACTS / "p_vs_pd.gif")
    save_gain_sweep(ARTIFACTS / "damping_sweep.png")
    free = run_closed_loop(initial, target, lambda s, g: pd_control(s, g, 1, 2), dt, steps)
    bounded = run_closed_loop(initial, target, lambda s, g: pd_control(s, g, 1, 2), dt, steps,
                              max_acceleration=2)
    save_saturation_plot(free, bounded, ARTIFACTS / "actuator_saturation.png")
    print("P CONTROL — no velocity feedback")
    print(f"  Final state: {p.states[-1]}; max position: {max(s.position for s in p.states):.2f} m")
    print("PD CONTROL — velocity damping")
    print(f"  Final state: {pd.states[-1]}; max position: {max(s.position for s in pd.states):.2f} m")
    print("ACTUATOR LIMIT — requested versus executed")
    print(f"  First command: {bounded.requested[0]:.2f} m/s²; executed: {bounded.executed[0]:.2f} m/s²")
    print(f"Saved 6 artifacts to: {ARTIFACTS}")
    print("Note: P-only control can gain oscillation amplitude under discrete sample-and-hold feedback.")


if __name__ == "__main__":
    main()
