"""Chapter 2: A dot discovers velocity.

Run from the repository root:
    uv run python chapters/ch02_velocity/chapter02.py

The world is intentionally simple: acceleration is constant during each step.
State: (position [m], velocity [m/s]); action: acceleration [m/s^2].
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Generate images/GIFs without requiring a display server.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# These colors are part of the chapter's visual language, not the physics.
TEAL = "#0C8986"
ORANGE = "#DB7E29"
INK = "#263442"
BACKGROUND = "#FAFAF8"


@dataclass(frozen=True)
class State:
    position: float  # metres
    velocity: float  # metres per second


def transition(state: State, acceleration: float, dt: float) -> State:
    """Return the next state under constant acceleration for dt seconds.

    Equations:
        x_next = x + v*dt + 0.5*a*dt**2
        v_next = v + a*dt
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    return State(
        position=state.position + state.velocity * dt + 0.5 * acceleration * dt**2,
        velocity=state.velocity + acceleration * dt,
    )


def rollout(initial_state: State, actions: list[float], dt: float) -> list[State]:
    """Generate a trajectory containing the initial state and every next state."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    trajectory = [initial_state]
    state = initial_state
    for action in actions:
        state = transition(state, acceleration=action, dt=dt)
        trajectory.append(state)
    return trajectory


def clip_acceleration(requested: float, limit: float) -> float:
    """Optional exercise: a simple symmetric actuator acceleration limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return max(-limit, min(limit, requested))


def save_state_plot(states: list[State], dt: float, path: Path, title: str) -> None:
    times = [i * dt for i in range(len(states))]
    fig, (ax_x, ax_v) = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for ax in (ax_x, ax_v):
        ax.set_facecolor(BACKGROUND)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    ax_x.plot(times, [s.position for s in states], color=TEAL, lw=2.6)
    ax_x.set_ylabel("Position x  [m]")
    ax_v.plot(times, [s.velocity for s in states], color=ORANGE, lw=2.6)
    ax_v.set_ylabel("Velocity v  [m/s]")
    ax_v.set_xlabel("Time  [s]")
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


def save_divergence_plot(a_states: list[State], b_states: list[State],
                         dt: float, path: Path) -> None:
    times = [i * dt for i in range(len(a_states))]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(times, [s.position for s in a_states], color=TEAL, lw=2.7, label="A: v₀ = +2 m/s")
    ax.plot(times, [s.position for s in b_states], color=ORANGE, lw=2.7, label="B: v₀ = −2 m/s")
    ax.scatter([0, 0], [5, 5], color=INK, zorder=5, s=38)
    ax.set(xlabel="Time  [s]", ylabel="Position x  [m]",
           title="Same position. Different velocity. Different future.")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


def save_divergence_gif(a_states: list[State], b_states: list[State],
                        dt: float, path: Path) -> None:
    """Animate two moving dots using the same horizontal position scale."""
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set(xlim=(0, 10), ylim=(-1.1, 1.1), xlabel="Position x  [m]")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.axhline(0, color=INK, alpha=0.28, lw=1)
    ax.axvline(5, color=INK, alpha=0.22, ls="--")
    a_dot, = ax.plot([], [], "o", color=TEAL, ms=13)
    b_dot, = ax.plot([], [], "o", color=ORANGE, ms=13)
    a_label = ax.text(0, 0, "A", fontsize=10, color=TEAL, weight="bold", ha="center")
    b_label = ax.text(0, 0, "B", fontsize=10, color=ORANGE, weight="bold", ha="center")
    status = ax.text(0.015, 0.94, "", va="top", transform=ax.transAxes,
                     fontsize=10.5, color=INK)
    ax.set_title("Two dots. Same action: a = 0.", color=INK, weight="bold")

    def update(frame: int):
        a = a_states[frame]
        b = b_states[frame]
        a_dot.set_data([a.position], [0.25])
        b_dot.set_data([b.position], [-0.25])
        a_label.set_position((a.position, 0.53))
        b_label.set_position((b.position, -0.55))
        status.set_text(f"t = {frame * dt:.2f} s       A: x = {a.position:.2f} m       B: x = {b.position:.2f} m")
        return a_dot, b_dot, a_label, b_label, status

    anim = FuncAnimation(fig, update, frames=len(a_states), interval=95,
                         blit=False, repeat=True)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def save_acceleration_gif(states: list[State], dt: float, path: Path) -> None:
    """The velocity arrow length grows as acceleration increases speed."""
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    final_x = states[-1].position
    ax.set(xlim=(-0.5, final_x + 3), ylim=(-1.0, 1.15), xlabel="Position x  [m]")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.axhline(0, color=INK, alpha=0.25)
    dot, = ax.plot([], [], "o", color=TEAL, ms=13)
    arrow = ax.annotate("", xy=(0, 0.25), xytext=(0, 0.25),
                        arrowprops={"arrowstyle":"->", "lw":2.8, "color":ORANGE})
    status = ax.text(0.015, 0.94, "", transform=ax.transAxes, va="top", color=INK)
    ax.set_title("Acceleration changes velocity. Velocity changes position.",
                 color=INK, weight="bold")

    def update(frame: int):
        s = states[frame]
        dot.set_data([s.position], [0])
        arrow.xy = (s.position + s.velocity * 0.48, 0.25)
        arrow.set_position((s.position, 0.25))
        status.set_text(f"t = {frame*dt:.1f} s    x = {s.position:.2f} m    v = {s.velocity:.2f} m/s")
        return dot, arrow, status

    anim = FuncAnimation(fig, update, frames=len(states), interval=95,
                         blit=False, repeat=True)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    # Experiment 1: no acceleration, but the dot still moves.
    dt = 0.1
    coasting = rollout(State(position=0.0, velocity=2.0), [0.0] * 50, dt)
    save_state_plot(coasting, dt, ARTIFACTS / "constant_velocity.png",
                    "Experiment 1: Zero acceleration does not mean zero motion")

    # Experiment 2: positive acceleration makes velocity grow steadily.
    accelerating = rollout(State(position=0.0, velocity=0.0), [1.0] * 50, dt)
    save_state_plot(accelerating, dt, ARTIFACTS / "acceleration.png",
                    "Experiment 2: Constant acceleration produces curved position")
    save_acceleration_gif(accelerating, dt, ARTIFACTS / "acceleration.gif")

    # Experiment 3: the same position is not enough to predict motion.
    dt = 0.05
    a_states = rollout(State(5.0, +2.0), [0.0] * 40, dt)
    b_states = rollout(State(5.0, -2.0), [0.0] * 40, dt)
    save_divergence_plot(a_states, b_states, dt,
                         ARTIFACTS / "same_position_different_future.png")
    save_divergence_gif(a_states, b_states, dt,
                        ARTIFACTS / "same_position_different_future.gif")

    print("EXPERIMENT 1 — COASTING (after 5 s)")
    print(f"  x = {coasting[-1].position:.2f} m, v = {coasting[-1].velocity:.2f} m/s")
    print("EXPERIMENT 2 — ACCELERATING (after 5 s)")
    print(f"  x = {accelerating[-1].position:.2f} m, v = {accelerating[-1].velocity:.2f} m/s")
    print("EXPERIMENT 3 — SAME POSITION, DIFFERENT FUTURES (after 2 s)")
    print(f"  A: x = {a_states[-1].position:.2f} m, v = {a_states[-1].velocity:.2f} m/s")
    print(f"  B: x = {b_states[-1].position:.2f} m, v = {b_states[-1].velocity:.2f} m/s")
    print("BREAK IT — a = 100 m/s², x0 = v0 = 0, dt = 0.5 s")
    unrealistic = transition(State(0.0, 0.0), acceleration=100, dt=0.5)
    bounded = transition(State(0.0, 0.0), acceleration=clip_acceleration(100, 2), dt=0.5)
    print(f"  Unbounded: {unrealistic}")
    print(f"  Limited to ±2 m/s²: {bounded}")
    print(f"Artifacts saved to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
