"""Chapter 1: The World's Stupidest Universe.

A state is a position; an action is instantaneous displacement.

Start by predicting the trajectory, then run this file and inspect the GIFs.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def transition(position: float, action: float) -> float:
    """Advance our deliberately simple universe by one action."""
    return position + action


def rollout(initial_position: float, actions: list[float]) -> list[float]:
    positions = [initial_position]
    for action in actions:
        positions.append(transition(positions[-1], action))
    return positions


def plot_trajectory(positions: list[float], filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(positions)), positions, marker="o")
    ax.set(xlabel="Time step", ylabel="Position", title="One dot, one transition rule")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def animate(positions: list[float], actions: list[float], filename: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    low, high = min(positions), max(positions)
    margin = max(1, (high - low) * 0.1)
    ax.set(xlim=(low - margin, high + margin), ylim=(-1, 1), yticks=[], xlabel="Position", title=title)
    ax.axhline(0, linewidth=1, alpha=0.45)
    dot, = ax.plot([], [], "o", markersize=15, color="#e97843")
    history, = ax.plot([], [], ".--", alpha=0.55, color="#297b87")
    label = ax.text(0.02, 0.82, "", transform=ax.transAxes)

    def update(frame: int):
        x = positions[frame]
        dot.set_data([x], [0])
        history.set_data(positions[:frame+1], [0] * (frame+1))
        last_action = "none" if frame == 0 else f"{actions[frame-1]:+g}"
        label.set_text(f"step {frame} | x = {x:+g} | last action = {last_action}")
        return dot, history, label

    animation = FuncAnimation(fig, update, frames=len(positions), interval=850, blit=False)
    animation.save(filename, writer=PillowWriter(fps=1.5))
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    experiments = {
        "trajectory": [1.0, 1.0, 2.0, -3.0, 4.0],
        "teleport": [1.0, 1.0, 50.0, -50.0, 1.0],
    }
    for name, actions in experiments.items():
        positions = rollout(0.0, actions)
        print(f"\n{name.upper()}: actions={actions}, positions={positions}")
        for t, action in enumerate(actions):
            print(f"  t={t}: ({positions[t]:+g}, {action:+g}, {positions[t+1]:+g})")
        animate(positions, actions, ARTIFACT_DIR / f"{name}.gif", name.title())
        if name == "trajectory":
            plot_trajectory(positions, ARTIFACT_DIR / "trajectory.png")
    print(f"\nFigures and GIFs saved in {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
