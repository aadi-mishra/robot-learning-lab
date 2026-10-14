# Robot Learning Lab

Small, visual experiments in robotics, control, robot learning, and Physical AI.

**Philosophy:** Build the intelligence. Borrow the machinery.

This repository accompanies [Robot Learning Lab on Substack](https://robotlearninglab.substack.com/). Each chapter starts with a small system, makes its behavior visible, and deliberately tests its assumptions.

## Chapter 1 — The World's Stupidest Universe

A one-dimensional dot lives by one law:

`next_position = position + action`

Here, **action means commanded displacement**, not force or acceleration.

Predict the sequence before running it:

- Initial position: `0`
- Actions: `[1, 1, 2, -3, 4]`

Expected positions: `[0, 1, 2, 4, 1, 5]`.

### Run

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv run python chapters/ch01_stupid_universe/chapter01.py
uv run python -m unittest discover -s tests -v
```

The first command generates in `chapters/ch01_stupid_universe/artifacts/`:

- `trajectory.png` — position against step number
- `trajectory.gif` — animation of the dot
- `teleport.gif` — what happens when large actions are permitted

Open both GIFs and explain why `+50` is allowed under this model.

### Extension

Modify `transition()` to clip actions to `[-2, 2]`, then compare the trajectories. Decide whether the dataset should record the requested action, the executed action, or both.

## Roadmap

Upcoming: state and velocity, feedback control, learning from data, imitation learning, reinforcement learning, world models, and more.

Code in this repository is educational. The tiny environments are intentionally simplified, not validated robot simulators.
