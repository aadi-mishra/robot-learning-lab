# Chapter 2 — The World's Stupidest Universe Discovers Velocity

**Central question:** Why is the current position sometimes insufficient to predict the next position?

Our world evolves a state containing **position** and **velocity**. The action is **acceleration** (not displacement, as in Chapter 1). Acceleration is assumed constant over each discrete time interval `dt`:

```text
x_next = x + v*dt + 0.5*a*dt²
v_next = v + a*dt
```

These are the exact constant-acceleration equations **for one interval**. The simulator does not include friction, contact, actuator dynamics, or automatic braking.

## Predict first

Two dots start at `x = 5 m`, with `v_A = +2 m/s`, `v_B = -2 m/s`, both with `a = 0` for 2 seconds. Predict both positions before running.

## Run

From the repository root:

```bash
uv run python chapters/ch02_velocity/chapter02.py
uv run python -m unittest discover -s tests -v
```

The script creates `artifacts/` here with:

| File | What to look for |
| --- | --- |
| `constant_velocity.png` | Position changes linearly while velocity remains constant. |
| `acceleration.png` | Velocity changes linearly; position changes quadratically. |
| `acceleration.gif` | The velocity arrow grows as the dot accelerates. |
| `same_position_different_future.png` | Same initial position, different trajectories. |
| `same_position_different_future.gif` | Two dots separate although their actions are identical. |

## Break it

Try acceleration `+100 m/s²` for `dt=0.5 s` from rest. Predict next position and velocity. Then compare that with a world that permits at most `±2 m/s²`.

**The engineering question:** Should a dataset record the requested action, the executed action, or both? Why?

## Quick retention check

1. Why is `[x, v]` a useful state in this world but `[x]` alone insufficient?
2. Why does zero acceleration not imply zero velocity?
3. What does changing `dt` mean? What else must stay constant for a fair comparison?
4. Why can observations that hide velocity make prediction appear inconsistent?
5. What would have to change for the dot to slow down without a commanded acceleration?

**Modern connections:** robot policy observations, state estimation, imitation learning data, trajectory prediction, model-based planning, robot controller design. This chapter is not training a machine-learning model yet.
