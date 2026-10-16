# Chapter 3 — The World's Stupidest Controller

One world, one target, two controllers. Why reaching a goal isn't the same as **settling** at a goal.

## The physics is reused, not reinvented

This chapter imports `State`, `transition()`, and `clip_acceleration()` from Chapter 2. There is **no neural network**. The new concept is the *decision rule* that chooses actions from observed state.

- State: `(position [m], velocity [m/s])`
- Action requested: acceleration `[m/s²]`
- Goal: position `10 m`, target velocity `0 m/s`
- Physical law: `x_next = x + v*dt + 0.5*a*dt²`, `v_next = v + a*dt`

## Equations

Error: `e = target - x`

P: `a_request = kp * e`

PD: `a_request = kp * e - kd * v`

Actuator saturation is part of execution (not the controller): `a_execute = clip(a_request, -limit, +limit)`.

## Predict first

1. At `x=10, v=+5`, P control requests `a=0`. Is the robot at rest?
2. At `x=12, v=+3`, target `10`, `kp=1`, `kd=2`, what acceleration does PD request?
3. Under a ±2 m/s² acceleration limit, what actually executes for question 2?
4. Does setting `a=0` stop the dot? Why or why not?

## Run

From the repository root:

```bash
uv run python -m chapters.ch03_feedback_control.chapter03
uv run python -m unittest discover -s tests -v
```

## Visual artifacts

- `artifacts/p_vs_pd.gif` — **primary publishing visual**. Same dynamics and target, dramatically different behavior.
- `artifacts/p_vs_pd.png` — position response comparison.
- `artifacts/p_signals.png` — P position, velocity, and acceleration together.
- `artifacts/pd_signals.png` — the same signals for PD.
- `artifacts/damping_sweep.png` — vary damping alone and see oscillatory/slow responses.
- `artifacts/actuator_saturation.png` — requested vs executed acceleration, plus position effects.

All figures are generated locally by the script. Matplotlib uses the `Agg` backend; no GUI display server is required.

## Break it

Change `kd` to 0, 0.5, 2, and 8 while keeping `kp=1`. Check the plots. Does adding damping always make convergence faster?

Add an actuator acceleration limit of ±2 m/s². Did the controller ask for a different acceleration? Did the world execute what it requested?

**Subtlety:** This is feedback sampled every `dt` seconds, with acceleration held constant until the next step. A P-only controller in an ideal continuous-time, undamped double integrator oscillates without settling. With our discrete sample-and-hold controller, oscillations can *grow slightly*, because the discrete update is not identical to continuous feedback. This is a feature to investigate, not a plotting bug.

## Completion criteria

Explain why P overshoots; explain why D damps; identify controller versus world; distinguish commanded and executed action; demonstrate behavior with plots/tests; answer the retention challenge without looking at code.

Next: **Chapter 4 — Boss Fight: Make the Dot Behave**. Random starts, goals, actuator limits, evaluation metrics, and robustness.
