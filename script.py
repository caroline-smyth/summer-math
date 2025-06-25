import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(42)


def simulate_branching(paths, points, mu, sigma, x0, rate, interval=[0.0, 1.0]):
    """
    Simulate `paths` geometric Brownian motions over `points` time steps,
    with death-and-rebirth events occurring at rate `rate` (Poisson clock).
    On each death, that path is reset to the lowest current value across all paths.
    Returns (t_axis, X) where X has shape (paths, points).
    """
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    # Gaussian increments
    Z = rng.normal(0, 1, size=(paths, points))
    # initialize
    X = np.zeros((paths, points))
    X[:, 0] = x0

    for idx in range(points - 1):
        dW = np.sqrt(dt) * Z[:, idx]
        # GBM step
        X[:, idx + 1] = X[:, idx] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        # branching: death with prob = rate * dt
        deaths = rng.random(paths) < rate * dt
        if deaths.any():
            current = X[:, idx + 1]
            min_val = current.min()  # lowest rank
            X[deaths, idx + 1] = min_val

    return t_axis, X

def simulate_offspring_branching(paths, points, mu, sigma, x0, rate, interval=[0.0, 1.0]):
    """
    Simulate `paths` geometric Brownian motions over `points` time steps,
    with pure‐birth branching at rate `rate` (Poisson clock).  Whenever
    a branch event occurs for a path, that path “splits” into two: the
    original continues, and a new path is born at the same current value.
    Returns (t_axis, histories) where histories is a list of length‐points
    for each path (including those born later, padded with NaNs before birth).
    """
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    histories = [[x0] for _ in range(paths)]  # each sub‐list will grow to length=points

    for idx in range(points - 1):
        current_vals = np.array([h[-1] for h in histories])
        # GBM increment for each alive firm
        dW = rng.normal(0, 1, size=current_vals.shape) * np.sqrt(dt)
        next_vals = current_vals * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

        # append next value to every history
        for h, nv in zip(histories, next_vals):
            h.append(nv)

        # check branching events (one extra offspring per event)
        births = rng.random(next_vals.shape) < rate * dt
        for j, did_branch in enumerate(births):
            if did_branch:
                # new history: pad with NaNs up to this step, then start at nv
                new_hist = [np.nan] * (idx + 1) + [next_vals[j]]
                histories.append(new_hist)

        # for any newly born histories, we need to pad them at subsequent steps—
        # but since they start at length idx+2, and others are at idx+2, they stay aligned

    return t_axis, histories

def simulate_random_defaults(paths, points, mu, sigma, x0, default_rate, interval=[0.0, 1.0]):
    """
    Simulate `paths` geometric Brownian motions over `points` time steps,
    with random default events at rate `default_rate` (Poisson clock).
    When a firm defaults, its history is stopped (values are set to NaN thereafter).
    Returns (t_axis, histories) where `histories` is a list of lists (len = # firms)
    each of length `points` (with NaNs after default).
    """
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    histories = [[x0] for _ in range(paths)]
    alive = [True] * paths

    for idx in range(points - 1):
        for j in range(len(histories)):
            if alive[j]:
                # one GBM step
                dW = rng.normal(0, 1) * np.sqrt(dt)
                prev = histories[j][-1]
                nxt = prev * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                # default test
                if rng.random() < default_rate * dt:
                    histories[j].append(np.nan)
                    alive[j] = False
                else:
                    histories[j].append(nxt)
            else:
                # once dead, just pad with NaN
                histories[j].append(np.nan)

    return t_axis, histories

paths = 5
points = 1000
x = 100
y = 100

mu, sigma = 0.0, 1.0

Z = rng.normal(mu, sigma, (paths, points))

interval = [0.0, 1.0]
dt = (interval[1] - interval[0]) / (points - 1)

t_axis = np.linspace(interval[0], interval[1], points)

W = np.zeros((paths, points))
for idx in range(points - 1):
  real_idx = idx + 1
  W[:, real_idx] = W[:, real_idx - 1] + np.sqrt(dt) * Z[:, idx]

branch_rate = 1.0    # expected death events per unit time per path
x0 = 100             # starting value for all paths

plt.figure(figsize=(12, 8))
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
  ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()

t_b, X_b = simulate_branching(paths, points, mu, sigma, x0, branch_rate, interval)
plt.figure(figsize=(12, 8))
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(paths):
    ax.plot(t_b, X_b[i, :])
ax.set_title("Geometric BM with Die-and-Rebirth Branching")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()

t_o, offs = simulate_offspring_branching(paths, points, mu, sigma, x0, branch_rate, interval)
plt.figure(figsize=(12, 8))
fig, ax = plt.subplots(figsize=(12, 8))
for hist in offs:
    ax.plot(t_o, hist, lw=1)
ax.set_title("Geometric BM with Pure‐Birth Branching (each splits into 2)")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.tight_layout()
plt.show()

paths       = 5
points      = 500
mu, sigma   = 0.0, 1.0
x0          = 100
default_rate = 2.0    # expected defaults per unit time per firm
interval    = [0.0, 1.0]

t_d, def_hist = simulate_random_defaults(paths, points, mu, sigma, x0, default_rate, interval)

fig, ax = plt.subplots(figsize=(12, 8))
for hist in def_hist:
    ax.plot(t_d, hist, lw=1)
ax.set_title("Geometric BM with Random Defaults (Absorbing)")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.tight_layout()
plt.show()