import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(42)

paths = 50
points = 1000
x = 100

mu, sigma = 0.0, 1.0

Z = rng.normal(mu, sigma, (paths, points))

interval = [0.0, 1.0]
dt = (interval[1] - interval[0]) / (points - 1)

t_axis = np.linspace(interval[0], interval[1], points)

W = np.zeros((paths, points))
for idx in range(points - 1):
  real_idx = idx + 1
  W[:, real_idx] = W[:, real_idx - 1] + np.sqrt(dt) * Z[:, idx]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
  ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()