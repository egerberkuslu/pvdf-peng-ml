#!/usr/bin/env python3
"""Regenerate the GP reliability map (manuscript fig07) and its quoted numbers.

Fits the paper's ARD Matern-5/2 Gaussian process on all 75 conditions,
evaluates the posterior mean and standard deviation on a dense
frequency-force grid at the 2 wt% CNT slice, marks the measured region, and
prints the summary statistics quoted in Section 4.7 of the manuscript (the
minimum in-box posterior standard deviation and its growth factor toward the
most sparsely supported corners).
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

_p = (
    "targets_design.parquet"
    if os.path.exists("targets_design.parquet")
    else os.path.join(os.path.dirname(__file__), "..", "data", "targets_design.parquet")
)
df = pd.read_parquet(_p)
X = df[["cnt_pct", "is_pristine", "force_N", "freq_Hz"]].values.astype(float)
y = df["rms_Voc"].values.astype(float)

k = C(1.0, (1e-3, 1e3)) * Matern(
    length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
) + WhiteKernel(1e-3, (1e-6, 1e1))
xs = StandardScaler().fit(X)
gp = GaussianProcessRegressor(
    kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
).fit(xs.transform(X), y)

F, Q = np.meshgrid(np.linspace(1, 3, 160), np.linspace(5, 25, 200), indexing="ij")
G = np.column_stack([np.full(F.size, 2.0), np.zeros(F.size), F.ravel(), Q.ravel()])
mean, std = gp.predict(xs.transform(G), return_std=True)
mean = mean.reshape(F.shape)
std = std.reshape(F.shape)

print(f"min in-box posterior std : {std.min():.4f} V")
print(f"max in-box posterior std : {std.max():.4f} V")
print(f"growth factor across box : {std.max() / std.min():.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
for ax, Z, title, unit in [
    (axes[0], mean, "Predicted RMS (mean)", "RMS (V)"),
    (axes[1], std, "GP predictive std", "std (V)"),
]:
    pc = ax.pcolormesh(
        Q, F, Z, cmap="viridis" if Z is mean else "magma", shading="gouraud"
    )
    ax.add_patch(plt.Rectangle((5, 1), 20, 2, fill=False, ls="--", lw=1.2, ec="white"))
    ax.text(5.6, 2.86, "measured region", color="white", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(pc, ax=ax, label=unit)

for ext in ("pdf", "png"):
    fig.savefig(f"fig07_uncertainty_map.{ext}", dpi=200)
print("wrote fig07_uncertainty_map.pdf/.png")
