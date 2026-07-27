#!/usr/bin/env python3
"""Regenerate the two-panel response-surface figure (manuscript Fig. 8).

(a) ARD-GP predicted RMS voltage and (b) random-forest predicted Vpp over the
frequency-force plane at the 2 wt% CNT slice, refit on all 75 conditions with
the paper's fixed hyperparameters. White points mark measured conditions and
the star in (a) marks the predicted optimum within this slice.
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
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

k = C(1.0, (1e-3, 1e3)) * Matern(
    length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
) + WhiteKernel(1e-3, (1e-6, 1e1))
xs = StandardScaler().fit(X)
gp = GaussianProcessRegressor(
    kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
).fit(xs.transform(X), df["rms_Voc"].values)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(
    X, df["Vpp"].values
)

F, Q = np.meshgrid(np.linspace(1, 3, 160), np.linspace(5, 25, 200), indexing="ij")
G = np.column_stack([np.full(F.size, 2.0), np.zeros(F.size), F.ravel(), Q.ravel()])
rms = gp.predict(xs.transform(G)).reshape(F.shape)
vpp = rf.predict(G).reshape(F.shape)

meas = df[df.cnt_pct == 2.0][["freq_Hz", "force_N"]].drop_duplicates()
iopt = np.unravel_index(np.argmax(rms), rms.shape)
print(
    f"slice optimum: {Q[iopt]:.1f} Hz, {F[iopt]:.2f} N, "
    f"RMS max {rms.max():.3f} V | Vpp max {vpp.max():.2f} V"
)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)

pc = axes[0].contourf(Q, F, rms, levels=12, cmap="viridis")
axes[0].scatter(
    meas.freq_Hz,
    meas.force_N,
    s=22,
    c="white",
    edgecolors="0.3",
    linewidths=0.5,
    zorder=3,
)
axes[0].plot(
    Q[iopt], F[iopt], marker="*", ms=17, mfc="crimson", mec="white", mew=1.0, zorder=4
)
cb = fig.colorbar(pc, ax=axes[0], shrink=0.92)
cb.set_label(r"$V_{rms}$ (V)", fontsize=9)
axes[0].set_title(
    r"a  Predicted $V_{rms}$ (V)", loc="left", fontsize=10, fontweight="bold"
)

pc2 = axes[1].pcolormesh(
        Q, F, vpp, cmap="viridis", shading="nearest", antialiased=False,
        rasterized=True
    )
axes[1].scatter(
    meas.freq_Hz,
    meas.force_N,
    s=22,
    c="white",
    edgecolors="0.3",
    linewidths=0.5,
    zorder=3,
)
cb2 = fig.colorbar(pc2, ax=axes[1], shrink=0.92)
cb2.set_label(r"$V_{pp}$ (V)", fontsize=9)
axes[1].set_title(
    r"b  Predicted $V_{pp}$ (V)", loc="left", fontsize=10, fontweight="bold"
)

for ax in axes:
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Force (N)", fontsize=9)
    ax.tick_params(labelsize=8)

for ext in ("pdf", "png"):
    fig.savefig(f"fig06_response_surface.{ext}", dpi=200)
print("wrote fig06_response_surface.pdf/.png")
