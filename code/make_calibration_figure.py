#!/usr/bin/env python3
"""Regenerate the leave-one-out calibration figure (manuscript fig05).

Refits the paper's fixed-hyperparameter pipeline (ARD Matern-5/2 GP for RMS
and the energy proxy, 150-tree random forest for Vpp and Vmax) under
leave-one-out cross-validation on the 75-condition design table, checks the
resulting CV R^2 against the published values, and draws a 2x2 predicted-vs-
measured grid with a shared frequency colorbar. Layout uses constrained_layout
so panel titles and axis labels cannot collide.
"""
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
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

import os
_p = "targets_design.parquet" if os.path.exists("targets_design.parquet") else os.path.join(os.path.dirname(__file__), "..", "data", "targets_design.parquet")
df = pd.read_parquet(_p)
feat = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[feat].values.astype(float)


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def rf():
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)


def loo_predict(make_model, y, scale=True):
    def one(tr, te):
        Xtr, Xte = X[tr], X[te]
        if scale:
            xs = StandardScaler().fit(Xtr)
            Xtr, Xte = xs.transform(Xtr), xs.transform(Xte)
        m = make_model().fit(Xtr, y[tr])
        return te[0], m.predict(Xte)[0]
    out = Parallel(n_jobs=-1)(
        delayed(one)(tr, te) for tr, te in LeaveOneOut().split(X)
    )
    pred = np.empty_like(y, dtype=float)
    for i, p in out:
        pred[i] = p
    return pred


PANELS = [
    ("rms_Voc", gp, True, "RMS", "ARD-GP", "Measured RMS (V)", 0.914),
    ("Vpp", rf, False, r"$V_{pp}$", "Random forest", r"Measured $V_{pp}$ (V)", 0.829),
    ("Vmax", rf, False, r"Peak $|V|$", "Random forest", r"Measured peak $|V|$ (V)", 0.816),
    ("energy", gp, True, "Energy", "ARD-GP", "Measured energy (a.u.)", 0.882),
]

fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), constrained_layout=True)
freq = df["freq_Hz"].values
sc = None
for ax, (col, make_model, scale, label, model_name, xlabel, published) in zip(
    axes.ravel(), PANELS
):
    y = df[col].values.astype(float)
    pred = loo_predict(make_model, y, scale=scale)
    r2 = r2_score(y, pred)
    print(f"{col}: LOO R2 = {r2:.3f} (published {published})")
    assert abs(r2 - published) < 0.02, f"{col} drifted from published value"
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    pad = 0.05 * (lim[1] - lim[0])
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, ls="--", c="gray", lw=1, zorder=1)
    sc = ax.scatter(
        y,
        pred,
        c=freq,
        cmap="viridis",
        s=26,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(f"{label}: {model_name} ($R^2$ {r2:.2f})", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.tick_params(labelsize=8)

cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Frequency (Hz)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

for ext in ("pdf", "png"):
    fig.savefig(f"fig05_calibration.{ext}", dpi=200)
print("wrote fig05_calibration.pdf/.png")
