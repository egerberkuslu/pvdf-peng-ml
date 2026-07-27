#!/usr/bin/env python3
"""PhysGP: physics-guided Gaussian-process surrogate.

The closed-form voltage law
    V(c, F, f) = (b0 + b1 c + b2 c^2) * F * A (g/2)^2 / ((f - f0)^2 + (g/2)^2)
is fit inside every training fold (scipy curve_fit, bounded) and used as the
mean function; the paper's ARD Matern-5/2 GP then models the residuals.
Evaluated head-to-head against the plain GP under leave-one-out and
leave-one-level-out cross-validation on the 75-condition design table.
Writes physgp_results.json next to the design table.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_p = (
    "targets_design.parquet"
    if os.path.exists("targets_design.parquet")
    else os.path.join(os.path.dirname(__file__), "..", "data", "targets_design.parquet")
)
df = pd.read_parquet(_p)
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
CNT = df["cnt_pct"].values.astype(float)
FRC = df["force_N"].values.astype(float)
FRQ = df["freq_Hz"].values.astype(float)

P0 = [1.0, 0.1, -0.02, 1.0, 19.0, 8.0]
BOUNDS = ([0, -5, -5, 0, 10, 1], [50, 5, 5, 50, 30, 30])


def law(Xt, b0, b1, b2, A, f0, g):
    c, F, f = Xt
    lor = A * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)
    return (b0 + b1 * c + b2 * c**2) * F * lor


def fit_law(tr, y):
    """Fit the law on the training rows of a scaled target; return predict fn."""
    scale = float(np.std(y[tr])) or 1.0
    try:
        p, _ = curve_fit(
            law,
            (CNT[tr], FRC[tr], FRQ[tr]),
            y[tr] / scale,
            p0=P0,
            bounds=BOUNDS,
            maxfev=20000,
        )
    except Exception:
        p = np.array(P0)
    return lambda idx: law((CNT[idx], FRC[idx], FRQ[idx]), *p) * scale, p


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fold_plain(tr, te, y):
    xs = StandardScaler().fit(X[tr])
    g = gp().fit(xs.transform(X[tr]), y[tr])
    mu, sd = g.predict(xs.transform(X[te]), return_std=True)
    return te, mu, sd


def fold_phys(tr, te, y):
    m, _ = fit_law(tr, y)
    resid = y[tr] - m(tr)
    xs = StandardScaler().fit(X[tr])
    g = gp().fit(xs.transform(X[tr]), resid)
    rmu, rsd = g.predict(xs.transform(X[te]), return_std=True)
    return te, m(te) + rmu, rsd


def run_cv(y, folds, fold_fn):
    n = len(y)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    out = Parallel(n_jobs=-1)(delayed(fold_fn)(tr, te, y) for tr, te in folds)
    for te, m, s in out:
        mu[te] = m
        sd[te] = s
    mask = ~np.isnan(mu)
    return mu, sd, mask


def loo_folds(n):
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


def group_folds(groups):
    idx = np.arange(len(groups))
    return [
        (np.where(groups != v)[0], np.where(groups == v)[0]) for v in np.unique(groups)
    ]


TARGETS = ["rms_Voc", "Vpp", "Vmax", "energy"]
GROUPS = {
    "composition": df["composition"].values,
    "force": df["force_N"].values,
    "frequency": df["freq_Hz"].values,
}

results = {}
loo_store = {}
for tgt in TARGETS:
    y = df[tgt].values.astype(float)
    block = {}
    for name, fn in [("gp", fold_plain), ("physgp", fold_phys)]:
        mu, sd, _ = run_cv(y, loo_folds(len(y)), fn)
        block[f"{name}_LOO_R2"] = float(r2_score(y, mu))
        block[f"{name}_LOO_MAE"] = float(mean_absolute_error(y, mu))
        if tgt == "rms_Voc":
            loo_store[name] = {
                "mu": mu.tolist(),
                "sd": sd.tolist(),
                "y": y.tolist(),
            }
        for gname, gvals in GROUPS.items():
            gmu, _, _ = run_cv(y, group_folds(gvals), fn)
            block[f"{name}_leave_one_{gname}_out_R2"] = float(r2_score(y, gmu))
            block[f"{name}_leave_one_{gname}_out_MAE"] = float(
                mean_absolute_error(y, gmu)
            )
    # drop-20Hz extrapolation probe
    tr = np.where(FRQ != 20)[0]
    te = np.where(FRQ == 20)[0]
    for name, fn in [("gp", fold_plain), ("physgp", fold_phys)]:
        _, mu, _ = fn(tr, te, y)
        block[f"{name}_drop20Hz_pred_mean"] = float(np.mean(mu))
        block[f"{name}_drop20Hz_meas_mean"] = float(np.mean(y[te]))
        block[f"{name}_drop20Hz_MAE"] = float(mean_absolute_error(y[te], mu))
        block[f"{name}_drop20Hz_rel_underprediction_pct"] = float(
            100 * (1 - np.mean(mu) / np.mean(y[te]))
        )
    results[tgt] = block

# full-data law fit (for reporting the fitted physics parameters on RMS)
m_all, p_all = fit_law(np.arange(len(df)), df["rms_Voc"].values.astype(float))
results["law_params_rms_fulldata"] = {
    k: float(v) for k, v in zip(["b0", "b1", "b2", "A", "f0_Hz", "gamma"], p_all)
}
results["loo_predictions_rms"] = loo_store

out = "physgp_results.json"
json.dump(results, open(out, "w"), indent=2)
print(
    json.dumps(
        {
            t: {k: round(v, 4) for k, v in results[t].items() if isinstance(v, float)}
            for t in TARGETS
        },
        indent=1,
    )
)
print("law params (rms, full data):", results["law_params_rms_fulldata"])
print("wrote", out)
