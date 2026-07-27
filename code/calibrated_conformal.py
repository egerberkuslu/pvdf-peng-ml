#!/usr/bin/env python3
"""GP-scaled jackknife+ intervals (calibrated conformal for the GP surrogate).

Leave-one-out residuals are standardized by the GP posterior sigma, and the
full jackknife+ construction of Barber et al. is applied to the standardized
scores, so the interval at x* is rescaled by sigma(x*). Coverage and width are
measured under an honest outer leave-one-out loop (train models never see the
evaluation point) and compared against the raw +-z sigma GP intervals and the
unnormalized jackknife+ on the same nested predictions.
Writes calibrated_conformal_results.json.
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
from sklearn.preprocessing import StandardScaler

_p = (
    "targets_design.parquet"
    if os.path.exists("targets_design.parquet")
    else os.path.join(os.path.dirname(__file__), "..", "data", "targets_design.parquet")
)
df = pd.read_parquet(_p)
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
N = len(df)
ALPHAS = {"90": 0.10, "95": 0.05}
Z = {"90": 1.6448536269514722, "95": 1.959963984540054}


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fit_predict(tr, tes, y):
    """Fit scaler+GP on tr, predict mean/std at the given test rows."""
    xs = StandardScaler().fit(X[tr])
    g = gp().fit(xs.transform(X[tr]), y[tr])
    mu, sd = g.predict(xs.transform(X[tes]), return_std=True)
    return mu, sd


def outer_fold(j, y):
    """For held-out j: inner LOO over the remaining points.

    Returns, for every i != j, the model-(minus i,j) prediction at x_i (its own
    LOO residual source) and at x_j (the jackknife+ ensemble member), plus the
    model-(minus j) prediction at x_j (jackknife / raw-GP evaluation).
    """
    rest = np.delete(np.arange(N), j)
    mu_j, sd_j = fit_predict(rest, np.array([j]), y)
    mem = []
    for i in rest:
        tr = np.setdiff1d(rest, [i])
        mu, sd = fit_predict(tr, np.array([i, j]), y)
        # (residual at i, sigma at i, prediction at j, sigma at j)
        mem.append((abs(y[i] - mu[0]), sd[0], mu[1], sd[1]))
    mem = np.array(mem)
    return j, float(mu_j[0]), float(sd_j[0]), mem


def jk_quantile(vals, alpha, side):
    """jackknife+ order statistic on n values."""
    n = len(vals)
    k = int(np.clip(np.ceil((1 - alpha) * (n + 1)), 1, n))
    v = np.sort(vals)
    return v[k - 1] if side == "hi" else v[n - k]


def evaluate(y):
    out = Parallel(n_jobs=-1)(delayed(outer_fold)(j, y) for j in range(N))
    res = {
        m: {a: {"hits": [], "widths": []} for a in ALPHAS}
        for m in ("raw_gp", "jackknife_plus", "scaled_jackknife_plus")
    }
    for j, mu_j, sd_j, mem in out:
        r_abs, sd_i, mu_at_j, sd_at_j = mem.T
        r_norm = r_abs / np.maximum(sd_i, 1e-12)
        for a, alpha in ALPHAS.items():
            # raw GP +- z sigma
            lo, hi = mu_j - Z[a] * sd_j, mu_j + Z[a] * sd_j
            res["raw_gp"][a]["hits"].append(lo <= y[j] <= hi)
            res["raw_gp"][a]["widths"].append(hi - lo)
            # vanilla jackknife+ (unnormalized)
            lo = jk_quantile(mu_at_j - r_abs, alpha, "lo")
            hi = jk_quantile(mu_at_j + r_abs, alpha, "hi")
            res["jackknife_plus"][a]["hits"].append(lo <= y[j] <= hi)
            res["jackknife_plus"][a]["widths"].append(hi - lo)
            # GP-scaled jackknife+ (proposed)
            lo = jk_quantile(mu_at_j - r_norm * sd_at_j, alpha, "lo")
            hi = jk_quantile(mu_at_j + r_norm * sd_at_j, alpha, "hi")
            res["scaled_jackknife_plus"][a]["hits"].append(lo <= y[j] <= hi)
            res["scaled_jackknife_plus"][a]["widths"].append(hi - lo)
    summary = {}
    for m, block in res.items():
        summary[m] = {}
        for a in ALPHAS:
            summary[m][a] = {
                "coverage": float(np.mean(block[a]["hits"])),
                "mean_width": float(np.mean(block[a]["widths"])),
                "median_width": float(np.median(block[a]["widths"])),
            }
    return summary


results = {}
for tgt in ["rms_Voc", "Vpp", "Vmax", "energy"]:
    y = df[tgt].values.astype(float)
    results[tgt] = evaluate(y)
    print(tgt)
    for m in ("raw_gp", "jackknife_plus", "scaled_jackknife_plus"):
        s = results[tgt][m]
        print(
            f"  {m:22s} cov90={s['90']['coverage']:.3f} w90={s['90']['mean_width']:.3g}"
            f"  cov95={s['95']['coverage']:.3f} w95={s['95']['mean_width']:.3g}"
        )

json.dump(results, open("calibrated_conformal_results.json", "w"), indent=2)
print("wrote calibrated_conformal_results.json")
