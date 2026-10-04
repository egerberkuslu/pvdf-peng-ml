#!/usr/bin/env python3
"""Jackknife+ coverage under group-wise splits, and paired McNemar tests.

Two referee demands on the RMS voltage target:
  1. the paired significance of raw-GP versus jackknife+ coverage under the
     nested leave-one-out evaluation (exact McNemar on per-point hits), and
  2. the empirical coverage of the jackknife+ band when a whole factor level
     is held out, which measures what the guarantee does NOT promise once
     exchangeability between train and query points is broken.
Writes groupwise_conformal.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import binom
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "..", "..", "data", "targets_design.parquet"))
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
y = df["rms_Voc"].values.astype(float)
n = len(y)


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fit_predict(tr, query):
    xs = StandardScaler().fit(X[tr])
    g = gp().fit(xs.transform(X[tr]), y[tr])
    mu, sd = g.predict(xs.transform(X[query]), return_std=True)
    return mu, sd


def jk_interval(preds, resids, alpha):
    """Jackknife+ interval from leave-one-out predictions at the query point."""
    m = len(resids)
    kk = int(np.clip(np.ceil((1 - alpha) * (m + 1)), 1, m))
    lo = np.sort(preds - resids)[m - kk]
    hi = np.sort(preds + resids)[kk - 1]
    return lo, hi


R = {}

# ---------- 1. nested LOO per-point hits + McNemar ----------
print("nested LOO per-point hits (75x74)")


def outer(i):
    rest = np.delete(np.arange(n), i)
    mu_i, sd_i = fit_predict(rest, np.array([i]))

    # inner LOO over the 74 training points
    def inner(j):
        tr = rest[rest != j]
        mu_pair, _ = fit_predict(tr, np.array([j, i]))
        return abs(y[j] - mu_pair[0]), mu_pair[1]

    out = [inner(j) for j in rest]
    resids = np.array([o[0] for o in out])
    preds_at_i = np.array([o[1] for o in out])
    row = {"mu": float(mu_i[0]), "sd": float(sd_i[0])}
    for alpha, tag in [(0.10, "90"), (0.05, "95")]:
        lo, hi = jk_interval(preds_at_i, resids, alpha)
        z = {"90": 1.645, "95": 1.96}[tag]
        row[f"jk_hit_{tag}"] = bool(lo <= y[i] <= hi)
        row[f"raw_hit_{tag}"] = bool(abs(y[i] - mu_i[0]) <= z * sd_i[0])
    return i, row


out = Parallel(n_jobs=-1)(delayed(outer)(i) for i in range(n))
rows = dict(out)
mc = {}
for tag in ["90", "95"]:
    raw = np.array([rows[i][f"raw_hit_{tag}"] for i in range(n)])
    jk = np.array([rows[i][f"jk_hit_{tag}"] for i in range(n)])
    b = int(np.sum(~raw & jk))
    c = int(np.sum(raw & ~jk))
    m = b + c
    p = float(2 * binom.cdf(min(b, c), m, 0.5)) if m > 0 else 1.0
    mc[tag] = {
        "raw_cov": float(raw.mean()),
        "jk_cov": float(jk.mean()),
        "jk_only_hits_b": b,
        "raw_only_hits_c": c,
        "mcnemar_exact_p": min(p, 1.0),
    }
R["mcnemar_rms"] = mc

# ---------- 2. group-wise jackknife+ coverage ----------
print("group-wise jackknife+ coverage")
GROUPS = {
    "composition": df["composition"].values,
    "force": df["force_N"].values,
    "frequency": df["freq_Hz"].values,
}
GW = {}
for gname, gvals in GROUPS.items():
    hits = {"90": [], "95": []}
    widths = {"90": [], "95": []}
    for v in np.unique(gvals):
        tr = np.where(gvals != v)[0]
        te = np.where(gvals == v)[0]

        def inner(j):
            trj = tr[tr != j]
            mu_all, _ = fit_predict(trj, np.concatenate(([j], te)))
            return abs(y[j] - mu_all[0]), mu_all[1:]

        out = Parallel(n_jobs=-1)(delayed(inner)(j) for j in tr)
        resids = np.array([o[0] for o in out])
        preds_te = np.stack([o[1] for o in out])  # (|tr|, |te|)
        for alpha, tag in [(0.10, "90"), (0.05, "95")]:
            for t_idx, t in enumerate(te):
                lo, hi = jk_interval(preds_te[:, t_idx], resids, alpha)
                hits[tag].append(lo <= y[t] <= hi)
                widths[tag].append(hi - lo)
    GW[gname] = {
        tag: {
            "coverage": float(np.mean(hits[tag])),
            "n": len(hits[tag]),
            "mean_width": float(np.mean(widths[tag])),
        }
        for tag in ["90", "95"]
    }
R["groupwise_jackknife_rms"] = GW

out_p = os.path.join(HERE, "..", "..", "results", "baseline", "groupwise_conformal.json")
json.dump(R, open(out_p, "w"), indent=1)
print("wrote", out_p)
