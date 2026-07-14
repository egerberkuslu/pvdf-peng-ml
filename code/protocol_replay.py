#!/usr/bin/env python3
"""External replication of the honest-evaluation protocol on two public
engineering datasets (UCI Airfoil Self-Noise; UCI Energy Efficiency ENB2012).
Same steps as the PENG study: record-wise CV vs leave-one-level-out CV,
GP interval coverage, CV+ conformal for RF, and an EI active-learning audit.
Writes protocol_replay_results.json."""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 0


def gp(dim):
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * dim, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fit_fold(X, y, tr, te):
    xs = StandardScaler().fit(X[tr])
    g = gp(X.shape[1]).fit(xs.transform(X[tr]), y[tr])
    mu, sd = g.predict(xs.transform(X[te]), return_std=True)
    return te, mu, sd


def recordwise_cv(X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = list(kf.split(X))
    out = Parallel(n_jobs=-1)(delayed(fit_fold)(X, y, tr, te) for tr, te in folds)
    mu = np.zeros_like(y)
    sd = np.zeros_like(y)
    for te, m, s in out:
        mu[te] = m
        sd[te] = s
    cov95 = float(np.mean(np.abs(y - mu) <= 1.96 * sd))
    return float(r2_score(y, mu)), cov95


def insample_r2(X, y):
    xs = StandardScaler().fit(X)
    g = gp(X.shape[1]).fit(xs.transform(X), y)
    return float(r2_score(y, g.predict(xs.transform(X))))


def leave_level_out(X, y, levels):
    vals = np.unique(levels)
    out = Parallel(n_jobs=-1)(
        delayed(fit_fold)(X, y, np.where(levels != v)[0], np.where(levels == v)[0])
        for v in vals
    )
    mu = np.zeros_like(y)
    for te, m, _ in out:
        mu[te] = m
    return float(r2_score(y, mu)), int(len(vals))


def cvplus_rf(X, y, k=10, alpha=0.05):
    n = len(y)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(n)))
    models = []
    for tr, _ in folds:
        m = RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)
        m.fit(X[tr], y[tr])
        models.append(m)
    resid = np.empty(n)
    fold_of = np.empty(n, dtype=int)
    for j, (_, te) in enumerate(folds):
        resid[te] = np.abs(y[te] - models[j].predict(X[te]))
        fold_of[te] = j
    # evaluate coverage on each point using calibration set without it
    preds_all = np.column_stack([m.predict(X) for m in models])  # n x k
    hits = []
    widths = []
    for j in range(n):
        cal = np.delete(np.arange(n), j)
        pa = preds_all[j, fold_of[cal]]
        r = resid[cal]
        lo_v = np.sort(pa - r)
        hi_v = np.sort(pa + r)
        kk = min(max(int(np.ceil((1 - alpha) * (len(cal) + 1))), 1), len(cal))
        lo, hi = lo_v[len(cal) - kk], hi_v[kk - 1]
        hits.append(lo <= y[j] <= hi)
        widths.append(hi - lo)
    return float(np.mean(hits)), float(np.mean(widths))


def active_learning(X, y, n_seeds=20, n0=10, max_q=150, top_frac=0.01):
    thr = np.quantile(y, 1 - top_frac)
    n = len(y)

    def run(seed):
        rng = np.random.default_rng(seed)
        labeled = list(rng.choice(n, n0, replace=False))
        if np.max(y[labeled]) >= thr:
            return n0
        pool = np.setdiff1d(np.arange(n), labeled).tolist()
        for _ in range(max_q):
            xs = StandardScaler().fit(X[labeled])
            g = gp(X.shape[1]).fit(xs.transform(X[labeled]), y[labeled])
            mu, sd = g.predict(xs.transform(X[pool]), return_std=True)
            best = np.max(y[labeled])
            z = (mu - best - 0.01) / np.maximum(sd, 1e-12)
            ei = (mu - best - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
            pick = pool[int(np.argmax(ei))]
            labeled.append(pick)
            pool.remove(pick)
            if y[pick] >= thr:
                return len(labeled)
        return n0 + max_q

    def run_rand(seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        pos = np.argmax(y[order] >= thr) + 1
        return int(max(pos, n0))

    ei = Parallel(n_jobs=-1)(delayed(run)(s) for s in range(n_seeds))
    rd = [run_rand(s) for s in range(n_seeds)]
    return {
        "EI_median": float(np.median(ei)),
        "EI_mean": float(np.mean(ei)),
        "random_median": float(np.median(rd)),
        "random_mean": float(np.mean(rd)),
        "budget_total": int(n),
        "EI_median_budget_pct": float(100 * np.median(ei) / n),
    }


results = {}

# ---------- Dataset 1: UCI Airfoil Self-Noise ----------
af = pd.read_csv(
    "external_data/airfoil_self_noise.dat",
    sep=r"\s+",
    names=["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m", "spl_dB"],
)
Xa = af[["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m"]].values.astype(
    float
)
ya = af["spl_dB"].values.astype(float)
r2_cv, cov95 = recordwise_cv(Xa, ya)
r2_vel, nvel = leave_level_out(Xa, ya, af["velocity_ms"].values)
r2_chord, nch = leave_level_out(Xa, ya, af["chord_m"].values)
cov_cvp, w_cvp = cvplus_rf(Xa, ya)
al = active_learning(Xa, ya)
results["airfoil"] = {
    "n": int(len(ya)),
    "in_sample_R2": insample_r2(Xa, ya),
    "recordwise_10fold_R2": r2_cv,
    "gp_coverage95_recordwise": cov95,
    "leave_one_velocity_out_R2": r2_vel,
    "n_velocity_levels": nvel,
    "leave_one_chord_out_R2": r2_chord,
    "n_chord_levels": nch,
    "cvplus_rf_coverage95": cov_cvp,
    "cvplus_rf_mean_width95": w_cvp,
    "active_learning_top1pct": al,
}

# ---------- Dataset 2: UCI Energy Efficiency (ENB2012, heating load Y1) ----------
enb = pd.read_excel("external_data/ENB2012_data.xlsx")
Xe = enb[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values.astype(float)
ye = enb["Y1"].values.astype(float)
r2_cv_e, cov95_e = recordwise_cv(Xe, ye)
r2_glz, ngl = leave_level_out(Xe, ye, enb["X7"].values)
r2_rc, nrc = leave_level_out(Xe, ye, enb["X1"].values)
cov_cvp_e, w_cvp_e = cvplus_rf(Xe, ye)
al_e = active_learning(Xe, ye)
results["energy_efficiency"] = {
    "n": int(len(ye)),
    "in_sample_R2": insample_r2(Xe, ye),
    "recordwise_10fold_R2": r2_cv_e,
    "gp_coverage95_recordwise": cov95_e,
    "leave_one_glazing_out_R2": r2_glz,
    "n_glazing_levels": ngl,
    "leave_one_compactness_out_R2": r2_rc,
    "n_compactness_levels": nrc,
    "cvplus_rf_coverage95": cov_cvp_e,
    "cvplus_rf_mean_width95": w_cvp_e,
    "active_learning_top1pct": al_e,
}

with open("protocol_replay_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("DONE")
