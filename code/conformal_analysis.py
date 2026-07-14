#!/usr/bin/env python3
"""M3: jackknife+ prediction intervals (Barber et al. 2021) for RF on Vpp/Vmax,
with honest coverage via an outer LOO around jackknife+; GP LOO comparison.
Writes conformal_results.json."""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = "targets_design.parquet"
FEATURES = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
TARGETS = ["Vpp", "Vmax"]
SEED = 0
N_TREES_REPORT = 300
N_TREES_OUTER = 150
ALPHAS = {"90": 0.10, "95": 0.05}
N_JOBS = -1


def rf(n_estimators):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=SEED, n_jobs=1)


def loo_predictions_and_residuals(X, y, n_estimators):
    n = len(y)

    def one(i):
        idx = np.delete(np.arange(n), i)
        m = rf(n_estimators)
        m.fit(X[idx], y[idx])
        return i, float(m.predict(X[i : i + 1])[0])

    results = Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n))
    loo_pred = np.empty(n)
    for i, p in results:
        loo_pred[i] = p
    return loo_pred, np.abs(y - loo_pred)


def jackknife_plus_interval(loo_pred_at_x, resid, alpha):
    n = len(resid)
    lo_vals = np.sort(loo_pred_at_x - resid)
    hi_vals = np.sort(loo_pred_at_x + resid)
    k = min(max(int(np.ceil((1.0 - alpha) * (n + 1))), 1), n)
    return lo_vals[n - k], hi_vals[k - 1]


def nested_loo_coverage(X, y, n_estimators):
    n = len(y)
    all_idx = np.arange(n)

    def outer(j):
        sub = np.delete(all_idx, j)

        def inner(i):
            train = np.setdiff1d(sub, [i])
            m = rf(n_estimators)
            m.fit(X[train], y[train])
            r_i = abs(y[i] - float(m.predict(X[i : i + 1])[0]))
            return r_i, float(m.predict(X[j : j + 1])[0])

        pairs = [inner(i) for i in sub]
        resid = np.array([p[0] for p in pairs])
        pred_at_j = np.array([p[1] for p in pairs])
        out = {}
        for name, a in ALPHAS.items():
            lo, hi = jackknife_plus_interval(pred_at_j, resid, a)
            out[name] = (lo, hi, float(y[j]), bool(lo <= y[j] <= hi))
        return j, out

    return dict(Parallel(n_jobs=N_JOBS)(delayed(outer)(j) for j in range(n)))


def cvplus_coverage(X, y, n_estimators, k_folds=10):
    n = len(y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(n)))
    fold_models = []
    for tr, _ in folds:
        m = rf(n_estimators)
        m.fit(X[tr], y[tr])
        fold_models.append(m)
    resid = np.empty(n)
    fold_of = np.empty(n, dtype=int)
    for k, (_, te) in enumerate(folds):
        resid[te] = np.abs(y[te] - fold_models[k].predict(X[te]))
        fold_of[te] = k
    cov = {}
    for j in range(n):
        cal = np.delete(np.arange(n), j)
        preds_at_j = np.array(
            [fold_models[fold_of[i]].predict(X[j : j + 1])[0] for i in cal]
        )
        r = resid[cal]
        out = {}
        for name, a in ALPHAS.items():
            lo, hi = jackknife_plus_interval(preds_at_j, r, a)
            out[name] = (lo, hi, float(y[j]), bool(lo <= y[j] <= hi))
        cov[j] = out
    return cov


def summarize(cov_by_j, y):
    n = len(y)
    top_mask = y >= np.quantile(y, 0.9)
    summary = {}
    for name in ALPHAS:
        widths = np.array(
            [cov_by_j[j][name][1] - cov_by_j[j][name][0] for j in range(n)]
        )
        hits = np.array([cov_by_j[j][name][3] for j in range(n)], dtype=float)
        summary[name] = {
            "empirical_coverage": float(hits.mean()),
            "mean_width": float(widths.mean()),
            "median_width": float(np.median(widths)),
            "mean_width_top_decile": float(widths[top_mask].mean()),
            "n_top_decile": int(top_mask.sum()),
        }
    return summary


def gp_loo_coverage(X, y):
    n = len(y)

    def one(i):
        idx = np.delete(np.arange(n), i)
        xs = StandardScaler().fit(X[idx])
        Xtr = xs.transform(X[idx])
        Xte = xs.transform(X[i : i + 1])
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * X.shape[1], nu=2.5, length_scale_bounds=(1e-2, 1e3)
        ) + WhiteKernel(1e-3, (1e-6, 1e1))
        g = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=SEED
        )
        g.fit(Xtr, y[idx])
        mu, sd = g.predict(Xte, return_std=True)
        mu, sd = float(mu[0]), float(sd[0])
        lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
        return bool(lo <= y[i] <= hi), hi - lo

    res = Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n))
    hits = np.array([r[0] for r in res], dtype=float)
    widths = np.array([r[1] for r in res])
    return {
        "coverage_95": float(hits.mean()),
        "mean_width_95": float(widths.mean()),
        "median_width_95": float(np.median(widths)),
    }


def main():
    t_start = time.time()
    df = pd.read_parquet(DATA)
    X = df[FEATURES].values.astype(float)
    n = len(df)
    y_probe = df[TARGETS[0]].values.astype(float)
    sub0 = np.delete(np.arange(n), 0)

    def probe(i):
        tr = np.setdiff1d(sub0, [i])
        rf(N_TREES_OUTER).fit(X[tr], y_probe[tr])
        return 0

    t0 = time.time()
    Parallel(n_jobs=N_JOBS)(delayed(probe)(i) for i in sub0)
    per_fold = time.time() - t0
    est_full_both = per_fold * n * len(TARGETS)
    use_full = est_full_both <= 540.0
    method = (
        "jackknife+ (nested outer LOO)"
        if use_full
        else "CV+ (K=10 outer, jackknife+ inner)"
    )
    results = {
        "method": method,
        "timing_probe_per_outer_fold_sec": round(per_fold, 3),
        "estimated_full_nested_loo_both_targets_sec": round(est_full_both, 1),
        "n_rows": int(n),
        "seed": SEED,
        "targets": {},
    }
    for tgt in TARGETS:
        y = df[tgt].values.astype(float)
        _, resid_full = loo_predictions_and_residuals(X, y, N_TREES_REPORT)
        cov = (
            nested_loo_coverage(X, y, N_TREES_OUTER)
            if use_full
            else cvplus_coverage(X, y, N_TREES_OUTER, 10)
        )
        results["targets"][tgt] = {
            "loo_rf_mae_300trees": float(np.mean(resid_full)),
            "loo_rf_rmse_300trees": float(np.sqrt(np.mean(resid_full**2))),
            "jackknife_plus": summarize(cov, y),
            "gp_loo": gp_loo_coverage(X, y),
        }
    results["total_runtime_sec"] = round(time.time() - t_start, 1)
    with open("conformal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
