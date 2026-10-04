#!/usr/bin/env python3
"""Batch-D additional analyses on the 75-condition table (no new experiments).
Addresses editorial points: E1 (energy=N*rms^2), E3 (group-wise CV, drop-20Hz),
E3b (per-group error, high-output bias, calibration slope), E7 (interval coverage).
Writes batchD_results.json."""
import json, warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import Parallel, delayed

df = pd.read_parquet("targets_design.parquet")
feat = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[feat].values.astype(float)
res = {}

# --- E1: energy is exactly N * rms^2 ---
res["energy_is_N_rms2_max_abs_dev"] = float(
    (df.energy - 1000 * df.rms_Voc**2).abs().max()
)


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fit_predict(tr_idx, te_idx, y, return_std=False):
    xs = StandardScaler().fit(X[tr_idx])
    g = gp().fit(xs.transform(X[tr_idx]), y[tr_idx])
    if return_std:
        m, s = g.predict(xs.transform(X[te_idx]), return_std=True)
        return te_idx, m, s
    return te_idx, g.predict(xs.transform(X[te_idx])), None


def cv_by_groups(y, groups):
    """Leave-one-group-out: hold out all rows sharing a group value."""
    out = Parallel(n_jobs=-1)(
        delayed(fit_predict)(
            np.where(groups != gval)[0], np.where(groups == gval)[0], y, True
        )
        for gval in np.unique(groups)
    )
    preds = np.zeros_like(y)
    std = np.zeros_like(y)
    for te, m, s in out:
        preds[te] = m
        std[te] = s
    return preds, std


y = df["rms_Voc"].values.astype(float)

# --- anchor: standard LOO (each condition held out singly) ---
loo_groups = np.arange(len(y))
p_loo, s_loo = cv_by_groups(y, loo_groups)
res["rms_LOO_R2"] = float(r2_score(y, p_loo))
res["rms_LOO_MAE"] = float(mean_absolute_error(y, p_loo))

# --- E3: leave-one-GROUP-out (harder, honest generalization) ---
for name, col in [
    ("composition", df["composition"].values),
    ("frequency", df["freq_Hz"].values),
    ("force", df["force_N"].values),
]:
    p, s = cv_by_groups(y, np.asarray(col))
    res[f"rms_leave_one_{name}_out_R2"] = float(r2_score(y, p))
    res[f"rms_leave_one_{name}_out_MAE"] = float(mean_absolute_error(y, p))

# --- E3: drop 20 Hz entirely, predict the resonance ---
tr = np.where(df["freq_Hz"].values != 20)[0]
te = np.where(df["freq_Hz"].values == 20)[0]
_, m20, _ = fit_predict(tr, te, y, True)
res["drop20Hz_pred_mean"] = float(np.mean(m20))
res["drop20Hz_meas_mean"] = float(np.mean(y[te]))
res["drop20Hz_MAE"] = float(mean_absolute_error(y[te], m20))
res["drop20Hz_rel_underprediction_pct"] = float(
    100 * (np.mean(y[te]) - np.mean(m20)) / np.mean(y[te])
)

# --- E3b: high-output bias (calibration slope/intercept, top-10% error) ---
b1, b0 = np.polyfit(y, p_loo, 1)  # pred = b1*meas + b0
res["calibration_slope"] = float(b1)
res["calibration_intercept"] = float(b0)
thr = np.percentile(y, 90)
hi = y >= thr
res["top10pct_MAE"] = float(mean_absolute_error(y[hi], p_loo[hi]))
res["top10pct_mean_rel_err_pct"] = float(
    100 * np.mean(np.abs(p_loo[hi] - y[hi]) / y[hi])
)
res["overall_mean_rel_err_pct"] = float(
    100 * np.mean(np.abs(p_loo - y) / np.maximum(y, 1e-9))
)

# --- E3b: per-frequency and per-composition MAE (standard LOO) ---
res["MAE_by_frequency"] = {
    int(f): float(mean_absolute_error(y[df.freq_Hz == f], p_loo[df.freq_Hz == f]))
    for f in sorted(df.freq_Hz.unique())
}
res["MAE_by_composition"] = {
    str(c): float(
        mean_absolute_error(y[df.composition == c], p_loo[df.composition == c])
    )
    for c in df.composition.unique()
}

# --- E7: empirical coverage of GP intervals under standard LOO ---
z68, z95 = 1.0, 1.959963985
res["coverage68_pct"] = float(100 * np.mean(np.abs(y - p_loo) <= z68 * s_loo))
res["coverage95_pct"] = float(100 * np.mean(np.abs(y - p_loo) <= z95 * s_loo))
res["mean_interval_width95"] = float(np.mean(2 * z95 * s_loo))
res["mean_posterior_std"] = float(np.mean(s_loo))

with open("batchD_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print("\nDONE")
