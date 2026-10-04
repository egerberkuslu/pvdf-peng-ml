#!/usr/bin/env python3
"""Extended external replication on five public engineering datasets.

Generic registry version of protocol_replay.py covering Airfoil Self-Noise,
Energy Efficiency, Concrete Compressive Strength, Yacht Hydrodynamics, and a
seeded 2000-record subsample of the Combined Cycle Power Plant set. For every
dataset it reports in-sample and record-wise GP accuracy, raw-sigma GP
coverage, CV+(K=10) jackknife-plus-style GP coverage and width (the
calibrated-interval procedure of the manuscript at public-set scale),
group-wise generalization on verified level columns, and the
expected-improvement active-learning audit.
Writes protocol_replay_extended.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SEED = 0
EXT = (
    "external_data"
    if os.path.isdir("external_data")
    else os.path.join(os.path.dirname(__file__), "..", "..", "data", "external")
)


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
    n = len(y)
    mu = np.zeros(n)
    sd = np.zeros(n)
    fold_of = np.zeros(n, dtype=int)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = list(kf.split(X))
    out = Parallel(n_jobs=-1)(delayed(fit_fold)(X, y, tr, te) for tr, te in folds)
    models_pred = []
    for j, (te, m, s) in enumerate(out):
        mu[te] = m
        sd[te] = s
        fold_of[te] = j
    r2 = r2_score(y, mu)
    cov95 = float(np.mean(np.abs(y - mu) <= 1.96 * sd))
    return r2, cov95, mu, sd, fold_of, folds


def gp_cvplus(X, y, mu, fold_of, folds, alphas=(0.10, 0.05)):
    """CV+ (jackknife+ with K=10 folds) applied to the GP out-of-fold errors."""
    n = len(y)
    resid = np.abs(y - mu)
    # fold-model predictions at every point
    preds_all = np.zeros((n, len(folds)))

    def one(j, tr, te):
        xs = StandardScaler().fit(X[tr])
        g = gp(X.shape[1]).fit(xs.transform(X[tr]), y[tr])
        return j, g.predict(xs.transform(X))

    out = Parallel(n_jobs=-1)(
        delayed(one)(j, tr, te) for j, (tr, te) in enumerate(folds)
    )
    for j, p in out:
        preds_all[:, j] = p
    res = {}
    for alpha in alphas:
        hits, widths = [], []
        for j in range(n):
            cal = np.delete(np.arange(n), j)
            pa = preds_all[j, fold_of[cal]]
            r = resid[cal]
            kk = int(np.clip(np.ceil((1 - alpha) * (len(cal) + 1)), 1, len(cal)))
            lo = np.sort(pa - r)[len(cal) - kk]
            hi = np.sort(pa + r)[kk - 1]
            hits.append(lo <= y[j] <= hi)
            widths.append(hi - lo)
        res[f"{int((1-alpha)*100)}"] = {
            "coverage": float(np.mean(hits)),
            "mean_width": float(np.mean(widths)),
        }
    return res


def insample_r2(X, y):
    xs = StandardScaler().fit(X)
    g = gp(X.shape[1]).fit(xs.transform(X), y)
    return float(r2_score(y, g.predict(xs.transform(X))))


def leave_level_out(X, y, levels):
    n = len(y)
    mu = np.zeros(n)
    vals = np.unique(levels)
    out = Parallel(n_jobs=-1)(
        delayed(fit_fold)(X, y, np.where(levels != v)[0], np.where(levels == v)[0])
        for v in vals
    )
    for te, m, _ in out:
        mu[te] = m
    return float(r2_score(y, mu)), int(len(vals))


def active_learning(X, y, n_seeds=20, n0=10, max_q=150, top_frac=0.01):
    thr = np.quantile(y, 1 - top_frac)
    n = len(y)

    def run(seed):
        rng = np.random.default_rng(seed)
        labeled = list(rng.choice(n, n0, replace=False))
        if np.max(y[labeled]) >= thr:
            return n0
        pool = [i for i in range(n) if i not in set(labeled)]
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
        pos = int(np.argmax(y[order] >= thr)) + 1
        return max(pos, n0)

    ei = Parallel(n_jobs=-1)(delayed(run)(s) for s in range(n_seeds))
    rnd = [run_rand(s) for s in range(n_seeds)]
    return {"EI_median": float(np.median(ei)), "random_median": float(np.median(rnd))}


# ---------------- dataset registry ----------------
def load_airfoil():
    af = pd.read_csv(
        os.path.join(EXT, "airfoil_self_noise.dat"),
        sep=r"\s+",
        names=["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m", "spl_dB"],
    )
    X = af[["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m"]].values
    return (
        X.astype(float),
        af["spl_dB"].values.astype(float),
        {"chord": af["chord_m"].values, "velocity": af["velocity_ms"].values},
    )


def load_energy():
    enb = pd.read_excel(os.path.join(EXT, "ENB2012_data.xlsx"))
    X = enb[[f"X{i}" for i in range(1, 9)]].values.astype(float)
    return (
        X,
        enb["Y1"].values.astype(float),
        {"compactness": enb["X1"].values, "glazing": enb["X7"].values},
    )


def load_concrete():
    cols = [
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_agg",
        "fine_agg",
        "age_day",
        "strength_MPa",
    ]
    con = pd.read_excel(os.path.join(EXT, "Concrete_Data.xls"))
    con.columns = cols
    binder = np.where(
        con.slag > 0,
        np.where(con.fly_ash > 0, "slag+flyash", "slag"),
        np.where(con.fly_ash > 0, "flyash", "opc"),
    )
    X = con[cols[:8]].values.astype(float)
    return (
        X,
        con["strength_MPa"].values.astype(float),
        {"binder": binder, "age": con["age_day"].values},
    )


def load_yacht():
    cols = [
        "long_pos",
        "prismatic_coef",
        "length_disp_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
        "residuary_resistance",
    ]
    ya = pd.read_csv(
        os.path.join(EXT, "yacht_hydrodynamics.data"),
        sep=r"\s+",
        header=None,
        names=cols,
    )
    hull = ya.groupby(cols[:5], sort=False).ngroup().values
    X = ya[cols[:6]].values.astype(float)
    return (
        X,
        ya["residuary_resistance"].values.astype(float),
        {"hull": hull, "froude": ya["froude_number"].values},
    )


def load_ccpp():
    cc = pd.read_excel(os.path.join(EXT, "Folds5x2_pp.xlsx"), sheet_name="Sheet1")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(cc), 2000, replace=False)
    cc = cc.iloc[np.sort(idx)].reset_index(drop=True)
    X = cc[["AT", "V", "AP", "RH"]].values.astype(float)
    at_band = (cc["AT"] // 5 * 5).astype(int).values
    return X, cc["PE"].values.astype(float), {"AT_band": at_band}


DATASETS = [
    ("airfoil", load_airfoil),
    ("energy_efficiency", load_energy),
    ("concrete", load_concrete),
    ("yacht", load_yacht),
    ("ccpp_2000sub", load_ccpp),
]

results = {}
for name, loader in DATASETS:
    X, y, groups = loader()
    block = {"n": int(len(y))}
    block["in_sample_R2"] = insample_r2(X, y)
    r2, cov95, mu, sd, fold_of, folds = recordwise_cv(X, y)
    block["recordwise_10fold_R2"] = float(r2)
    block["gp_raw_coverage95"] = cov95
    block["gp_cvplus"] = gp_cvplus(X, y, mu, fold_of, folds)
    for gname, gvals in groups.items():
        gr2, nlev = leave_level_out(X, y, gvals)
        block[f"leave_one_{gname}_out_R2"] = gr2
        block[f"n_{gname}_levels"] = nlev
    block["active_learning_top1pct"] = active_learning(X, y)
    results[name] = block
    print(name, json.dumps(block, indent=1)[:400])

json.dump(results, open("protocol_replay_extended.json", "w"), indent=2)
print("wrote protocol_replay_extended.json")
