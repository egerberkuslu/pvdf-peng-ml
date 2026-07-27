#!/usr/bin/env python3
"""Referee-response experiments for the PhysGP / calibrated-interval revision.

Answers, with the existing 75-condition design table and the five public
datasets, the quantitative demands of both referees:
  A. per-fold force-axis results for GP and PhysGP (within-fold R2 and MAE)
  B. mean-function ablations under group-wise CV: law alone, linear-force
     mean, and a force-offset variant of the law
  C. exact binomial (Clopper-Pearson) intervals for every reported coverage
  D. Lorentzian parameter uncertainties (per-material and global fits)
  E. multi-seed stability of the stochastic learners under LOO
  F. small-sample (n=75) subsampled replication on the five public sets,
     with CV+ widths normalized by the target standard deviation
  G. expected-improvement audit with a realizable stopping rule and a
     one-factor-at-a-time laboratory baseline
Writes revision_experiments.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import beta, norm
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "..", "data", "targets_design.parquet"))
EXT = os.path.join(HERE, "..", "data", "external")

FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
CNT = df["cnt_pct"].values.astype(float)
FRC = df["force_N"].values.astype(float)
FRQ = df["freq_Hz"].values.astype(float)
TARGETS = ["rms_Voc", "Vpp", "Vmax", "energy"]

P0 = [1.0, 0.1, -0.02, 1.0, 19.0, 8.0]
BOUNDS = ([0, -5, -5, 0, 10, 1], [50, 5, 5, 50, 30, 30])


def law(Xt, b0, b1, b2, A, f0, g):
    c, F, f = Xt
    lor = A * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)
    return (b0 + b1 * c + b2 * c**2) * F * lor


def law_off(Xt, b0, b1, b2, A, f0, g, d):
    return law(Xt, b0, b1, b2, A, f0, g) + d


def fit_curve(fn, tr, y, p0, bounds):
    scale = float(np.std(y[tr])) or 1.0
    try:
        p, pcov = curve_fit(
            fn,
            (CNT[tr], FRC[tr], FRQ[tr]),
            y[tr] / scale,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        p, pcov = np.array(p0), None
    return (
        lambda idx: fn((CNT[idx], FRC[idx], FRQ[idx]), *p) * scale,
        p,
        pcov,
        scale,
    )


def gp(dim=4):
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * dim, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fold_gp_mean(tr, te, y, mean_fn=None):
    """GP on y (mean_fn None) or on residuals about mean_fn (refit on tr)."""
    if mean_fn is None:
        target = y[tr]
        base_te = np.zeros(len(te))
    else:
        m = mean_fn(tr, y)
        target = y[tr] - m(tr)
        base_te = m(te)
    xs = StandardScaler().fit(X[tr])
    g = gp().fit(xs.transform(X[tr]), target)
    mu, sd = g.predict(xs.transform(X[te]), return_std=True)
    return te, base_te + mu, sd


def mean_law(tr, y):
    m, _, _, _ = fit_curve(law, tr, y, P0, BOUNDS)
    return m


def mean_law_off(tr, y):
    m, _, _, _ = fit_curve(
        law_off, tr, y, P0 + [0.0], (BOUNDS[0] + [-10], BOUNDS[1] + [10])
    )
    return m


def mean_linF(tr, y):
    a, b = np.polyfit(FRC[tr], y[tr], 1)
    return lambda idx: a * FRC[idx] + b


def group_folds(groups):
    idx = np.arange(len(groups))
    return [
        (np.where(groups != v)[0], np.where(groups == v)[0]) for v in np.unique(groups)
    ]


def loo_folds(n):
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


def pooled_cv(y, folds, fold_fn):
    mu = np.full(len(y), np.nan)
    out = Parallel(n_jobs=-1)(delayed(fold_fn)(tr, te, y) for tr, te in folds)
    for te, m, _ in out:
        mu[te] = m
    return mu


GROUPS = {
    "composition": df["composition"].values,
    "force": df["force_N"].values,
    "frequency": df["freq_Hz"].values,
}

R = {}

# ---------- A. per-fold force axis ----------
print("A: per-fold force results")
A = {}
force_folds = group_folds(FRC)
for tgt in TARGETS:
    y = df[tgt].values.astype(float)
    A[tgt] = {}
    for mname, mfn in [("gp", None), ("physgp", mean_law)]:
        rows = []
        for tr, te in force_folds:
            _, mu, _ = fold_gp_mean(tr, te, y, mfn)
            rows.append(
                {
                    "held_out_force_N": float(FRC[te[0]]),
                    "within_fold_R2": float(r2_score(y[te], mu)),
                    "MAE": float(mean_absolute_error(y[te], mu)),
                }
            )
        mu_all = pooled_cv(
            y, force_folds, lambda tr, te, yy, m=mfn: fold_gp_mean(tr, te, yy, m)
        )
        A[tgt][mname] = {"per_fold": rows, "pooled_R2": float(r2_score(y, mu_all))}
R["per_fold_force"] = A

# ---------- B. mean-function ablations (group-wise, all 3 axes) ----------
print("B: ablations")
B = {}
for tgt in TARGETS:
    y = df[tgt].values.astype(float)
    B[tgt] = {}
    for aname, gvals in GROUPS.items():
        folds = group_folds(gvals)
        row = {}
        # law alone (no residual GP)
        mu = np.full(len(y), np.nan)
        for tr, te in folds:
            m = mean_law(tr, y)
            mu[te] = m(te)
        row["law_alone_R2"] = float(r2_score(y, mu))
        # GP with linear-force mean
        mu = pooled_cv(y, folds, lambda tr, te, yy: fold_gp_mean(tr, te, yy, mean_linF))
        row["gp_linF_mean_R2"] = float(r2_score(y, mu))
        # PhysGP with additive offset law
        mu = pooled_cv(
            y, folds, lambda tr, te, yy: fold_gp_mean(tr, te, yy, mean_law_off)
        )
        row["physgp_offset_R2"] = float(r2_score(y, mu))
        B[tgt][aname] = row
R["ablations_groupwise"] = B

# ---------- C. binomial CIs for reported coverages ----------
print("C: binomial CIs")


def clopper(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


cal = json.load(
    open(os.path.join(HERE, "..", "results", "calibrated_conformal_results.json"))
)
CC = {}
for tgt in TARGETS:
    CC[tgt] = {}
    for meth in ["raw_gp", "jackknife_plus", "scaled_jackknife_plus"]:
        for lvl in ["90", "95"]:
            cov = cal[tgt][meth][lvl]["coverage"]
            k = int(round(cov * 75))
            CC[tgt][f"{meth}_{lvl}"] = {
                "coverage": cov,
                "hits": k,
                "ci95": clopper(k, 75),
            }
R["coverage_binomial_ci"] = CC

# ---------- D. Lorentzian uncertainties ----------
print("D: Lorentzian uncertainties")


def lor_only(f, A_, f0, g):
    return A_ * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)


DD = {"per_material": {}}
for comp, sub in df.groupby("composition"):
    m = sub.groupby("freq_Hz")["rms_Voc"].mean()
    fx, vy = m.index.values.astype(float), m.values
    try:
        p, pc = curve_fit(
            lor_only,
            fx,
            vy,
            p0=[vy.max(), 19.0, 8.0],
            bounds=([0, 10, 1], [10 * vy.max(), 30, 30]),
            maxfev=20000,
        )
        se = np.sqrt(np.diag(pc))
        DD["per_material"][str(comp)] = {
            "f0": float(p[1]),
            "f0_se": float(se[1]),
            "gamma": float(p[2]),
            "gamma_se": float(se[2]),
            "Q": float(p[1] / p[2]),
            "resid_df": int(len(fx) - 3),
        }
    except Exception as e:
        DD["per_material"][str(comp)] = {"error": str(e)}
# global law fit on rms with parameter SEs
y = df["rms_Voc"].values.astype(float)
tr = np.arange(len(y))
_, p, pcov, scale = fit_curve(law, tr, y, P0, BOUNDS)
se = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 6
DD["global_law_rms"] = {
    "f0": float(p[4]),
    "f0_se": float(se[4]),
    "gamma": float(p[5]),
    "gamma_se": float(se[5]),
    "b2": float(p[2]),
    "b2_se": float(se[2]),
    "Q": float(p[4] / p[5]),
}
R["lorentzian_uncertainty"] = DD

# ---------- E. multi-seed stability under LOO ----------
print("E: multi-seed stability")


def loo_model(model_fn, y, seed):
    mu = np.zeros(len(y))

    def one(i):
        tr = np.delete(np.arange(len(y)), i)
        m = model_fn(seed)
        m.fit(X[tr], y[tr])
        return i, m.predict(X[[i]])[0]

    out = Parallel(n_jobs=-1)(delayed(one)(i) for i in range(len(y)))
    for i, v in out:
        mu[i] = v
    return float(r2_score(y, mu))


MODELS = {
    "RandomForest": lambda s: RandomForestRegressor(n_estimators=100, random_state=s),
    "ExtraTrees": lambda s: ExtraTreesRegressor(n_estimators=100, random_state=s),
    "GradientBoosting": lambda s: GradientBoostingRegressor(random_state=s),
    "MLP": lambda s: MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=5000, random_state=s
    ),
}
EE = {}
y = df["rms_Voc"].values.astype(float)
for name, fn in MODELS.items():
    scores = [loo_model(fn, y, s) for s in range(5)]
    EE[name] = {
        "seeds_R2": [float(v) for v in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
R["multiseed_rms"] = EE

# ---------- F. n=75 subsampled replication ----------
print("F: n=75 subsampled replication")


def load_airfoil():
    af = pd.read_csv(
        os.path.join(EXT, "airfoil_self_noise.dat"),
        sep=r"\s+",
        names=["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m", "spl_dB"],
    )
    return af.iloc[:, :5].values.astype(float), af["spl_dB"].values.astype(float)


def load_energy():
    enb = pd.read_excel(os.path.join(EXT, "ENB2012_data.xlsx"))
    return (
        enb[[f"X{i}" for i in range(1, 9)]].values.astype(float),
        enb["Y1"].values.astype(float),
    )


def load_concrete():
    con = pd.read_excel(os.path.join(EXT, "Concrete_Data.xls"))
    return con.iloc[:, :8].values.astype(float), con.iloc[:, 8].values.astype(float)


def load_yacht():
    ya = pd.read_csv(
        os.path.join(EXT, "yacht_hydrodynamics.data"), sep=r"\s+", header=None
    )
    return ya.iloc[:, :6].values.astype(float), ya.iloc[:, 6].values.astype(float)


def load_ccpp():
    cc = pd.read_excel(os.path.join(EXT, "Folds5x2_pp.xlsx"), sheet_name="Sheet1")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cc), 2000, replace=False)
    cc = cc.iloc[np.sort(idx)]
    return cc[["AT", "V", "AP", "RH"]].values.astype(float), cc["PE"].values.astype(
        float
    )


PUB = {
    "airfoil": load_airfoil,
    "energy_efficiency": load_energy,
    "concrete": load_concrete,
    "yacht": load_yacht,
    "ccpp_2000sub": load_ccpp,
}


def gp_generic(dim):
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * dim, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def small_sample_run(Xp, yp, rng):
    n = len(yp)
    idx = rng.choice(n, 75, replace=False)
    Xs, ys = Xp[idx], yp[idx]
    # LOO GP point accuracy + CV+ (K=10) coverage and width at 95%
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    folds = list(kf.split(Xs))
    mu = np.zeros(75)
    fold_of = np.zeros(75, dtype=int)
    preds_all = np.zeros((75, 10))
    for j, (tr, te) in enumerate(folds):
        xs = StandardScaler().fit(Xs[tr])
        g = gp_generic(Xs.shape[1]).fit(xs.transform(Xs[tr]), ys[tr])
        mu[te] = g.predict(xs.transform(Xs[te]))
        fold_of[te] = j
        preds_all[:, j] = g.predict(xs.transform(Xs))
    r2 = r2_score(ys, mu)
    resid = np.abs(ys - mu)
    hits, widths = [], []
    alpha = 0.05
    for j in range(75):
        calx = np.delete(np.arange(75), j)
        pa = preds_all[j, fold_of[calx]]
        rr = resid[calx]
        kk = int(np.clip(np.ceil((1 - alpha) * (len(calx) + 1)), 1, len(calx)))
        lo = np.sort(pa - rr)[len(calx) - kk]
        hi = np.sort(pa + rr)[kk - 1]
        hits.append(lo <= ys[j] <= hi)
        widths.append(hi - lo)
    return r2, float(np.mean(hits)), float(np.mean(widths) / np.std(ys))


FF = {}
for name, loader in PUB.items():
    Xp, yp = loader()
    rng = np.random.default_rng(1)
    runs = [small_sample_run(Xp, yp, rng) for _ in range(10)]
    r2s = [r[0] for r in runs]
    covs = [r[1] for r in runs]
    ws = [r[2] for r in runs]
    FF[name] = {
        "n_pool": int(len(yp)),
        "R2_median": float(np.median(r2s)),
        "R2_iqr": [float(np.percentile(r2s, 25)), float(np.percentile(r2s, 75))],
        "cvplus95_cov_median": float(np.median(covs)),
        "cvplus95_cov_range": [float(min(covs)), float(max(covs))],
        "cvplus95_width_over_sd_median": float(np.median(ws)),
    }
    print(" ", name, FF[name])
R["subsample75_replication"] = FF

# full-size width normalization for Table 9
ext = json.load(
    open(os.path.join(HERE, "..", "results", "protocol_replay_extended.json"))
)
WN = {}
for name, loader in PUB.items():
    _, yp = loader()
    w = ext[name]["gp_cvplus"]["95"]["mean_width"]
    WN[name] = {
        "width95": float(w),
        "target_sd": float(np.std(yp)),
        "width_over_sd": float(w / np.std(yp)),
    }
R["fullsize_width_normalized"] = WN

# ---------- G. realizable stopping + OFAT baseline ----------
print("G: stopping rules and OFAT")
y = df["rms_Voc"].values.astype(float)
best_idx = int(np.argmax(y))
thr = np.max(y)


def ei_run(seed, stop_rel=0.01, patience=3, max_q=65, n0=10):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(75, n0, replace=False))
    pool = [i for i in range(75) if i not in set(labeled)]
    quiet = 0
    found_at = None
    stopped_at = None
    for step in range(max_q):
        if np.max(y[labeled]) >= thr and found_at is None:
            found_at = len(labeled)
        xs = StandardScaler().fit(X[labeled])
        g = gp().fit(xs.transform(X[labeled]), y[labeled])
        mu, sd = g.predict(xs.transform(X[pool]), return_std=True)
        best = np.max(y[labeled])
        z = (mu - best - 0.01) / np.maximum(sd, 1e-12)
        ei = (mu - best - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
        if np.max(ei) < stop_rel * best:
            quiet += 1
            if quiet >= patience:
                stopped_at = len(labeled)
                break
        else:
            quiet = 0
        pick = pool[int(np.argmax(ei))]
        labeled.append(pick)
        pool.remove(pick)
    if found_at is None and np.max(y[labeled]) >= thr:
        found_at = len(labeled)
    if stopped_at is None:
        stopped_at = len(labeled)
    regret = float(thr - np.max(y[labeled]))
    return stopped_at, found_at is not None and found_at <= stopped_at, regret


GG = {}
runs = [ei_run(s) for s in range(30)]
GG["ei_stopping"] = {
    "budget_median": float(np.median([r[0] for r in runs])),
    "budget_iqr": [
        float(np.percentile([r[0] for r in runs], 25)),
        float(np.percentile([r[0] for r in runs], 75)),
    ],
    "success_rate": float(np.mean([r[1] for r in runs])),
    "regret_median": float(np.median([r[2] for r in runs])),
    "regret_max": float(np.max([r[2] for r in runs])),
    "rule": "stop after 3 consecutive rounds with max EI < 1% of current best",
}

# OFAT: sweep frequency at (start composition, start force), then force at the
# best frequency, then composition at the best (force, frequency).
comps = df["composition"].unique().tolist()
forces = sorted(df["force_N"].unique().tolist())
freqs = sorted(df["freq_Hz"].unique().tolist())


def cell(comp, F, f):
    m = (df["composition"] == comp) & (df["force_N"] == F) & (df["freq_Hz"] == f)
    return int(np.where(m)[0][0])


ofat_rows = []
for c0 in comps:
    for F0 in forces:
        seen = []
        for f in freqs:
            seen.append(cell(c0, F0, f))
        f_best = FRQ[seen[int(np.argmax(y[seen]))]]
        for F in forces:
            i = cell(c0, F, f_best)
            if i not in seen:
                seen.append(i)
        sub = [i for i in seen if FRQ[i] == f_best]
        F_best = FRC[sub[int(np.argmax(y[sub]))]]
        for c in comps:
            i = cell(c, F_best, f_best)
            if i not in seen:
                seen.append(i)
        found = best_idx in seen
        ofat_rows.append(
            {
                "start": [str(c0), float(F0)],
                "n_experiments": len(seen),
                "found_true_best": bool(found),
                "best_found": float(np.max(y[seen])),
                "regret": float(thr - np.max(y[seen])),
            }
        )
GG["ofat"] = {
    "runs": ofat_rows,
    "n_experiments_median": float(np.median([r["n_experiments"] for r in ofat_rows])),
    "success_rate": float(np.mean([r["found_true_best"] for r in ofat_rows])),
    "regret_median": float(np.median([r["regret"] for r in ofat_rows])),
    "regret_max": float(np.max([r["regret"] for r in ofat_rows])),
}
R["stopping_and_ofat"] = GG

out = os.path.join(HERE, "..", "results", "revision_experiments.json")
json.dump(R, open(out, "w"), indent=1)
print("wrote", out)
