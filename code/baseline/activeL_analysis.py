#!/usr/bin/env python3
"""M2/E9b: pool-based active learning over the 75-point grid.
EI-driven GP queries vs random baseline; experiments-to-optimum over 30 seeds.
Writes activeL_results.json."""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df = pd.read_parquet("targets_design.parquet")
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
y = df["rms_Voc"].values.astype(float)
OPT = int(np.argmax(y))  # 2% CNT, 3 N, 20 Hz
N = len(y)
N0 = 10
SEEDS = range(30)


def gp():
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def run_ei(seed):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(N, N0, replace=False))
    if OPT in labeled:
        return N0
    pool = [i for i in range(N) if i not in labeled]
    while pool:
        xs = StandardScaler().fit(X[labeled])
        g = gp().fit(xs.transform(X[labeled]), y[labeled])
        mu, sd = g.predict(xs.transform(X[pool]), return_std=True)
        best = np.max(y[labeled])
        xi = 0.01
        z = (mu - best - xi) / np.maximum(sd, 1e-12)
        ei = (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)
        pick = pool[int(np.argmax(ei))]
        labeled.append(pick)
        pool.remove(pick)
        if pick == OPT:
            return len(labeled)
    return N


def run_random(seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    pos = int(np.where(order == OPT)[0][0]) + 1
    return N0 if pos <= N0 else pos


ei_counts = Parallel(n_jobs=-1)(delayed(run_ei)(s) for s in SEEDS)
rnd_counts = [run_random(s) for s in SEEDS]


def summarize(cnts):
    a = np.array(cnts, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": int(a.min()),
        "max": int(a.max()),
        "pct_within_19exp_25pct_budget": float(100 * np.mean(a <= 19)),
        "pct_within_30exp_40pct_budget": float(100 * np.mean(a <= 30)),
        "counts": [int(v) for v in cnts],
    }


res = {
    "n0": N0,
    "n_seeds": 30,
    "optimum_index": OPT,
    "optimum_condition": "2% CNT, 3 N, 20 Hz",
    "EI": summarize(ei_counts),
    "random": summarize(rnd_counts),
}
with open("activeL_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print("DONE")
