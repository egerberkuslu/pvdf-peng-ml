#!/usr/bin/env python3
"""Genetic-algorithm cross-check of the grid optimum.

A plain generational GA with tournament selection, blend crossover, and
Gaussian mutation searches the continuous composition-force-frequency box
over the RMS-voltage Gaussian process refit on all 75 conditions. The run
cross-checks the dense-grid optimum reported in the manuscript. Writes
ga_search.json.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "..", "..", "data", "targets_design.parquet"))
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
y = df["rms_Voc"].values.astype(float)

k = C(1.0, (1e-3, 1e3)) * Matern(
    length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
) + WhiteKernel(1e-3, (1e-6, 1e1))
xs = StandardScaler().fit(X)
gp = GaussianProcessRegressor(
    kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
).fit(xs.transform(X), y)

LO = np.array([0.0, 1.0, 5.0])  # cnt_pct, force_N, freq_Hz
HI = np.array([3.0, 3.0, 25.0])


def fitness(pop):
    full = np.column_stack(
        [pop[:, 0], np.zeros(len(pop)), pop[:, 1], pop[:, 2]]
    )  # is_pristine=0 inside the composite series
    return gp.predict(xs.transform(full))


def run_ga(seed, pop_size=60, gens=80, tourn=3, mut_sd=0.08, elite=2):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LO, HI, size=(pop_size, 3))
    for _ in range(gens):
        fit = fitness(pop)
        order = np.argsort(fit)[::-1]
        nxt = [pop[i].copy() for i in order[:elite]]
        while len(nxt) < pop_size:
            picks = rng.integers(0, pop_size, size=(2, tourn))
            pa = pop[picks[0][np.argmax(fit[picks[0]])]]
            pb = pop[picks[1][np.argmax(fit[picks[1]])]]
            w = rng.uniform(-0.1, 1.1, size=3)
            child = w * pa + (1 - w) * pb
            child += rng.normal(0, mut_sd * (HI - LO), size=3)
            nxt.append(np.clip(child, LO, HI))
        pop = np.array(nxt)
    fit = fitness(pop)
    b = int(np.argmax(fit))
    return pop[b], float(fit[b])


results = []
for seed in range(10):
    xbest, fbest = run_ga(seed)
    results.append(
        {
            "seed": seed,
            "cnt_pct": float(xbest[0]),
            "force_N": float(xbest[1]),
            "freq_Hz": float(xbest[2]),
            "predicted_rms_V": fbest,
        }
    )
    print(seed, np.round(xbest, 3), round(fbest, 4))

out = {
    "runs": results,
    "cnt_range": [
        min(r["cnt_pct"] for r in results),
        max(r["cnt_pct"] for r in results),
    ],
    "force_range": [
        min(r["force_N"] for r in results),
        max(r["force_N"] for r in results),
    ],
    "freq_range": [
        min(r["freq_Hz"] for r in results),
        max(r["freq_Hz"] for r in results),
    ],
}
json.dump(
    out, open(os.path.join(HERE, "..", "..", "results", "baseline", "ga_search.json"), "w"), indent=1
)
print("wrote ga_search.json")
