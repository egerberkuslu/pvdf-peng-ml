#!/usr/bin/env python3
"""M4: nested CV for the MLP — outer LOO, inner 5-fold hyperparameter selection.
Does tuning close the gap to the GP? Writes mlp_tuned_results.json."""
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df = pd.read_parquet("targets_design.parquet")
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)

GRID = [
    {"hidden": h, "alpha": a}
    for h in [(16,), (32,), (64,), (32, 16)]
    for a in [1e-4, 1e-2, 1.0]
]

FIXED_BASELINE = {"rms_Voc": 0.831, "Vpp": 0.769, "Vmax": 0.754}
GP_BASELINE = {"rms_Voc": 0.914, "Vpp": 0.775, "Vmax": 0.788}


def make_model(hidden, alpha):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        solver="lbfgs",
        max_iter=5000,
        random_state=0,
        early_stopping=False,
    )
    pipe = Pipeline([("xs", StandardScaler()), ("mlp", mlp)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def outer_fold(i, y):
    n = len(y)
    tr = np.delete(np.arange(n), i)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_cfg, best_score = None, -np.inf
    for cfg in GRID:
        preds = np.zeros(len(tr))
        for itr, ite in kf.split(tr):
            m = make_model(cfg["hidden"], cfg["alpha"])
            m.fit(X[tr[itr]], y[tr[itr]])
            preds[ite] = m.predict(X[tr[ite]])
        s = r2_score(y[tr], preds)
        if s > best_score:
            best_score, best_cfg = s, cfg
    m = make_model(best_cfg["hidden"], best_cfg["alpha"])
    m.fit(X[tr], y[tr])
    return (
        i,
        float(m.predict(X[i : i + 1])[0]),
        json.dumps({"hidden": list(best_cfg["hidden"]), "alpha": best_cfg["alpha"]}),
    )


results = {}
for tgt in ["rms_Voc", "Vpp", "Vmax"]:
    y = df[tgt].values.astype(float)
    out = Parallel(n_jobs=-1)(delayed(outer_fold)(i, y) for i in range(len(y)))
    preds = np.zeros(len(y))
    cfgs = []
    for i, p, c in out:
        preds[i] = p
        cfgs.append(c)
    r2 = float(r2_score(y, preds))
    mae = float(mean_absolute_error(y, preds))
    most_cfg, most_n = Counter(cfgs).most_common(1)[0]
    delta_gp = r2 - GP_BASELINE[tgt]
    results[tgt] = {
        "tuned_nestedCV_R2": r2,
        "tuned_nestedCV_MAE": mae,
        "fixed_config_LOO_R2": FIXED_BASELINE[tgt],
        "GP_LOO_R2": GP_BASELINE[tgt],
        "delta_tuned_minus_fixed": r2 - FIXED_BASELINE[tgt],
        "delta_tuned_minus_GP": delta_gp,
        "most_selected_config": json.loads(most_cfg),
        "most_selected_count": int(most_n),
        "tuning_closes_gap": bool(delta_gp >= -0.02),
    }

with open("mlp_tuned_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("DONE")
