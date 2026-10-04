#!/usr/bin/env python3
"""E5 METHODS (analysis protocol section 1, prefix a10, macros mth).

Three blocks, one command from the repository root:
    python code/a10_methods.py

(a) Fair tuning. Nested, equal-budget random search for every regressor of the
    benchmark (the 14 factories of code/reg_common.py): 30 configurations per
    model drawn with ParameterSampler(random_state=42) from a declared space, inner
    folds group-aware by specimen (leave-one-composition-out inside the outer
    training set), selection by pooled inner mean squared error, outer LOO on all
    three targets. Reported beside the fixed-setting benchmark of numbers_a3.
(b) Held-out frequency and composition rescue on V_rms. Five new models against the
    plain GP, LawGP and ProductGP of A5, on within-grid LOO and the three held-out
    axes with the paper's folds and per-level metrics (within-level R2, MAE):
      hier     hierarchical LawGP, partial pooling of (f0, gamma) across compositions,
               pooling strength chosen by inner folds of the same scheme
      lorenv   law-as-kernel GP: covariance F L(f) F' L(f') x (constant + ARD Matern),
               the law's force and Lorentzian factors as a covariance envelope
      lorspec  ProductGP whose frequency factor is a damped-cosine kernel, the
               stationary kernel with a Lorentzian spectral density, initialized
               from the fitted law width
      loghet   log-target LawGP (law fitted to log y) with additive plus proportional
               noise (log-scale noise variance s_p + s_a (mbar/m(x))^2)
      mono     plain GP conditioned on dV/dF >= 0 at 125 virtual points (hard-constraint
               limit of virtual derivative observations, truncated-Gaussian posterior)
    A twin-informed multi-fidelity GP is skipped: no E1 generator is importable.
(c) Active learning with the paper's replay protocol (same initial designs, oracle
    stop and the realizable stop of revision_experiments.py) on the real grid
    (LawGP and plain-GP models) and on the five public datasets of
    protocol_replay_extended.py (plain-GP model, top 1 percent target):
    EI, UCB, Thompson sampling, batch EI (q = 3, kriging believer), OFAT-then-EI
    hybrid, OFAT sweeps and random order.
Every headline is repeated without recordings 52, 54, 73 (n = 72, duplicated-recordings note in the README).

Outputs (results/): a10_methods.json, a10_predictions.csv,
a10_figdata_{fair,levels,al}.csv, numbers_a10.tex, tables/tab_a10_*.tex,
cache in a10_cache/. Nothing is selected on outer test folds.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
for _p in (str(ROOT), str(ROOT / "code"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(HERE), str(ROOT / "code"), str(ROOT)]
    + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.linalg import cho_solve  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
import scipy.special as sps  # noqa: E402
from scipy.stats import (
    kendalltau,
    loguniform,
    norm,
    randint,
    spearmanr,
    uniform,
)  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel as CK  # noqa: E402
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel  # noqa: E402
from sklearn.gaussian_process.kernels import Matern, WhiteKernel  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, ParameterSampler  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

import a2_common as A2  # noqa: E402
import a3_common as C3  # noqa: E402
import a5_common as A5  # noqa: E402

SEED = 42
N_CFG = 30
DUP_ROWS = [52, 54, 73]
ALL_ROWS = np.arange(A2.N)
KEEP = np.array([i for i in range(A2.N) if i not in DUP_ROWS])
ROWSETS = {"n75": ALL_ROWS, "n72": KEEP}
X = A2.X
Y = {t: A2.df[t].values.astype(float) for t in A2.TARGETS}
COMP = A2.COMP
COMP_ID = np.array([A2.COMP_ORDER.index(c) for c in COMP])
CAMEL_T = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}

RESULTS = ROOT / "results"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if np.isnan(x):
            return None
        if np.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return o


def fnum(x, nd):
    return C3.fnum(x, nd)


# ====================================================================== cache
class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        return self.root / f"{key}.pkl"

    def has(self, key):
        return self.path(key).exists()

    def get(self, key):
        with open(self.path(key), "rb") as fh:
            return pickle.load(fh)

    def put(self, key, val):
        tmp = self.path(key).with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(val, fh)
        tmp.replace(self.path(key))


# ====================================================================== (a) fair tuning
import a3_bench  # noqa: E402  (ORDER, NAME; imports reg_common)
import reg_common  # noqa: E402

BENCH = a3_bench.ORDER
TREE_SPACE = {
    "n_estimators": [100, 200, 400],
    "max_features": [0.25, 0.5, 0.75, 1.0],
    "max_depth": [None, 3, 5, 8],
    "min_samples_leaf": randint(1, 6),
    "bootstrap": [True, False],
}
SPACES = {
    "Lineer": {"fit_intercept": [True, False], "positive": [True, False]},
    "Ridge": {"alpha": loguniform(1e-4, 1e3)},
    "Lasso": {"alpha": loguniform(1e-5, 1e0)},
    "Polinom2-OLS": {"degree": [1, 2, 3], "interaction_only": [True, False]},
    "Polinom2-Ridge": {"degree": [2, 3], "alpha": loguniform(1e-4, 1e3)},
    "SVR-RBF": {
        "C": loguniform(1e-2, 1e3),
        "gamma": loguniform(1e-3, 1e1),
        "epsilon": loguniform(1e-3, 1e0),
    },
    "KernelRidge": {
        "kernel": ["rbf", "laplacian"],
        "alpha": loguniform(1e-4, 1e1),
        "gamma": loguniform(1e-3, 1e1),
    },
    "KNN": {
        "n_neighbors": randint(1, 16),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "RandomForest": dict(TREE_SPACE),
    "ExtraTrees": dict(TREE_SPACE),
    "GradientBoosting": {
        "n_estimators": randint(50, 501),
        "learning_rate": loguniform(1e-2, 3e-1),
        "max_depth": randint(2, 6),
        "subsample": uniform(0.5, 0.5),
        "min_samples_leaf": randint(1, 6),
    },
    "XGBoost": {
        "n_estimators": randint(50, 601),
        "max_depth": randint(2, 7),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-3, 1e1),
        "min_child_weight": loguniform(0.5, 1e1),
    },
    "ANN-MLP": {
        "hidden_layer_sizes": [(16,), (32,), (64,), (16, 16), (32, 16), (64, 32)],
        "alpha": loguniform(1e-5, 1e0),
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "activation": ["relu", "tanh"],
    },
    "ARD-GP": {
        "nu": [0.5, 1.5, 2.5, np.inf],
        "ard": [True, False],
        "restarts": [0, 2],
        "standardize": [True, False],
    },
}


def space_text(name):
    out = {}
    for k, v in SPACES[name].items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            d = v.dist.name if hasattr(v, "dist") else type(v).__name__
            out[k] = f"{d}{tuple(float(a) for a in v.args)}"
    return out


def configs(name, n_cfg=N_CFG):
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
        for p in ParameterSampler(SPACES[name], n_iter=n_cfg, random_state=SEED)
    ]


def build(name, p):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    def ttr(r):
        return TransformedTargetRegressor(regressor=r, transformer=StandardScaler())

    if name == "Lineer":
        return LinearRegression(**p)
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "Lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=p["alpha"], max_iter=10000))
    if name == "Polinom2-OLS":
        return make_pipeline(
            PolynomialFeatures(p["degree"], interaction_only=p["interaction_only"]),
            LinearRegression(),
        )
    if name == "Polinom2-Ridge":
        return make_pipeline(
            StandardScaler(), PolynomialFeatures(p["degree"]), Ridge(alpha=p["alpha"])
        )
    if name == "SVR-RBF":
        return ttr(
            make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"]),
            )
        )
    if name == "KernelRidge":
        return ttr(
            make_pipeline(
                StandardScaler(),
                KernelRidge(kernel=p["kernel"], alpha=p["alpha"], gamma=p["gamma"]),
            )
        )
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=int(p["n_neighbors"]), weights=p["weights"], p=int(p["p"])
            ),
        )
    if name in ("RandomForest", "ExtraTrees"):
        cls = RandomForestRegressor if name == "RandomForest" else ExtraTreesRegressor
        return cls(
            n_estimators=int(p["n_estimators"]),
            max_features=p["max_features"],
            max_depth=p["max_depth"],
            min_samples_leaf=int(p["min_samples_leaf"]),
            bootstrap=p["bootstrap"],
            random_state=SEED,
            n_jobs=1,
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=int(p["n_estimators"]),
            learning_rate=p["learning_rate"],
            max_depth=int(p["max_depth"]),
            subsample=p["subsample"],
            min_samples_leaf=int(p["min_samples_leaf"]),
            random_state=SEED,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            min_child_weight=p["min_child_weight"],
            random_state=SEED,
            verbosity=0,
            n_jobs=1,
        )
    if name == "ANN-MLP":
        return ttr(
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                    alpha=p["alpha"],
                    learning_rate_init=p["learning_rate_init"],
                    activation=p["activation"],
                    max_iter=4000,
                    random_state=SEED,
                ),
            )
        )
    if name == "ARD-GP":
        if p["ard"]:
            ls = [1.0] * 4 if p["standardize"] else [1.0, 1.0, 1.0, 5.0]
        else:
            ls = 1.0
        base = RBF(ls) if np.isinf(p["nu"]) else Matern(ls, nu=p["nu"])
        g = GaussianProcessRegressor(
            kernel=CK(1.0) * base + WhiteKernel(1e-3),
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=int(p["restarts"]),
            random_state=SEED,
        )
        return make_pipeline(StandardScaler(), g) if p["standardize"] else g
    raise ValueError(name)


def inner_folds_a(tr, inner):
    if inner == "group":
        out = []
        for v in A2.COMP_ORDER:
            m = COMP[tr] == v
            if m.any() and (~m).any():
                out.append((tr[~m], tr[m]))
        return out
    kf = KFold(5, shuffle=True, random_state=SEED)
    return [(tr[a], tr[b]) for a, b in kf.split(tr)]


def nested_one(name, t, rows, i, inner, cfgs):
    warnings.filterwarnings("ignore")
    y = Y[t]
    tr = rows[rows != i]
    folds = inner_folds_a(tr, inner)
    mse = np.full(len(cfgs), np.inf)
    for c, p in enumerate(cfgs):
        try:
            sse = 0.0
            for a, b in folds:
                m = build(name, p).fit(X[a], y[a])
                sse += float(np.sum((y[b] - m.predict(X[b])) ** 2))
            mse[c] = sse / len(tr)
        except Exception:
            mse[c] = np.inf
        if not np.isfinite(mse[c]):
            mse[c] = np.inf
    best = int(np.argmin(mse))
    m = build(name, cfgs[best]).fit(X[tr], y[tr])
    return int(i), best, float(m.predict(X[[i]])[0]), mse


def fixed_one(name, t, rows, i):
    warnings.filterwarnings("ignore")
    y = Y[t]
    tr = rows[rows != i]
    est = reg_common.model_factories()[name]()
    if name == "XGBoost":
        est.set_params(n_jobs=1)
    est.fit(X[tr], y[tr])
    return int(i), float(est.predict(X[[i]])[0])


def ranks_desc(vals):
    """Rank 1 = best (largest R2); ties share the lower rank."""
    order = sorted(vals, key=lambda k: -vals[k])
    out, prev, r = {}, None, 0
    for k, m in enumerate(order, 1):
        if prev is None or vals[m] != prev:
            r = k
        out[m], prev = r, vals[m]
    return out


def block_a(cache, n_jobs, quick=False):
    T0 = time.time()
    models = BENCH
    n_cfg = 4 if quick else N_CFG
    CF = {m: configs(m, n_cfg) for m in models}
    targets = ["rms_Voc"] if quick else A2.TARGETS
    runs = []  # (tag, model, target, rowset, inner)
    for rs in ("n75", "n72"):
        for t in targets:
            for m in models:
                runs.append((f"a_nested_{m}_{t}_{rs}_group", m, t, rs, "group"))
    if not quick:
        for m in models:
            runs.append(
                (f"a_nested_{m}_rms_Voc_n75_kfold", m, "rms_Voc", "n75", "kfold")
            )
    fixed_runs = []
    for rs in ("n75", "n72"):
        for t in targets:
            for m in models:
                fixed_runs.append((f"a_fixed_{m}_{t}_{rs}", m, t, rs))

    jobs, jmap = [], []
    for tag, m, t, rs, inner in runs:
        if cache.has(tag):
            continue
        rows = ROWSETS[rs] if not quick else ROWSETS[rs][:20]
        for i in rows:
            jobs.append(delayed(nested_one)(m, t, rows, i, inner, CF[m]))
            jmap.append(tag)
    for tag, m, t, rs in fixed_runs:
        if cache.has(tag):
            continue
        rows = ROWSETS[rs] if not quick else ROWSETS[rs][:20]
        for i in rows:
            jobs.append(delayed(fixed_one)(m, t, rows, i))
            jmap.append(tag)
    print(f"[a10/a] {len(jobs)} outer-fold jobs to run", flush=True)
    if jobs:
        out = Parallel(n_jobs=n_jobs, verbose=0)(jobs)
        bucket = {}
        for tag, r in zip(jmap, out):
            bucket.setdefault(tag, []).append(r)
        for tag, lst in bucket.items():
            cache.put(tag, lst)
    print(f"[a10/a] nested tuning done in {time.time() - T0:.0f} s", flush=True)

    A3 = json.loads((RESULTS / "a3_tables.json").read_text())["benchmark"]
    res = {
        "design": {
            "models": models,
            "n_configurations": n_cfg,
            "sampler": "sklearn ParameterSampler(random_state=42); spaces with fewer distinct"
            " configurations than the budget use all of them",
            "inner_folds": "leave-one-composition-out (group-aware by specimen) on the outer"
            " training rows; 5 inner folds at n = 75 and n = 72",
            "selection_score": "pooled inner mean squared error over all outer-training rows",
            "outer": "LOO over the rows of the set",
            "refit": "selected configuration refitted on the full outer training set",
            "seed": SEED,
            "spaces": {m: space_text(m) for m in models},
            "fixed_setting_source": "results/a3_tables.json benchmark.<target>.models"
            ".<model>.cv_r2 (the values behind tab_bench_* and numbers_a3.tex)",
            "sensitivity_inner_kfold": "V_rms only: inner KFold(5, shuffle, 42) record-level",
        },
        "n75": {},
        "n72": {},
        "inner_kfold_rms": {},
    }
    PRED = []
    for rs in ("n75", "n72"):
        rows = ROWSETS[rs] if not quick else ROWSETS[rs][:20]
        for t in targets:
            y = Y[t]
            per = {}
            for m in models:
                lst = sorted(cache.get(f"a_nested_{m}_{t}_{rs}_group"))
                idx = np.array([r[0] for r in lst])
                pred = np.array([r[2] for r in lst])
                sel = np.array([r[1] for r in lst])
                mse = np.array([r[3] for r in lst])
                flst = sorted(cache.get(f"a_fixed_{m}_{t}_{rs}"))
                fpred = np.array([r[1] for r in flst])
                vals, cnt = np.unique(sel, return_counts=True)
                mode = int(vals[np.argmax(cnt)])
                per[m] = {
                    "tuned_loo_r2": float(r2_score(y[idx], pred)),
                    "tuned_loo_mae": float(mean_absolute_error(y[idx], pred)),
                    "fixed_recomputed_loo_r2": float(r2_score(y[idx], fpred)),
                    "fixed_recomputed_loo_mae": float(
                        mean_absolute_error(y[idx], fpred)
                    ),
                    "selected_config_per_fold": sel.tolist(),
                    "modal_config_index": mode,
                    "modal_config": CF[m][mode],
                    "modal_config_share": float(np.max(cnt) / len(sel)),
                    "n_distinct_configs_selected": int(len(vals)),
                    "inner_mse_best_median": float(np.median(np.min(mse, axis=1))),
                    "n": int(len(idx)),
                }
                if rs == "n75" and not quick:
                    a3m = A3[t]["models"][m]
                    per[m]["fixed_loo_r2"] = float(a3m["cv_r2"])
                    per[m]["fixed_loo_mae"] = float(a3m["cv_mae"])
                    per[m]["fixed_source"] = a3m.get("cv_r2_source", "a3_tables.json")
                else:
                    per[m]["fixed_loo_r2"] = per[m]["fixed_recomputed_loo_r2"]
                    per[m]["fixed_loo_mae"] = per[m]["fixed_recomputed_loo_mae"]
                    per[m]["fixed_source"] = "recomputed here with reg_common factories"
                per[m]["delta_r2_tuned_minus_fixed"] = (
                    per[m]["tuned_loo_r2"] - per[m]["fixed_loo_r2"]
                )
                for k, i in enumerate(idx):
                    PRED.append(
                        {
                            "block": "a_fair_tuning",
                            "condition_id": int(i),
                            "target": t,
                            "model": m,
                            "split_scheme": "loo"
                            + ("" if rs == "n75" else "_excl_dup"),
                            "held_out_level": str(int(i)),
                            "y_true": y[i],
                            "y_pred": pred[k],
                            "y_pred_fixed_recomputed": fpred[k],
                            "selected_config": int(sel[k]),
                        }
                    )
            fx = {m: per[m]["fixed_loo_r2"] for m in models}
            tu = {m: per[m]["tuned_loo_r2"] for m in models}
            rf, rt = ranks_desc(fx), ranks_desc(tu)
            for m in models:
                per[m]["rank_fixed"] = rf[m]
                per[m]["rank_tuned"] = rt[m]
            best_f = min(models, key=lambda m: rf[m])
            best_t = min(models, key=lambda m: rt[m])
            kt = kendalltau([fx[m] for m in models], [tu[m] for m in models])
            sp = spearmanr([fx[m] for m in models], [tu[m] for m in models])
            top3_f = sorted(models, key=lambda m: rf[m])[:3]
            top3_t = sorted(models, key=lambda m: rt[m])[:3]
            res[rs][t] = {
                "models": per,
                "best_fixed": best_f,
                "best_tuned": best_t,
                "winner_changes": best_f != best_t,
                "top3_fixed": top3_f,
                "top3_tuned": top3_t,
                "top3_set_changes": set(top3_f) != set(top3_t),
                "n_models_rank_changed": int(sum(rf[m] != rt[m] for m in models)),
                "max_abs_rank_shift": int(max(abs(rf[m] - rt[m]) for m in models)),
                "kendall_tau": float(kt.statistic),
                "kendall_p": float(kt.pvalue),
                "spearman_rho": float(sp.statistic),
                "gp_rank_fixed": rf["ARD-GP"],
                "gp_rank_tuned": rt["ARD-GP"],
                "gp_tuned_minus_best_other_tuned": tu["ARD-GP"]
                - max(tu[m] for m in models if m != "ARD-GP"),
                "n_models_improved": int(sum(tu[m] > fx[m] for m in models)),
            }
    if not quick:
        y = Y["rms_Voc"]
        for m in models:
            lst = sorted(cache.get(f"a_nested_{m}_rms_Voc_n75_kfold"))
            idx = np.array([r[0] for r in lst])
            pred = np.array([r[2] for r in lst])
            res["inner_kfold_rms"][m] = {
                "tuned_loo_r2": float(r2_score(y[idx], pred)),
                "tuned_loo_mae": float(mean_absolute_error(y[idx], pred)),
            }
        tu = {m: res["inner_kfold_rms"][m]["tuned_loo_r2"] for m in models}
        rk = ranks_desc(tu)
        for m in models:
            res["inner_kfold_rms"][m]["rank"] = rk[m]
        res["inner_kfold_rms_best"] = min(models, key=lambda m: rk[m])
    res["runtime_s"] = time.time() - T0
    return res, PRED


# ====================================================================== (b) new models
class Env(Kernel):
    """Fixed rank-one envelope k(x, x') = s(x) s(x'), s = F * Lorentz(f; f0, gamma).

    F and f are read in raw units from columns dims of the augmented input. No
    hyperparameters (f0 and gamma come from the law fitted on the training fold).
    """

    def __init__(self, dims=(4, 5), f0=19.0, gamma=8.0):
        self.dims = dims
        self.f0 = f0
        self.gamma = gamma

    def _s(self, Z):
        Z = np.atleast_2d(Z)
        return Z[:, self.dims[0]] * A2.lorentz(Z[:, self.dims[1]], self.f0, self.gamma)

    def __call__(self, X_, Y_=None, eval_gradient=False):
        sx = self._s(X_)
        sy = sx if Y_ is None else self._s(Y_)
        K = np.outer(sx, sy)
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X_):
        return self._s(X_) ** 2

    def is_stationary(self):
        return False


class DampedCos(Kernel):
    """k(d) = exp(-|d| / l) cos(2 pi d / P) on one input column.

    The spectral density is a pair of Lorentzian lines at +-1/P with half width
    1/(2 pi l): the stationary kernel with a Lorentzian spectrum.
    """

    def __init__(
        self,
        dim=3,
        length_scale=1.0,
        period=2.0,
        length_scale_bounds=(1e-2, 1e3),
        period_bounds=(1e-1, 1e3),
    ):
        self.dim = dim
        self.length_scale = length_scale
        self.period = period
        self.length_scale_bounds = length_scale_bounds
        self.period_bounds = period_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_period(self):
        return Hyperparameter("period", "numeric", self.period_bounds)

    def __call__(self, X_, Y_=None, eval_gradient=False):
        x = np.atleast_2d(X_)[:, self.dim]
        yv = x if Y_ is None else np.atleast_2d(Y_)[:, self.dim]
        d = x[:, None] - yv[None, :]
        a = np.abs(d)
        E = np.exp(-a / self.length_scale)
        ph = 2 * np.pi * d / self.period
        K = E * np.cos(ph)
        if eval_gradient:
            g = []
            if not self.hyperparameter_length_scale.fixed:
                g.append(K * a / self.length_scale)
            if not self.hyperparameter_period.fixed:
                g.append(E * np.sin(ph) * ph)
            G = np.dstack(g) if g else np.empty((K.shape[0], K.shape[1], 0))
            return K, G
        return K

    def diag(self, X_):
        return np.ones(np.atleast_2d(X_).shape[0])

    def is_stationary(self):
        return True


class HeteroWhite(Kernel):
    """Diagonal noise noise_level * w(x), w read from column wdim (fixed weights)."""

    def __init__(self, wdim=4, noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)):
        self.wdim = wdim
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X_, Y_=None, eval_gradient=False):
        X_ = np.atleast_2d(X_)
        if Y_ is not None and eval_gradient:
            raise ValueError("gradient only for Y=None")
        if Y_ is None:
            K = np.diag(self.noise_level * X_[:, self.wdim])
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return K, K[:, :, None]
                return K, np.empty((K.shape[0], K.shape[1], 0))
            return K
        return np.zeros((X_.shape[0], np.atleast_2d(Y_).shape[0]))

    def diag(self, X_):
        return self.noise_level * np.atleast_2d(X_)[:, self.wdim]

    def is_stationary(self):
        return False


def law_rows(c, F, f, a0, a1, a2, f0, g):
    return (a0 + a1 * c + a2 * c**2) * F * A2.lorentz(f, f0, g)


KAPPAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]


def fit_hier(tr, y, kappa):
    """Partial pooling of f0 and gamma across compositions (MAP, Gaussian prior).

    f0_k = f0 (1 + u_k), gamma_k = gamma exp(v_k), u_k, v_k ~ N(0, kappa^2); the data
    term is scaled by the RMS residual of the completely pooled fit (kappa = 0 is
    exactly the LawGP law). Unseen compositions get u = v = 0.
    """
    tr = np.asarray(tr)
    _, pv, info = A2.fit_law5(tr, y)
    p = np.asarray(info["p_scaled"], float)
    s = info["scale"]
    if kappa == 0:
        return (lambda idx: A2.law5(A2._xt(idx), *p) * s), {
            "kappa": 0.0,
            "f0_k": [float(pv[3])] * 5,
            "gamma_k": [float(pv[4])] * 5,
        }
    kid = COMP_ID[tr]
    present = np.array(sorted(set(kid.tolist())))
    P = len(present)
    r0 = A2.law5(A2._xt(tr), *p) - y[tr] / s
    sig = max(float(np.sqrt(np.mean(r0**2))), 1e-6)
    pos = {k: j for j, k in enumerate(present)}
    rowpos = np.array([pos[k] for k in kid])

    def res(th):
        u = th[5 : 5 + P]
        v = th[5 + P :]
        m = law_rows(
            A2.CNT[tr],
            A2.FRC[tr],
            A2.FRQ[tr],
            th[0],
            th[1],
            th[2],
            th[3] * (1 + u[rowpos]),
            th[4] * np.exp(v[rowpos]),
        )
        return np.concatenate([(m - y[tr] / s) / sig, th[5:] / kappa])

    lb = np.array(A2.BOUNDS_5[0] + [-0.5] * P + [-1.5] * P, float)
    ub = np.array(A2.BOUNDS_5[1] + [0.5] * P + [1.5] * P, float)
    x0 = np.clip(np.concatenate([p, np.zeros(2 * P)]), lb + 1e-12, ub - 1e-12)
    sol = least_squares(
        res, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000
    )
    th = sol.x
    u = np.zeros(5)
    v = np.zeros(5)
    u[present] = th[5 : 5 + P]
    v[present] = th[5 + P :]

    def pred(idx):
        idx = np.asarray(idx)
        k = COMP_ID[idx]
        return (
            law_rows(
                A2.CNT[idx],
                A2.FRC[idx],
                A2.FRQ[idx],
                th[0],
                th[1],
                th[2],
                th[3] * (1 + u[k]),
                th[4] * np.exp(v[k]),
            )
            * s
        )

    return pred, {
        "kappa": float(kappa),
        "f0_k": (th[3] * (1 + u)).tolist(),
        "gamma_k": (th[4] * np.exp(v)).tolist(),
    }


def hier_fit_predict(tr, te, y, kappa):
    m, info = fit_hier(tr, y, kappa)
    xs = StandardScaler().fit(X[tr])
    g = A2.gp().fit(xs.transform(X[tr]), y[tr] - m(tr))
    return m(te) + g.predict(xs.transform(X[te])), info


def inner_folds_b(scheme, tr):
    tr = np.asarray(tr)
    if scheme == "loo":
        kf = KFold(5, shuffle=True, random_state=SEED)
        return [(tr[a], tr[b]) for a, b in kf.split(tr)]
    g = A2.AXES[scheme][tr]
    return [
        (tr[g != v], tr[g == v])
        for v in A2.axis_levels(scheme)
        if (g == v).any() and (g != v).any()
    ]


def predict_hier(tr, te, y, scheme):
    folds = inner_folds_b(scheme, tr)
    score = []
    for kap in KAPPAS:
        sse = 0.0
        for a, b in folds:
            mu, _ = hier_fit_predict(a, b, y, kap)
            sse += float(np.sum((y[b] - mu) ** 2))
        score.append(sse)
    score = np.array(score)
    k = int(np.argmin(score))
    mu, info = hier_fit_predict(tr, te, y, KAPPAS[k])
    info = dict(info)
    info["inner_sse"] = score.tolist()
    return mu, info


def predict_lorenv(tr, te, y):
    _, pv, _ = A2.fit_law5(tr, y)
    f0, gam = float(pv[3]), float(pv[4])
    xs = StandardScaler().fit(X[tr])

    def aug(idx):
        return np.column_stack([xs.transform(X[idx]), A2.FRC[idx], A2.FRQ[idx]])

    sub = A5.SubMatern(
        dims=(0, 1, 2, 3),
        length_scale=[1.0] * 4,
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5,
    )
    k = Env((4, 5), f0, gam) * (
        CK(1.0, (1e-3, 1e3)) * sub + CK(1.0, (1e-3, 1e3))
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    sc = float(np.std(y[tr])) or 1.0
    g = GaussianProcessRegressor(
        kernel=k,
        normalize_y=False,
        alpha=1e-10,
        n_restarts_optimizer=4,
        random_state=SEED,
    ).fit(aug(tr), y[tr] / sc)
    return g.predict(aug(te)) * sc, {"f0": f0, "gamma": gam, "kernel": str(g.kernel_)}


def predict_lorspec(tr, te, y):
    _, pv, _ = A2.fit_law5(tr, y)
    gam = float(pv[4])
    xs = StandardScaler().fit(X[tr])
    sdf = float(xs.scale_[3])
    l0 = float(np.clip(gam / sdf, 2e-2, 5e2))
    k = CK(1.0, (1e-3, 1e3)) * A5.SubMatern(
        dims=(0, 1), length_scale=[1.0, 1.0]
    ) * A5.SubMatern(dims=(2,), length_scale=1.0) * DampedCos(
        dim=3, length_scale=l0, period=2 * l0
    ) + WhiteKernel(
        1e-3, (1e-6, 1e1)
    )
    g = GaussianProcessRegressor(
        kernel=k,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=4,
        random_state=SEED,
    ).fit(xs.transform(X[tr]), y[tr])
    return g.predict(xs.transform(X[te])), {
        "gamma_law": gam,
        "init_length_scale_std": l0,
        "kernel": str(g.kernel_),
    }


def fit_loglaw(tr, y):
    _, _, info = A2.fit_law5(tr, y)
    s = info["scale"]
    p0 = np.clip(
        np.asarray(info["p_scaled"], float),
        np.array(A2.BOUNDS_5[0]) + 1e-9,
        np.array(A2.BOUNDS_5[1]) - 1e-9,
    )
    z = np.log(y[tr] / s)

    def res(th):
        return np.log(np.maximum(A2.law5(A2._xt(tr), *th), 1e-9)) - z

    sol = least_squares(
        res, p0, bounds=A2.BOUNDS_5, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000
    )
    p = sol.x
    return (lambda idx: np.maximum(A2.law5(A2._xt(idx), *p), 1e-9) * s), p


def predict_loghet(tr, te, y):
    m, p = fit_loglaw(tr, y)
    xs = StandardScaler().fit(X[tr])
    mbar = float(np.mean(m(tr)))

    def aug(idx):
        return np.column_stack([xs.transform(X[idx]), (mbar / m(idx)) ** 2])

    sub = A5.SubMatern(
        dims=(0, 1, 2, 3),
        length_scale=[1.0] * 4,
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5,
    )
    k = (
        CK(1.0, (1e-3, 1e3)) * sub
        + WhiteKernel(1e-3, (1e-6, 1e1))
        + HeteroWhite(4, 1e-3, (1e-8, 1e1))
    )
    r = np.log(y[tr]) - np.log(m(tr))
    g = GaussianProcessRegressor(
        kernel=k,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=4,
        random_state=SEED,
    ).fit(aug(tr), r)
    mu = np.exp(np.log(m(te)) + g.predict(aug(te)))
    return mu, {"kernel": str(g.kernel_)}


SQ5 = np.sqrt(5.0)
F_VIRT = np.linspace(1.0, 3.0, 5)


def _virtual_raw():
    comps = np.unique(X[:, :2], axis=0)
    fr = np.unique(A2.FRQ)
    V = [[c[0], c[1], F, f] for c in comps for f in fr for F in F_VIRT]
    return np.array(V, float)


V_RAW = _virtual_raw()


def trunc_above(a, rng):
    """One draw of Z ~ N(0,1) conditioned on Z >= a."""
    p = sps.ndtr(-a)
    if p < 1e-300:
        return a
    u = rng.random()
    return float(-sps.ndtri(max(u * p, 1e-300)))


def tmvn_mean(mu, S, rng, burn=200, keep=1000):
    d = len(mu)
    Q = np.linalg.inv(S)
    qd = np.diag(Q).copy()
    x = np.maximum(mu, 1e-9)
    acc = np.zeros(d)
    for it in range(burn + keep):
        for j in range(d):
            cm = mu[j] - (Q[j] @ (x - mu) - qd[j] * (x[j] - mu[j])) / qd[j]
            cs = 1.0 / np.sqrt(qd[j])
            x[j] = cm + cs * trunc_above((0.0 - cm) / cs, rng)
        if it >= burn:
            acc += x
    return acc / keep


def predict_mono(tr, te, y, rng_key):
    tr, te = np.asarray(tr), np.asarray(te)
    xs = StandardScaler().fit(X[tr])
    Xtr = xs.transform(X[tr])
    g = A2.gp().fit(Xtr, y[tr])
    kk = g.kernel_
    amp = float(kk.k1.k1.constant_value)
    ls = np.atleast_1d(kk.k1.k2.length_scale).astype(float)
    if ls.size == 1:
        ls = np.repeat(ls, 4)
    lF = ls[2]
    T = xs.transform(X[te])
    V = xs.transform(V_RAW)

    def rd(Aa, Bb):
        D = (Aa[:, None, :] - Bb[None, :, :]) / ls
        return np.sqrt(np.sum(D**2, -1)), Aa[:, None, 2] - Bb[None, :, 2]

    def kval(r):
        return amp * (1 + SQ5 * r + 5 * r**2 / 3) * np.exp(-SQ5 * r)

    def hfun(r):
        return (5.0 / 3.0) * (1 + SQ5 * r) * np.exp(-SQ5 * r)

    rTX, _ = rd(T, Xtr)
    rVX, uVX = rd(V, Xtr)
    K_TX = kval(rTX)
    dK_VX = -amp * hfun(rVX) * uVX / lF**2
    Ks = np.vstack([K_TX, dK_VX])
    mean = Ks @ g.alpha_.ravel()
    vv = cho_solve((g.L_, True), Ks.T)
    rTT, _ = rd(T, T)
    rTV, uTV = rd(T, V)
    rVV, uVV = rd(V, V)
    K_TT = kval(rTT)
    K_TV = amp * hfun(rTV) * uTV / lF**2
    K_VV = amp * (
        hfun(rVV) / lF**2 - (25.0 / 3.0) * np.exp(-SQ5 * rVV) * uVV**2 / lF**4
    )
    prior = np.block([[K_TT, K_TV], [K_TV.T, K_VV]])
    Sig = prior - Ks @ vv
    nT = len(te)
    muT, muV = mean[:nT], mean[nT:]
    S_VV = Sig[nT:, nT:]
    S_VV = 0.5 * (S_VV + S_VV.T) + 1e-8 * np.mean(np.diag(S_VV)) * np.eye(len(muV))
    S_TV = Sig[:nT, nT:]
    sdV = np.sqrt(np.clip(np.diag(S_VV), 1e-300, None))
    zmin = float(np.min(muV / sdV))
    active = zmin < 5.0
    if active:
        rng = np.random.default_rng(rng_key)
        EV = tmvn_mean(muV, S_VV, rng)
        gT = muT + S_TV @ np.linalg.solve(S_VV, EV - muV)
    else:
        gT = muT
    ym = float(np.asarray(g._y_train_mean).ravel()[0])
    ysd = float(np.asarray(g._y_train_std).ravel()[0])
    return ym + ysd * gT, {
        "constraint_active": bool(active),
        "min_z_derivative": zmin,
        "n_virtual_negative_mean": int(np.sum(muV < 0)),
        "n_virtual": int(len(muV)),
    }


REF_MODELS = ["gp", "lawgp", "prod"]
NEW_MODELS = ["hier", "lorenv", "lorspec", "loghet", "mono"]
B_MODELS = REF_MODELS + NEW_MODELS
B_LABEL = {
    "gp": r"\GP",
    "lawgp": r"\LawGP",
    "prod": r"\Prod",
    "hier": "Hier.\\ LawGP",
    "lorenv": "Law kernel",
    "lorspec": "Lor.\\ spectrum",
    "loghet": "Log het.\\ LawGP",
    "mono": "Monotone GP",
}
B_TEXT = {
    "gp": "plain GP",
    "lawgp": "LawGP",
    "prod": "ProductGP",
    "hier": "hierarchical LawGP (partial pooling of f0, gamma across compositions)",
    "lorenv": "law-as-kernel GP (force and Lorentzian factors as covariance envelope)",
    "lorspec": "ProductGP with a Lorentzian-spectrum (damped cosine) kernel in f",
    "loghet": "log-target LawGP with additive plus proportional noise",
    "mono": "monotone-in-force GP (dV/dF >= 0 at virtual points)",
}
B_MAC = {
    "gp": "Gp",
    "lawgp": "Law",
    "prod": "Prod",
    "hier": "Hier",
    "lorenv": "LawKern",
    "lorspec": "LorSpec",
    "loghet": "LogHet",
    "mono": "Mono",
}
SCHEMES = ["loo", "composition", "force", "frequency"]
SCHEME_MAC = {
    "loo": "Loo",
    "composition": "Comp",
    "force": "Force",
    "frequency": "Freq",
}


def folds_b(scheme, rows):
    if scheme == "loo":
        return [
            (np.delete(rows, i), rows[[i]], str(int(rows[i]))) for i in range(len(rows))
        ]
    g = A2.AXES[scheme][rows]
    return [
        (rows[g != v], rows[g == v], A2.level_key(scheme, v))
        for v in A2.axis_levels(scheme)
        if (g == v).any()
    ]


def b_job(model, rs, scheme, fi, target):
    warnings.filterwarnings("ignore")
    rows = ROWSETS[rs]
    tr, te, key = folds_b(scheme, rows)[fi]
    y = Y[target]
    info = {}
    if model == "gp":
        mu, _ = A5.fold_predict("gp", tr, te, y)
    elif model == "lawgp":
        mu, _ = A5.fold_predict("physgp", tr, te, y)
    elif model == "prod":
        mu, _ = A5.fold_predict("m1", tr, te, y, seed=0)
    elif model == "hier":
        mu, info = predict_hier(tr, te, y, scheme)
    elif model == "lorenv":
        mu, info = predict_lorenv(tr, te, y)
    elif model == "lorspec":
        mu, info = predict_lorspec(tr, te, y)
    elif model == "loghet":
        mu, info = predict_loghet(tr, te, y)
    elif model == "mono":
        mu, info = predict_mono(
            tr, te, y, [SEED, SCHEMES.index(scheme), fi, 0 if rs == "n75" else 1]
        )
    else:
        raise ValueError(model)
    return (model, rs, scheme, fi), np.asarray(mu, float).ravel(), info


def lvl_metrics(y, mu):
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, mu)) if len(y) > 1 else None,
        "MAE": float(mean_absolute_error(y, mu)),
        "bias": float(np.mean(mu - y)),
        "y_mean": float(np.mean(y)),
    }


def block_b(cache, n_jobs, quick=False):
    T0 = time.time()
    target = "rms_Voc"
    models = B_MODELS
    rowsets = ["n75"] if quick else ["n75", "n72"]
    schemes = ["frequency"] if quick else SCHEMES
    jobs, tags = [], set()
    for m in models:
        for rs in rowsets:
            tag = f"b_{m}_{target}_{rs}"
            if cache.has(tag):
                continue
            tags.add(tag)
            for sc in schemes:
                for fi in range(len(folds_b(sc, ROWSETS[rs]))):
                    jobs.append(delayed(b_job)(m, rs, sc, fi, target))
    print(f"[a10/b] {len(jobs)} fold jobs to run", flush=True)
    if jobs:
        out = Parallel(n_jobs=n_jobs)(jobs)
        bucket = {}
        for key, mu, info in out:
            m, rs = key[0], key[1]
            bucket.setdefault(f"b_{m}_{target}_{rs}", {})[(key[2], key[3])] = (mu, info)
        for tag, d in bucket.items():
            cache.put(tag, d)
    print(f"[a10/b] models done in {time.time() - T0:.0f} s", flush=True)

    y = Y[target]
    res = {
        "target": target,
        "models": {m: B_TEXT[m] for m in models},
        "reference_models_note": "gp, lawgp and prod call a5_common.fold_predict('gp'/'physgp'/'m1',"
        " seed 0), the exact A5 references; new models use seed 42 where an optimizer restarts",
        "hier_kappa_grid": KAPPAS,
        "hier_selection": "inner folds of the same scheme on the outer training rows"
        " (leave-one-level-out along the held-out axis; KFold(5, shuffle, 42) for LOO),"
        " minimum pooled inner squared error, ties to the smaller kappa",
        "mono_virtual_points": "5 compositions x 5 frequency levels x F in {1, 1.5, 2, 2.5, 3} N"
        " = 125 derivative constraints, Gibbs sampler 200 burn-in + 1000 draws, skipped when"
        " every virtual derivative has posterior z > 5",
        "multi_fidelity": "skipped: no twin generator (E1 a6_twin.py) is importable at run time",
        "rescue_criterion": "a priori: held-out-frequency pooled R2 > 0 (better than the global"
        " mean of the held-out predictions' reference) is called a rescue; a model 'softens'"
        " the collapse when its pooled R2 exceeds the best reference model",
        "pooled": {},
        "per_level": {},
        "fold_info": {},
    }
    PRED, FIG = [], []
    for rs in rowsets:
        rows = ROWSETS[rs]
        res["pooled"][rs], res["per_level"][rs], res["fold_info"][rs] = {}, {}, {}
        for sc in schemes:
            (
                res["pooled"][rs][sc],
                res["per_level"][rs][sc],
                res["fold_info"][rs][sc],
            ) = ({}, {}, {})
            F = folds_b(sc, rows)
            for m in models:
                d = cache.get(f"b_{m}_{target}_{rs}")
                mu = np.full(A2.N, np.nan)
                lvl = np.empty(A2.N, dtype=object)
                infos = []
                for fi, (tr, te, key) in enumerate(F):
                    a, inf = d[(sc, fi)]
                    mu[te] = a
                    lvl[te] = key
                    infos.append(jsonable(inf))
                p = {
                    "n": int(len(rows)),
                    "R2": float(r2_score(y[rows], mu[rows])),
                    "MAE": float(mean_absolute_error(y[rows], mu[rows])),
                }
                if sc != "loo":
                    lv = {key: lvl_metrics(y[te], mu[te]) for tr, te, key in F}
                    res["per_level"][rs][sc][m] = lv
                    wl = np.array([lv[k]["R2"] for k in lv])
                    p["mean_within_level_R2"] = float(np.mean(wl))
                    p["median_within_level_R2"] = float(np.median(wl))
                    p["n_levels_R2_positive"] = int(np.sum(wl > 0))
                    for k in lv:
                        FIG.append(
                            {
                                "rowset": rs,
                                "scheme": sc,
                                "level": k,
                                "model": m,
                                "R2_within": lv[k]["R2"],
                                "MAE": lv[k]["MAE"],
                                "n": lv[k]["n"],
                            }
                        )
                res["pooled"][rs][sc][m] = p
                if m in ("hier", "mono"):
                    res["fold_info"][rs][sc][m] = infos
                for i in rows:
                    PRED.append(
                        {
                            "block": "b_new_models",
                            "condition_id": int(i),
                            "target": target,
                            "model": m,
                            "split_scheme": ("loo" if sc == "loo" else f"held_out_{sc}")
                            + ("" if rs == "n75" else "_excl_dup"),
                            "held_out_level": lvl[i],
                            "y_true": y[i],
                            "y_pred": mu[i],
                        }
                    )
    # reference agreement with A5 (n75)
    if not quick:
        A5J = json.loads((RESULTS / "a5_structured.json").read_text())
        amap = {"gp": "gp", "lawgp": "physgp", "prod": "m1"}
        agree = {}
        for m, am in amap.items():
            for sc in SCHEMES:
                agree[f"{m}.{sc}"] = abs(
                    res["pooled"]["n75"][sc][m]["R2"]
                    - A5J["pooled"][target][sc][am]["R2"]
                )
        res["a5_reference_abs_diff_pooled_R2"] = agree
        res["a5_reference_max_abs_diff"] = max(agree.values())
        # hier kappa selections
        ks = {}
        for rs in rowsets:
            ks[rs] = {
                sc: [f["kappa"] for f in res["fold_info"][rs][sc]["hier"]]
                for sc in SCHEMES
            }
        res["hier_selected_kappa"] = ks
    # headline
    head = {}
    for rs in rowsets:
        P = res["pooled"][rs]
        hs = {}
        for sc in schemes:
            best_ref = max(REF_MODELS, key=lambda m: P[sc][m]["R2"])
            best_new = max(NEW_MODELS, key=lambda m: P[sc][m]["R2"])
            hs[sc] = {
                "best_reference": best_ref,
                "best_reference_R2": P[sc][best_ref]["R2"],
                "best_new": best_new,
                "best_new_R2": P[sc][best_new]["R2"],
                "new_beats_all_references": [
                    m for m in NEW_MODELS if P[sc][m]["R2"] > P[sc][best_ref]["R2"]
                ],
            }
        fr = P["frequency"]
        hs["frequency_rescued_by"] = [m for m in NEW_MODELS if fr[m]["R2"] > 0]
        hs["frequency_softened_by"] = hs["frequency"]["new_beats_all_references"]
        head[rs] = hs
    res["headline"] = head
    res["runtime_s"] = time.time() - T0
    return res, PRED, FIG


# ====================================================================== (c) active learning
XI = 0.01
UCB_KAPPA = 2.0
STOP_REL, PATIENCE = 0.01, 3
BATCH_Q = 3


def gp_generic(dim):
    k = CK(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * dim, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


EXT = ROOT / "data" / "external"


def load_public():
    """Loaders of code/baseline/protocol_replay_extended.py (X, y only)."""
    out = {}
    af = pd.read_csv(
        EXT / "airfoil_self_noise.dat",
        sep=r"\s+",
        names=["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m", "spl_dB"],
    )
    out["airfoil"] = (
        af[
            ["freq_Hz", "aoa_deg", "chord_m", "velocity_ms", "thickness_m"]
        ].values.astype(float),
        af["spl_dB"].values.astype(float),
    )
    enb = pd.read_excel(EXT / "ENB2012_data.xlsx")
    out["energy_efficiency"] = (
        enb[[f"X{i}" for i in range(1, 9)]].values.astype(float),
        enb["Y1"].values.astype(float),
    )
    con = pd.read_excel(EXT / "Concrete_Data.xls")
    out["concrete"] = (
        con.iloc[:, :8].values.astype(float),
        con.iloc[:, 8].values.astype(float),
    )
    ya = pd.read_csv(EXT / "yacht_hydrodynamics.data", sep=r"\s+", header=None)
    out["yacht"] = (
        ya.iloc[:, :6].values.astype(float),
        ya.iloc[:, 6].values.astype(float),
    )
    cc = pd.read_excel(EXT / "Folds5x2_pp.xlsx", sheet_name="Sheet1")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cc), 2000, replace=False)
    cc = cc.iloc[np.sort(idx)].reset_index(drop=True)
    out["ccpp_2000sub"] = (
        cc[["AT", "V", "AP", "RH"]].values.astype(float),
        cc["PE"].values.astype(float),
    )
    return out


PUB_ORDER = ["energy_efficiency", "airfoil", "concrete", "yacht", "ccpp_2000sub"]
PUB_LABEL = {
    "energy_efficiency": "Energy Eff.",
    "airfoil": "Airfoil",
    "concrete": "Concrete",
    "yacht": "Yacht",
    "ccpp_2000sub": "Power Plant",
}
PUB_MAC = {
    "energy_efficiency": "EnergyEfficiency",
    "airfoil": "Airfoil",
    "concrete": "Concrete",
    "yacht": "Yacht",
    "ccpp_2000sub": "PowerPlant",
}


class Prob:
    """One pool-based search problem."""

    def __init__(self, name, Xp, yp, pool, law, thr, stop_mode, n0, max_q, grid):
        self.name = name
        self.X = Xp
        self.y = yp
        self.pool = np.asarray(pool)
        self.law = law
        self.thr = thr
        self.stop_mode = stop_mode
        self.n0 = n0
        self.max_q = max_q
        self.grid = grid
        self.ymax = float(np.max(yp[self.pool]))
        self.ysd = float(np.std(yp[self.pool]))
        if not grid:
            Z = (Xp - Xp.mean(0)) / np.where(Xp.std(0) > 0, Xp.std(0), 1.0)
            self.Z = Z
            self.bins = []
            for j in range(Xp.shape[1]):
                u = np.unique(Xp[:, j])
                if len(u) <= 20:
                    self.bins.append(np.searchsorted(u, Xp[:, j]))
                else:
                    edges = np.quantile(Xp[:, j], np.linspace(0, 1, 11)[1:-1])
                    self.bins.append(np.digitize(Xp[:, j], edges))

    def hit(self, i):
        return self.y[i] >= self.thr

    def stop_thr(self, lab):
        yl = self.y[lab]
        if self.stop_mode == "best":
            return STOP_REL * float(np.max(yl))
        return STOP_REL * float(np.max(yl) - np.min(yl))


def fit_sur(P, lab):
    lab = np.asarray(lab)
    xs = StandardScaler().fit(P.X[lab])
    base = None
    t = P.y[lab].copy()
    if P.law:
        m, _, _ = A2.fit_law5(lab, P.y)
        base = m
        t = t - m(lab)
    g = gp_generic(P.X.shape[1]).fit(xs.transform(P.X[lab]), t)
    return {"xs": xs, "g": g, "base": base, "lab": list(lab), "t": list(t)}


def sur_pred(P, M, idx, cov=False, g=None):
    g = M["g"] if g is None else g
    Z = M["xs"].transform(P.X[idx])
    if cov:
        mu, S = g.predict(Z, return_cov=True)
    else:
        mu, S = g.predict(Z, return_std=True)
    if M["base"] is not None:
        mu = mu + M["base"](np.asarray(idx))
    return mu, S


def ei_fn(mu, sd, best):
    z = (mu - best - XI) / np.maximum(sd, 1e-12)
    return (mu - best - XI) * norm.cdf(z) + sd * norm.pdf(z)


def ts_draw(mu, S, rng):
    n = len(mu)
    S = 0.5 * (S + S.T)
    jit = 1e-10 * max(float(np.mean(np.diag(S))), 1e-30)
    for _ in range(8):
        try:
            L = np.linalg.cholesky(S + jit * np.eye(n))
            return mu + L @ rng.standard_normal(n)
        except np.linalg.LinAlgError:
            jit *= 100
    w, V = np.linalg.eigh(S)
    return mu + V @ (np.sqrt(np.clip(w, 0, None)) * rng.standard_normal(n))


def believer(P, M, pool, best, q):
    g = M["g"]
    lab, t = list(M["lab"]), list(M["t"])
    pool = list(pool)
    picks = []
    cur = g
    for _ in range(q):
        if not pool:
            break
        Z = M["xs"].transform(P.X[pool])
        mu_r, sd = cur.predict(Z, return_std=True)
        mu = mu_r + (M["base"](np.asarray(pool)) if M["base"] is not None else 0.0)
        e = ei_fn(mu, sd, best)
        j = int(np.argmax(e))
        picks.append(pool[j])
        lab.append(pool[j])
        t.append(float(mu_r[j]))
        best = max(best, float(mu[j]))
        pool.pop(j)
        cur = GaussianProcessRegressor(
            kernel=g.kernel_, optimizer=None, normalize_y=True, alpha=1e-10
        ).fit(M["xs"].transform(P.X[lab]), np.array(t))
    return picks


# ---- OFAT on the real grid (cell based, legacy revision_experiments.py order)
DF = A2.df
COMPS = DF["composition"].unique().tolist()
FORCES = sorted(DF["force_N"].unique().tolist())
FREQS = sorted(DF["freq_Hz"].unique().tolist())


def cell(comp, F, f, avail):
    m = (
        (DF["composition"].values == comp)
        & (DF["force_N"].values == F)
        & (DF["freq_Hz"].values == f)
    )
    ids = [int(i) for i in np.where(m)[0] if i in avail]
    return ids[0] if ids else None


def ofat_grid(P, comp0, F0, known, stages=("freq", "force", "comp")):
    """Legacy OFAT: frequency sweep at (comp0, F0), force sweep at the best frequency,
    composition sweep at the best (force, frequency). Known cells are not re-run."""
    avail = set(P.pool.tolist())
    y = P.y
    seq = list(known)
    seen = set(seq)

    def add(i):
        if i is not None and i not in seen:
            seq.append(i)
            seen.add(i)

    sweep = [cell(comp0, F0, f, avail) for f in FREQS]
    for i in sweep:
        add(i)
    sw = [i for i in sweep if i is not None]
    f_best = A2.FRQ[sw[int(np.argmax(y[sw]))]]
    if "force" in stages:
        for F in FORCES:
            add(cell(comp0, F, f_best, avail))
        sub = [i for i in seen if A2.FRQ[i] == f_best and COMP[i] == comp0]
        F_best = A2.FRC[sub[int(np.argmax(y[sub]))]]
    else:
        F_best = F0
    if "comp" in stages:
        for c in COMPS:
            add(cell(c, F_best, f_best, avail))
    return seq


def ofat_generic(P, start, known, factors, cyclic, budget):
    """Pool OFAT: for each factor, one record per level (or decile bin) nearest to the
    incumbent in the other standardized inputs; the incumbent moves to the best seen."""
    y = P.y
    seq = list(known)
    seen = set(seq)
    inc = int(start)
    pool_mask = np.zeros(len(y), bool)
    pool_mask[P.pool] = True
    while True:
        before = y[inc]
        for j in factors:
            other = [k for k in range(P.X.shape[1]) if k != j]
            dist = np.sqrt(np.sum((P.Z[:, other] - P.Z[inc, other]) ** 2, axis=1))
            cand = []
            for b in np.unique(P.bins[j][pool_mask]):
                mem = np.where((P.bins[j] == b) & pool_mask)[0]
                cand.append(int(mem[np.argmin(dist[mem])]))
            for c in cand:
                if c not in seen:
                    seq.append(c)
                    seen.add(c)
                    if len(seq) >= budget:
                        return seq
            inc = max(cand + [inc], key=lambda i: y[i])
        if not cyclic or y[inc] <= before:
            return seq


def first_hit(P, seq, n_known):
    """Experiments until the optimum is first run; a hit inside the known (parallel)
    initial design counts the whole design, as in activeL_analysis.py."""
    seq = list(seq)
    if any(P.hit(i) for i in seq[:n_known]):
        return n_known
    for k in range(n_known, len(seq)):
        if P.hit(seq[k]):
            return k + 1
    return None


def summarize_seq(P, seq, n_known):
    return first_hit(P, seq, n_known), float(np.max(P.y[list(seq)]))


def run_al(P, strat, sur_law, seed, start=None):
    """One replay. Returns found_at (oracle), stopped_at and regret at the realizable stop."""
    warnings.filterwarnings("ignore")
    P.law = sur_law
    rng = np.random.default_rng(seed)
    rng_ts = np.random.default_rng([seed, 7])
    y = P.y
    budget = P.n0 + P.max_q
    if strat == "random":
        order = rng.permutation(P.pool)
        pos = int(np.argmax(y[order] >= P.thr)) + 1
        return {
            "found_at": max(pos, P.n0),
            "stopped_at": None,
            "regret_stop": None,
            "n_used": max(pos, P.n0),
        }
    if strat == "ofat_paper":
        comp0, F0 = start
        seq = ofat_grid(P, comp0, F0, [])
        fa, best = summarize_seq(P, seq, 0)
        return {
            "found_at": fa,
            "stopped_at": len(seq),
            "regret_stop": P.ymax - best,
            "n_used": len(seq),
            "start": list(start),
        }
    if strat == "hyb_ofatstart":
        lab = []
    else:
        lab = [int(i) for i in rng.choice(P.pool, P.n0, replace=False)]
    if strat in (
        "ofat_init",
        "ofat_cold_single",
        "ofat_cold_cyclic",
        "ofat_init_single",
        "ofat_init_cyclic",
    ):
        if P.grid:
            b = lab[int(np.argmax(y[lab]))]
            seq = ofat_grid(P, COMP[b], A2.FRC[b], lab)
        else:
            if strat.startswith("ofat_cold"):
                lab = [int(rng.choice(P.pool))]
            b = lab[int(np.argmax(y[lab]))]
            seq = ofat_generic(
                P, b, lab, list(range(P.X.shape[1])), strat.endswith("cyclic"), budget
            )
        fa, best = summarize_seq(P, seq, len(lab))
        return {
            "found_at": fa,
            "stopped_at": len(seq),
            "regret_stop": P.ymax - best,
            "n_used": len(seq),
        }
    # prefix of the hybrid
    if strat == "hyb_ofatstart":
        comp0, F0 = start
        lab = ofat_grid(P, comp0, F0, [], stages=("freq", "force"))
    elif strat == "hyb":
        b = lab[int(np.argmax(y[lab]))]
        if P.grid:
            lab = ofat_grid(P, COMP[b], A2.FRC[b], lab, stages=("freq", "force"))
        else:
            lab = ofat_generic(P, b, lab, [0, 1], False, budget)
    found_at = first_hit(P, lab, 0 if strat == "hyb_ofatstart" else P.n0)
    n_pre = len(lab)
    pool = [int(i) for i in P.pool if i not in set(lab)]
    quiet, stopped_at, regret_stop, rounds = 0, None, None, 0
    acq = {
        "ei": "ei",
        "ucb": "ucb",
        "ts": "ts",
        "bei": "bei",
        "hyb": "ei",
        "hyb_ofatstart": "ei",
    }[strat]
    while pool and len(lab) < budget:
        if found_at is not None and stopped_at is not None:
            break
        M = fit_sur(P, lab)
        best = float(np.max(y[lab]))
        need_cov = acq == "ts"
        mu, S = sur_pred(P, M, pool, cov=need_cov)
        sd = np.sqrt(np.clip(np.diag(S), 0, None)) if need_cov else S
        e = ei_fn(mu, sd, best)
        if stopped_at is None:
            if np.max(e) < P.stop_thr(lab):
                quiet += 1
                if quiet >= PATIENCE:
                    stopped_at = len(lab)
                    regret_stop = P.ymax - best
            else:
                quiet = 0
        if acq in ("ei",):
            picks = [pool[int(np.argmax(e))]]
        elif acq == "ucb":
            picks = [pool[int(np.argmax(mu + UCB_KAPPA * sd))]]
        elif acq == "ts":
            picks = [pool[int(np.argmax(ts_draw(mu, S, rng_ts)))]]
        else:
            picks = believer(P, M, pool, best, BATCH_Q)
        rounds += 1
        for pk in picks:
            lab.append(pk)
            pool.remove(pk)
        if found_at is None and any(P.hit(pk) for pk in picks):
            found_at = len(lab)
    if stopped_at is None:
        stopped_at = len(lab)
        regret_stop = P.ymax - float(np.max(y[lab]))
    return {
        "found_at": found_at,
        "stopped_at": stopped_at,
        "regret_stop": regret_stop,
        "n_used": len(lab),
        "n_prefix": n_pre,
        "rounds": rounds,
    }


def al_summary(runs, budget, norm_sd=None):
    fa = [r["found_at"] for r in runs]
    ok = [f is not None and f <= budget for f in fa]
    cens = np.array(
        [f if (f is not None and f <= budget) else budget for f in fa], float
    )
    out = {
        "runs": len(runs),
        "oracle_median": float(np.median(cens)),
        "oracle_q25": float(np.percentile(cens, 25)),
        "oracle_q75": float(np.percentile(cens, 75)),
        "oracle_success_hits": int(sum(ok)),
        "oracle_success_rate": float(np.mean(ok)),
        "oracle_success_ci95": C3.clopper(int(sum(ok)), len(runs)),
        "oracle_counts": [None if f is None else int(f) for f in fa],
        "censored_at": budget,
    }
    if runs[0].get("stopped_at") is not None:
        st = np.array([r["stopped_at"] for r in runs], float)
        succ = [
            r["found_at"] is not None and r["found_at"] <= r["stopped_at"] for r in runs
        ]
        reg = np.array([r["regret_stop"] for r in runs], float)
        out.update(
            {
                "stop_median": float(np.median(st)),
                "stop_q25": float(np.percentile(st, 25)),
                "stop_q75": float(np.percentile(st, 75)),
                "stop_success_hits": int(sum(succ)),
                "stop_success_rate": float(np.mean(succ)),
                "stop_success_ci95": C3.clopper(int(sum(succ)), len(runs)),
                "stop_regret_median": float(np.median(reg)),
                "stop_regret_max": float(np.max(reg)),
                "stop_counts": st.astype(int).tolist(),
            }
        )
        if norm_sd:
            out["stop_regret_median_over_sd"] = float(np.median(reg) / norm_sd)
    return out


GRID_STRATS = [
    ("ofat_paper", False),
    ("ofat_init", False),
    ("random", False),
    ("ei", False),
    ("ei", True),
    ("ucb", False),
    ("ucb", True),
    ("ts", False),
    ("ts", True),
    ("bei", False),
    ("bei", True),
    ("hyb", False),
    ("hyb", True),
    ("hyb_ofatstart", False),
    ("hyb_ofatstart", True),
]
PUB_STRATS = [
    ("ofat_cold_single", False),
    ("ofat_cold_cyclic", False),
    ("ofat_init_single", False),
    ("ofat_init_cyclic", False),
    ("random", False),
    ("ei", False),
    ("ucb", False),
    ("ts", False),
    ("bei", False),
    ("hyb", False),
]
STRAT_LABEL = {
    "ofat_paper": "OFAT sweep (15 cell starts)",
    "ofat_init": "OFAT after the initial design",
    "ofat_cold_single": "OFAT, one pass, single-record start",
    "ofat_cold_cyclic": "OFAT, repeated passes, single-record start",
    "ofat_init_single": "OFAT, one pass, after the initial design",
    "ofat_init_cyclic": "OFAT, repeated passes, after the initial design",
    "random": "Random order",
    "ei": "EI",
    "ucb": "UCB",
    "ts": "Thompson sampling",
    "bei": "Batch EI ($q=3$)",
    "hyb": "OFAT-then-EI hybrid",
    "hyb_ofatstart": "OFAT-then-EI hybrid (15 cell starts)",
}
STRAT_MAC = {
    "ofat_paper": "OfatPaper",
    "ofat_init": "OfatInit",
    "ofat_cold_single": "OfatColdOne",
    "ofat_cold_cyclic": "OfatColdCyc",
    "ofat_init_single": "OfatInitOne",
    "ofat_init_cyclic": "OfatInitCyc",
    "random": "Random",
    "ei": "Ei",
    "ucb": "Ucb",
    "ts": "Ts",
    "bei": "BatchEi",
    "hyb": "Hybrid",
    "hyb_ofatstart": "HybridOfatStart",
}


def skey(s, law):
    return (
        f"{s}_{'law' if law else 'gp'}"
        if s not in ("ofat_paper", "ofat_init", "random") and not s.startswith("ofat")
        else s
    )


def grid_prob(rs):
    rows = ROWSETS[rs]
    y = Y["rms_Voc"]
    return Prob(
        "grid_" + rs, X, y, rows, False, float(np.max(y[rows])), "best", 10, 65, True
    )


def block_c(cache, n_jobs, quick=False):
    T0 = time.time()
    seeds_grid = list(range(4 if quick else 30))
    seeds_pub = list(range(3 if quick else 20))
    starts = [(c, F) for c in COMPS for F in FORCES]
    jobs, jmap = [], []
    probs = {}
    for rs in ("n75", "n72"):
        P = grid_prob(rs)
        probs[f"grid_{rs}"] = P
        for s, law in GRID_STRATS:
            tag = f"c_grid_{rs}_{skey(s, law)}"
            if cache.has(tag):
                continue
            if s in ("ofat_paper", "hyb_ofatstart"):
                for st in starts:
                    jobs.append(delayed(run_al)(P, s, law, 0, st))
                    jmap.append(tag)
            else:
                for sd in seeds_grid:
                    jobs.append(delayed(run_al)(P, s, law, sd))
                    jmap.append(tag)
    pub = load_public()
    names = PUB_ORDER[:2] if quick else PUB_ORDER
    for nm in names:
        Xp, yp = pub[nm]
        P = Prob(
            nm,
            Xp,
            yp,
            np.arange(len(yp)),
            False,
            float(np.quantile(yp, 0.99)),
            "range",
            10,
            150,
            False,
        )
        probs[nm] = P
        for s, law in PUB_STRATS:
            tag = f"c_pub_{nm}_{skey(s, law)}"
            if cache.has(tag):
                continue
            for sd in seeds_pub:
                jobs.append(delayed(run_al)(P, s, law, sd))
                jmap.append(tag)
    print(f"[a10/c] {len(jobs)} replay runs to execute", flush=True)
    if jobs:
        out = Parallel(n_jobs=n_jobs, batch_size=1)(jobs)
        bucket = {}
        for tag, r in zip(jmap, out):
            bucket.setdefault(tag, []).append(r)
        for tag, lst in bucket.items():
            cache.put(tag, lst)
    print(f"[a10/c] replays done in {time.time() - T0:.0f} s", flush=True)

    res = {
        "protocol": {
            "grid": "V_rms, pool = the 75 (or 72) conditions, optimum = best measured condition"
            " (row 58), initial design of 10 random conditions from np.random.default_rng(seed),"
            " seeds 0..29, oracle stop when the optimum is queried, realizable stop after 3"
            " consecutive rounds with max EI < 1% of the current best (revision_experiments.py),"
            " at most 65 queries after the initial design (the whole pool)",
            "public": "plain GP (legacy gp(dim)), pool = full dataset (CCPP 2000-record"
            " subsample), target = top 1% of y, n0 = 10, seeds 0..19, at most 150 queries"
            " (protocol_replay_extended.py); realizable stop after 3 consecutive rounds with max"
            " EI < 1% of the observed range of the labeled targets (the grid rule is relative to"
            " the best value, which is not scale free for dB or MW targets)",
            "acquisitions": {
                "ei": "expected improvement with xi = 0.01 (legacy)",
                "ucb": f"mu + {UCB_KAPPA} sd",
                "ts": "one joint posterior draw over the pool per round, rng default_rng([seed, 7])",
                "bei": "batch of 3 per round by the kriging believer (fantasy = posterior mean,"
                " kernel hyperparameters frozen within the batch); a batch counts 3 experiments",
                "hyb": "grid: the initial design, then the OFAT frequency sweep at the best initial"
                " cell's (composition, force) and the force sweep at the best frequency, then EI;"
                " public: the initial design, one OFAT pass over the first two inputs, then EI",
                "hyb_ofatstart": "grid only: the paper's 15 OFAT start cells, frequency and force"
                " sweeps (7 conditions, no random design), then EI",
                "ofat_paper": "revision_experiments.py OFAT: frequency sweep at the start"
                " (composition, force), force sweep at the best frequency, composition sweep",
                "ofat_init": "the same sweep started from the best cell of the seed's 10-condition"
                " initial design; the initial design is counted",
                "ofat_generic": "public sets: per input in column order, one record per level"
                " (unique values if at most 20, else deciles) nearest to the incumbent in the other"
                " standardized inputs; incumbent moves to the best record seen; one pass, or"
                " repeated passes until a pass brings no improvement",
                "random": "random permutation of the pool (legacy)",
            },
            "models": "grid: plain GP (legacy AL GP) and LawGP (five-parameter law refitted on"
            " the labeled conditions, GP on the residual); public: plain GP only (the voltage law"
            " has no meaning there)",
            "stopping_rule_for_ucb_ts_batch": "the EI-based realizable rule evaluated on the same"
            " model, so all model-based strategies share one stopping rule",
        },
        "grid": {},
        "public": {},
    }
    for rs in ("n75", "n72"):
        P = probs[f"grid_{rs}"]
        d = {}
        for s, law in GRID_STRATS:
            k = skey(s, law)
            runs = cache.get(f"c_grid_{rs}_{k}")
            d[k] = al_summary(runs, len(P.pool))
            d[k]["label"] = STRAT_LABEL[s] + (
                ""
                if s.startswith("ofat") or s == "random"
                else (", LawGP" if law else ", plain GP")
            )
        res["grid"][rs] = d
    for nm in names:
        P = probs[nm]
        d = {"n": int(len(P.y)), "threshold_top1pct": P.thr, "y_sd": P.ysd}
        for s, law in PUB_STRATS:
            k = skey(s, law)
            runs = cache.get(f"c_pub_{nm}_{k}")
            d[k] = al_summary(runs, P.n0 + P.max_q, norm_sd=P.ysd)
            d[k]["label"] = STRAT_LABEL[s]
        res["public"][nm] = d
    # legacy agreement
    if not quick:
        LA = json.loads((C3.LEG / "activeL_results.json").read_text())
        g = res["grid"]["n75"]["ei_gp"]
        LE = json.loads((C3.LEG / "revision_experiments.json").read_text())[
            "stopping_and_ofat"
        ]
        EXTJ = json.loads((C3.LEG / "protocol_replay_extended.json").read_text())
        res["legacy_agreement"] = {
            "grid_ei_oracle_counts_equal": g["oracle_counts"] == LA["EI"]["counts"],
            "grid_ei_oracle_median": [g["oracle_median"], LA["EI"]["median"]],
            "grid_ei_stop_median": [
                g["stop_median"],
                LE["ei_stopping"]["budget_median"],
            ],
            "grid_ei_stop_success": [
                g["stop_success_rate"],
                LE["ei_stopping"]["success_rate"],
            ],
            "grid_ofat_median": [
                res["grid"]["n75"]["ofat_paper"]["stop_median"],
                LE["ofat"]["n_experiments_median"],
            ],
            "public_ei_median": {
                nm: [
                    res["public"][nm]["ei_gp"]["oracle_median"],
                    EXTJ[nm]["active_learning_top1pct"]["EI_median"],
                ]
                for nm in names
            },
            "public_random_median": {
                nm: [
                    res["public"][nm]["random"]["oracle_median"],
                    EXTJ[nm]["active_learning_top1pct"]["random_median"],
                ]
                for nm in names
            },
        }
    # headline: does any model-based strategy beat OFAT?
    head = {}
    for rs in ("n75", "n72"):
        d = res["grid"][rs]
        of = d["ofat_paper"]
        model_keys = [
            skey(s, l)
            for s, l in GRID_STRATS
            if not (s.startswith("ofat") or s == "random")
        ]
        beat_oracle = [
            k
            for k in model_keys
            if d[k]["oracle_median"] < of["oracle_median"]
            and d[k]["oracle_success_rate"] >= of["oracle_success_rate"]
        ]
        beat_real = [
            k
            for k in model_keys
            if d[k]["stop_median"] <= of["stop_median"]
            and d[k]["stop_success_rate"] >= of["stop_success_rate"]
            and not (
                d[k]["stop_median"] == of["stop_median"]
                and d[k]["stop_success_rate"] == of["stop_success_rate"]
            )
        ]
        best_or = min(
            model_keys,
            key=lambda k: (d[k]["oracle_median"], -d[k]["oracle_success_rate"]),
        )
        best_re = max(
            model_keys, key=lambda k: (d[k]["stop_success_rate"], -d[k]["stop_median"])
        )
        head[rs] = {
            "ofat_median": of["stop_median"],
            "ofat_success_rate": of["oracle_success_rate"],
            "beats_ofat_oracle_stop": beat_oracle,
            "beats_ofat_realizable_stop": beat_real,
            "best_oracle": best_or,
            "best_oracle_median": d[best_or]["oracle_median"],
            "best_realizable": best_re,
            "best_realizable_success": d[best_re]["stop_success_rate"],
            "best_realizable_median": d[best_re]["stop_median"],
        }
    pubhead = {}
    for nm in names:
        d = res["public"][nm]
        mk = [
            skey(s, l)
            for s, l in PUB_STRATS
            if not (s.startswith("ofat") or s == "random")
        ]
        ok = [
            "ofat_cold_single",
            "ofat_cold_cyclic",
            "ofat_init_single",
            "ofat_init_cyclic",
        ]
        best_m = min(
            mk, key=lambda k: (-d[k]["oracle_success_rate"], d[k]["oracle_median"])
        )
        best_o = max(
            ok, key=lambda k: (d[k]["oracle_success_rate"], -d[k]["oracle_median"])
        )

        def beats(k, o=best_o):
            a, b = d[k], d[o]
            return bool(
                a["oracle_success_rate"] > b["oracle_success_rate"]
                or (
                    a["oracle_success_rate"] == b["oracle_success_rate"]
                    and a["oracle_median"] < b["oracle_median"]
                )
            )

        def beats_stop(k, o=best_o):
            a, b = d[k], d[o]
            return bool(
                a["stop_success_rate"] > b["stop_success_rate"]
                or (
                    a["stop_success_rate"] == b["stop_success_rate"]
                    and a["stop_median"] < b["stop_median"]
                )
            )

        pubhead[nm] = {
            "best_model_strategy": best_m,
            "best_model_median": d[best_m]["oracle_median"],
            "best_model_success": d[best_m]["oracle_success_rate"],
            "best_ofat": best_o,
            "best_ofat_success": d[best_o]["oracle_success_rate"],
            "best_ofat_oracle_median": d[best_o]["oracle_median"],
            "best_ofat_median_used": d[best_o]["stop_median"],
            "best_ofat_stop_success": d[best_o]["stop_success_rate"],
            "ei_beats_best_ofat": beats("ei_gp"),
            "strategies_beating_best_ofat_oracle": [k for k in mk if beats(k)],
            "strategies_beating_best_ofat_realizable": [k for k in mk if beats_stop(k)],
            "comparison_rule": "like for like: oracle count and success within budget against"
            " the OFAT variant with the highest success (ties: lower median); realizable stop"
            " against the same OFAT variant's own stopping point",
        }
    head["public"] = pubhead
    res["headline"] = head
    res["runtime_s"] = time.time() - T0
    AL_CSV = []
    for tag_prefix, group in (("grid", ("n75", "n72")), ("pub", names)):
        for g_ in group:
            strats = GRID_STRATS if tag_prefix == "grid" else PUB_STRATS
            for s, law in strats:
                k = skey(s, law)
                runs = cache.get(f"c_{tag_prefix}_{g_}_{k}")
                for j, r in enumerate(runs):
                    AL_CSV.append(
                        {
                            "problem": ("grid_" + g_) if tag_prefix == "grid" else g_,
                            "strategy": k,
                            "run": j,
                            "found_at": r["found_at"],
                            "stopped_at": r["stopped_at"],
                            "regret_stop": r["regret_stop"],
                            "n_used": r["n_used"],
                        }
                    )
    return res, AL_CSV


# ====================================================================== outputs
MACROS = []


def mac(name, val, key):
    assert name.isalpha(), name
    MACROS.append((name, str(val), key))


def tex_model(m):
    return a3_bench.NAME[m]


def write_tables(out, A, B, Cc, quick):
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    T = {}
    if A is not None and not quick:
        L = [
            r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            r" & \multicolumn{2}{c}{\RMS} & \multicolumn{2}{c}{\Vpp} & \multicolumn{2}{c}{\Vmax} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Model & Fixed & Tuned & Fixed & Tuned & Fixed & Tuned \\",
            r"\midrule",
        ]
        for m in BENCH:
            cells = []
            for t in A2.TARGETS:
                r = A["n75"][t]["models"][m]
                bf = r["rank_fixed"] == 1
                bt = r["rank_tuned"] == 1
                f_ = fnum(r["fixed_loo_r2"], 3) + f" ({r['rank_fixed']})"
                u_ = fnum(r["tuned_loo_r2"], 3) + f" ({r['rank_tuned']})"
                cells += [
                    rf"\textbf{{{f_}}}" if bf else f_,
                    rf"\textbf{{{u_}}}" if bt else u_,
                ]
            L.append(f"{tex_model(m)} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
        cells = []
        for t in A2.TARGETS:
            cells.append(
                r"\multicolumn{2}{c}{" + fnum(A["n75"][t]["kendall_tau"], 2) + "}"
            )
        L.append(r"Kendall $\tau$, fixed vs tuned & " + " & ".join(cells) + r" \\")
        cells = []
        for t in A2.TARGETS:
            r = A["n72"][t]
            cells.append(
                r"\multicolumn{2}{c}{"
                + f"{a3_bench.SHORTNAME[r['best_fixed']]} / {a3_bench.SHORTNAME[r['best_tuned']]}"
                + "}"
            )
        L.append(r"Best at $n=72$, fixed / tuned & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
        (tab / "tab_a10_fair.tex").write_text("\n".join(L))
        T[
            "tab_a10_fair.tex"
        ] = "fair_tuning.n75.<target>.models.<model>.{fixed_loo_r2,tuned_loo_r2,rank_fixed,rank_tuned}; n75.<t>.kendall_tau; n72.<t>.best_fixed/best_tuned"
        L = [
            r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            r" & \multicolumn{2}{c}{\RMS} & \multicolumn{2}{c}{\Vpp} & \multicolumn{2}{c}{\Vmax} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Model & Fixed & Tuned & Fixed & Tuned & Fixed & Tuned \\",
            r"\midrule",
        ]
        for m in BENCH:
            cells = []
            for t in A2.TARGETS:
                r = A["n72"][t]["models"][m]
                cells += [
                    fnum(r["fixed_loo_r2"], 3) + f" ({r['rank_fixed']})",
                    fnum(r["tuned_loo_r2"], 3) + f" ({r['rank_tuned']})",
                ]
            L.append(f"{tex_model(m)} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
        (tab / "tab_a10_fair_n72.tex").write_text("\n".join(L))
        T[
            "tab_a10_fair_n72.tex"
        ] = "fair_tuning.n72.<target>.models.<model>.{fixed_loo_r2 (recomputed), tuned_loo_r2, ranks}"
    if B is not None:
        schemes = [s for s in SCHEMES if s in B["pooled"]["n75"]]
        cols = "l" + "c" * len(B_MODELS)
        head = r"Level & " + " & ".join(B_LABEL[m] for m in B_MODELS) + r" \\"
        for metric, fname, nd in (
            ("R2", "tab_a10_levels.tex", 2),
            ("MAE", "tab_a10_levels_mae.tex", 3),
        ):
            L = [
                r"\footnotesize\setlength{\tabcolsep}{2pt}%",
                rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
                r"\toprule",
                head,
                r"\midrule",
            ]
            for sc in schemes:
                if sc == "loo":
                    continue
                L.append(
                    rf"\multicolumn{{{len(B_MODELS) + 1}}}{{@{{}}l}}{{Held-out {sc}}} \\"
                )
                for lv in A2.axis_levels(sc):
                    k = A2.level_key(sc, lv)
                    vals = [B["per_level"]["n75"][sc][m][k][metric] for m in B_MODELS]
                    bi = (
                        int(np.argmax(vals)) if metric == "R2" else int(np.argmin(vals))
                    )
                    cells = [
                        (rf"\textbf{{{fnum(v, nd)}}}" if j == bi else fnum(v, nd))
                        for j, v in enumerate(vals)
                    ]
                    L.append(f"{A2.level_tex(sc, lv)} & " + " & ".join(cells) + r" \\")
                if sc != schemes[-1]:
                    L.append(r"\midrule")
            L += [r"\bottomrule", r"\end{tabular}", ""]
            (tab / fname).write_text("\n".join(L))
            T[fname] = f"new_models.per_level.n75.<axis>.<model>.<level>.{metric}"
        L = [
            r"\footnotesize\setlength{\tabcolsep}{2pt}%",
            rf"\begin{{tabular}}{{@{{}}ll{'c' * len(B_MODELS)}@{{}}}}",
            r"\toprule",
            r"Scheme & Metric & " + " & ".join(B_LABEL[m] for m in B_MODELS) + r" \\",
            r"\midrule",
        ]
        for sc in schemes:
            nm = {
                "loo": "LOO",
                "composition": "Composition",
                "force": "Force",
                "frequency": "Frequency",
            }[sc]
            rows = [("R2", r"$R^2$", 3 if sc == "loo" else 2, "max")]
            if sc != "loo":
                rows.append(
                    ("mean_within_level_R2", r"mean within-level $R^2$", 2, "max")
                )
            rows.append(("MAE", "MAE (V)", 3, "min"))
            if "n72" in B["pooled"]:
                rows.append(("R2_72", r"$R^2$, $n=72$", 3 if sc == "loo" else 2, "max"))
            for j, (key, lab, nd, how) in enumerate(rows):
                if key == "R2_72":
                    vals = [B["pooled"]["n72"][sc][m]["R2"] for m in B_MODELS]
                else:
                    vals = [B["pooled"]["n75"][sc][m][key] for m in B_MODELS]
                bi = int(np.argmax(vals)) if how == "max" else int(np.argmin(vals))
                cells = [
                    (rf"\textbf{{{fnum(v, nd)}}}" if i == bi else fnum(v, nd))
                    for i, v in enumerate(vals)
                ]
                L.append(
                    f"{nm if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\"
                )
            if sc != schemes[-1]:
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}", ""]
        (tab / "tab_a10_pooled.tex").write_text("\n".join(L))
        T[
            "tab_a10_pooled.tex"
        ] = "new_models.pooled.{n75,n72}.<scheme>.<model>.{R2,mean_within_level_R2,MAE}"
    if Cc is not None:

        def fm(v):
            v = float(v)
            return str(int(v)) if v.is_integer() else f"{v:g}"

        for rs, fname in (
            ("n75", "tab_a10_al_grid.tex"),
            ("n72", "tab_a10_al_grid_n72.tex"),
        ):
            d = Cc["grid"][rs]
            L = [
                r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
                r"\begin{tabular}{@{}llccccc@{}}",
                r"\toprule",
                r" & & \multicolumn{2}{c}{Oracle stop} & \multicolumn{3}{c}{Realizable stop} \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
                r"Strategy & Model & Median [IQR] & Success & Median [IQR] & Success & Regret (V) \\",
                r"\midrule",
            ]
            for s, law in GRID_STRATS:
                k = skey(s, law)
                r = d[k]
                sur = (
                    "--"
                    if (s.startswith("ofat") or s == "random")
                    else (r"\LawGP" if law else r"\GP")
                )
                orc = f"{fm(r['oracle_median'])} [{fm(r['oracle_q25'])}, {fm(r['oracle_q75'])}]"
                osc = rf"{r['oracle_success_hits']}/{r['runs']}"
                if "stop_median" in r and s != "random":
                    rea = f"{fm(r['stop_median'])} [{fm(r['stop_q25'])}, {fm(r['stop_q75'])}]"
                    rsc = rf"{r['stop_success_hits']}/{r['runs']}"
                    reg = fnum(r["stop_regret_median"], 3)
                else:
                    rea, rsc, reg = "--", "--", "--"
                lab = STRAT_LABEL[s]
                L.append(
                    f"{lab} & {sur} & {orc} & {osc} & {rea} & {rsc} & {reg}" + r" \\"
                )
            L += [r"\bottomrule", r"\end{tabular}", ""]
            (tab / fname).write_text("\n".join(L))
            T[fname] = f"active_learning.grid.{rs}.<strategy>.{{oracle_*,stop_*}}"
        names = [n for n in PUB_ORDER if n in Cc["public"]]
        for mode, fname in (
            ("oracle", "tab_a10_al_public.tex"),
            ("stop", "tab_a10_al_public_stop.tex"),
        ):
            L = [
                r"\footnotesize\setlength{\tabcolsep}{2pt}%",
                rf"\begin{{tabular}}{{@{{}}l{'c' * len(names)}@{{}}}}",
                r"\toprule",
                r"Strategy & " + " & ".join(PUB_LABEL[n] for n in names) + r" \\",
                r"\midrule",
            ]
            for s, law in PUB_STRATS:
                k = skey(s, law)
                if mode == "stop" and s == "random":
                    continue
                cells = []
                for n in names:
                    r = Cc["public"][n][k]
                    if mode == "oracle":
                        cells.append(
                            f"{fm(r['oracle_median'])} ({C3.fpct(r['oracle_success_rate'])}\\%)"
                        )
                    else:
                        cells.append(
                            f"{fm(r['stop_median'])} ({C3.fpct(r['stop_success_rate'])}\\%)"
                        )
                L.append(f"{STRAT_LABEL[s]} & " + " & ".join(cells) + r" \\")
            L += [r"\bottomrule", r"\end{tabular}", ""]
            (tab / fname).write_text("\n".join(L))
            T[
                fname
            ] = f"active_learning.public.<dataset>.<strategy>.{mode}_median and {mode}_success_rate"
    return T


def camel(s):
    return "".join(w[:1].upper() + w[1:] for w in s.replace("-", "_").split("_") if w)


def model_mac_a(m):
    return {
        "Lineer": "Linear",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "Polinom2-OLS": "PolyOls",
        "Polinom2-Ridge": "PolyRidge",
        "SVR-RBF": "Svr",
        "KernelRidge": "Krr",
        "KNN": "Knn",
        "RandomForest": "Rf",
        "ExtraTrees": "Et",
        "GradientBoosting": "Gb",
        "XGBoost": "Xgb",
        "ANN-MLP": "Mlp",
        "ARD-GP": "Gp",
    }[m]


def build_macros(A, B, Cc):
    if A is not None:
        mac("mthFairConfigs", N_CFG, "fair_tuning.design.n_configurations")
        for rs, pre in (("n75", "mthFair"), ("n72", "mthDupFair")):
            for t in A2.TARGETS:
                if t not in A[rs]:
                    continue
                r = A[rs][t]
                T_ = CAMEL_T[t]
                for m in BENCH:
                    mm = r["models"][m]
                    mac(
                        f"{pre}{T_}{model_mac_a(m)}Tuned",
                        fnum(mm["tuned_loo_r2"], 3),
                        f"fair_tuning.{rs}.{t}.models.{m}.tuned_loo_r2",
                    )
                    mac(
                        f"{pre}{T_}{model_mac_a(m)}Fixed",
                        fnum(mm["fixed_loo_r2"], 3),
                        f"fair_tuning.{rs}.{t}.models.{m}.fixed_loo_r2",
                    )
                    mac(
                        f"{pre}{T_}{model_mac_a(m)}RankTuned",
                        mm["rank_tuned"],
                        f"fair_tuning.{rs}.{t}.models.{m}.rank_tuned",
                    )
                    mac(
                        f"{pre}{T_}{model_mac_a(m)}RankFixed",
                        mm["rank_fixed"],
                        f"fair_tuning.{rs}.{t}.models.{m}.rank_fixed",
                    )
                mac(
                    f"{pre}{T_}BestFixed",
                    a3_bench.SHORTNAME[r["best_fixed"]],
                    f"fair_tuning.{rs}.{t}.best_fixed",
                )
                mac(
                    f"{pre}{T_}BestTuned",
                    a3_bench.SHORTNAME[r["best_tuned"]],
                    f"fair_tuning.{rs}.{t}.best_tuned",
                )
                mac(
                    f"{pre}{T_}Kendall",
                    fnum(r["kendall_tau"], 2),
                    f"fair_tuning.{rs}.{t}.kendall_tau",
                )
                mac(
                    f"{pre}{T_}RankChanged",
                    r["n_models_rank_changed"],
                    f"fair_tuning.{rs}.{t}.n_models_rank_changed",
                )
                mac(
                    f"{pre}{T_}MaxShift",
                    r["max_abs_rank_shift"],
                    f"fair_tuning.{rs}.{t}.max_abs_rank_shift",
                )
                mac(
                    f"{pre}{T_}WinnerChanges",
                    "yes" if r["winner_changes"] else "no",
                    f"fair_tuning.{rs}.{t}.winner_changes",
                )
                mac(
                    f"{pre}{T_}NImproved",
                    r["n_models_improved"],
                    f"fair_tuning.{rs}.{t}.n_models_improved",
                )
        if A.get("inner_kfold_rms"):
            mac(
                "mthFairKfoldRmsBest",
                a3_bench.SHORTNAME[A["inner_kfold_rms_best"]],
                "fair_tuning.inner_kfold_rms_best",
            )
            mac(
                "mthFairKfoldRmsGp",
                fnum(A["inner_kfold_rms"]["ARD-GP"]["tuned_loo_r2"], 3),
                "fair_tuning.inner_kfold_rms.ARD-GP.tuned_loo_r2",
            )
    if B is not None:
        for rs, pre in (("n75", "mth"), ("n72", "mthDup")):
            if rs not in B["pooled"]:
                continue
            for sc, P in B["pooled"][rs].items():
                for m, p in P.items():
                    nd = 3 if sc == "loo" else 2
                    mac(
                        f"{pre}{SCHEME_MAC[sc]}{B_MAC[m]}Rsq",
                        fnum(p["R2"], nd),
                        f"new_models.pooled.{rs}.{sc}.{m}.R2",
                    )
                    if rs == "n75":
                        mac(
                            f"{pre}{SCHEME_MAC[sc]}{B_MAC[m]}Mae",
                            fnum(p["MAE"], 3),
                            f"new_models.pooled.{rs}.{sc}.{m}.MAE",
                        )
                        if "mean_within_level_R2" in p:
                            mac(
                                f"{pre}{SCHEME_MAC[sc]}{B_MAC[m]}Within",
                                fnum(p["mean_within_level_R2"], 2),
                                f"new_models.pooled.{rs}.{sc}.{m}.mean_within_level_R2",
                            )
            if rs == "n75":
                for sc, PL in B["per_level"][rs].items():
                    for m, lv in PL.items():
                        for k, v in lv.items():
                            lvl = next(
                                x
                                for x in A2.axis_levels(sc)
                                if A2.level_key(sc, x) == k
                            )
                            mac(
                                f"mth{B_MAC[m]}{A2.level_macro(sc, lvl)}Rsq",
                                fnum(v["R2"], 2),
                                f"new_models.per_level.n75.{sc}.{m}.{k}.R2",
                            )
        h = B["headline"]["n75"]
        mac(
            "mthFreqRescuedN",
            len(h["frequency_rescued_by"]),
            "new_models.headline.n75.frequency_rescued_by",
        )
        mac(
            "mthFreqSoftenedN",
            len(h["frequency_softened_by"]),
            "new_models.headline.n75.frequency_softened_by",
        )
        mac(
            "mthFreqBestNew",
            B_TEXT[h["frequency"]["best_new"]],
            "new_models.headline.n75.frequency.best_new",
        )
        mac(
            "mthFreqBestNewRsq",
            fnum(h["frequency"]["best_new_R2"], 2),
            "new_models.headline.n75.frequency.best_new_R2",
        )
        mac(
            "mthFreqBestRefRsq",
            fnum(h["frequency"]["best_reference_R2"], 2),
            "new_models.headline.n75.frequency.best_reference_R2",
        )
        if "n72" in B["headline"]:
            h7 = B["headline"]["n72"]
            mac(
                "mthDupFreqRescuedN",
                len(h7["frequency_rescued_by"]),
                "new_models.headline.n72.frequency_rescued_by",
            )
            mac(
                "mthDupFreqBestNewRsq",
                fnum(h7["frequency"]["best_new_R2"], 2),
                "new_models.headline.n72.frequency.best_new_R2",
            )
    if Cc is not None:
        for rs, pre in (("n75", "mthAl"), ("n72", "mthDupAl")):
            for k, r in Cc["grid"][rs].items():
                K = "".join(camel(x) for x in k.split("_"))
                mac(
                    f"{pre}{K}Median",
                    f"{r['oracle_median']:g}",
                    f"active_learning.grid.{rs}.{k}.oracle_median",
                )
                mac(
                    f"{pre}{K}Success",
                    r["oracle_success_hits"],
                    f"active_learning.grid.{rs}.{k}.oracle_success_hits",
                )
                mac(f"{pre}{K}Runs", r["runs"], f"active_learning.grid.{rs}.{k}.runs")
                if "stop_median" in r:
                    mac(
                        f"{pre}{K}StopMedian",
                        f"{r['stop_median']:g}",
                        f"active_learning.grid.{rs}.{k}.stop_median",
                    )
                    mac(
                        f"{pre}{K}StopSuccess",
                        r["stop_success_hits"],
                        f"active_learning.grid.{rs}.{k}.stop_success_hits",
                    )
                    mac(
                        f"{pre}{K}StopRegret",
                        fnum(r["stop_regret_median"], 3),
                        f"active_learning.grid.{rs}.{k}.stop_regret_median",
                    )
            h = Cc["headline"][rs]
            mac(
                f"{pre}BeatsOfatN",
                len(h["beats_ofat_oracle_stop"]),
                f"active_learning.headline.{rs}.beats_ofat_oracle_stop",
            )
            mac(
                f"{pre}BeatsOfatStopN",
                len(h["beats_ofat_realizable_stop"]),
                f"active_learning.headline.{rs}.beats_ofat_realizable_stop",
            )
        for nm, d in Cc["public"].items():
            P_ = PUB_MAC[nm]
            for s, law in PUB_STRATS:
                k = skey(s, law)
                r = d[k]
                K = "".join(camel(x) for x in k.split("_"))
                mac(
                    f"mthAlPub{P_}{K}Median",
                    f"{r['oracle_median']:g}",
                    f"active_learning.public.{nm}.{k}.oracle_median",
                )
                mac(
                    f"mthAlPub{P_}{K}Success",
                    C3.fpct(r["oracle_success_rate"]),
                    f"active_learning.public.{nm}.{k}.oracle_success_rate",
                )
                if "stop_median" in r and s != "random":
                    mac(
                        f"mthAlPub{P_}{K}StopMedian",
                        f"{r['stop_median']:g}",
                        f"active_learning.public.{nm}.{k}.stop_median",
                    )
                    mac(
                        f"mthAlPub{P_}{K}StopSuccess",
                        C3.fpct(r["stop_success_rate"]),
                        f"active_learning.public.{nm}.{k}.stop_success_rate",
                    )
        n_ei_beats = sum(
            v["ei_beats_best_ofat"] for v in Cc["headline"]["public"].values()
        )
        mac(
            "mthAlPubEiBeatsOfatN",
            n_ei_beats,
            "active_learning.headline.public.<dataset>.ei_beats_best_ofat",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="abc")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    T0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / ("a10_cache_quick" if args.quick else "a10_cache"))
    J = {
        "task": "E5 METHODS (analysis protocol section 1)",
        "seed": SEED,
        "n_conditions": int(A2.N),
        "excluded_rows_n72": DUP_ROWS,
        "excluded_reason": "duplicated-recordings note in the README: 52 and 54 are exact scalar multiples of 47 and 49, 73 is a scaled copy of 5",
        "quick_mode": bool(args.quick),
    }
    A = B = Cc = None
    PRED, FIGB, ALCSV = [], [], []
    if "a" in args.blocks:
        A, pa = block_a(cache, args.n_jobs, args.quick)
        J["fair_tuning"] = A
        PRED += pa
    if "b" in args.blocks:
        B, pb, FIGB = block_b(cache, args.n_jobs, args.quick)
        J["new_models"] = B
        PRED += pb
    if "c" in args.blocks:
        Cc, ALCSV = block_c(cache, args.n_jobs, args.quick)
        J["active_learning"] = Cc
    tabs = write_tables(out, A, B, Cc, args.quick)
    build_macros(A, B, Cc)
    names = [m[0] for m in MACROS]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1][:5]
    lines = [
        "% numbers_a10.tex -- generated by code/a10_methods.py (E5 METHODS); do not edit",
    ]
    for n, v, k in MACROS:
        lines.append(rf"\newcommand{{\{n}}}{{{v}}}% {k}")
    (out / "numbers_a10.tex").write_text("\n".join(lines) + "\n")
    J["index"] = {
        "tables": tabs,
        "macros": "numbers_a10.tex: every macro line carries its JSON key as a comment",
        "a10_predictions.csv": "out-of-fold predictions of block a (tuned, and fixed recomputed) and block b",
        "a10_figdata_fair.csv": "fair_tuning.<rowset>.<target>.models.<model>",
        "a10_figdata_levels.csv": "new_models.per_level",
        "a10_figdata_al.csv": "per-run active-learning outcomes behind active_learning.*",
    }
    J["macros"] = {n: {"value": v, "key": k} for n, v, k in MACROS}
    J["runtime_s"] = time.time() - T0
    (out / "a10_methods.json").write_text(json.dumps(jsonable(J), indent=1))
    if PRED:
        pd.DataFrame(PRED).to_csv(out / "a10_predictions.csv", index=False)
    if A is not None:
        rows = []
        for rs in ("n75", "n72"):
            for t, r in A[rs].items():
                for m, mm in r["models"].items():
                    rows.append(
                        {
                            "rowset": rs,
                            "target": t,
                            "model": m,
                            "fixed_loo_r2": mm["fixed_loo_r2"],
                            "tuned_loo_r2": mm["tuned_loo_r2"],
                            "fixed_loo_mae": mm["fixed_loo_mae"],
                            "tuned_loo_mae": mm["tuned_loo_mae"],
                            "rank_fixed": mm["rank_fixed"],
                            "rank_tuned": mm["rank_tuned"],
                        }
                    )
        pd.DataFrame(rows).to_csv(out / "a10_figdata_fair.csv", index=False)
    if FIGB:
        pd.DataFrame(FIGB).to_csv(out / "a10_figdata_levels.csv", index=False)
    if ALCSV:
        pd.DataFrame(ALCSV).to_csv(out / "a10_figdata_al.csv", index=False)
    summ = []
    if A is not None and "rms_Voc" in A["n75"]:
        r = A["n75"]["rms_Voc"]
        summ.append(
            f"fair V_rms best fixed {r['best_fixed']} tuned {r['best_tuned']} tau {r['kendall_tau']:.2f}"
        )
    if B is not None:
        h = B["headline"]["n75"]
        summ.append(
            f"held-out freq best new {h['frequency']['best_new']} R2 {h['frequency']['best_new_R2']:.2f} vs ref {h['frequency']['best_reference_R2']:.2f}; rescued by {h['frequency_rescued_by']}"
        )
    if Cc is not None:
        h = Cc["headline"]["n75"]
        summ.append(
            f"AL grid OFAT {h['ofat_median']:g}; beats OFAT (oracle) {h['beats_ofat_oracle_stop']}"
        )
    print(f"[a10] done in {time.time() - T0:.0f} s | " + " | ".join(summ))


if __name__ == "__main__":
    main()
