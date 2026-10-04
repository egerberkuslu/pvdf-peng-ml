"""A3 benchmark helpers: hyperparameter table from the live model factories,
benchmark and significance tables from the legacy JSON, and LOO predictions
of the best model per target recomputed with the canonical factories.

Sources
  code/reg_common.py::model_factories()      fourteen regressors
  code/reg_common.py::svr_tuned_estimator()  exploratory tuned SVR
  code/baseline/mlp_tuned_analysis.py    MLP nested-tuning grid
  results/baseline/reg_stats.json   CV R2, bootstrap CI, Wilcoxon
  results/baseline/reg_models.json  CV R2, in-sample R2, MAE (all 14)
  results/baseline/mlp_tuned_results.json
"""

import ast
import sys

import numpy as np

import a3_common as C

sys.path[:0] = [str(C.ROOT / "code")]
import reg_common  # noqa: E402

ORDER = [
    "Lineer",
    "Ridge",
    "Lasso",
    "Polinom2-OLS",
    "Polinom2-Ridge",
    "SVR-RBF",
    "KernelRidge",
    "KNN",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "XGBoost",
    "ANN-MLP",
    "ARD-GP",
]
NAME = {
    "Lineer": "Linear regression",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "Polinom2-OLS": "Polynomial deg.\\ 2, OLS",
    "Polinom2-Ridge": "Polynomial deg.\\ 2, ridge",
    "SVR-RBF": "Support-vector reg.",
    "KernelRidge": "Kernel ridge",
    "KNN": "$k$-nearest neighbors",
    "RandomForest": "Random forest",
    "ExtraTrees": "Extra-trees",
    "GradientBoosting": "Gradient boosting",
    "XGBoost": "XGBoost",
    "ANN-MLP": "Multilayer perceptron",
    "ARD-GP": "Gaussian process (ARD)",
}
SHORTNAME = {
    "Lineer": "Linear",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "Polinom2-OLS": "Poly2-OLS",
    "Polinom2-Ridge": "Poly2-ridge",
    "SVR-RBF": "SVR",
    "KernelRidge": "Kernel ridge",
    "KNN": "kNN",
    "RandomForest": "Random forest",
    "ExtraTrees": "Extra-trees",
    "GradientBoosting": "Grad. boosting",
    "XGBoost": "XGBoost",
    "ANN-MLP": "MLP",
    "ARD-GP": "GP (ARD)",
}
CAMEL_M = {
    "ARD-GP": "Gp",
    "RandomForest": "Rf",
    "ExtraTrees": "Et",
    "XGBoost": "Xgb",
    "GradientBoosting": "Gb",
    "Lineer": "Linear",
    "ANN-MLP": "Mlp",
}


# ---------------------------------------------------------------- hyperparameters
def _unwrap(est):
    """Return (target_standardized, steps[(name, est)]) of a factory product."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    tstd = False
    if isinstance(est, TransformedTargetRegressor):
        tstd = type(est.transformer).__name__
        est = est.regressor
    steps = est.steps if isinstance(est, Pipeline) else [("model", est)]
    return tstd, steps


SHOW = {
    "LinearRegression": ["fit_intercept"],
    "Ridge": ["alpha", "fit_intercept"],
    "Lasso": ["alpha", "max_iter"],
    "SVR": ["kernel", "C", "gamma", "epsilon"],
    "KernelRidge": ["kernel", "alpha", "gamma"],
    "KNeighborsRegressor": ["n_neighbors", "weights", "p"],
    "RandomForestRegressor": [
        "n_estimators",
        "max_features",
        "max_depth",
        "min_samples_leaf",
        "bootstrap",
        "random_state",
    ],
    "ExtraTreesRegressor": [
        "n_estimators",
        "max_features",
        "max_depth",
        "min_samples_leaf",
        "bootstrap",
        "random_state",
    ],
    "GradientBoostingRegressor": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "loss",
        "random_state",
    ],
    "XGBRegressor": [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "reg_lambda",
        "random_state",
    ],
    "MLPRegressor": [
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "learning_rate_init",
        "max_iter",
        "random_state",
    ],
    "GaussianProcessRegressor": [
        "normalize_y",
        "alpha",
        "n_restarts_optimizer",
        "random_state",
    ],
}


def _xgb_default(key):
    """xgboost leaves unset params as None; report the library default instead."""
    return {"subsample": 1.0, "reg_lambda": 1.0}.get(key)


def _val(v):
    """Literal rendering: floats stay floats (max_features=1.0 means all features)."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def _bounds(b):
    """(1e-05, 100000.0) -> [10^{-5}, 10^{5}] in math mode."""
    def one(x):
        e = np.log10(float(x))
        if abs(e - round(e)) < 1e-12:
            return f"10^{{{int(round(e))}}}"
        return f"{float(x):g}"

    lo, hi = b
    return f"$[{one(lo)}, {one(hi)}]$"


def _list(v):
    return "[" + ", ".join(f"{float(x):g}" for x in np.atleast_1d(v)) + "]"


def describe(name, est):
    tstd, steps = _unwrap(est)
    pre, model_params, model_cls = [], {}, None
    for sname, s in steps:
        cls = type(s).__name__
        if cls == "StandardScaler":
            pre.append("input standardization")
        elif cls == "PolynomialFeatures":
            pre.append(
                f"PolynomialFeatures(degree={s.degree}, include_bias={s.include_bias})"
            )
        else:
            model_cls = cls
            p = s.get_params(deep=False)
            for k in SHOW.get(cls, []):
                v = p.get(k)
                if v is None and cls == "XGBRegressor":
                    v = _xgb_default(k)
                    if v is not None:
                        v = f"{_val(v)} (default)"
                model_params[k] = v if isinstance(v, str) and k != "kernel" else _val(v)
            if cls == "GaussianProcessRegressor":
                k = s.kernel
                model_params["kernel"] = str(k)
                mat = k.k1.k2
                white = k.k2
                const = k.k1.k1
                model_params["length_scale_init"] = _list(mat.length_scale)
                model_params["length_scale_bounds"] = _bounds(mat.length_scale_bounds)
                model_params["nu"] = str(mat.nu)
                model_params["constant_bounds"] = _bounds(const.constant_value_bounds)
                model_params["noise_level_init"] = f"{float(white.noise_level):g}"
                model_params["noise_level_bounds"] = _bounds(white.noise_level_bounds)
    return {
        "model": name,
        "class": model_cls,
        "preprocessing": pre if pre else ["none (raw inputs)"],
        "target_standardization": tstd if tstd else "none",
        "params": model_params,
    }


def mlp_grid_from_source():
    """Parse GRID and make_model() constants from the legacy MLP script (not run)."""
    src = (C.REV / "code" / "baseline" / "mlp_tuned_analysis.py").read_text()
    tree = ast.parse(src)
    hidden, alphas = None, None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "GRID"
        ):
            comp = node.value
            for gen in comp.generators:
                vals = ast.literal_eval(gen.iter)
                if gen.target.id == "h":
                    hidden = vals
                elif gen.target.id == "a":
                    alphas = vals
    mlp_kw = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "MLPRegressor"
        ):
            for kw in node.keywords:
                try:
                    mlp_kw[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    mlp_kw[kw.arg] = ast.unparse(kw.value)
    kf = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "KFold":
            for kw in node.keywords:
                kf[kw.arg] = ast.literal_eval(kw.value)
    return {
        "hidden_layer_sizes_grid": [list(h) for h in hidden],
        "alpha_grid": list(alphas),
        "n_configs": len(hidden) * len(alphas),
        "mlp_fixed_kwargs": mlp_kw,
        "inner_cv": {"type": "KFold", **kf},
        "selection_score": "inner 5-fold R2 on the LOO training fold",
        "outer": "LOO (75 folds)",
        "input_standardization": True,
        "target_standardization": "StandardScaler (TransformedTargetRegressor)",
    }


def svr_tuned_description():
    est = reg_common.svr_tuned_estimator()
    grid = {k: list(v) for k, v in est.param_grid.items()}
    return {
        "estimator": "GridSearchCV(Pipeline(StandardScaler, SVR(kernel='rbf')))",
        "param_grid": grid,
        "inner_cv": est.cv,
        "inner_cv_note": "integer cv on a regressor = unshuffled KFold(5) in scikit-learn",
        "scoring": est.scoring,
        "target_standardization": "none",
        "outer": "LOO via cross_val_predict (regression/reg_01_models.py), so the search"
        " is refit inside every LOO training fold",
    }


def interval_gp_description():
    g = C.gp()
    return describe("interval/PhysGP GP", g) | {
        "preprocessing": [
            "input standardization fitted on the training rows of each fold"
        ]
    }


def hyperparams():
    facs = reg_common.model_factories()
    rows = [describe(n, facs[n]()) for n in ORDER]
    return {
        "source": "code/reg_common.py::model_factories()",
        "seed_constant": reg_common.SEED,
        "models": rows,
        "mlp_nested_tuning": mlp_grid_from_source(),
        "svr_tuned_exploratory": svr_tuned_description(),
        "interval_and_physgp_gp": interval_gp_description(),
        "reg_stats_gp_note": (
            "regression/reg_stats.py rebuilds the ARD-GP for the bootstrap CI and Wilcoxon"
            " rows with Matern length_scale init [1,1,1,1] instead of the factory's"
            " [1,1,1,5]; all other GP settings are identical"
        ),
        "selection_protocol": (
            "all settings fixed a priori; GP designated primary before scoring;"
            " MLP nested tuning as the declared exception"
        ),
    }


def _tex(s):
    return (
        s.replace("_", r"\_")
        .replace("**", "^")
    )


def hyperparam_table(H):
    """Tabular body listing every regressor with its exact settings."""
    lines = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.46\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.11\textwidth}@{}}",
        r"\toprule",
        r"Model & Settings (scikit-learn / xgboost names) & Preprocessing & Tuning \\",
        r"\midrule",
    ]
    for row in H["models"]:
        n = row["model"]
        p = dict(row["params"])
        if row["class"] == "GaussianProcessRegressor":
            ls = p.pop("length_scale_init")
            lb = p.pop("length_scale_bounds")
            nu = p.pop("nu")
            cb = p.pop("constant_bounds")
            nl = p.pop("noise_level_init")
            nb = p.pop("noise_level_bounds")
            p.pop("kernel")
            kern = (
                rf"ConstantKernel(1.0, bounds {cb}) $\times$ Matern($\nu$={nu},"
                rf" length\_scale init {ls}, bounds {lb}) + WhiteKernel({nl}, bounds {nb});"
            )
            settings = (
                kern + " " + ", ".join(f"{_tex(k)}={_tex(v)}" for k, v in p.items())
            )
            tune = "kernel by marginal likelihood in every training fold"
        else:
            settings = f"{row['class']}: " + ", ".join(
                f"{_tex(k)}={_tex(v)}" for k, v in p.items()
            )
            tune = "fixed a priori"
        pre = "; ".join(
            _tex(x).replace("(", r"(\allowbreak ").replace(", ", r",\allowbreak ")
            for x in row["preprocessing"]
        )
        if row["target_standardization"] != "none":
            pre += "; target standardized (StandardScaler, inverted before scoring)"
        lines.append(rf"{NAME[n]} & {settings} & {pre} & {tune} \\")
    lines.append(r"\midrule")
    M = H["mlp_nested_tuning"]
    kw = M["mlp_fixed_kwargs"]
    hid = ", ".join(
        "(" + ",".join(str(x) for x in h) + ")" for h in M["hidden_layer_sizes_grid"]
    )
    alp = ", ".join(_val(a) if a >= 1 else f"{a:g}" for a in M["alpha_grid"])
    lines.append(
        r"Multilayer perceptron, nested tuning & "
        rf"MLPRegressor: hidden\_layer\_sizes $\in$ \{{{hid}\}}, alpha $\in$ \{{{alp}\}}"
        rf" ({M['n_configs']} configurations), solver={kw.get('solver')},"
        rf" max\_iter={kw.get('max_iter')}, random\_state={kw.get('random_state')};"
        rf" selected by inner KFold(n\_splits={M['inner_cv'].get('n_splits')},"
        rf" shuffle={M['inner_cv'].get('shuffle')}, random\_state={M['inner_cv'].get('random_state')})"
        r" $R^2$ inside every LOO training fold"
        r" & input standardization; target standardized & nested grid search (declared exception) \\"
    )
    S = H["svr_tuned_exploratory"]
    g = S["param_grid"]
    lines.append(
        r"Support-vector reg., tuned (exploratory, not in the benchmark) & "
        rf"SVR(kernel=rbf): C $\in$ \{{{', '.join(_val(v) for v in g['svr__C'])}\}},"
        rf" gamma $\in$ \{{{', '.join(_val(v) for v in g['svr__gamma'])}\}};"
        rf" GridSearchCV(cv={S['inner_cv']}, unshuffled KFold, scoring={S['scoring']})"
        r" refit inside every LOO training fold"
        r" & input standardization; target not standardized & nested grid search \\"
    )
    I = H["interval_and_physgp_gp"]
    p = I["params"]
    lines.append(
        r"Gaussian process for intervals and PhysGP (not in the benchmark) & "
        rf"ConstantKernel(1.0, bounds {p['constant_bounds']}) $\times$ Matern($\nu$={p['nu']},"
        rf" length\_scale init {p['length_scale_init']}, bounds {p['length_scale_bounds']})"
        rf" + WhiteKernel({p['noise_level_init']}, bounds {p['noise_level_bounds']});"
        rf" normalize\_y={p['normalize_y']}, alpha={p['alpha']},"
        rf" n\_restarts\_optimizer={p['n_restarts_optimizer']}"
        r" & input standardization in every fold & kernel by marginal likelihood in every training fold \\"
    )
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{0.98\textwidth}@{}}{Selection protocol. Every setting above was"
        r" fixed a priori and not tuned per target, the Gaussian process was designated the"
        r" primary model before any score was computed, and the multilayer perceptron with"
        r" nested tuning is the one declared exception. All preprocessing is fitted on the"
        r" training rows of each fold, and scores are computed on the original voltage scale.} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- benchmark numbers
def bench_numbers():
    """Merge reg_stats.json (canonical where it reports a model) with reg_models.json."""
    RS = C.load_legacy("reg_stats.json")
    RM = C.load_legacy("reg_models.json")
    MT = C.load_legacy("mlp_tuned_results.json")
    out = {}
    for t in C.TARGETS:
        lk = C.LEGACY_TKEY[t]
        rs = RS["per_target"][lk]
        rm = RM["targets"][lk]["models"]
        mods = {}
        for m in ORDER:
            rec = {
                "cv_r2_reg_models": rm[m]["cv_r2"],
                "in_sample_r2": rm[m]["in_sample_r2"],
                "cv_mae": rm[m]["cv_mae"],
                "cv_rmse": rm[m]["cv_rmse"],
                "in_sample_source": "reg_models.json",
            }
            if m in rs["models"]:
                s = rs["models"][m]
                rec["cv_r2"] = s["cv_r2"]
                rec["cv_r2_source"] = "reg_stats.json"
                rec["boot_ci95"] = s["ci95"]
                rec[
                    "boot_ci_method"
                ] = "record-level percentile bootstrap, 2000 resamples"
                if "wilcoxon_p_vs_best" in s:
                    rec["wilcoxon_p_vs_best"] = s["wilcoxon_p_vs_best"]
            else:
                rec["cv_r2"] = rm[m]["cv_r2"]
                rec["cv_r2_source"] = "reg_models.json"
            mods[m] = rec
        out[t] = {
            "best_reg_stats": rs["best"],
            "models": mods,
            "svr_tuned": RM["targets"][lk]["models"]["SVR-ayarli"],
            "mlp_nested_tuned": MT[t],
            "gp_insample_by_init": gp_insample_by_init(t) if t == "Vmax" else None,
            "wilcoxon_test": "paired Wilcoxon signed-rank on absolute LOO errors vs best",
            "n": 75,
        }
    return out


def _bold(s, on):
    return rf"\textbf{{{s}}}" if on else s


MAIN_ROWS = [
    ("ARD-GP", "Gaussian process"),
    ("RandomForest", "Random forest"),
    ("ExtraTrees", "Extra-trees"),
    ("XGBoost", "XGBoost"),
    ("GradientBoosting", "Gradient boosting"),
    ("ANN-MLP", "Multilayer perceptron"),
    ("SVR-RBF", "Support-vector reg."),
    ("KernelRidge", "Kernel ridge"),
    ("Polinom2-Ridge", "Polynomial deg.\\ 2, ridge"),
    ("Ridge", "Ridge"),
]


def gp_vmax_note(B):
    """Generated table note on the kernel-initialization sensitivity of the GP |V|max score."""
    g = B["Vmax"]["models"]["ARD-GP"]
    ins = B["Vmax"]["gp_insample_by_init"]
    same = C.fnum(ins["init_1111"], 2) == C.fnum(ins["init_1115"], 2)
    s = (
        rf"Leave-one-out $R^2$ of the Gaussian process on \Vmax{{}} is {C.fnum(g['cv_r2'], 3)}"
        r" with Mat\'ern length scales initialized at [1, 1, 1, 1], the setting of the"
        rf" significance analysis, and {C.fnum(g['cv_r2_reg_models'], 3)} with the benchmark"
        r" initialization [1, 1, 1, 5]."
    )
    if same:
        s += rf" The in-sample value {C.fnum(ins['init_1111'], 2)} is the same under both."
    else:
        s += (
            rf" In-sample values are {C.fnum(ins['init_1111'], 2)} and"
            rf" {C.fnum(ins['init_1115'], 2)}, and the table shows the second."
        )
    return s


def gp_insample_by_init(t):
    """In-sample R2 of the benchmark GP under both legacy length-scale initializations."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C_
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.metrics import r2_score

    y = C.yv(t)
    out = {}
    for tag, init in (("init_1111", [1.0] * 4), ("init_1115", [1.0, 1.0, 1.0, 5.0])):
        k = C_(1.0) * Matern(length_scale=init, nu=2.5) + WhiteKernel(1e-3)
        g = GaussianProcessRegressor(
            kernel=k, normalize_y=True, alpha=1e-6, n_restarts_optimizer=4,
            random_state=reg_common.SEED,
        ).fit(C.X, y)
        out[tag] = float(r2_score(y, g.predict(C.X)))
    out["note"] = "recomputed on the full 75 conditions, reg_stats.py/_est and reg_common.gp settings"
    return out


def main_table(B):
    best = {
        t: max(B[t]["models"], key=lambda m: B[t]["models"][m]["cv_r2"])
        for t in C.TARGETS
    }
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Model & \RMS & \Vpp & \Vmax \\",
        r"\midrule",
    ]
    for m, lab in MAIN_ROWS:
        cells = []
        for t in C.TARGETS:
            r = B[t]["models"][m]
            cells.append(
                _bold(C.fnum(r["cv_r2"], 3), m == best[t])
                + f" ({C.fnum(r['in_sample_r2'], 2)})"
            )
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = [
        C.fnum(B[t]["mlp_nested_tuned"]["tuned_nestedCV_R2"], 3) + " (--)"
        for t in C.TARGETS
    ]
    L.append(r"MLP, nested tuning & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(
        r"\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{0.95\columnwidth}@{}}{\footnotesize "
        + gp_vmax_note(B)
        + r"} \\"
    )
    L += [r"\end{tabular}", ""]
    return "\n".join(L), best


def all_table(B):
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}lccccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\RMS} & \multicolumn{3}{c}{\Vpp} & \multicolumn{3}{c}{\Vmax} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Model & CV $R^2$ & in-sample & MAE (V) & CV $R^2$ & in-sample & MAE (V) & CV $R^2$ & in-sample & MAE (V) \\",
        r"\midrule",
    ]
    for m in ORDER:
        c = []
        for t in C.TARGETS:
            r = B[t]["models"][m]
            c += [
                C.fnum(r["cv_r2"], 3),
                C.fnum(r["in_sample_r2"], 2),
                C.fnum(r["cv_mae"], 3),
            ]
        L.append(f"{NAME[m]} & " + " & ".join(c) + r" \\")
    L.append(r"\midrule")
    c = []
    for t in C.TARGETS:
        r = B[t]["svr_tuned"]
        c += [C.fnum(r["cv_r2"], 3), "--", C.fnum(r["cv_mae"], 3)]
    L.append(r"Support-vector reg., tuned (exploratory) & " + " & ".join(c) + r" \\")
    c = []
    for t in C.TARGETS:
        r = B[t]["mlp_nested_tuned"]
        c += [
            C.fnum(r["tuned_nestedCV_R2"], 3),
            "--",
            C.fnum(r["tuned_nestedCV_MAE"], 3),
        ]
    L.append(r"Multilayer perceptron, nested tuning & " + " & ".join(c) + r" \\")
    L.append(r"\bottomrule")
    L.append(
        r"\multicolumn{10}{@{}>{\raggedright\arraybackslash}p{0.97\textwidth}@{}}{"
        + gp_vmax_note(B)
        + r" The Gaussian-process \Vmax{} MAE comes from the run initialized at [1, 1, 1, 5].} \\"
    )
    L += [r"\end{tabular}", ""]
    return "\n".join(L)


def _p(p):
    if p is None:
        return "--"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        mant = p / 10**e
        return rf"${C.rhu(mant, 1):.1f}\times10^{{{e}}}$"
    if p < 0.1:
        return C.fnum(p, 3)
    return C.fnum(p, 2)


def sig_table(B):
    RS = C.load_legacy("reg_stats.json")
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Target & Model & CV $R^2$ & $95\%$ bootstrap CI & $p$ vs best \\",
        r"\midrule",
    ]
    for k, t in enumerate(C.TARGETS):
        rs = RS["per_target"][C.LEGACY_TKEY[t]]
        best = rs["best"]
        models = list(rs["models"].keys())
        for i, m in enumerate(models):
            r = B[t]["models"][m]
            lab = NAME[m].replace(" (ARD)", "")
            if m == "Lineer":
                lab = "Linear regression"
            lo, hi = r["boot_ci95"]
            first = (
                rf"\multirow{{{len(models)}}}{{*}}{{{C.TEX_T[t]}}}" if i == 0 else ""
            )
            L.append(
                f"{first} & {lab} & {_bold(C.fnum(r['cv_r2'], 3), m == best)} & [{C.fnum(lo, 2)}, {C.fnum(hi, 2)}]"
                f" & {_p(r.get('wilcoxon_p_vs_best')) if m != best else '--'}" + r" \\"
            )
        if k < len(C.TARGETS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- calibration LOO
def recompute_loo_best(B):
    """LOO predictions of the reg_stats best model per target with canonical factories."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    facs = reg_common.model_factories()
    out = {}
    for t in C.TARGETS:
        m = B[t]["best_reg_stats"]
        y = C.yv(t)
        pred = cross_val_predict(facs[m](), C.X, y, cv=LeaveOneOut(), n_jobs=-1)
        out[t] = {
            "model": m,
            "y_true": y.tolist(),
            "y_pred": pred.tolist(),
            "freq_Hz": C.FRQ.tolist(),
            "loo_r2_recomputed": float(r2_score(y, pred)),
            "loo_mae_recomputed": float(mean_absolute_error(y, pred)),
            "legacy_cv_r2": B[t]["models"][m]["cv_r2"],
            "factory": f"code/reg_common.py::model_factories()['{m}']",
            "inputs": "cnt_pct, is_pristine, force_N, freq_Hz from targets_design.parquet",
        }
    return out
