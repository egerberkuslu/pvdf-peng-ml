#!/usr/bin/env python3
"""Task A2: PhysGP identifiability, per-level axis results and the axis-wise selection rule.

One command (run from the repo root):
    python code/a2_physgp.py
regenerates
    results/a2_physgp.json            every number, with an "index" dict
    results/a2_predictions.csv        out-of-fold predictions
    results/tables/tab_phys_*.tex     booktabs tabular bodies
    results/numbers_a2.tex            \\newcommand macros, prefix phys
    figures/fig_phys_axes.pdf/.png, fig_phys_law.pdf/.png
    figures/captions_a2.md

Sections
    1. equivalence of the five-parameter law with the legacy six-parameter law
    2. identifiability of (a0, a1, a2, f0, gamma): Jacobian SE, residual bootstrap,
       correlations, per-fold spread over LOO and group-wise folds
    3. per-level group-wise results for plain GP and PhysGP, three axes, three targets
    4. mean-function ablation (law alone, linear-in-force mean, law with offset)
    5. axis-wise selection rules scored out of sample (nested leave-one-level-out)
Targets: V_rms, V_pp, |V|_max. The squared-voltage proxy is not analysed.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_common as A  # noqa: E402
import a2_report as RPT  # noqa: E402

SEED = 0
N_BOOT = 500
STAB_F0_SPREAD_HZ = 2.0
T0 = time.time()

LEGACY_PHYS = os.path.join(
    A.REV, "results", "baseline", "physgp_results.json"
)
LEGACY_REV = os.path.join(
    A.REV, "results", "baseline", "revision_experiments.json"
)
OUT_JSON = os.path.join(A.REV, "results", "a2_physgp.json")
OUT_CSV = os.path.join(A.REV, "results", "a2_predictions.csv")

Y = {t: A.df[t].values.astype(float) for t in A.TARGETS}
cache = A.FoldCache()


def f(x):
    """Plain float (NaN -> None) for JSON."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def lvl_metrics(y, mu):
    e = mu - y
    r = np.corrcoef(y, mu)[0, 1] if np.std(mu) > 0 and np.std(y) > 0 else np.nan
    return {
        "n": int(len(y)),
        "R2": f(r2_score(y, mu)),
        "MAE": f(mean_absolute_error(y, mu)),
        "bias": f(np.mean(e)),
        "pearson_r": f(r),
        "y_mean": f(np.mean(y)),
        "pred_mean": f(np.mean(mu)),
        "pred_sd": f(np.std(mu)),
        "y_sd": f(np.std(y)),
    }


# =============================================================== register jobs
MAIN_MODELS = ["gp", "physgp"]
ABL_MODELS = ["law", "gp_linF", "physgp_offset"]
EQUIV_MODELS = ["physgp6", "physgp_ltol", "physgp6_ltol"]
RESTART_MODELS = ["gp_r4", "physgp_r4", "physgp6_r4"]  # optimizer-restart check, LOO only

for t in A.TARGETS:
    for tr, te in A.loo_folds():
        for m in MAIN_MODELS + EQUIV_MODELS + RESTART_MODELS:
            cache.want(m, t, tr, te)
    for ax in A.AXES:
        for tr, te, _ in A.group_folds(ax):
            for m in MAIN_MODELS + ABL_MODELS:
                cache.want(m, t, tr, te)
            # inner leave-one-level-out along the same axis on the outer training rows
            for itr, ite, _ in A.group_folds(ax, idx=tr):
                for m in MAIN_MODELS:
                    cache.want(m, t, itr, ite)
# sensitivity without the three duplicated recordings (duplicated-recordings note in the README)
DUP_IDS = [52, 54, 73]
KEEP = np.array([i for i in range(A.N) if i not in DUP_IDS])
for t in A.TARGETS:
    for tr, te in A.loo_folds(KEEP):
        for m in MAIN_MODELS:
            cache.want(m, t, tr, te)
    for ax in A.AXES:
        for tr, te, _ in A.group_folds(ax, idx=KEEP):
            for m in MAIN_MODELS + (["law", "gp_linF"] if ax == "force" else []):
                cache.want(m, t, tr, te)
n_jobs = cache.run()
print(f"[a2] {n_jobs} fold fits in {time.time() - T0:.1f} s")


def oof(model, t, folds):
    """Out-of-fold predictions and law extras over a list of folds."""
    mu = np.full(A.N, np.nan)
    sd = np.full(A.N, np.nan)
    extras = []
    for fo in folds:
        tr, te = fo[0], fo[1]
        m, s, ex = cache.get(model, t, tr, te)
        mu[te], sd[te] = m, s
        extras.append(ex)
    return mu, sd, extras


R = {"meta": {}, "index": {}}
R["meta"] = {
    "task": "A2",
    "seed": SEED,
    "n_conditions": A.N,
    "targets": A.TARGETS,
    "law_form": "m(x) = (a0 + a1 c + a2 c^2) * F * (gamma/2)^2 / ((f - f0)^2 + (gamma/2)^2)",
    "law_bounds": {
        "a0": [0.0, 2500.0],
        "a1": [-250.0, 250.0],
        "a2": [-250.0, 250.0],
        "f0": [10.0, 30.0],
        "gamma": [1.0, 30.0],
    },
    "law_bounds_note": "bounds apply to the scaled fit on y/std(y_train); legacy (b0,b1,b2,A) box maps exactly onto this (a0,a1,a2) box",
    "law_start_scaled": A.P0_5,
    "law_fit_tolerance": A.TOL_TIGHT,
    "param_units": {
        "a0": "V/N",
        "a1": "V/(N wt%)",
        "a2": "V/(N wt%^2)",
        "f0": "Hz",
        "gamma": "Hz",
    },
    "gp": "ConstantKernel(1,(1e-3,1e3)) * Matern(ls=[1]*4, nu=2.5, (1e-2,1e3)) + WhiteKernel(1e-3,(1e-6,1e1)), normalize_y, n_restarts 0, alpha 1e-10, X standardized in fold",
    "n_bootstrap": N_BOOT,
    "stability_threshold_Hz": STAB_F0_SPREAD_HZ,
    "stability_rule": f"PhysGP if inner-fold f0 spread (max - min) < {STAB_F0_SPREAD_HZ} Hz and sign(a2) constant over inner folds, else plain GP",
    "inner_cv_rule": "PhysGP if inner leave-one-level-out pooled R2 (global denominator over outer training rows) of PhysGP exceeds that of plain GP, else plain GP (ties -> plain GP)",
    "condition_id": "row index of data/targets_design.parquet (0-74)",
    "predictions_csv_intervals": "lower/upper = y_pred -/+ 1.96 sd, the Gaussian 95% posterior interval of the GP (not jackknife+); empty for law-alone and rule rows",
    "n_fold_fits": n_jobs,
}

# =============================================================== 1. equivalence
leg = json.load(open(LEGACY_PHYS))
legmu = np.array(leg["loo_predictions_rms"]["physgp"]["mu"])
leg_gp_mu = np.array(leg["loo_predictions_rms"]["gp"]["mu"])
EQ = {}
for t in A.TARGETS:
    lf = A.loo_folds()
    mu5, _, _ = oof("physgp", t, lf)
    mu6, _, _ = oof("physgp6", t, lf)
    mu5l, _, _ = oof("physgp_ltol", t, lf)
    mu6l, _, _ = oof("physgp6_ltol", t, lf)
    blk = {
        "max_abs_diff_physgp5_vs_physgp6_tight_tol_V": f(np.max(np.abs(mu5 - mu6))),
        "max_abs_diff_physgp5_vs_physgp6_legacy_tol_V": f(np.max(np.abs(mu5l - mu6l))),
        "max_abs_diff_physgp5_tight_vs_physgp6_legacy_tol_V": f(
            np.max(np.abs(mu5 - mu6l))
        ),
        "loo_R2_physgp5_tight": f(r2_score(Y[t], mu5)),
        "loo_R2_physgp6_tight": f(r2_score(Y[t], mu6)),
        "loo_R2_physgp5_legacy_tol": f(r2_score(Y[t], mu5l)),
        "loo_R2_physgp6_legacy_tol": f(r2_score(Y[t], mu6l)),
    }
    # where the prediction gaps sit, and whether the two GP fits reached different optima there
    _, _, ex5 = oof("physgp", t, lf)
    _, _, ex6 = oof("physgp6", t, lf)
    _, _, ex6l = oof("physgp6_ltol", t, lf)
    for tag, a_, b_, ea, eb in [
        ("physgp5_vs_physgp6_tight_tol", mu5, mu6, ex5, ex6),
        ("physgp5_tight_vs_physgp6_legacy_tol", mu5, mu6l, ex5, ex6l),
    ]:
        d_ = np.abs(a_ - b_)
        i_ = int(np.argmax(d_))
        blk[f"argmax_condition_{tag}"] = i_
        blk[f"n_conditions_diff_above_1e-3V_{tag}"] = int(np.sum(d_ > 1e-3))
        blk[f"gp_log_marginal_likelihood_at_argmax_{tag}"] = [
            f(ea[i_]["gp_log_marginal_likelihood"]),
            f(eb[i_]["gp_log_marginal_likelihood"]),
        ]
    if t == "rms_Voc":
        gmu, _, _ = oof("gp", t, lf)
        blk["max_abs_diff_physgp5_tight_vs_legacy_json_V"] = f(
            np.max(np.abs(mu5 - legmu))
        )
        blk["max_abs_diff_physgp5_legacy_tol_vs_legacy_json_V"] = f(
            np.max(np.abs(mu5l - legmu))
        )
        blk["max_abs_diff_physgp6_legacy_tol_rerun_vs_legacy_json_V"] = f(
            np.max(np.abs(mu6l - legmu))
        )
        blk["max_abs_diff_gp_rerun_vs_legacy_json_V"] = f(
            np.max(np.abs(gmu - leg_gp_mu))
        )
        blk["legacy_json_loo_R2_physgp"] = f(leg["rms_Voc"]["physgp_LOO_R2"])
    # law-mean agreement and objective values over the 75 LOO training sets
    dmean, dsse, dmean_l = [], [], []
    for tr, te in lf:
        m5, _, _ = A.fit_law5(tr, Y[t])
        m6, _, _ = A.fit_law6(tr, Y[t])
        m5l, _, _ = A.fit_law5(tr, Y[t], tol=A.TOL_LEGACY)
        m6l, _, _ = A.fit_law6(tr, Y[t], tol=A.TOL_LEGACY)
        allr = np.arange(A.N)
        dmean.append(np.max(np.abs(m5(allr) - m6(allr))))
        dmean_l.append(np.max(np.abs(m5l(allr) - m6l(allr))))
        dsse.append(np.sum((Y[t][tr] - m5(tr)) ** 2) - np.sum((Y[t][tr] - m6(tr)) ** 2))
    blk["law_mean_max_abs_diff_tight_tol_V"] = f(np.max(dmean))
    blk["law_mean_max_abs_diff_legacy_tol_V"] = f(np.max(dmean_l))
    blk["law_sse_diff_5_minus_6_tight_tol_range_V2"] = [
        f(np.min(dsse)),
        f(np.max(dsse)),
    ]
    EQ[t] = blk
# sensitivity of the plain GP to a target perturbation of the size of the law-mean difference
_rng = np.random.default_rng(SEED)
_eps = EQ["rms_Voc"]["law_mean_max_abs_diff_tight_tol_V"]
_ypert = Y["rms_Voc"] + _rng.normal(0.0, _eps, A.N)


def _gp_pert(tr, te):
    return A.fold_predict("gp", tr, te, Y["rms_Voc"])[0][0] - A.fold_predict("gp", tr, te, _ypert)[0][0]


_d = Parallel(n_jobs=-1)(delayed(_gp_pert)(tr, te) for tr, te in A.loo_folds())
EQ["gp_optimizer_sensitivity_rms"] = {
    "perturbation_sd_V": f(_eps),
    "max_abs_change_gp_loo_prediction_V": f(np.max(np.abs(_d))),
    "method": "plain GP LOO on V_rms refit after adding N(0, sd^2) noise with sd equal to the tight-tolerance law-mean difference, seed 0",
}
# optimizer-restart check: PhysGP (and plain GP) LOO with n_restarts_optimizer = 4, random_state 0
RESTART_TOL_R2 = 0.005
RC = {"n_restarts_optimizer": A.RESTARTS_CHECK, "random_state": 0, "threshold_R2": RESTART_TOL_R2}
for t in A.TARGETS:
    lf = A.loo_folds()
    blk = {}
    for base, rm in [("gp", "gp_r4"), ("physgp", "physgp_r4"), ("physgp6", "physgp6_r4")]:
        mu0, _, _ = oof(base, t, lf)
        mur, _, _ = oof(rm, t, lf)
        r0, rr = r2_score(Y[t], mu0), r2_score(Y[t], mur)
        blk[base] = {
            "loo_R2_restarts0": f(r0),
            "loo_R2_restarts4": f(rr),
            "delta_R2": f(rr - r0),
            "moves_more_than_threshold": bool(abs(rr - r0) > RESTART_TOL_R2),
            "max_abs_pred_change_V": f(np.max(np.abs(mur - mu0))),
            "argmax_condition": int(np.argmax(np.abs(mur - mu0))),
        }
    mu5r, _, _ = oof("physgp_r4", t, lf)
    mu6r, _, _ = oof("physgp6_r4", t, lf)
    blk["max_abs_diff_physgp5_vs_physgp6_restarts4_V"] = f(np.max(np.abs(mu5r - mu6r)))
    RC[t] = blk
EQ["restart_check"] = RC
print("[a2] restart check:", {t: (RC[t]["physgp"]["loo_R2_restarts0"], RC[t]["physgp"]["loo_R2_restarts4"]) for t in A.TARGETS})

EQ["explanation"] = (
    "Both forms reach the same least-squares minimum: at tolerance 1e-15 the law means agree to the "
    "reported law_mean_max_abs_diff_tight_tol_V and the SSE difference is at rounding level. At the legacy "
    "curve_fit default tolerance (1e-8) both forms stop early at slightly different points; this, not a "
    "different optimum, is the source of differences of order 1e-6 V against the legacy JSON, and the legacy "
    "six-parameter code rerun at its own tolerance also differs from its stored JSON by a similar amount. "
    "The remaining difference between the two PhysGP forms at tight tolerance (about 1e-6 V) comes from the GP "
    "hyperparameter optimizer (L-BFGS-B at scikit-learn default tolerances), which moves plain-GP LOO predictions "
    "by an amount of the same order when the target is perturbed at the size of the law-mean difference "
    "(gp_optimizer_sensitivity_rms). For V_pp and |V|_max the law means also agree to about 1e-8 V, yet single "
    "LOO folds differ by up to 0.38 V (V_pp, forms at tolerance 1e-15) and 0.105 V (|V|_max, five-parameter "
    "tight versus six-parameter legacy tolerance). There the GP hyperparameter optimizer, started once "
    "(n_restarts_optimizer = 0), lands in different local optima of the marginal likelihood for residuals that "
    "are equal to about 1e-8 V; the log marginal likelihoods at the worst fold are recorded. restart_check "
    "repeats the LOO with four extra optimizer starts."
)
R["equivalence"] = EQ
print("[a2] equivalence rms:", {k: v for k, v in EQ["rms_Voc"].items() if "diff" in k})


# =============================================================== 2. identifiability
def num_jac(fn, p, xt, h=1e-7):
    J = np.zeros((len(xt[0]), len(p)))
    for j in range(len(p)):
        dp = np.zeros(len(p))
        dp[j] = h * max(1.0, abs(p[j]))
        J[:, j] = (fn(xt, *(p + dp)) - fn(xt, *(p - dp))) / (2 * dp[j])
    return J


def boot_one(t, b, m_hat, resid):
    rng = np.random.default_rng([SEED, b])
    ystar = m_hat + rng.choice(resid, size=len(resid), replace=True)
    _, pv, info = A.fit_law5(np.arange(A.N), ystar)
    return pv, bool(info["ok"])


def at_bound(pv_scaled):
    lo, hi = np.array(A.BOUNDS_5[0]), np.array(A.BOUNDS_5[1])
    p = np.asarray(pv_scaled)
    return (
        (np.abs(p - lo) < 1e-6 * (1 + np.abs(lo)))
        | (np.abs(p - hi) < 1e-6 * (1 + np.abs(hi)))
    ).tolist()


def fg_at_bound(P):
    """True where f0 or gamma of a volt-scale parameter row sits on its bound."""
    P = np.atleast_2d(P)
    lo, hi = np.array(A.BOUNDS_5[0][3:]), np.array(A.BOUNDS_5[1][3:])
    return np.any((np.abs(P[:, 3:] - lo) < 1e-6) | (np.abs(P[:, 3:] - hi) < 1e-6), axis=1)


ID = {}
BOOT_PARAMS = {}
FULL_FIT = {}
for t in A.TARGETS:
    y = Y[t]
    allr = np.arange(A.N)
    m, pv, info = A.fit_law5(allr, y)
    FULL_FIT[t] = pv
    cov = info["cov_volt"]
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    resid = y - m(allr)
    m_hat = m(allr)
    # six-parameter Jacobian rank deficiency, for contrast
    m6, p6, info6 = A.fit_law6(allr, y)
    xt = A._xt(allr)
    J5 = num_jac(A.law5, info["p_scaled"], xt)
    J6 = num_jac(A.law6, p6, xt)
    s5 = np.linalg.svd(J5, compute_uv=False)
    s6 = np.linalg.svd(J6, compute_uv=False)
    # residual bootstrap
    out = Parallel(n_jobs=-1)(
        delayed(boot_one)(t, b, m_hat, resid) for b in range(N_BOOT)
    )
    bp = np.array([o[0] for o in out])
    bok = np.array([o[1] for o in out])
    BOOT_PARAMS[t] = bp
    bse = bp.std(axis=0, ddof=1)
    bcorr = np.corrcoef(bp.T)
    pct = np.percentile(bp, [2.5, 25, 50, 75, 97.5], axis=0)
    # per-fold spread
    _, _, ex_loo = oof("physgp", t, A.loo_folds())
    loo_p = np.array([e["params"] for e in ex_loo])
    grp = {}
    grp_all = []
    for ax in A.AXES:
        folds = A.group_folds(ax)
        _, _, ex = oof("physgp", t, folds)
        P = np.array([e["params"] for e in ex])
        grp_all.append(P)
        grp[ax] = {
            "levels": [A.level_key(ax, fo[2]) for fo in folds],
            "params": {n: [f(v) for v in P[:, j]] for j, n in enumerate(A.PNAMES)},
            "fit_ok": [bool(e["ok"]) for e in ex],
            "f0_or_gamma_at_bound": [bool(v) for v in fg_at_bound(P)],
        }
    grp_p = np.vstack(grp_all)

    def spread(P):
        return {
            n: {
                "min": f(P[:, j].min()),
                "median": f(np.median(P[:, j])),
                "max": f(P[:, j].max()),
            }
            for j, n in enumerate(A.PNAMES)
        }

    ID[t] = {
        "full_grid": {
            "estimate": {n: f(pv[j]) for j, n in enumerate(A.PNAMES)},
            "se_jacobian": {n: f(se[j]) for j, n in enumerate(A.PNAMES)},
            "corr_jacobian": [[f(v) for v in row] for row in corr],
            "at_bound": dict(zip(A.PNAMES, at_bound(info["p_scaled"]))),
            "fit_ok": bool(info["ok"]),
            "scale_std_y": f(info["scale"]),
            "resid_sd_V": f(np.sqrt(np.sum(resid**2) / (A.N - 5))),
            "law_alone_in_sample_R2": f(r2_score(y, m_hat)),
            "Q_f0_over_gamma": f(pv[3] / pv[4]),
            "c_star_wt_pct_if_a2_neg": f(-pv[1] / (2 * pv[2])) if pv[2] < 0 else None,
            "jacobian_singular_values_5param_scaled": [f(v) for v in s5],
            "jacobian_condition_5param": f(s5[0] / s5[-1]),
            "jacobian_singular_values_6param_scaled_legacy": [f(v) for v in s6],
            "jacobian_condition_6param_legacy": f(s6[0] / s6[-1])
            if s6[-1] > 0
            else None,
            "legacy_6param_estimate_scaled": dict(
                zip(["b0", "b1", "b2", "A", "f0", "gamma"], [f(v) for v in p6])
            ),
        },
        "bootstrap": {
            "n_resamples": N_BOOT,
            "method": "residual bootstrap on the full grid, law refit from the fold start point, percentile CI",
            "n_fit_failures": int((~bok).sum()),
            "se": {n: f(bse[j]) for j, n in enumerate(A.PNAMES)},
            "ci95": {n: [f(pct[0, j]), f(pct[4, j])] for j, n in enumerate(A.PNAMES)},
            "percentiles_2.5_25_50_75_97.5": {
                n: [f(v) for v in pct[:, j]] for j, n in enumerate(A.PNAMES)
            },
            "corr": [[f(v) for v in row] for row in bcorr],
            "a2_negative_fraction": f(np.mean(bp[:, 2] < 0)),
            "resamples_volt_scale": [[f(v) for v in row] for row in bp],
            "resample_fit_ok": [bool(v) for v in bok],
        },
        "loo_folds": {
            "n": int(len(loo_p)),
            "spread": spread(loo_p),
            "params": {n: [f(v) for v in loo_p[:, j]] for j, n in enumerate(A.PNAMES)},
            "a2_negative_count": int(np.sum(loo_p[:, 2] < 0)),
            "a2_sign_stable": bool(np.all(loo_p[:, 2] < 0) or np.all(loo_p[:, 2] > 0)),
            "n_fit_failures": int(sum(not e["ok"] for e in ex_loo)),
            "n_f0_or_gamma_at_bound": int(np.sum(fg_at_bound(loo_p))),
        },
        "group_folds": {
            "n": int(len(grp_p)),
            "spread": spread(grp_p),
            "per_axis": grp,
            "per_axis_spread": {
                ax: spread(np.array([grp[ax]["params"][n] for n in A.PNAMES]).T)
                for ax in A.AXES
            },
            "a2_negative_count": int(np.sum(grp_p[:, 2] < 0)),
            "a2_sign_stable": bool(np.all(grp_p[:, 2] < 0) or np.all(grp_p[:, 2] > 0)),
            "n_f0_or_gamma_at_bound": int(np.sum(fg_at_bound(grp_p))),
        },
        "bootstrap_n_f0_or_gamma_at_bound": int(np.sum(fg_at_bound(bp))),
    }
R["identifiability"] = ID
R["identifiability_legacy_reference"] = json.load(open(LEGACY_REV))[
    "lorentzian_uncertainty"
]["global_law_rms"]
print("[a2] law params rms:", ID["rms_Voc"]["full_grid"]["estimate"])

# =============================================================== 3. per-level results
PL = {}
POOLED = {}
CSV_ROWS = []
Z95 = 1.959963984540054  # lower/upper = GP posterior mean -/+ 1.96 sd (Gaussian 95%); NaN where no sd


def add_csv(t, model, scheme, mu, sd, lvl_of, rows=None):
    for i in range(A.N) if rows is None else rows:
        CSV_ROWS.append(
            {
                "condition_id": i,
                "composition": A.COMP[i],
                "cnt_pct": A.CNT[i],
                "force_N": A.FRC[i],
                "freq_Hz": A.FRQ[i],
                "target": t,
                "model": model,
                "split_scheme": scheme,
                "held_out_level": lvl_of[i],
                "y_true": Y[t][i],
                "y_pred": mu[i],
                "sd": sd[i],
                "lower": mu[i] - Z95 * sd[i],
                "upper": mu[i] + Z95 * sd[i],
            }
        )


for t in A.TARGETS:
    PL[t], POOLED[t] = {}, {}
    y = Y[t]
    POOLED[t]["loo"] = {}
    for mdl in MAIN_MODELS:
        mu, sd, _ = oof(mdl, t, A.loo_folds())
        POOLED[t]["loo"][mdl] = {
            "R2": f(r2_score(y, mu)),
            "MAE": f(mean_absolute_error(y, mu)),
        }
        add_csv(t, mdl, "loo", mu, sd, [str(i) for i in range(A.N)])
    for ax in A.AXES:
        folds = A.group_folds(ax)
        PL[t][ax] = {}
        POOLED[t][ax] = {}
        for mdl in MAIN_MODELS + ABL_MODELS:
            mu, sd, _ = oof(mdl, t, folds)
            lv = {}
            for tr, te, v in folds:
                lv[A.level_key(ax, v)] = lvl_metrics(y[te], mu[te])
            PL[t][ax][mdl] = lv
            wl = [lv[k]["R2"] for k in lv]
            POOLED[t][ax][mdl] = {
                "R2": f(r2_score(y, mu)),
                "MAE": f(mean_absolute_error(y, mu)),
                "mean_within_level_R2": f(np.mean(wl)),
                "n_levels_R2_negative": int(np.sum(np.array(wl) < 0)),
            }
            lvl_of = [A.level_key(ax, v) for v in A.AXES[ax]]
            add_csv(t, mdl, f"held_out_{ax}", mu, sd, lvl_of)
R["per_level"] = PL
R["pooled"] = POOLED
R["per_level_note"] = (
    "within-level R2 uses the mean of the held-out level as reference (1 - SSE/SST within the level); "
    "pooled R2 uses all 75 out-of-fold predictions with the global mean (legacy definition). "
    "bias = mean(pred - y) within the level; pearson_r = correlation of pred and y within the level."
)

# =============================================================== 5. selection rules
SEL = {}
for t in A.TARGETS:
    y = Y[t]
    SEL[t] = {}
    for ax in A.AXES:
        folds = A.group_folds(ax)
        rows = []
        mu_rule = {
            k: np.full(A.N, np.nan)
            for k in ["inner_cv", "stability", "always_gp", "always_physgp", "oracle"]
        }
        for tr, te, v in folds:
            inner = A.group_folds(ax, idx=tr)
            inner_r2 = {}
            inner_levels = [A.level_key(ax, fo[2]) for fo in inner]
            for mdl in MAIN_MODELS:
                mu_in = np.full(A.N, np.nan)
                for itr, ite, _ in inner:
                    mu_in[ite] = cache.get(mdl, t, itr, ite)[0]
                inner_r2[mdl] = r2_score(y[tr], mu_in[tr])
            pick_cv = "physgp" if inner_r2["physgp"] > inner_r2["gp"] else "gp"
            inner_par = np.array(
                [cache.get("physgp", t, itr, ite)[2]["params"] for itr, ite, _ in inner]
            )
            f0_spread = float(inner_par[:, 3].max() - inner_par[:, 3].min())
            a2_const = bool(np.all(inner_par[:, 2] < 0) or np.all(inner_par[:, 2] > 0))
            stable = (f0_spread < STAB_F0_SPREAD_HZ) and a2_const
            pick_st = "physgp" if stable else "gp"
            outer = {mdl: cache.get(mdl, t, tr, te)[0] for mdl in MAIN_MODELS}
            outer_r2 = {mdl: r2_score(y[te], outer[mdl]) for mdl in MAIN_MODELS}
            best = max(MAIN_MODELS, key=lambda k: (outer_r2[k], k == "gp"))
            for key, mdl in [
                ("inner_cv", pick_cv),
                ("stability", pick_st),
                ("always_gp", "gp"),
                ("always_physgp", "physgp"),
                ("oracle", best),
            ]:
                mu_rule[key][te] = outer[mdl]
            rows.append(
                {
                    "held_out_level": A.level_key(ax, v),
                    "n_outer_train": int(len(tr)),
                    "n_inner_folds": int(len(inner)),
                    "inner_levels": inner_levels,
                    "inner_pooled_R2": {k: f(v_) for k, v_ in inner_r2.items()},
                    "pick_inner_cv": pick_cv,
                    "inner_f0": [f(v_) for v_ in inner_par[:, 3]],
                    "inner_a2": [f(v_) for v_ in inner_par[:, 2]],
                    "inner_f0_spread_Hz": f(f0_spread),
                    "inner_a2_sign_constant": a2_const,
                    "pick_stability": pick_st,
                    "outer_within_level_R2": {k: f(v_) for k, v_ in outer_r2.items()},
                    "outer_MAE": {
                        mdl: f(mean_absolute_error(y[te], outer[mdl]))
                        for mdl in MAIN_MODELS
                    },
                    "realized_R2_inner_cv": f(outer_r2[pick_cv]),
                    "realized_MAE_inner_cv": f(
                        mean_absolute_error(y[te], outer[pick_cv])
                    ),
                    "realized_R2_stability": f(outer_r2[pick_st]),
                    "realized_MAE_stability": f(
                        mean_absolute_error(y[te], outer[pick_st])
                    ),
                    "oracle_pick": best,
                }
            )
        summ = {}
        for key, mu in mu_rule.items():
            wl = [r2_score(y[te], mu[te]) for _, te, _ in folds]
            summ[key] = {
                "pooled_R2": f(r2_score(y, mu)),
                "pooled_MAE": f(mean_absolute_error(y, mu)),
                "mean_within_level_R2": f(np.mean(wl)),
            }
        for key in ["inner_cv", "stability"]:
            summ[key]["n_pick_physgp"] = int(
                sum(r[f"pick_{key}"] == "physgp" for r in rows)
            )
            summ[key]["n_match_oracle"] = int(
                sum(r[f"pick_{key}"] == r["oracle_pick"] for r in rows)
            )
        summ["n_outer_folds"] = len(rows)
        SEL[t][ax] = {"per_outer_fold": rows, "summary": summ}
        for key in ["inner_cv", "stability"]:
            add_csv(
                t,
                f"rule_{key}",
                f"held_out_{ax}",
                mu_rule[key],
                np.full(A.N, np.nan),
                [A.level_key(ax, v) for v in A.AXES[ax]],
            )
R["selection_rule"] = SEL
for ax in A.AXES:
    s = SEL["rms_Voc"][ax]["summary"]
    print(
        f"[a2] selection rms {ax}: inner_cv picks physgp {s['inner_cv']['n_pick_physgp']}/{s['n_outer_folds']}, "
        f"stability {s['stability']['n_pick_physgp']}/{s['n_outer_folds']}, pooled R2 rule {s['inner_cv']['pooled_R2']:.3f} "
        f"gp {s['always_gp']['pooled_R2']:.3f} physgp {s['always_physgp']['pooled_R2']:.3f}"
    )

# =============================================================== 6. sensitivity, n = 72
DUP = {
    "excluded_condition_ids": DUP_IDS,
    "reason": "duplicated-recordings note in the README: recordings 52 and 54 are exact scalar multiples of 47 and 49 (2 wt% CNT, 1 N), 73 is a near copy of 5",
    "n": int(len(KEEP)),
    "loo": {},
    "pooled": {},
    "per_level": {},
}
for t in A.TARGETS:
    y = Y[t]
    DUP["loo"][t], DUP["pooled"][t], DUP["per_level"][t] = {}, {}, {}
    lf = A.loo_folds(KEEP)
    for mdl in MAIN_MODELS:
        mu, sd, _ = oof(mdl, t, lf)
        DUP["loo"][t][mdl] = {"R2": f(r2_score(y[KEEP], mu[KEEP])), "MAE": f(mean_absolute_error(y[KEEP], mu[KEEP])), "n": int(len(KEEP))}
        add_csv(t, mdl, "loo_excl_dup", mu, sd, [str(i) for i in range(A.N)], rows=KEEP)
    for ax in A.AXES:
        folds = A.group_folds(ax, idx=KEEP)
        DUP["pooled"][t][ax], DUP["per_level"][t][ax] = {}, {}
        for mdl in MAIN_MODELS + (["law", "gp_linF"] if ax == "force" else []):
            mu, sd, _ = oof(mdl, t, folds)
            lv = {A.level_key(ax, v): lvl_metrics(y[te], mu[te]) for tr, te, v in folds}
            wl = [lv[k]["R2"] for k in lv]
            DUP["per_level"][t][ax][mdl] = lv
            DUP["pooled"][t][ax][mdl] = {
                "R2": f(r2_score(y[KEEP], mu[KEEP])),
                "MAE": f(mean_absolute_error(y[KEEP], mu[KEEP])),
                "mean_within_level_R2": f(np.mean(wl)),
                "n_levels_R2_negative": int(np.sum(np.array(wl) < 0)),
                "n": int(len(KEEP)),
            }
            add_csv(t, mdl, f"held_out_{ax}_excl_dup", mu, sd, [A.level_key(ax, v) for v in A.AXES[ax]], rows=KEEP)
R["sensitivity_excluding_duplicates"] = DUP
print("[a2] n=72 force pooled R2 rms:", {m: round(DUP["pooled"]["rms_Voc"]["force"][m]["R2"], 3) for m in DUP["pooled"]["rms_Voc"]["force"]})

# legacy cross-check of per-level force numbers (same model, legacy tolerance differs)
legrev = json.load(open(LEGACY_REV))["per_fold_force"]
R["legacy_crosscheck_force_rms"] = {
    mdl: {
        "legacy_within_level_R2": [
            f(r["within_fold_R2"]) for r in legrev["rms_Voc"][mdl]["per_fold"]
        ],
        "a2_within_level_R2": [
            PL["rms_Voc"]["force"][mdl][k]["R2"] for k in ["1", "2", "3"]
        ],
    }
    for mdl in MAIN_MODELS
}

# =============================================================== outputs
pd.DataFrame(CSV_ROWS).to_csv(OUT_CSV, index=False)
RPT.write_all(R, BOOT_PARAMS, FULL_FIT)  # tables, macros, figures, captions, index
json.dump(R, open(OUT_JSON, "w"), indent=1)
print(
    f"[a2] wrote {OUT_JSON}, {OUT_CSV}; {len(R['index'])} index entries; {time.time() - T0:.1f} s"
)

