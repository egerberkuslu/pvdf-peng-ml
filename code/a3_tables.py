#!/usr/bin/env python3
"""Task A3: benchmark, coverage, active-learning and replication tables.

One command regenerates every A3 artifact from the design table and the
legacy result files:

    python code/a3_tables.py [--use-cache]

Outputs
  results/tables/tab_bench_{hyperparams,main,all,significance}.tex
  results/tables/tab_cov_{within,groupwise,groupwise_raw,mcnemar}.tex
  results/tables/tab_al_audit.tex
  results/tables/tab_rep_{public,subsample}.tex
  results/numbers_a3.tex             macros, prefixes bench and cov
  results/a3_tables.json             every number with its source
  results/a3_groupwise_coverage.json held-out-level and within-grid McNemar
  results/a3_predictions.csv         per-condition predictions and intervals
  figures/fig_bench_{models,calibration}.pdf/.png
  figures/fig_cov_tradeoff.pdf/.png
  figures/captions_a3.md
  results/a3_manifest.json
--use-cache reuses the model fits in results/a3_cache when the code
that determines them is unchanged (the cache key hashes that code and the data).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_bench as BN  # noqa: E402
import a3_common as C  # noqa: E402
import a3_conformal as K  # noqa: E402
import a3_sensitivity as S  # noqa: E402

FAM_T = {
    "raw_gp": r"Raw GP $\pm z\sigma$",
    "jackknife_plus": "Jackknife+",
    "scaled_jackknife_plus": r"$\sigma$-scaled jackknife+",
}
FAM_LEG = {
    "raw_gp": "raw GP ± zσ",
    "jackknife_plus": "jackknife+",
    "scaled_jackknife_plus": "σ-scaled jackknife+",
}
FAM_C = {"raw_gp": "Raw", "jackknife_plus": "Jk", "scaled_jackknife_plus": "ScaledJk"}
LVL_C = {"90": "Ninety", "95": "NinetyFive"}
AX_C = {"composition": "Composition", "force": "Force", "frequency": "Frequency"}
AX_T = {
    "composition": "composition",
    "force": "force level",
    "frequency": "frequency level",
}
MOD_C = {"gp": "Gp", "physgp": "Phys"}
MOD_T = {"gp": "plain GP", "physgp": "PhysGP"}
M_CAMEL = {
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
}

MACROS = []  # (name, value, json_key, source)


def mac(name, value, key, source):
    assert not any(ch.isdigit() for ch in name), name
    assert name.startswith(("bench", "cov")), name
    MACROS.append((name, value, key, source))


# ====================================================================== benchmark
def do_bench(J):
    H = BN.hyperparams()
    J["hyperparameters"] = H
    C.write_text(C.TAB / "tab_bench_hyperparams.tex", BN.hyperparam_table(H))
    B = BN.bench_numbers()
    J["benchmark"] = B
    tex, best = BN.main_table(B)
    C.write_text(C.TAB / "tab_bench_main.tex", tex)
    C.write_text(C.TAB / "tab_bench_all.tex", BN.all_table(B))
    C.write_text(C.TAB / "tab_bench_significance.tex", BN.sig_table(B))
    J["benchmark_best_by_cv_r2"] = best
    mac("benchNModels", str(len(BN.ORDER)), "hyperparameters.models", "code/reg_common.py")
    for t in C.TARGETS:
        T = C.CAMEL_T[t]
        for m in BN.ORDER:
            r = B[t]["models"][m]
            M = M_CAMEL[m]
            src = r["cv_r2_source"]
            mac(
                f"benchLoo{T}{M}",
                C.fnum(r["cv_r2"], 3),
                f"benchmark.{t}.models.{m}.cv_r2",
                src,
            )
            mac(
                f"benchIns{T}{M}",
                C.fnum(r["in_sample_r2"], 2),
                f"benchmark.{t}.models.{m}.in_sample_r2",
                "reg_models.json",
            )
            mac(
                f"benchMae{T}{M}",
                C.fnum(r["cv_mae"], 3),
                f"benchmark.{t}.models.{m}.cv_mae",
                "reg_models.json",
            )
            if "boot_ci95" in r:
                lo, hi = r["boot_ci95"]
                mac(
                    f"benchCiLo{T}{M}",
                    C.fnum(lo, 2),
                    f"benchmark.{t}.models.{m}.boot_ci95",
                    src,
                )
                mac(
                    f"benchCiHi{T}{M}",
                    C.fnum(hi, 2),
                    f"benchmark.{t}.models.{m}.boot_ci95",
                    src,
                )
            if "wilcoxon_p_vs_best" in r:
                mac(
                    f"benchP{T}{M}",
                    BN._p(r["wilcoxon_p_vs_best"]),
                    f"benchmark.{t}.models.{m}.wilcoxon_p_vs_best",
                    src,
                )
        mt = B[t]["mlp_nested_tuned"]
        mac(
            f"benchLoo{T}MlpTuned",
            C.fnum(mt["tuned_nestedCV_R2"], 3),
            f"benchmark.{t}.mlp_nested_tuned.tuned_nestedCV_R2",
            "mlp_tuned_results.json",
        )
        mac(
            f"benchLoo{T}SvrTuned",
            C.fnum(B[t]["svr_tuned"]["cv_r2"], 3),
            f"benchmark.{t}.svr_tuned.cv_r2",
            "reg_models.json",
        )
    # GP |V|max initialization sensitivity (two legacy runs of the same model)
    g = B["Vmax"]["models"]["ARD-GP"]
    J["gp_vmax_init_sensitivity"] = {
        "reg_stats_cv_r2_init_1111": g["cv_r2"],
        "reg_models_cv_r2_init_1115": g["cv_r2_reg_models"],
        "note": "same ARD-GP, different Matern length-scale initialization;"
        " reg_stats.json is canonical per the task card",
    }
    mac(
        "benchLooVmaxGpAltInit",
        C.fnum(g["cv_r2_reg_models"], 3),
        "gp_vmax_init_sensitivity.reg_models_cv_r2_init_1115",
        "reg_models.json",
    )

    # LOO predictions of the best model per target (recomputed, canonical factories)
    L = BN.recompute_loo_best(B)
    J["calibration_loo_recomputed"] = L
    for t in C.TARGETS:
        mac(
            f"benchRecompLoo{C.CAMEL_T[t]}{M_CAMEL[L[t]['model']]}",
            C.fnum(L[t]["loo_r2_recomputed"], 3),
            f"calibration_loo_recomputed.{t}.loo_r2_recomputed",
            "a3_tables.json",
        )
    return B, L


# ====================================================================== coverage
def do_within(J, W):
    """Within-grid table from legacy JSON, checked against the recomputation."""
    CAL = C.load_legacy("calibrated_conformal_results.json")
    BCI = C.load_legacy("revision_experiments.json")["coverage_binomial_ci"]
    within = {}
    check = {}
    for t in C.TARGETS:
        within[t] = {}
        for f in FAM_T:
            within[t][f] = {}
            for a in C.LEVELS:
                leg = CAL[t][f][a]
                b = BCI[t][f"{f}_{a}"]
                rec = {
                    "coverage": leg["coverage"],
                    "hits": b["hits"],
                    "n": 75,
                    "ci95": b["ci95"],
                    "ci_method": "Clopper-Pearson exact, 95%",
                    "mean_width": leg["mean_width"],
                    "median_width": leg["median_width"],
                    "source": "calibrated_conformal_results.json (coverage, width);"
                    " revision_experiments.json coverage_binomial_ci (hits, CI)",
                }
                # recheck CI with our own Clopper-Pearson and scipy's exact interval
                from scipy.stats import binomtest

                ci_s = binomtest(b["hits"], 75).proportion_ci(0.95, method="exact")
                assert (
                    abs(ci_s.low - b["ci95"][0]) < 1e-9
                    and abs(ci_s.high - b["ci95"][1]) < 1e-9
                )
                within[t][f][a] = rec
                rc = W[t][f][a]
                check[f"{t}.{f}.{a}"] = {
                    "legacy_hits": b["hits"],
                    "recomputed_hits": rc["hits"],
                    "legacy_mean_width": leg["mean_width"],
                    "recomputed_mean_width": rc["mean_width"],
                    "hits_equal": b["hits"] == rc["hits"],
                    "width_rel_diff": abs(rc["mean_width"] - leg["mean_width"])
                    / leg["mean_width"],
                }
    J["coverage_within_grid"] = within
    J["coverage_within_grid_recompute_check"] = check

    L = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Nominal $90\%$} & \multicolumn{2}{c}{Nominal $95\%$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Target & Interval & Coverage (hits/$n$) [$95\%$ CI] & Width (V) & Coverage (hits/$n$) [$95\%$ CI] & Width (V) \\",
        r"\midrule",
    ]
    for k, t in enumerate(C.TARGETS):
        for i, f in enumerate(FAM_T):
            first = rf"\multirow{{3}}{{*}}{{{C.TEX_T[t]}}}" if i == 0 else ""
            c = []
            for a in C.LEVELS:
                r = within[t][f][a]
                c += [C.cov_cell(r), C.fnum(r["mean_width"], C.width_nd(t))]
            L.append(f"{first} & {FAM_T[f]} & " + " & ".join(c) + r" \\")
        if k < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_cov_within.tex", "\n".join(L))

    mac("covN", "75", "coverage_within_grid.rms_Voc.jackknife_plus.95.n", "design")
    for t in C.TARGETS:
        T = C.CAMEL_T[t]
        for f in FAM_T:
            for a in C.LEVELS:
                r = within[t][f][a]
                base = f"covWithin{FAM_C[f]}{LVL_C[a]}{T}"
                key = f"coverage_within_grid.{t}.{f}.{a}"
                mac(
                    base,
                    C.fpct(r["coverage"]),
                    key,
                    "calibrated_conformal_results.json",
                )
                mac(base + "Hits", str(r["hits"]), key, "revision_experiments.json")
                mac(
                    base + "CiLo",
                    C.fpct(r["ci95"][0]),
                    key,
                    "revision_experiments.json",
                )
                mac(
                    base + "CiHi",
                    C.fpct(r["ci95"][1]),
                    key,
                    "revision_experiments.json",
                )
                mac(
                    base + "Width",
                    C.fnum(r["mean_width"], C.width_nd(t)),
                    key,
                    "calibrated_conformal_results.json",
                )
        prem = (
            within[t]["jackknife_plus"]["95"]["mean_width"]
            / within[t]["raw_gp"]["95"]["mean_width"]
            - 1
        )
        J.setdefault("width_premium_jk_vs_raw_95", {})[t] = prem
        mac(
            f"covWidthPremium{T}",
            C.fpct(prem),
            f"width_premium_jk_vs_raw_95.{t}",
            "calibrated_conformal_results.json",
        )
    mac(
        "covJkRms",
        C.fpct(within["rms_Voc"]["jackknife_plus"]["95"]["coverage"]),
        "coverage_within_grid.rms_Voc.jackknife_plus.95",
        "calibrated_conformal_results.json",
    )
    return within


def do_mcnemar(J, W, G):
    GLEG = C.load_legacy("groupwise_conformal.json")
    mc = {t: W[t]["mcnemar_raw_vs_jackknife_plus"] for t in C.TARGETS}
    J["mcnemar_within_grid"] = mc
    J["mcnemar_legacy_rms"] = GLEG["mcnemar_rms"]
    L = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Target & Nominal & Raw GP hits & Jackknife+ hits & Jackknife+ only & Raw only & Exact McNemar $p$ \\",
        r"\midrule",
    ]
    for k, t in enumerate(C.TARGETS):
        if k > 0:
            L.append(r"\midrule")
        for i, a in enumerate(C.LEVELS):
            r = mc[t][a]
            first = rf"\multirow{{2}}{{*}}{{{C.TEX_T[t]}}}" if i == 0 else ""
            L.append(
                f"{first} & ${a}\\%$ & {W[t]['raw_gp'][a]['hits']}/75 & "
                f"{W[t]['jackknife_plus'][a]['hits']}/75 & {r['jk_only_hits_b']} & "
                f"{r['raw_only_hits_c']} & {C.fnum(r['exact_two_sided_p'], 3)}" + r" \\"
            )
            T = C.CAMEL_T[t]
            key = f"mcnemar_within_grid.{t}.{a}"
            mac(
                f"covMcnemarP{LVL_C[a]}{T}",
                C.fnum(r["exact_two_sided_p"], 3),
                key,
                "a3_groupwise_coverage.json",
            )
            mac(
                f"covMcnemarJkOnly{LVL_C[a]}{T}",
                str(r["jk_only_hits_b"]),
                key,
                "a3_groupwise_coverage.json",
            )
            mac(
                f"covMcnemarRawOnly{LVL_C[a]}{T}",
                str(r["raw_only_hits_c"]),
                key,
                "a3_groupwise_coverage.json",
            )
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_cov_mcnemar.tex", "\n".join(L))


def do_groupwise(J, G):
    GLEG = C.load_legacy("groupwise_conformal.json")["groupwise_jackknife_rms"]
    check = {}
    for ax in C.AXES:
        for a in C.LEVELS:
            leg = GLEG[ax][a]
            rc = G["rms_Voc"]["gp"][ax]["jackknife_plus"][a]
            check[f"rms_Voc.gp.{ax}.{a}"] = {
                "legacy_coverage": leg["coverage"],
                "recomputed_coverage": rc["coverage"],
                "legacy_mean_width": leg["mean_width"],
                "recomputed_mean_width": rc["mean_width"],
                "coverage_equal": abs(leg["coverage"] - rc["coverage"]) < 1e-12,
            }
    J["coverage_groupwise"] = G
    J["coverage_groupwise_legacy_check"] = check

    def table(fam, fname):
        L = [
            r"\begin{tabular}{lllcccc}",
            r"\toprule",
            r" & & & \multicolumn{2}{c}{Nominal $90\%$} & \multicolumn{2}{c}{Nominal $95\%$} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Target & Held-out & Model & Coverage (hits/$n$) [$95\%$ CI] & Width (V) & Coverage (hits/$n$) [$95\%$ CI] & Width (V) \\",
            r"\midrule",
        ]
        for k, t in enumerate(C.TARGETS):
            first_t = True
            for ax in C.AXES:
                for mi, m in enumerate(("gp", "physgp")):
                    tcell = rf"\multirow{{6}}{{*}}{{{C.TEX_T[t]}}}" if first_t else ""
                    first_t = False
                    acell = rf"\multirow{{2}}{{*}}{{{AX_T[ax]}}}" if mi == 0 else ""
                    c = []
                    for a in C.LEVELS:
                        r = G[t][m][ax][fam][a]
                        c += [C.cov_cell(r), C.fnum(r["mean_width"], C.width_nd(t))]
                    L.append(
                        f"{tcell} & {acell} & {MOD_T[m]} & " + " & ".join(c) + r" \\"
                    )
            if k < 2:
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}", ""]
        C.write_text(C.TAB / fname, "\n".join(L))

    table("jackknife_plus", "tab_cov_groupwise.tex")
    table("raw", "tab_cov_groupwise_raw.tex")
    for t in C.TARGETS:
        T = C.CAMEL_T[t]
        for m in ("gp", "physgp"):
            for ax in C.AXES:
                for fam, pre in (
                    ("jackknife_plus", "covGroup"),
                    ("raw", "covGroupRaw"),
                ):
                    for a in C.LEVELS:
                        r = G[t][m][ax][fam][a]
                        base = f"{pre}{MOD_C[m]}{AX_C[ax]}{LVL_C[a]}{T}"
                        key = f"coverage_groupwise.{t}.{m}.{ax}.{fam}.{a}"
                        mac(
                            base,
                            C.fpct(r["coverage"]),
                            key,
                            "a3_groupwise_coverage.json",
                        )
                        mac(
                            base + "Width",
                            C.fnum(r["mean_width"], C.width_nd(t)),
                            key,
                            "a3_groupwise_coverage.json",
                        )
                        if fam == "jackknife_plus":
                            mac(
                                base + "Hits",
                                str(r["hits"]),
                                key,
                                "a3_groupwise_coverage.json",
                            )
                            mac(
                                base + "CiLo",
                                C.fpct(r["ci95"][0]),
                                key,
                                "a3_groupwise_coverage.json",
                            )
                            mac(
                                base + "CiHi",
                                C.fpct(r["ci95"][1]),
                                key,
                                "a3_groupwise_coverage.json",
                            )


# ====================================================================== active learning
def do_al(J):
    AL = C.load_legacy("activeL_results.json")
    SO = C.load_legacy("revision_experiments.json")["stopping_and_ofat"]
    y = C.yv("rms_Voc")
    ei_c = np.array(AL["EI"]["counts"])
    rd_c = np.array(AL["random"]["counts"])
    # oracle runs stop the moment the best condition is queried; a count below the
    # pool size (75) certifies that it was found, the random order always reaches it
    ei_found = int(np.sum(ei_c < C.N))
    rd_found = int(np.sum(rd_c <= C.N))
    ofat = SO["ofat"]
    eis = SO["ei_stopping"]
    n_stop = 30
    k_stop = int(round(eis["success_rate"] * n_stop))
    rows = {
        "ei_oracle": {
            "label": "EI, oracle stop",
            "runs": AL["n_seeds"],
            "median": AL["EI"]["median"],
            "iqr": [AL["EI"]["q25"], AL["EI"]["q75"]],
            "success_hits": ei_found,
            "success_n": AL["n_seeds"],
            "regret_median_V": 0.0,
            "regret_note": "zero by construction, the run stops when the best condition is queried",
            "source": "activeL_results.json EI",
        },
        "ei_realizable": {
            "label": "EI, realizable stop",
            "runs": n_stop,
            "median": eis["budget_median"],
            "iqr": eis["budget_iqr"],
            "success_hits": k_stop,
            "success_n": n_stop,
            "regret_median_V": eis["regret_median"],
            "regret_max_V": eis["regret_max"],
            "rule": eis["rule"],
            "source": "revision_experiments.json stopping_and_ofat.ei_stopping",
        },
        "random": {
            "label": "Random sampling, oracle stop",
            "runs": AL["n_seeds"],
            "median": AL["random"]["median"],
            "iqr": [AL["random"]["q25"], AL["random"]["q75"]],
            "success_hits": rd_found,
            "success_n": AL["n_seeds"],
            "regret_median_V": 0.0,
            "regret_note": "zero by construction, sampling continues until the best condition is drawn",
            "source": "activeL_results.json random",
        },
        "ofat": {
            "label": "OFAT sweep",
            "runs": len(ofat["runs"]),
            "median": ofat["n_experiments_median"],
            "iqr": [
                float(np.percentile([r["n_experiments"] for r in ofat["runs"]], 25)),
                float(np.percentile([r["n_experiments"] for r in ofat["runs"]], 75)),
            ],
            "success_hits": int(sum(r["found_true_best"] for r in ofat["runs"])),
            "success_n": len(ofat["runs"]),
            "regret_median_V": ofat["regret_median"],
            "source": "revision_experiments.json stopping_and_ofat.ofat",
        },
    }
    for r in rows.values():
        r["success_rate"] = r["success_hits"] / r["success_n"]
        r["success_ci95"] = C.clopper(r["success_hits"], r["success_n"])
    assert abs(rows["ei_realizable"]["success_rate"] - eis["success_rate"]) < 1e-12
    meta = {
        "initial_design": AL["n0"],
        "pool": C.N,
        "seeds": AL["n_seeds"],
        "optimum_condition": AL["optimum_condition"],
        "target": "rms_Voc",
        "best_value_V": float(y.max()),
        "experiments_count_includes_initial_design": True,
        "regret_units": "V (V_rms), best measured minus best found",
    }
    J["active_learning"] = {"rows": rows, "meta": meta}

    def fmt_n(v):
        v = float(v)
        return str(int(v)) if v.is_integer() else f"{v:g}"

    L = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Strategy & Runs & Median experiments & IQR & Success (hits/$n$) & $95\%$ CI & Median regret (V) \\",
        r"\midrule",
    ]
    for key in ("ei_oracle", "ei_realizable", "random", "ofat"):
        r = rows[key]
        lo, hi = r["success_ci95"]
        L.append(
            f"{r['label']} & {r['runs']} & {fmt_n(r['median'])} & "
            f"{fmt_n(r['iqr'][0])}--{fmt_n(r['iqr'][1])} & "
            rf"{C.fpct(r['success_rate'])}\% ({r['success_hits']}/{r['success_n']}) & "
            rf"[{C.fpct(lo)}, {C.fpct(hi)}] & {C.fnum(r['regret_median_V'], 3)}"
            + r" \\"
        )
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_al_audit.tex", "\n".join(L))
    P = {
        "ei_oracle": "EiOracle",
        "ei_realizable": "EiStop",
        "random": "Random",
        "ofat": "Ofat",
    }
    for k, r in rows.items():
        b = f"benchAl{P[k]}"
        key = f"active_learning.rows.{k}"
        mac(b + "Median", fmt_n(r["median"]), key, r["source"])
        mac(b + "QLo", fmt_n(r["iqr"][0]), key, r["source"])
        mac(b + "QHi", fmt_n(r["iqr"][1]), key, r["source"])
        mac(b + "Success", C.fpct(r["success_rate"]), key, r["source"])
        mac(b + "SuccessHits", str(r["success_hits"]), key, r["source"])
        mac(b + "Runs", str(r["runs"]), key, r["source"])
        mac(b + "Regret", C.fnum(r["regret_median_V"], 3), key, r["source"])
    mac(
        "benchAlInitial",
        str(meta["initial_design"]),
        "active_learning.meta.initial_design",
        "activeL_results.json",
    )
    mac(
        "benchAlEiStopRegretMax",
        C.fnum(rows["ei_realizable"]["regret_max_V"], 3),
        "active_learning.rows.ei_realizable.regret_max_V",
        "revision_experiments.json",
    )


# ====================================================================== replication
PUB = [
    (
        "energy_efficiency",
        r"Energy Efficiency \cite{tsanasAccurateQuantitativeEstimation2012}",
        "Energy Efficiency",
    ),
    (
        "airfoil",
        r"Airfoil Self-Noise \cite{brooksAirfoilSelfnoisePrediction1989}",
        "Airfoil Self-Noise",
    ),
    (
        "concrete",
        r"Concrete Strength \cite{yehModelingStrengthHigh1998}",
        "Concrete Strength",
    ),
    (
        "yacht",
        r"Yacht Hydrodynamics \cite{gerritsmaGeometryResistanceStability1981}",
        "Yacht Hydrodynamics",
    ),
    (
        "ccpp_2000sub",
        r"Power Plant \cite{tufekciPredictionFullLoad2014}",
        "Power Plant",
    ),
]
GROUP_NAME = {
    "chord": "chord",
    "velocity": "velocity",
    "compactness": "compactness",
    "glazing": "glazing",
    "binder": "binder",
    "age": "curing age",
    "hull": "hull",
    "froude": "Froude number",
    "AT_band": "temperature band",
}
REP_C = {
    "energy_efficiency": "EnergyEfficiency",
    "airfoil": "Airfoil",
    "concrete": "Concrete",
    "yacht": "Yacht",
    "ccpp_2000sub": "PowerPlant",
}


def do_rep(J, B):
    EXT = C.load_legacy("protocol_replay_extended.json")
    REX = C.load_legacy("revision_experiments.json")
    WN = REX["fullsize_width_normalized"]
    SUB = REX["subsample75_replication"]
    PG = C.load_legacy("physgp_results.json")["rms_Voc"]
    CAL = C.load_legacy("calibrated_conformal_results.json")["rms_Voc"]
    AL = C.load_legacy("activeL_results.json")
    RM = C.load_legacy("reg_models.json")["targets"]["rms"]["models"]["ARD-GP"]
    y = C.yv("rms_Voc")
    gw = {ax: PG[f"gp_leave_one_{ax}_out_R2"] for ax in C.AXES}
    worst_ax = min(gw, key=gw.get)
    peng = {
        "n": 75,
        "in_sample_R2": RM["in_sample_r2"],
        "recordwise_R2": B["rms_Voc"]["models"]["ARD-GP"]["cv_r2"],
        "worst_groupwise": {"R2": gw[worst_ax], "factor": worst_ax},
        "raw_cov95": CAL["raw_gp"]["95"]["coverage"],
        "jk_cov95": CAL["jackknife_plus"]["95"]["coverage"],
        "jk_width95_over_sd": CAL["jackknife_plus"]["95"]["mean_width"]
        / float(np.std(y)),
        "target_sd_ddof0": float(np.std(y)),
        "al_ei_median": AL["EI"]["median"],
        "al_random_median": AL["random"]["median"],
        "sources": {
            "in_sample_R2": "reg_models.json",
            "recordwise_R2": "reg_stats.json",
            "worst_groupwise": "physgp_results.json gp_leave_one_*_out_R2",
            "coverage": "calibrated_conformal_results.json",
            "width_over_sd": "calibrated_conformal_results.json width / np.std(rms_Voc)",
            "al": "activeL_results.json",
        },
    }
    pub = {}
    for key, _, _ in PUB:
        e = EXT[key]
        grp = {
            k[len("leave_one_") : -len("_out_R2")]: v
            for k, v in e.items()
            if k.startswith("leave_one_") and k.endswith("_out_R2")
        }
        wk = min(grp, key=grp.get)
        pub[key] = {
            "n": e["n"],
            "in_sample_R2": e["in_sample_R2"],
            "recordwise_R2": e["recordwise_10fold_R2"],
            "worst_groupwise": {"R2": grp[wk], "factor": wk},
            "raw_cov95": e["gp_raw_coverage95"],
            "jk_cov95": e["gp_cvplus"]["95"]["coverage"],
            "jk_width95_over_sd": WN[key]["width_over_sd"],
            "al_ei_median": e["active_learning_top1pct"]["EI_median"],
            "al_random_median": e["active_learning_top1pct"]["random_median"],
            "sources": "protocol_replay_extended.json; width from revision_experiments.json"
            " fullsize_width_normalized",
        }
    J["replication_public"] = {"peng": peng, "public": pub}

    def row(lab, r):
        wg = r["worst_groupwise"]
        return (
            f"{lab} & {r['n']} & {C.fnum(r['in_sample_R2'], 2)} & {C.fnum(r['recordwise_R2'], 3)} & "
            f"{C.fnum(wg['R2'], 2)} ({GROUP_NAME.get(wg['factor'], wg['factor'])}) & "
            rf"{C.fpct(r['raw_cov95'])}\% & {C.fpct(r['jk_cov95'])}\% & "
            f"{C.fnum(r['jk_width95_over_sd'], 2)} & "
            f"{int(C.rhu(r['al_ei_median']))} ({int(C.rhu(r['al_random_median']))})"
            + r" \\"
        )

    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"Dataset & $N$ & \shortstack{In-sample\\$R^2$} & \shortstack{Record-wise\\CV $R^2$} & "
        r"\shortstack{Worst group-wise\\$R^2$} & \shortstack{Raw GP cov.\\@95\%} & "
        r"\shortstack{J+ cov.\\@95\%} & \shortstack{J+ width\\$/\sigma_y$} & "
        r"\shortstack{AL median\\(rand.)} \\",
        r"\midrule",
        row("PENG (this work)", peng),
    ]
    for key, lab, _ in PUB:
        L.append(row(lab, pub[key]))
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_rep_public.tex", "\n".join(L))

    subs = {k: SUB[k] for k, _, _ in PUB}
    J["replication_subsample75"] = {
        "rows": subs,
        "design": "10 random subsamples of 75 records per public set (numpy default_rng(1)),"
        " 10-fold GP with CV+ at nominal 95%, per revision_experiments.py small_sample_run",
        "source": "revision_experiments.json subsample75_replication",
    }
    L = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Dataset & Pool size & Median CV $R^2$ [IQR] & CV+ coverage @95\%, median [range] & CV+ width$/\sigma_y$, median \\",
        r"\midrule",
    ]
    for key, lab, _ in PUB:
        r = subs[key]
        lo, hi = r["R2_iqr"]
        clo, chi = r["cvplus95_cov_range"]
        L.append(
            f"{lab} & {r['n_pool']} & {C.fnum(r['R2_median'], 2)} [{C.fnum(lo, 2)}, {C.fnum(hi, 2)}] & "
            rf"{C.fpct(r['cvplus95_cov_median'])}\% [{C.fpct(clo)}, {C.fpct(chi)}] & "
            f"{C.fnum(r['cvplus95_width_over_sd_median'], 2)}" + r" \\"
        )
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_rep_subsample.tex", "\n".join(L))

    mac(
        "benchRepPengWidthSd",
        C.fnum(peng["jk_width95_over_sd"], 2),
        "replication_public.peng.jk_width95_over_sd",
        "calibrated_conformal_results.json",
    )
    mac(
        "benchRepSubSamples",
        "10",
        "replication_subsample75.design",
        "revision_experiments.py",
    )
    for key, _, _ in PUB:
        P = REP_C[key]
        r = pub[key]
        mac(
            f"benchRep{P}Cv",
            C.fnum(r["recordwise_R2"], 3),
            f"replication_public.public.{key}",
            "protocol_replay_extended.json",
        )
        mac(
            f"benchRep{P}Worst",
            C.fnum(r["worst_groupwise"]["R2"], 2),
            f"replication_public.public.{key}",
            "protocol_replay_extended.json",
        )
        mac(
            f"benchRep{P}WidthSd",
            C.fnum(r["jk_width95_over_sd"], 2),
            f"replication_public.public.{key}",
            "revision_experiments.json",
        )
        s = subs[key]
        mac(
            f"benchRepSub{P}Rsq",
            C.fnum(s["R2_median"], 2),
            f"replication_subsample75.rows.{key}",
            "revision_experiments.json",
        )
        mac(
            f"benchRepSub{P}WidthSd",
            C.fnum(s["cvplus95_width_over_sd_median"], 2),
            f"replication_subsample75.rows.{key}",
            "revision_experiments.json",
        )
    ws = [subs[k]["cvplus95_width_over_sd_median"] for k, _, _ in PUB]
    mac(
        "benchRepSubWidthMin",
        C.fnum(min(ws), 1),
        "replication_subsample75.rows",
        "revision_experiments.json",
    )
    mac(
        "benchRepSubWidthMax",
        C.fnum(max(ws), 1),
        "replication_subsample75.rows",
        "revision_experiments.json",
    )


# ====================================================================== figures
FAMILY_OF = {
    "Lineer": "linear",
    "Ridge": "linear",
    "Lasso": "linear",
    "Polinom2-OLS": "linear",
    "Polinom2-Ridge": "linear",
    "SVR-RBF": "kernel",
    "KernelRidge": "kernel",
    "KNN": "kernel",
    "RandomForest": "tree",
    "ExtraTrees": "tree",
    "GradientBoosting": "tree",
    "XGBoost": "tree",
    "ANN-MLP": "mlp",
    "ARD-GP": "gp",
}
FAM_COL = {
    "linear": C.C_GREY,
    "kernel": C.C_TEAL,
    "tree": C.C_BLUE,
    "mlp": C.C_AMBER,
    "gp": C.C_RED,
}
FAM_LAB = {
    "linear": "linear and polynomial",
    "kernel": "kernel and neighbors",
    "tree": "tree ensembles",
    "mlp": "multilayer perceptron",
    "gp": "Gaussian process",
}


def fig_models(B):
    plt = C.apply_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axs = plt.subplots(1, 3, figsize=(C.TEXT_W, 3.3), sharey=True)
    ypos = np.arange(len(BN.ORDER))[::-1]
    for k, (ax, t) in enumerate(zip(axs, C.TARGETS)):
        cv = [B[t]["models"][m]["cv_r2"] for m in BN.ORDER]
        ins = [B[t]["models"][m]["in_sample_r2"] for m in BN.ORDER]
        cols = [FAM_COL[FAMILY_OF[m]] for m in BN.ORDER]
        ax.barh(ypos, cv, color=cols, height=0.7, edgecolor="none")
        ax.scatter(ins, ypos, marker="D", s=12, color="black", zorder=3, linewidths=0)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r"$R^2$")
        ax.set_title(f"({'abc'[k]}) {C.TARGET_LABEL[t]}", loc="left")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(ypos)
    axs[0].set_yticklabels([BN.SHORTNAME[m] for m in BN.ORDER])
    handles = [Patch(color=FAM_COL[f], label=f"LOO, {FAM_LAB[f]}") for f in FAM_COL]
    handles.append(
        Line2D(
            [], [], marker="D", color="black", lw=0, markersize=3.5, label="in-sample"
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    C.save_fig(fig, "fig_bench_models")
    plt.close(fig)


def fig_calibration(L):
    plt = C.apply_style()
    import matplotlib as mpl

    freqs = sorted(set(C.FRQ.tolist()))
    cmap = mpl.colormaps["viridis"]
    fcol = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}
    fig, axs = plt.subplots(1, 3, figsize=(C.TEXT_W, 2.55))
    for k, (ax, t) in enumerate(zip(axs, C.TARGETS)):
        d = L[t]
        yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo)
        ax.plot(
            [lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=C.C_GREY, lw=0.8
        )
        for f in freqs:
            m = C.FRQ == f
            ax.scatter(
                yt[m], yp[m], s=11, color=fcol[f], label=f"{int(f)} Hz", linewidths=0
            )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"measured {C.TARGET_LABEL[t]} (V)")
        ax.set_ylabel(f"LOO predicted {C.TARGET_LABEL[t]} (V)")
        ax.set_title(
            f"({'abc'[k]}) {C.TARGET_LABEL[t]}, {BN.SHORTNAME[d['model']]}", loc="left"
        )
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    C.save_fig(fig, "fig_bench_calibration")
    plt.close(fig)


def fig_tradeoff(within, G):
    plt = C.apply_style()
    from matplotlib.lines import Line2D

    fam_col = {
        "raw_gp": C.C_GREY,
        "jackknife_plus": C.C_BLUE,
        "scaled_jackknife_plus": C.C_TEAL,
    }
    ax_mk = {"composition": "s", "force": "^", "frequency": "D"}
    mod_col = {"gp": C.C_AMBER, "physgp": C.C_RED}
    fig, axs = plt.subplots(1, 3, figsize=(C.TEXT_W, 2.9))
    for k, (ax, t) in enumerate(zip(axs, C.TARGETS)):
        ax.axhline(90, ls=":", color="black", lw=0.7)
        ax.axhline(95, ls="--", color="black", lw=0.7)

        def series(recs, color, marker, filled):
            xs = [r["mean_width"] for r in recs]
            ys = [100 * r["coverage"] for r in recs]
            ax.plot(xs, ys, color=color, lw=0.7, zorder=2)
            for r, x, yv in zip(recs, xs, ys):
                lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
                ax.errorbar(
                    x,
                    yv,
                    yerr=[[yv - lo], [hi - yv]],
                    color=color,
                    lw=0.5,
                    capsize=0,
                    zorder=1,
                    alpha=0.6,
                )
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=16,
                zorder=3,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=0.8,
            )

        for f in FAM_T:
            series([within[t][f][a] for a in C.LEVELS], fam_col[f], "o", True)
        for m in ("gp", "physgp"):
            for axn in C.AXES:
                series(
                    [G[t][m][axn]["jackknife_plus"][a] for a in C.LEVELS],
                    mod_col[m],
                    ax_mk[axn],
                    m == "gp",
                )
        ax.set_xlabel(f"mean interval width, {C.TARGET_LABEL[t]} (V)")
        ax.set_ylabel("coverage (%)")
        ax.set_title(f"({'abc'[k]}) {C.TARGET_LABEL[t]}", loc="left")
        ax.set_ylim(0, 102)
    hs = [
        Line2D(
            [],
            [],
            color=fam_col[f],
            marker="o",
            lw=0.7,
            markersize=4,
            label=f"within grid, {FAM_LEG[f]}",
        )
        for f in FAM_T
    ]
    hs += [
        Line2D(
            [],
            [],
            color=mod_col[m],
            marker="o",
            lw=0,
            markerfacecolor=mod_col[m] if m == "gp" else "white",
            markersize=4,
            label=f"held-out level, jackknife+, {MOD_T[m]}",
        )
        for m in ("gp", "physgp")
    ]
    hs += [
        Line2D(
            [],
            [],
            color="black",
            marker=ax_mk[a],
            lw=0,
            markerfacecolor="white",
            markersize=4,
            label=f"held-out {AX_T[a]}",
        )
        for a in C.AXES
    ]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    C.save_fig(fig, "fig_cov_tradeoff")
    plt.close(fig)


# ====================================================================== captions
def do_sensitivity(J, B, within, G):
    """n = 72 sensitivity without the three duplicated recordings (duplicated-recordings note in the README)."""
    R = S.run()
    J["sensitivity_excluding_duplicates"] = R
    key0 = "sensitivity_excluding_duplicates"
    mac("benchDupN", str(R["n"]), f"{key0}.n", "a3_tables.json")
    mac("covDupN", str(R["n"]), f"{key0}.n", "a3_tables.json")
    mc = {"ARD-GP": "Gp", "RandomForest": "Rf", "ExtraTrees": "Et", "XGBoost": "Xgb"}
    # ---- benchmark table
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{$n=75$} & $n=72$ \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}",
        r"Target & Model & canonical & recomputed & recomputed \\",
        r"\midrule",
    ]
    for k, t in enumerate(C.TARGETS):
        T = C.CAMEL_T[t]
        for i, m in enumerate(S.BENCH_MODELS):
            r = R["benchmark_loo"][t][m]
            first = rf"\multirow{{4}}{{*}}{{{C.TEX_T[t]}}}" if i == 0 else ""
            canon = B[t]["models"][m]["cv_r2"]
            L.append(
                f"{first} & {BN.NAME[m].replace(' (ARD)', '')} & {C.fnum(canon, 3)} & "
                f"{C.fnum(r['n75']['loo_r2'], 3)} & {C.fnum(r['n72']['loo_r2'], 3)}" + r" \\"
            )
            kk = f"{key0}.benchmark_loo.{t}.{m}"
            mac(f"benchDupLoo{T}{mc[m]}", C.fnum(r["n72"]["loo_r2"], 3), kk + ".n72", "a3_tables.json")
            mac(f"benchDupFullLoo{T}{mc[m]}", C.fnum(r["n75"]["loo_r2"], 3), kk + ".n75", "a3_tables.json")
        if k < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_bench_sensitivity.tex", "\n".join(L))
    # ---- coverage table
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabular}{@{}lllcccc@{}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{$n=75$} & \multicolumn{2}{c}{$n=72$} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Target & Interval & Nominal & Coverage (hits/$n$) [$95\%$ CI] & Width (V) & Coverage (hits/$n$) [$95\%$ CI] & Width (V) \\",
        r"\midrule",
    ]
    for t in C.TARGETS:
        T = C.CAMEL_T[t]
        first_t = True
        for f in ("raw_gp", "jackknife_plus"):
            for a in C.LEVELS:
                r75 = within[t][f][a]
                r72 = R["within_grid"][t][f][a]
                tc = rf"\multirow{{4}}{{*}}{{{C.TEX_T[t]}}}" if first_t else ""
                first_t = False
                L.append(
                    f"{tc} & {FAM_T[f]} & ${a}\\%$ & {C.cov_cell(r75)} & "
                    f"{C.fnum(r75['mean_width'], C.width_nd(t))} & {C.cov_cell(r72)} & "
                    f"{C.fnum(r72['mean_width'], C.width_nd(t))}" + r" \\"
                )
                base = f"covDupWithin{FAM_C[f]}{LVL_C[a]}{T}"
                kk = f"{key0}.within_grid.{t}.{f}.{a}"
                mac(base, C.fpct(r72["coverage"]), kk, "a3_tables.json")
                mac(base + "Hits", str(r72["hits"]), kk, "a3_tables.json")
                mac(base + "CiLo", C.fpct(r72["ci95"][0]), kk, "a3_tables.json")
                mac(base + "CiHi", C.fpct(r72["ci95"][1]), kk, "a3_tables.json")
                mac(base + "Width", C.fnum(r72["mean_width"], C.width_nd(t)), kk, "a3_tables.json")
        L.append(r"\midrule")
    first = True
    for m in ("gp", "physgp"):
        for a in C.LEVELS:
            r75 = G["rms_Voc"][m]["force"]["jackknife_plus"][a]
            r72 = R["held_out_force_rms_jackknife_plus"][m][a]
            tc = r"\multirow{4}{*}{\RMS}" if first else ""
            first = False
            L.append(
                f"{tc} & held-out force, {MOD_T[m]} & ${a}\\%$ & {C.cov_cell(r75)} & "
                f"{C.fnum(r75['mean_width'], 3)} & {C.cov_cell(r72)} & "
                f"{C.fnum(r72['mean_width'], 3)}" + r" \\"
            )
            base = f"covDupGroup{MOD_C[m]}Force{LVL_C[a]}Rms"
            kk = f"{key0}.held_out_force_rms_jackknife_plus.{m}.{a}"
            mac(base, C.fpct(r72["coverage"]), kk, "a3_tables.json")
            mac(base + "Hits", str(r72["hits"]), kk, "a3_tables.json")
            mac(base + "CiLo", C.fpct(r72["ci95"][0]), kk, "a3_tables.json")
            mac(base + "CiHi", C.fpct(r72["ci95"][1]), kk, "a3_tables.json")
            mac(base + "Width", C.fnum(r72["mean_width"], 3), kk, "a3_tables.json")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    C.write_text(C.TAB / "tab_cov_sensitivity.tex", "\n".join(L))
    return R


def captions(J):
    txt = r"""# Captions for A3 figures and tables

Written by task A3. Every number quoted in the manuscript comes from `results/numbers_a3.tex` or the `\input` tables; captions carry no result numbers.

## fig_bench_models (figures/fig_bench_models.pdf, double column)

Leave-one-out and in-sample $R^2$ of the fourteen regressors on the three voltage targets, $N=75$ conditions. Bars give the leave-one-out $R^2$ computed from 75 out-of-fold predictions per model and target, colored by model family, and black diamonds give the in-sample $R^2$ of the same model refitted on all 75 conditions. Leave-one-out values come from `reg_stats.json` for the models it reports and from `reg_models.json` otherwise, and in-sample values come from `reg_models.json`. Hyperparameters are those of Table `tab_bench_hyperparams`, fixed before scoring.

## fig_bench_calibration (figures/fig_bench_calibration.pdf, double column)

Leave-one-out predicted against measured voltage for the best model per target, the Gaussian process for \RMS{} and the random forest for \Vpp{} and \Vmax{}. Each point is one of the 75 conditions predicted by the model trained on the other 74, colored by excitation frequency, and the dashed line is the identity. Predictions were recomputed with the canonical model factories of the benchmark, and their $R^2$ values are the macros `\benchRecompLoo...`, which differ from the legacy scores in the third decimal because of library versions.

## fig_cov_tradeoff (figures/fig_cov_tradeoff.pdf, double column)

Empirical coverage against mean interval width for the three voltage targets at nominal $90\%$ and $95\%$. Filled circles are within-grid intervals of the plain Gaussian process under nested leave-one-out evaluation over the 75 conditions, namely the raw posterior band, the jackknife+ band and the $\sigma$-scaled jackknife+ band. Squares, triangles and diamonds are group-wise jackknife+ bands when a whole composition, force level or frequency level is held out, pooled over the 75 conditions of that axis, with filled markers for the plain Gaussian process and open markers for PhysGP. Each line joins the $90\%$ and $95\%$ bands of one interval family, vertical bars are exact Clopper-Pearson $95\%$ intervals on the coverage, and the dotted and dashed horizontal lines mark the two nominal levels. Within-grid and held-out-level coverage answer different questions and are never averaged.

## tab_bench_hyperparams

Hyperparameters of the fourteen regressors, of the two tuned variants and of the Gaussian process used for intervals and PhysGP, generated from the model factories in the released code. Names follow scikit-learn and xgboost, values not listed are library defaults of scikit-learn 1.7.2 and the installed xgboost.

## tab_bench_main

Leave-one-out $R^2$ per target with in-sample $R^2$ in parentheses and the best leave-one-out value per column in bold, $N=75$. Each row names the exact variant it shows, and all fourteen models appear in Table `tab_bench_all`. The table note states the kernel-initialization sensitivity of the Gaussian-process \Vmax{} score. The nested-tuning row reports the multilayer perceptron (MLP) selected by inner five-fold cross-validation inside every leave-one-out training fold, which has no single in-sample fit.

## tab_bench_all

Leave-one-out $R^2$, in-sample $R^2$ and leave-one-out mean absolute error in volts of all fourteen regressors and the two tuned variants.

## tab_bench_significance

The best model per target is in bold. Record-level bootstrap $95\%$ intervals over 2000 resamples of the leave-one-out $R^2$ and paired Wilcoxon signed-rank $p$ values on absolute leave-one-out errors against the best model per target. The paired tests are descriptive because the leave-one-out training sets overlap, and the bootstrap treats the 75 conditions as independent although they come from five specimens.

## tab_cov_within

Within-grid coverage of the plain Gaussian process intervals under nested leave-one-out evaluation, 75 conditions per target. Each coverage cell gives the percentage, the number of covered conditions out of 75 and the exact Clopper-Pearson $95\%$ interval in percent. Width is the mean interval width in volts. The jackknife+ guarantee is $1-2\alpha$ under exchangeability.

## tab_cov_groupwise

Held-out-level coverage of group-wise jackknife+ bands for the plain Gaussian process and PhysGP. For each held-out composition, force level or frequency level the residuals come from leave-one-out models inside the remaining conditions, with the PhysGP law refitted inside every one of those training sets, and the same models predict the held-out conditions. Coverage is pooled over the 75 conditions of each axis and reported with hits over 75 and the exact Clopper-Pearson $95\%$ interval.

## tab_cov_groupwise_raw

Held-out-level coverage of the raw posterior band $\hat\mu\pm z\hat\sigma$ of the model trained on all conditions outside the held-out level, for the plain Gaussian process and PhysGP, with hits over 75 and exact Clopper-Pearson $95\%$ intervals.

## tab_cov_mcnemar

Paired comparison of raw Gaussian process and jackknife+ coverage on the same 75 within-grid conditions. The two discordant counts are the conditions covered by only one of the two bands, and the $p$ value is the exact two-sided McNemar test on those counts.

## tab_al_audit

Active-learning audit on the \RMS{} grid of 75 conditions with a random initial design of ten conditions. Experiments count the initial design. The oracle rows stop the moment the best measured condition is queried, so their regret is zero by construction and their experiment counts are lower bounds that no laboratory can realize. The realizable rule stops after three consecutive rounds whose largest expected improvement falls below one percent of the current best. The one-factor-at-a-time sweep starts from each of the fifteen composition and force pairs, sweeps frequency, then force, then composition. Success is the fraction of runs that query the best measured condition, with exact Clopper-Pearson $95\%$ intervals, and regret is the best measured \RMS{} minus the best value found, in volts.

## tab_rep_public

Replication of the evaluation protocol on five public engineering datasets, identical in every number to the submitted version. The replication checks the evaluation protocol and does not test the nanogenerator voltage law. Worst group-wise $R^2$ holds out all records that share one level of the named factor. J+ denotes jackknife+ on the nanogenerator grid and its ten-fold CV+ form on the public sets, and J+ width$/\sigma_y$ divides the mean nominal-$95\%$ band width by the target standard deviation. Active-learning medians count experiments to the target under the oracle stopping rule with random-sampling medians in parentheses. Record-wise cross-validation is leave-one-out on the nanogenerator grid and ten-fold on the public sets, and the Power Plant row is a seeded 2000-record subsample.

## tab_bench_sensitivity

Leave-one-out $R^2$ of the four leading regressors with and without recordings 52, 54 and 73, which a waveform audit identified as scaled copies of other recordings. The canonical column repeats the benchmark values of the submitted analysis, and the two recomputed columns use the same model factories with the current libraries on all 75 conditions and on the remaining 72, so they differ only in the data.

## tab_cov_sensitivity

Within-grid coverage of the raw Gaussian-process band and the jackknife+ band under nested leave-one-out evaluation, and held-out force level jackknife+ coverage on \RMS{} for the plain Gaussian process and PhysGP, on all 75 conditions and on the 72 conditions left after removing the three duplicated recordings. Each coverage cell gives hits over $n$ and the exact Clopper-Pearson $95\%$ interval, and width is the mean interval width in volts.

## tab_rep_subsample

The protocol replayed at the size of the nanogenerator grid. Ten random 75-record subsamples were drawn from every public set, and each was scored by a ten-fold Gaussian process with its CV+ band at nominal $95\%$. The table gives the median and interquartile range of the cross-validated $R^2$ over the ten subsamples, the median and range of the CV+ coverage and the median band width divided by the target standard deviation.
"""
    C.write_text(C.FIG / "captions_a3.md", txt)


# ====================================================================== outputs
def write_macros():
    L = [
        "% numbers_a3.tex: generated by code/a3_tables.py, do not edit.",
        "% Prefixes: bench (benchmark, active-learning audit, replication), cov (coverage).",
        "% Source of every macro: results/a3_tables.json key given in the comment.",
    ]
    seen = set()
    for name, val, key, src in MACROS:
        assert name not in seen, name
        seen.add(name)
        L.append(f"\\newcommand{{\\{name}}}{{{val}}} % {key} [{src}]")
    C.write_text(C.RES / "numbers_a3.tex", "\n".join(L) + "\n")


def write_predictions(pts_w, pts_g, L):
    rows = pts_w + pts_g
    for t in C.TARGETS:
        d = L[t]
        for i in range(C.N):
            rows.append(
                dict(
                    condition_id=i,
                    target=t,
                    model=d["model"],
                    split_scheme="loo_benchmark",
                    held_out_level="",
                    interval="",
                    nominal="",
                    y_true=d["y_true"][i],
                    y_pred=d["y_pred"][i],
                    lower=np.nan,
                    upper=np.nan,
                    hit="",
                )
            )
    pd.DataFrame(rows).to_csv(
        C.RES / "a3_predictions.csv", index=False, float_format="%.10g"
    )
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    np.random.seed(C.SEED)
    J = {
        "seed": C.SEED,
        "n_conditions": C.N,
        "targets": C.TARGETS,
        "energy_target": "excluded from every table, figure and macro",
    }
    B, L = do_bench(J)
    nested, group = K.compute_raw(use_cache=a.use_cache)
    W, pts_w = K.within_grid(nested)
    G, pts_g = K.groupwise(group)
    within = do_within(J, W)
    J["coverage_within_grid_recomputed"] = {
        t: {f: W[t][f] for f in FAM_T} for t in C.TARGETS
    }
    do_mcnemar(J, W, G)
    do_groupwise(J, G)
    do_al(J)
    do_rep(J, B)
    do_sensitivity(J, B, within, G)
    fig_models(B)
    fig_calibration(L)
    fig_tradeoff(within, G)
    captions(J)
    write_macros()
    n_pred = write_predictions(pts_w, pts_g, L)

    J["index"] = {
        "tab_bench_hyperparams.tex": ["hyperparameters"],
        "tab_bench_main.tex": [
            "benchmark.<target>.models",
            "benchmark.<target>.mlp_nested_tuned",
        ],
        "tab_bench_all.tex": [
            "benchmark.<target>.models",
            "benchmark.<target>.svr_tuned",
            "benchmark.<target>.mlp_nested_tuned",
        ],
        "tab_bench_significance.tex": [
            "benchmark.<target>.models.<m>.boot_ci95",
            "benchmark.<target>.models.<m>.wilcoxon_p_vs_best",
        ],
        "tab_cov_within.tex": ["coverage_within_grid"],
        "tab_cov_groupwise.tex": [
            "coverage_groupwise.<target>.<model>.<axis>.jackknife_plus"
        ],
        "tab_cov_groupwise_raw.tex": ["coverage_groupwise.<target>.<model>.<axis>.raw"],
        "tab_cov_mcnemar.tex": [
            "mcnemar_within_grid",
            "coverage_within_grid_recomputed",
        ],
        "tab_al_audit.tex": ["active_learning"],
        "tab_rep_public.tex": ["replication_public"],
        "tab_rep_subsample.tex": ["replication_subsample75"],
        "tab_bench_sensitivity.tex": ["sensitivity_excluding_duplicates.benchmark_loo",
                                      "benchmark.<target>.models.<m>.cv_r2"],
        "tab_cov_sensitivity.tex": ["sensitivity_excluding_duplicates.within_grid",
                                    "sensitivity_excluding_duplicates.held_out_force_rms_jackknife_plus",
                                    "coverage_within_grid", "coverage_groupwise.rms_Voc.<model>.force"],
        "fig_bench_models": [
            "benchmark.<target>.models.<m>.cv_r2",
            "benchmark.<target>.models.<m>.in_sample_r2",
        ],
        "fig_bench_calibration": ["calibration_loo_recomputed"],
        "fig_cov_tradeoff": ["coverage_within_grid", "coverage_groupwise"],
        "numbers_a3.tex": ["macros"],
        "a3_predictions.csv": "columns condition_id,target,model,split_scheme,held_out_level,"
        "interval,nominal,y_true,y_pred,lower,upper,hit; y_pred of a jackknife+ row is the mean"
        " of the leave-one-out ensemble predictions at that condition",
    }
    J["macros"] = {n: {"value": v, "json_key": k, "source": s} for n, v, k, s in MACROS}
    J["runtime_s"] = time.time() - t0
    json.dump(C.to_plain(J), open(C.RES / "a3_tables.json", "w"), indent=1)

    GW = {
        "description": K.__doc__,
        "groupwise": C.to_plain(G),
        "within_grid_mcnemar": C.to_plain(J["mcnemar_within_grid"]),
        "within_grid_recomputed": C.to_plain(J["coverage_within_grid_recomputed"]),
        "legacy_check_rms_gp": J["coverage_groupwise_legacy_check"],
        "seed": C.SEED,
        "models": {
            "gp": "legacy physgp_analysis.gp(), standardized inputs",
            "physgp": "legacy law refitted by bounded curve_fit inside every training set",
        },
    }
    json.dump(C.to_plain(GW), open(C.RES / "a3_groupwise_coverage.json", "w"), indent=1)

    pass  # run manifest omitted in the public version

    g = G["rms_Voc"]["gp"]["frequency"]["jackknife_plus"]["95"]
    print(
        f"A3 done in {time.time() - t0:.0f}s: {len(MACROS)} macros, {n_pred} prediction rows;"
        f" V_rms within-grid J+95 {within['rms_Voc']['jackknife_plus']['95']['hits']}/75,"
        f" held-out frequency J+95 GP {g['hits']}/75"
    )


if __name__ == "__main__":
    main()
