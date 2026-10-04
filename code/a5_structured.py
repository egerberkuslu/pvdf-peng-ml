#!/usr/bin/env python3
"""A5: structured kernels (separable product, additive, product + linear force mean)
against the plain GP and PhysGP on within-grid LOO and the three held-out axes.

One command regenerates everything:
    python code/a5_structured.py
Outputs
    results/a5_structured.json        all statistics + index dict
    results/a5_predictions.csv        out-of-fold predictions
    results/tables/tab_alt_per_level.tex, tab_alt_pooled.tex
    results/numbers_a5.tex            \\alt* macros
    figures/fig_alt_axes.pdf/.png
    figures/captions_a5.md
    results/a5_manifest.json
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
import a5_common as A  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

T0 = time.time()
SEED = 0
RESTART_SEEDS = [0, 1, 2, 3, 4]
Z95 = 1.959963984540054
CLIP = -3.0

RES = os.path.join(A.REV, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(A.REV, "figures")
OUT_JSON = os.path.join(RES, "a5_structured.json")
OUT_CSV = os.path.join(RES, "a5_predictions.csv")
OUT_MAC = os.path.join(RES, "numbers_a5.tex")
OUT_CAP = os.path.join(FIG, "captions_a5.md")
A2_JSON = os.path.join(RES, "a2_physgp.json")

Y = {t: A.df[t].values.astype(float) for t in A.TARGETS}
SCHEMES = ["loo"] + list(A.AXES)

MODEL_LABEL = {
    "gp": "plain GP",
    "physgp": "PhysGP",
    "m1": "product kernel",
    "m2": "additive kernel",
    "m3": "product + linear force mean",
    "m4": "product + linear force mean, log10 target",
}
MODEL_TEX = {
    "gp": "GP",
    "physgp": "PhysGP",
    "m1": "Prod.",
    "m2": "Add.",
    "m3": r"Prod.$+\beta F$",
    "m4": r"log Prod.$+\beta F$",
}
MODEL_MACRO = {
    "gp": "GP",
    "physgp": "PhysGP",
    "m1": "Product",
    "m2": "Additive",
    "m3": "ProductLin",
    "m4": "ProductLinLog",
    "gp_r4": "GPRestarts",
    "physgp_r4": "PhysGPRestarts",
}
MODEL_COLOR = {
    "gp": "#3B6FB6",
    "physgp": "#E08D2F",
    "m1": "#2A9D8F",
    "m2": "#6C757D",
    "m3": "#C8553D",
    "m4": "#8FB8E0",
}
TARGET_MACRO = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}
TARGET_TEX = {
    "rms_Voc": r"$V_\mathrm{rms}$",
    "Vpp": r"$V_\mathrm{pp}$",
    "Vmax": r"$|V|_\mathrm{max}$",
}
TARGET_PLOT = {
    "rms_Voc": r"$V_{\mathrm{rms}}$",
    "Vpp": r"$V_{\mathrm{pp}}$",
    "Vmax": r"$|V|_{\mathrm{max}}$",
}
SCHEME_MACRO = {
    "loo": "Loo",
    "composition": "Comp",
    "force": "Force",
    "frequency": "Freq",
}
NUM_WORD = {
    1: "One",
    2: "Two",
    3: "Three",
    5: "Five",
    10: "Ten",
    15: "Fifteen",
    20: "Twenty",
    25: "TwentyFive",
}
COMP_MACRO = {
    "PVDF": "Pristine",
    "PVDF+BaTiO3": "Bto",
    "PVDF+BaTiO3+%1CNT": "CntOne",
    "PVDF+BaTiO3+%2CNT": "CntTwo",
    "PVDF+BaTiO3+%3CNT": "CntThree",
}
COMP_TEX = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": r"PVDF/BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": r"+1 wt\% CNT",
    "PVDF+BaTiO3+%2CNT": r"+2 wt\% CNT",
    "PVDF+BaTiO3+%3CNT": r"+3 wt\% CNT",
}
COMP_TICK = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "BTO",
    "PVDF+BaTiO3+%1CNT": "1%",
    "PVDF+BaTiO3+%2CNT": "2%",
    "PVDF+BaTiO3+%3CNT": "3%",
}
FEAT_MACRO = {
    "cnt_pct": "Cnt",
    "is_pristine": "Pristine",
    "force_N": "Force",
    "freq_Hz": "Freq",
}
FEAT_UNIT = {
    "cnt_pct": "wt%",
    "is_pristine": "indicator units",
    "force_N": "N",
    "freq_Hz": "Hz",
}


def f(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def level_macro(axis, lvl):
    if axis == "composition":
        return "Comp" + COMP_MACRO[lvl]
    if axis == "force":
        return "Force" + NUM_WORD[int(lvl)] + "N"
    return "Freq" + NUM_WORD[int(lvl)] + "Hz"


def level_tex(axis, lvl):
    if axis == "composition":
        return COMP_TEX[lvl]
    return f"{int(lvl)} N" if axis == "force" else f"{int(lvl)} Hz"


def folds_of(scheme):
    if scheme == "loo":
        return [(tr, te, str(int(te[0]))) for tr, te in A.loo_folds()]
    return [(tr, te, A.level_key(scheme, v)) for tr, te, v in A.group_folds(scheme)]


# =============================================================== 1. run all folds
JOBS = []
for t in A.TARGETS:
    for sc in SCHEMES:
        for fi, (tr, te, _) in enumerate(folds_of(sc)):
            for m in A.MODELS_MAIN + A.MODELS_SENS:
                JOBS.append((m, t, sc, fi, SEED))
    for s in RESTART_SEEDS[1:]:
        for fi, _ in enumerate(A.loo_folds()):
            JOBS.append(("m1", t, "loo", fi, s))

FOLDS = {sc: folds_of(sc) for sc in SCHEMES}


def run_job(m, t, sc, fi, s):
    tr, te, _ = FOLDS[sc][fi]
    mu, sd = A.fold_predict(m, tr, te, Y[t], seed=s)
    return (m, t, sc, fi, s), np.asarray(mu, float), np.asarray(sd, float)


out = Parallel(n_jobs=-1)(delayed(run_job)(*j) for j in JOBS)
STORE = {k: (mu, sd) for k, mu, sd in out}
print(f"[a5] {len(JOBS)} fold fits in {time.time() - T0:.1f} s")


def oof(m, t, sc, s=SEED):
    mu = np.full(A.N, np.nan)
    sd = np.full(A.N, np.nan)
    lvl = np.empty(A.N, dtype=object)
    for fi, (tr, te, key) in enumerate(FOLDS[sc]):
        a, b = STORE[(m, t, sc, fi, s)]
        mu[te], sd[te], lvl[te] = a, b, key
    return mu, sd, lvl


def interval(m, mu, sd):
    if m == "m4":
        lm = np.log10(mu)
        return 10 ** (lm - Z95 * sd), 10 ** (lm + Z95 * sd)
    return mu - Z95 * sd, mu + Z95 * sd


def lvl_metrics(y, mu):
    e = mu - y
    return {
        "n": int(len(y)),
        "R2": f(r2_score(y, mu)),
        "MAE": f(mean_absolute_error(y, mu)),
        "bias": f(np.mean(e)),
        "y_mean": f(np.mean(y)),
        "pred_mean": f(np.mean(mu)),
    }


# =============================================================== 2. metrics
ALL_MODELS = A.MODELS_MAIN + A.MODELS_SENS
PL, POOLED, CSV = {}, {}, []
for t in A.TARGETS:
    y = Y[t]
    PL[t], POOLED[t] = {}, {}
    for sc in SCHEMES:
        POOLED[t][sc] = {}
        if sc != "loo":
            PL[t][sc] = {}
        for m in ALL_MODELS:
            mu, sd, lvl = oof(m, t, sc)
            lo, hi = interval(m, mu, sd)
            p = {
                "n": A.N,
                "R2": f(r2_score(y, mu)),
                "MAE": f(mean_absolute_error(y, mu)),
            }
            if sc != "loo":
                lv = {}
                for tr, te, key in FOLDS[sc]:
                    lv[key] = lvl_metrics(y[te], mu[te])
                PL[t][sc][m] = lv
                wl = np.array([lv[k]["R2"] for k in lv])
                p["mean_within_level_R2"] = f(np.mean(wl))
                p["median_within_level_R2"] = f(np.median(wl))
                p["n_levels"] = int(len(wl))
                p["n_levels_R2_negative"] = int(np.sum(wl < 0))
            POOLED[t][sc][m] = p
            for i in range(A.N):
                CSV.append(
                    {
                        "condition_id": i,
                        "composition": A.COMP[i],
                        "cnt_pct": A.CNT[i],
                        "force_N": A.FRC[i],
                        "freq_Hz": A.FRQ[i],
                        "target": t,
                        "model": m,
                        "split_scheme": "loo" if sc == "loo" else f"held_out_{sc}",
                        "held_out_level": lvl[i],
                        "seed": SEED,
                        "y_true": y[i],
                        "y_pred": mu[i],
                        "sd": sd[i],
                        "sd_scale": "log10" if m == "m4" else "volt",
                        "lower": lo[i],
                        "upper": hi[i],
                    }
                )

# =============================================================== 3. restart check (m1, LOO)
RESTART = {}
for t in A.TARGETS:
    y = Y[t]
    r2s, maes = [], []
    for s in RESTART_SEEDS:
        mu, sd, lvl = oof("m1", t, "loo", s)
        r2s.append(float(r2_score(y, mu)))
        maes.append(float(mean_absolute_error(y, mu)))
        if s != SEED:
            lo, hi = interval("m1", mu, sd)
            for i in range(A.N):
                CSV.append(
                    {
                        "condition_id": i,
                        "composition": A.COMP[i],
                        "cnt_pct": A.CNT[i],
                        "force_N": A.FRC[i],
                        "freq_Hz": A.FRQ[i],
                        "target": t,
                        "model": "m1",
                        "split_scheme": "loo_restart_check",
                        "held_out_level": lvl[i],
                        "seed": s,
                        "y_true": y[i],
                        "y_pred": mu[i],
                        "sd": sd[i],
                        "sd_scale": "volt",
                        "lower": lo[i],
                        "upper": hi[i],
                    }
                )
    RESTART[t] = {
        "seeds": RESTART_SEEDS,
        "loo_R2_per_seed": r2s,
        "loo_MAE_per_seed": maes,
        "loo_R2_mean": float(np.mean(r2s)),
        "loo_R2_sd": float(np.std(r2s, ddof=1)),
        "loo_R2_range": float(np.max(r2s) - np.min(r2s)),
        "loo_MAE_sd": float(np.std(maes, ddof=1)),
        "sd_definition": "sample SD (ddof=1) over the 5 optimizer seeds of the pooled LOO R2",
    }

# =============================================================== 4. full-grid hyperparameters
FULL = {
    t: {m: A.fit_full(m, Y[t], seed=SEED) for m in ("m1", "m2", "m3", "m4")}
    for t in A.TARGETS
}

# =============================================================== 5. M4 reporting rule
M4_RULE = (
    "m4 enters tables and figure only if its pooled R2 exceeds that of m3 on at least one "
    "scheme (LOO or a held-out axis) for at least one target"
)
m4_better = [
    {
        "target": t,
        "scheme": sc,
        "m4_R2": POOLED[t][sc]["m4"]["R2"],
        "m3_R2": POOLED[t][sc]["m3"]["R2"],
    }
    for t in A.TARGETS
    for sc in SCHEMES
    if POOLED[t][sc]["m4"]["R2"] > POOLED[t][sc]["m3"]["R2"]
]
REPORT_M4 = len(m4_better) > 0
SHOWN = ["gp", "physgp", "m1", "m2", "m3"] + (["m4"] if REPORT_M4 else [])
NEW = [m for m in SHOWN if m not in ("gp", "physgp")]

# =============================================================== 6. comparisons against references
COMP = {}
for t in A.TARGETS:
    COMP[t] = {}
    for sc in SCHEMES:
        COMP[t][sc] = {}
        ref_best = max(POOLED[t][sc]["gp"]["R2"], POOLED[t][sc]["physgp"]["R2"])
        for m in ["m1", "m2", "m3", "m4"]:
            d = {
                "pooled_R2_minus_gp": POOLED[t][sc][m]["R2"]
                - POOLED[t][sc]["gp"]["R2"],
                "pooled_R2_minus_physgp": POOLED[t][sc][m]["R2"]
                - POOLED[t][sc]["physgp"]["R2"],
                "beats_both_refs_pooled_R2": bool(POOLED[t][sc][m]["R2"] > ref_best),
            }
            if sc != "loo":
                lv = PL[t][sc]
                keys = list(lv[m].keys())
                d["n_levels_beats_gp"] = int(
                    sum(lv[m][k]["R2"] > lv["gp"][k]["R2"] for k in keys)
                )
                d["n_levels_beats_physgp"] = int(
                    sum(lv[m][k]["R2"] > lv["physgp"][k]["R2"] for k in keys)
                )
                d["n_levels_beats_both"] = int(
                    sum(
                        lv[m][k]["R2"] > max(lv["gp"][k]["R2"], lv["physgp"][k]["R2"])
                        for k in keys
                    )
                )
                d["n_levels"] = len(keys)
                d["mean_within_level_R2_minus_best_ref"] = POOLED[t][sc][m][
                    "mean_within_level_R2"
                ] - max(
                    POOLED[t][sc]["gp"]["mean_within_level_R2"],
                    POOLED[t][sc]["physgp"]["mean_within_level_R2"],
                )
            COMP[t][sc][m] = d
        COMP[t][sc]["best_model_pooled_R2"] = max(
            SHOWN, key=lambda m: POOLED[t][sc][m]["R2"]
        )
        if sc != "loo":
            COMP[t][sc]["best_model_mean_within_level_R2"] = max(
                SHOWN, key=lambda m: POOLED[t][sc][m]["mean_within_level_R2"]
            )

# headline: does any structured model beat both references on the pooled R2 of an axis
HEAD = {}
for m in ["m1", "m2", "m3", "m4"]:
    HEAD[m] = {
        sc: {t: COMP[t][sc][m]["beats_both_refs_pooled_R2"] for t in A.TARGETS}
        for sc in SCHEMES
    }
HEAD["n_scheme_target_cells"] = len(SCHEMES) * len(A.TARGETS)
HEAD["n_cells_any_new_model_beats_both_refs"] = int(
    sum(
        any(COMP[t][sc][m]["beats_both_refs_pooled_R2"] for m in NEW)
        for t in A.TARGETS
        for sc in SCHEMES
    )
)
HEAD["m3_vs_m1_force_pooled_R2_gain"] = {
    t: POOLED[t]["force"]["m3"]["R2"] - POOLED[t]["force"]["m1"]["R2"]
    for t in A.TARGETS
}
HEAD["m3_vs_m1_other_axes_pooled_R2_change"] = {
    t: {
        sc: POOLED[t][sc]["m3"]["R2"] - POOLED[t][sc]["m1"]["R2"]
        for sc in ("loo", "composition", "frequency")
    }
    for t in A.TARGETS
}

# =============================================================== 7. agreement with A2 references
A2CHK = {"available": False}
if os.path.exists(A2_JSON):
    a2 = json.load(open(A2_JSON))
    diffs = []
    try:
        for t in A.TARGETS:
            for sc in SCHEMES:
                for m in ("gp", "physgp"):
                    diffs.append(
                        abs(a2["pooled"][t][sc][m]["R2"] - POOLED[t][sc][m]["R2"])
                    )
        A2CHK = {
            "available": True,
            "max_abs_pooled_R2_diff": float(max(diffs)),
            "n_compared": len(diffs),
        }
    except (KeyError, TypeError):
        A2CHK = {"available": True, "comparable": False}


# =============================================================== 8. tables
def fmt(v, d):
    s = f"{v:.{d}f}"
    return s.replace("-", "$-$") if v < 0 else s


def bold(s):
    return r"\textbf{" + s + "}"


def row_cells(vals, d, best="max"):
    r = [round(v, d) for v in vals]
    b = max(r) if best == "max" else min(r)
    return [bold(fmt(v, d)) if rv == b else fmt(v, d) for v, rv in zip(vals, r)]


nm = len(SHOWN)
ncols = 2 + nm * len(A.TARGETS)
colspec = "ll" + ("r" * nm) * len(A.TARGETS)


def header():
    h = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    h.append(
        " & & "
        + " & ".join(
            r"\multicolumn{" + str(nm) + r"}{c}{" + TARGET_TEX[t] + "}"
            for t in A.TARGETS
        )
        + r" \\"
    )
    h.append(
        " ".join(
            r"\cmidrule(lr){" + f"{3 + i * nm}-{2 + (i + 1) * nm}" + "}"
            for i in range(len(A.TARGETS))
        )
    )
    h.append(
        r"Level & Metric & "
        + " & ".join([MODEL_TEX[m] for m in SHOWN] * len(A.TARGETS))
        + r" \\"
    )
    h.append(r"\midrule")
    return h


AX_HEAD = {
    "composition": "Held-out composition",
    "force": "Held-out force",
    "frequency": "Held-out frequency",
}
lines = header()
for ai, ax in enumerate(A.AXES):
    if ai:
        lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(ncols) + r"}{l}{\textit{" + AX_HEAD[ax] + r"}} \\"
    )
    for v in A.axis_levels(ax):
        k = A.level_key(ax, v)
        r2c, maec = [], []
        for t in A.TARGETS:
            r2c += row_cells([PL[t][ax][m][k]["R2"] for m in SHOWN], 2, "max")
            maec += row_cells([PL[t][ax][m][k]["MAE"] for m in SHOWN], 3, "min")
        lines.append(level_tex(ax, v) + r" & $R^2$ & " + " & ".join(r2c) + r" \\")
        lines.append(r" & MAE (V) & " + " & ".join(maec) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
TAB_PL = os.path.join(TAB, "tab_alt_per_level.tex")
open(TAB_PL, "w").write(
    "% generated by code/a5_structured.py\n" + "\n".join(lines) + "\n"
)

lines = header()
lines[-2] = (
    r"Scheme & Metric & "
    + " & ".join([MODEL_TEX[m] for m in SHOWN] * len(A.TARGETS))
    + r" \\"
)
SC_TEX = {
    "loo": "Within-grid LOO",
    "composition": "Held-out composition",
    "force": "Held-out force",
    "frequency": "Held-out frequency",
}
for si, sc in enumerate(SCHEMES):
    if si:
        lines.append(r"\midrule")
    d = 3 if sc == "loo" else 2
    r2c, maec, wlc = [], [], []
    for t in A.TARGETS:
        r2c += row_cells([POOLED[t][sc][m]["R2"] for m in SHOWN], d, "max")
        maec += row_cells([POOLED[t][sc][m]["MAE"] for m in SHOWN], 3, "min")
        if sc != "loo":
            wlc += row_cells(
                [POOLED[t][sc][m]["mean_within_level_R2"] for m in SHOWN], 2, "max"
            )
    lines.append(SC_TEX[sc] + r" & pooled $R^2$ & " + " & ".join(r2c) + r" \\")
    if sc != "loo":
        lines.append(r" & mean level $R^2$ & " + " & ".join(wlc) + r" \\")
    lines.append(r" & MAE (V) & " + " & ".join(maec) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
TAB_PO = os.path.join(TAB, "tab_alt_pooled.tex")
open(TAB_PO, "w").write(
    "% generated by code/a5_structured.py\n" + "\n".join(lines) + "\n"
)

# =============================================================== 9. macros
MAC = {}
MAC_KEYS = {}


def mac(name, val, key, d=None):
    assert not any(ch.isdigit() for ch in name), name
    if d is not None:
        s = f"{val:.{d}f}"
        s = s.replace("-", "\\ensuremath{-}") if val < 0 else s
    else:
        s = str(val)
    MAC["alt" + name] = s
    MAC_KEYS["alt" + name] = key


for t in A.TARGETS:
    T = TARGET_MACRO[t]
    for sc in SCHEMES:
        S = SCHEME_MACRO[sc]
        d = 3 if sc == "loo" else 2
        for m in A.MODELS_MAIN + A.MODELS_SENS:
            M = MODEL_MACRO[m]
            base = f"pooled.{t}.{sc}.{m}"
            mac(f"{M}{S}{T}Rsq", POOLED[t][sc][m]["R2"], base + ".R2", d)
            mac(f"{M}{S}{T}Mae", POOLED[t][sc][m]["MAE"], base + ".MAE", 3)
            if sc != "loo":
                mac(
                    f"{M}{S}{T}MeanLevelRsq",
                    POOLED[t][sc][m]["mean_within_level_R2"],
                    base + ".mean_within_level_R2",
                    2,
                )
                for v in A.axis_levels(sc):
                    k = A.level_key(sc, v)
                    mac(
                        f"{M}{level_macro(sc, v)}{T}Rsq",
                        PL[t][sc][m][k]["R2"],
                        f"per_level.{t}.{sc}.{m}.{k}.R2",
                        2,
                    )
        if sc != "loo":
            for m in ["m1", "m2", "m3", "m4"]:
                M = MODEL_MACRO[m]
                mac(
                    f"{M}{S}{T}LevelsBeatBoth",
                    COMP[t][sc][m]["n_levels_beats_both"],
                    f"comparison.{t}.{sc}.{m}.n_levels_beats_both",
                )
    rc = RESTART[t]
    mac(f"ProductRestartSdLoo{T}", rc["loo_R2_sd"], f"restart_check.{t}.loo_R2_sd", 4)
    mac(
        f"ProductRestartMeanLoo{T}",
        rc["loo_R2_mean"],
        f"restart_check.{t}.loo_R2_mean",
        3,
    )
    mac(
        f"ProductRestartRangeLoo{T}",
        rc["loo_R2_range"],
        f"restart_check.{t}.loo_R2_range",
        4,
    )
    for feat, lsd in FULL[t]["m1"]["length_scales"].items():
        mac(
            f"ProductLs{FEAT_MACRO[feat]}{T}",
            lsd["original_units"],
            f"full_grid_fit.{t}.m1.length_scales.{feat}.original_units",
            2,
        )
    mac(
        f"ProductLinBeta{T}",
        FULL[t]["m3"]["beta_force"],
        f"full_grid_fit.{t}.m3.beta_force",
        4,
    )
mac("NRestartSeeds", len(RESTART_SEEDS), "meta.restart_seeds")
mac("NRestarts", A.N_RESTARTS_NEW, "meta.n_restarts_optimizer_new")
mac(
    "CellsAnyBeatsBoth",
    HEAD["n_cells_any_new_model_beats_both_refs"],
    "headline.n_cells_any_new_model_beats_both_refs",
)
mac("CellsTotal", HEAD["n_scheme_target_cells"], "headline.n_scheme_target_cells")
mac("LogModelReported", "yes" if REPORT_M4 else "no", "m4_reporting.reported")

# =============================================================== 9b. sensitivity without duplicated recordings
# duplicated-recordings note in the README: rows 52, 54 and 73 of targets_design.parquet are scaled copies of
# rows 47, 49 and 5. The primary analysis keeps n = 75; this block refits on n = 72.
DUP_ROWS = [52, 54, 73]
KEEP = np.array([i for i in range(A.N) if i not in DUP_ROWS])
DUP_MODELS = ["gp", "physgp", "m1", "m3"]
DUP_MACRO = {"gp": "Gp", "physgp": "Physgp", "m1": "Product", "m3": "ProductLin"}


def folds_keep(scheme):
    if scheme == "loo":
        return [
            (np.delete(KEEP, i), KEEP[[i]], str(int(KEEP[i]))) for i in range(len(KEEP))
        ]
    g = A.AXES[scheme][KEEP]
    return [
        (KEEP[g != v], KEEP[g == v], A.level_key(scheme, v))
        for v in A.axis_levels(scheme)
        if (g == v).any()
    ]


FOLDS_DUP = {sc: folds_keep(sc) for sc in SCHEMES}


def run_dup(m, t, sc, fi):
    tr, te, _ = FOLDS_DUP[sc][fi]
    mu, sd = A.fold_predict(m, tr, te, Y[t], seed=SEED)
    return (m, t, sc, fi), np.asarray(mu, float), np.asarray(sd, float)


DUP_JOBS = [
    (m, t, sc, fi)
    for t in A.TARGETS
    for sc in SCHEMES
    for fi in range(len(FOLDS_DUP[sc]))
    for m in DUP_MODELS
]
T_DUP = time.time()
DSTORE = {
    k: (mu, sd)
    for k, mu, sd in Parallel(n_jobs=-1)(delayed(run_dup)(*j) for j in DUP_JOBS)
}
print(
    f"[a5] sensitivity n={len(KEEP)}: {len(DUP_JOBS)} fold fits in {time.time() - T_DUP:.1f} s"
)

SENS = {
    "excluded_rows": DUP_ROWS,
    "reason": "duplicated-recordings note in the README: rows 52 and 54 are exact scalar multiples of rows 47 and 49, row 73 is a scaled copy of row 5 (r = 0.988)",
    "n": int(len(KEEP)),
    "models": DUP_MODELS,
    "seed": SEED,
    "note": "same models, folds and metrics as the primary analysis, restricted to the 72 kept rows; pooled R2 uses the mean of the 72 kept rows",
    "pooled": {},
    "per_level": {},
    "delta_vs_n75_pooled_R2": {},
}
DUP_CSV = []
for t in A.TARGETS:
    y = Y[t]
    SENS["pooled"][t], SENS["per_level"][t], SENS["delta_vs_n75_pooled_R2"][t] = (
        {},
        {},
        {},
    )
    for sc in SCHEMES:
        SENS["pooled"][t][sc], SENS["delta_vs_n75_pooled_R2"][t][sc] = {}, {}
        if sc != "loo":
            SENS["per_level"][t][sc] = {}
        for m in DUP_MODELS:
            mu = np.full(A.N, np.nan)
            sd = np.full(A.N, np.nan)
            lvl = np.empty(A.N, dtype=object)
            for fi, (tr, te, key) in enumerate(FOLDS_DUP[sc]):
                a, b = DSTORE[(m, t, sc, fi)]
                mu[te], sd[te], lvl[te] = a, b, key
            yk, mk = y[KEEP], mu[KEEP]
            p = {
                "n": int(len(KEEP)),
                "R2": f(r2_score(yk, mk)),
                "MAE": f(mean_absolute_error(yk, mk)),
            }
            if sc != "loo":
                lv = {key: lvl_metrics(y[te], mu[te]) for tr, te, key in FOLDS_DUP[sc]}
                SENS["per_level"][t][sc][m] = lv
                p["mean_within_level_R2"] = f(np.mean([lv[k]["R2"] for k in lv]))
            SENS["pooled"][t][sc][m] = p
            SENS["delta_vs_n75_pooled_R2"][t][sc][m] = f(
                p["R2"] - POOLED[t][sc][m]["R2"]
            )
            lo, hi = interval(m, mu, sd)
            for i in KEEP:
                DUP_CSV.append(
                    {
                        "condition_id": int(i),
                        "composition": A.COMP[i],
                        "cnt_pct": A.CNT[i],
                        "force_N": A.FRC[i],
                        "freq_Hz": A.FRQ[i],
                        "target": t,
                        "model": m,
                        "split_scheme": ("loo" if sc == "loo" else f"held_out_{sc}")
                        + "_excl_dup",
                        "held_out_level": lvl[i],
                        "seed": SEED,
                        "y_true": y[i],
                        "y_pred": mu[i],
                        "sd": sd[i],
                        "sd_scale": "volt",
                        "lower": lo[i],
                        "upper": hi[i],
                    }
                )
            d = 3 if sc == "loo" else 2
            mac(
                f"Dup{DUP_MACRO[m]}{SCHEME_MACRO[sc]}{TARGET_MACRO[t]}Rsq",
                p["R2"],
                f"sensitivity_excluding_duplicates.pooled.{t}.{sc}.{m}.R2",
                d,
            )
CSV.extend(DUP_CSV)
mac("DupN", int(len(KEEP)), "sensitivity_excluding_duplicates.n")

# table: n = 75 beside n = 72
lines = [r"\begin{tabular}{ll" + "rr" * len(DUP_MODELS) + "}", r"\toprule"]
lines.append(
    " & & "
    + " & ".join(r"\multicolumn{2}{c}{" + MODEL_TEX[m] + "}" for m in DUP_MODELS)
    + r" \\"
)
lines.append(
    " ".join(
        r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}"
        for i in range(len(DUP_MODELS))
    )
)
lines.append(
    r"Scheme & Target & " + " & ".join([r"$n=75$ & $n=72$"] * len(DUP_MODELS)) + r" \\"
)
lines.append(r"\midrule")
for si, sc in enumerate(SCHEMES):
    if si:
        lines.append(r"\midrule")
    d = 3 if sc == "loo" else 2
    for ti, t in enumerate(A.TARGETS):
        cells = []
        for m in DUP_MODELS:
            cells += [
                fmt(POOLED[t][sc][m]["R2"], d),
                fmt(SENS["pooled"][t][sc][m]["R2"], d),
            ]
        lab = SC_TEX[sc] if ti == 0 else ""
        lines.append(f"{lab} & {TARGET_TEX[t]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
TAB_SE = os.path.join(TAB, "tab_alt_sensitivity.tex")
open(TAB_SE, "w").write(
    "% generated by code/a5_structured.py\n" + "\n".join(lines) + "\n"
)


# =============================================================== 10. figure
def style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#444444",
            "grid.linewidth": 0.4,
            "grid.color": "#DDDDDD",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "standard",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.bottom": True,
            "ytick.left": True,
        }
    )


def fig_axes():
    style()
    fig, axs = plt.subplots(
        3, 3, figsize=(6.85, 8.6), sharey=True, layout="constrained"
    )
    clipped = []
    stack = [0]
    LINE_PT = 10.5
    w = 0.84 / nm
    letters = iter("abcdefghi")
    for i, t in enumerate(A.TARGETS):
        for j, axn in enumerate(A.AXES):
            ax = axs[i, j]
            lv = A.axis_levels(axn)
            x = np.arange(len(lv))
            for k, m in enumerate(SHOWN):
                vals = np.array([PL[t][axn][m][A.level_key(axn, v)]["R2"] for v in lv])
                xs = x + (k - (nm - 1) / 2) * w
                ax.bar(
                    xs,
                    np.maximum(vals, CLIP),
                    width=w,
                    color=MODEL_COLOR[m],
                    label=MODEL_LABEL[m],
                    edgecolor="none",
                    zorder=2,
                )
                for xx, v, lvl in zip(xs, vals, lv):
                    if v < CLIP:
                        clipped.append(
                            {
                                "target": t,
                                "axis": axn,
                                "level": A.level_key(axn, lvl),
                                "model": m,
                                "R2": float(v),
                            }
                        )
            # clipped values: one stacked label per level group in a strip BELOW the clip line,
            # so no bar is covered; colored square = model, black number = actual R2
            for li, lvl in enumerate(lv):
                cl = [
                    c
                    for c in clipped
                    if c["target"] == t and c["axis"] == axn and c["level"] == A.level_key(axn, lvl)
                ]
                cl = sorted(cl, key=lambda c: SHOWN.index(c["model"]))
                stack[0] = max(stack[0], len(cl))
                for ci, c in enumerate(cl):
                    v = c["R2"]
                    s = (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}").replace("-", "\u2212")
                    yoff = -3 - LINE_PT * ci
                    XOFF = -16 if li == len(lv) - 1 else -7
                    ax.annotate("\u25a0", xy=(x[li], CLIP), xytext=(XOFF, yoff), textcoords="offset points",
                                ha="right", va="top", color=MODEL_COLOR[c["model"]], fontsize=9, zorder=5)
                    ax.annotate(s, xy=(x[li], CLIP), xytext=(XOFF, yoff), textcoords="offset points",
                                ha="left", va="top", color="black", fontsize=9, zorder=5)
            ax.axhline(0, color="black", lw=0.6, zorder=1)
            ax.axhline(CLIP, color="#6C757D", lw=0.5, ls=(0, (2, 2)), zorder=1)
            ax.set_yticks([-3, -2, -1, 0, 1])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [COMP_TICK[v] if axn == "composition" else f"{int(v)}" for v in lv]
            )
            ax.set_title(f"({next(letters)}) {TARGET_PLOT[t]}, {axn}", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"Within-level $R^2$")
            if i == 2:
                ax.set_xlabel(
                    {
                        "composition": "Held-out composition",
                        "force": "Held-out force (N)",
                        "frequency": "Held-out frequency (Hz)",
                    }[axn]
                )
            ax.grid(axis="x", visible=False)
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="outside upper center", ncol=3)
    # size the label strip below the clip line from the rendered axes height (shared y)
    need_pt = 3 + LINE_PT * stack[0] + 2
    top = 1.05
    lo = CLIP - 0.1
    for _ in range(3):
        axs[0, 0].set_ylim(lo, top)
        fig.canvas.draw()
        h_pt = axs[0, 0].get_window_extent().height * 72.0 / fig.dpi
        lo = CLIP - need_pt * (top - CLIP) / (h_pt - need_pt)
    axs[0, 0].set_ylim(lo, top)
    fig.savefig(os.path.join(FIG, "fig_alt_axes.pdf"))
    fig.savefig(os.path.join(FIG, "fig_alt_axes.png"), dpi=150)
    plt.close(fig)
    return clipped


CLIPPED = fig_axes()
mac("FigClippedBars", len(CLIPPED), "figure.clipped")

# =============================================================== 11. JSON, CSV, macros, captions
R = {
    "meta": {
        "task": "A5",
        "script": "code/a5_structured.py",
        "seed": SEED,
        "restart_seeds": RESTART_SEEDS,
        "n_restarts_optimizer_new": A.N_RESTARTS_NEW,
        "n_restarts_optimizer_references": 0,
        "n_conditions": A.N,
        "targets": A.TARGETS,
        "schemes": SCHEMES,
        "models": {
            "gp": "legacy plain GP: C(1,(1e-3,1e3)) * ARD Matern52 over (c, pristine, F, f), ls bounds (1e-2,1e3) + White(1e-3,(1e-6,1e1)), normalize_y, alpha 1e-10, n_restarts 0",
            "physgp": "five-parameter law (a0 + a1 c + a2 c^2) F Lorentz(f; f0, gamma) refitted per training fold (A2 form, tol 1e-15) + legacy plain GP on the residual",
            "m1": "C * Matern52(c, pristine; ARD) * Matern52(F) * Matern52(f) + White; each factor ls bounds (1e-2,1e3); normalize_y; n_restarts 4, random_state seed",
            "m2": "C1*Matern52(c, pristine; ARD) + C2*Matern52(F) + C3*Matern52(f) + White; C bounds (1e-3,1e3); same settings as m1",
            "m3": "m1 on y - beta*F, beta = sum(F y)/sum(F^2) on training rows (no intercept)",
            "m4": "m3 on log10(y), prediction 10**mu (posterior median), interval 10**(mu +- 1.96 sd)",
            "gp_r4": "sensitivity: plain GP with n_restarts 4, random_state seed",
            "physgp_r4": "sensitivity: PhysGP with n_restarts 4, random_state seed",
        },
        "inputs": "X = (cnt_pct, is_pristine, force_N, freq_Hz), standardized inside every fold",
        "interval_note": "lower/upper in the CSV are raw posterior mean +- 1.96 sd (volt scale; m4 on log10 scale then back-transformed); not calibrated",
        "per_level_note": "within-level R2 uses the mean of the held-out level as reference; pooled R2 uses all 75 out-of-fold predictions with the global mean (legacy definition)",
        "runtime_s": None,
    },
    "pooled": POOLED,
    "per_level": PL,
    "restart_check": RESTART,
    "full_grid_fit": FULL,
    "m4_reporting": {
        "rule": M4_RULE,
        "reported": REPORT_M4,
        "cells_where_m4_beats_m3": m4_better,
    },
    "comparison": COMP,
    "headline": HEAD,
    "a2_reference_agreement": A2CHK,
    "figure": {"clip_at": CLIP, "clipped": CLIPPED, "models_shown": SHOWN},
}
R["index"] = {
    "results/tables/tab_alt_per_level.tex": [
        f"per_level.<target>.<axis>.<model>.<level>.{{R2,MAE}} for models {SHOWN}"
    ],
    "results/tables/tab_alt_pooled.tex": [
        f"pooled.<target>.<scheme>.<model>.{{R2,MAE,mean_within_level_R2}} for models {SHOWN}"
    ],
    "figures/fig_alt_axes.pdf": [
        f"per_level.<target>.<axis>.<model>.<level>.R2 for models {SHOWN}",
        "figure.clipped",
    ],
    "results/a5_predictions.csv": [
        "out-of-fold predictions behind pooled, per_level and restart_check"
    ],
    "results/tables/tab_alt_sensitivity.tex": [
        "pooled.<target>.<scheme>.<model>.R2 (n = 75) for models gp, physgp, m1, m3",
        "sensitivity_excluding_duplicates.pooled.<target>.<scheme>.<model>.R2 (n = 72)",
    ],
    "results/numbers_a5.tex": MAC_KEYS,
}
R["sensitivity_excluding_duplicates"] = SENS
R["index"]["results/a5_predictions.csv"].append(
    "rows with split_scheme ending in _excl_dup behind sensitivity_excluding_duplicates"
)
R["meta"]["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(OUT_JSON, "w"), indent=1)

pd.DataFrame(CSV).to_csv(OUT_CSV, index=False)

with open(OUT_MAC, "w") as fh:
    fh.write(
        "% numbers_a5.tex: generated by code/a5_structured.py; keys in a5_structured.json -> index\n"
    )
    for k in sorted(MAC):
        fh.write(f"\\newcommand{{\\{k}}}{{{MAC[k]}}}\n")

CAP_LABEL = {
    "gp": "the plain GP",
    "physgp": "PhysGP",
    "m1": "the separable product kernel",
    "m2": "the additive kernel",
    "m3": "the product kernel about a linear force mean",
    "m4": "the same model fitted to the log10 target",
}
shown_words = (
    ", ".join(CAP_LABEL[m] for m in SHOWN[:-1]) + " and " + CAP_LABEL[SHOWN[-1]]
)
cap = f"""# Captions for task A5 (generated by code/a5_structured.py)

## fig_alt_axes

Within-level coefficient of determination of {shown_words} when one level of one design factor is withheld from training. Rows show the three voltage targets and columns show the held-out composition, the held-out force level and the held-out frequency level. Each bar is computed only from the out-of-fold predictions of the conditions in one held-out level, which are 15 conditions for a composition or frequency level and 25 conditions for a force level, and its reference mean is the mean of that level. Nothing is averaged across levels, targets or optimizer seeds. The PhysGP law and the linear force coefficient are refitted on the training rows of every fold, and every kernel is refitted in every fold with inputs standardized on the training rows. Bars below $R^2 = -3$ are clipped at $-3$ and labeled with their actual value in a strip below the dashed clip line, one line per clipped bar with a square in the color of its model, so that no bar is covered, which applies to {len(CLIPPED)} bars. Values of magnitude below 10 carry one decimal and larger values are rounded to an integer. On the composition axis PVDF denotes pristine PVDF, BTO denotes PVDF/BaTiO$_3$, and 1%, 2% and 3% denote PVDF/BaTiO$_3$ with that MWCNT loading in wt%.

## tab_alt_per_level

Within-level $R^2$ and mean absolute error in volts for every held-out composition, force level and frequency level, three voltage targets and {len(SHOWN)} models. Each entry uses only the out-of-fold predictions of the withheld level and the mean of that level as reference. Bold marks the best model per target in each row, which is the highest $R^2$ or the lowest error.

## tab_alt_pooled

Pooled out-of-fold $R^2$ over all 75 conditions with the global mean as reference, mean of the within-level $R^2$ values over the held-out levels of one axis, and pooled mean absolute error in volts, for within-grid leave-one-out and the three held-out axes. Leave-one-out $R^2$ carries three decimals and group-wise $R^2$ two. Bold marks the best model per target in each row. The structured kernels use four optimizer restarts at seed 0 and the two reference models keep the single-start optimizer of the first-stage analysis.
"""
open(OUT_CAP, "w").write(cap)

# manifest

summ = {
    t: {sc: {m: round(POOLED[t][sc][m]["R2"], 3) for m in SHOWN} for sc in SCHEMES}
    for t in A.TARGETS
}
print(json.dumps(summ))
print(
    f"[a5] done in {time.time() - T0:.1f} s; m4 reported={REPORT_M4}; "
    f"cells where a structured model beats both refs: {HEAD['n_cells_any_new_model_beats_both_refs']}/"
    f"{HEAD['n_scheme_target_cells']}; A2 agreement {A2CHK}; {len(MAC)} macros; {len(CLIPPED)} clipped bars"
)
