#!/usr/bin/env python3
r"""E7 AXISRULE (Protocol V4 section 5): the axis choice for LawGP as an inner-fold audit.

One command from the repository root
    python code/a12_axisrule.py [--reps 20] [--reps-sens 10] [--subsets 6]
                                                       [--n-jobs 4] [--seed 0] [--seed-sens 42] [--smoke]
writes, under results/ only,
    a12_axisrule.json              every number, with an "index" dict (macro -> json key)
    a12_units.csv                  one row per (case, axis): levels, outer gain, inner audit, truth
    tables/tab_a12_*.tex           booktabs tabular bodies
    numbers_a12.tex                \newcommand macros, prefix axr

Question. Before any held-out level is scored, can the data say along which axis the law mean helps?
The audit is the paper's inner-CV selection rule (a2_physgp.py, "inner_cv_rule") applied to one axis:
    for every outer fold (one level of the axis held out) run leave-one-level-out along the same axis
    on the outer training rows only; the fold picks LawGP if its pooled inner R2 (global denominator
    over the outer training rows) exceeds that of the plain GP, else GP (ties -> GP).
The axis verdict is the majority of the fold picks (ties -> GP), so it never sees a held-out level.
Truth is the outer held-out result on the same data: LawGP "helps" on the axis if its pooled
leave-one-level-out R2 exceeds the GP's by more than MARGIN (0.02), "hurts" if it falls short by more
than MARGIN, else "neutral". Accuracy is computed on the non-neutral units; neutral units are reported
with their realized R2. The truth is therefore the realized outer gain on the same data, not the
generator's truth: on a synthetic grid it is scored from the simulated observations exactly as on
real data, never from the known response surface. An axis with two levels cannot be audited (one training level, no inner fold).
Per outer fold the audit is also scored against the fold truth (smaller squared error on the held-out
level), and the realized pooled R2 of "use the fold's inner pick" is set beside always-GP,
always-LawGP and the oracle.

Axis choice. For a case with two or more auditable axes, the picked axis is the argmax of the mean
inner gain (LawGP minus GP inner R2, averaged over outer folds), "none" if that maximum is <= MARGIN;
the true axis is the argmax of the outer gain, "none" if that maximum is <= MARGIN (same margin).

Cases.
  real      the paper's grid (a2_common; GP = a2_common.gp(), LawGP = 'physgp'), targets V_rms, V_pp,
            |V|max, n = 75 and the n = 72 sensitivity without recordings 52, 54, 73.
  external  the six digitized datasets of a9_external.py (loaders, law forms and fold predictor
            imported unchanged; Park with the GP optimizer restarted from five starts as in a9's
            headline), full level sets, plus level sub-samples: every axis with >= 5 levels is cut to
            3 and to 5 levels (all subsets when there are at most --subsets of them, otherwise
            --subsets random ones drawn with the seed); the other axes keep all levels.
  twin      synthetic grids (not E1's waveform twin) built from E1's ground-truth surfaces (a6_twin.response: 'law' = the A2
            five-parameter fit to the real V_rms grid, 'twomode' = its two-Lorentzian fit), one specimen
            per composition with a lognormal multiplier of CV 0.10 or 0.20 (a6_twin.spec_mult), and
            per-recording Gaussian relative noise whose SD is resampled from the real recordings'
            cycle-averaging error (a1_recordings.csv rms_se_rel_pct, recordings 52, 54, 73 excluded).
            The waveform simulator of a6_twin is not used (it is tied to the real 75-row design and its
            calibration was still running); these grids share its response surfaces and specimen model
            only.  Designs: F3 (paper: 5 compositions x 1,2,3 N x 5..25 Hz), F5 (forces 1, 1.5, 2, 2.5,
            3 N; same span), f3 (frequencies 5, 15, 25 Hz).  Seed 0 is the primary replicate set,
            seed 42 a sensitivity set.
All GP fits use one BLAS thread per worker and n_jobs <= 4 (the twin run shares the machine).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_common as A  # noqa: E402
import a6_twin as T  # noqa: E402  (module import only: constants, response(), fit_twomode(), spec_mult())
import a9_external as E  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
RES = os.path.join(A.REV, "results")
TAB = os.path.join(RES, "tables")
CACHE = os.path.join(RES, "a12_cache")
OUT_JSON = os.path.join(RES, "a12_axisrule.json")
VERSION = "v1"
MARGIN = 0.02
DUP_IDS = [52, 54, 73]
TWIN_DESIGNS = {
    "F3": {"forces": [1.0, 2.0, 3.0], "freqs": [5.0, 10.0, 15.0, 20.0, 25.0]},
    "F5": {"forces": [1.0, 1.5, 2.0, 2.5, 3.0], "freqs": [5.0, 10.0, 15.0, 20.0, 25.0]},
    "f3": {"forces": [1.0, 2.0, 3.0], "freqs": [5.0, 15.0, 25.0]},
}
TWIN_CV = [0.10, 0.20]
TWIN_GENS = ["law", "twomode"]
COMP_C = [0.0, 0.0, 1.0, 2.0, 3.0]
COMP_PR = [1.0, 0.0, 0.0, 0.0, 0.0]
PARK_RESTARTS = ("ParkG", "ParkT")
AX_WORD = {
    "composition": "Comp",
    "force": "Force",
    "frequency": "Freq",
    "preload": "Preload",
}
DS_WORD = {
    "SalamaForce": "SalamaImp",
    "SalamaFreq": "SalamaFreq",
    "Shi": "Shi",
    "Li": "Li",
    "ParkG": "ParkG",
    "ParkT": "ParkT",
}
TGT_WORD = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}


# =============================================================================== cases
def real_case(target, rows, tag):
    rows = np.asarray(rows)
    return {
        "kind": "real",
        "name": f"real/{target}/{tag}",
        "target": target,
        "rows": rows,
        "y": A.df[target].values.astype(float)[rows],
        "X": A.X[rows],
        "axes": {
            ax: np.asarray(A.AXES[ax])[rows]
            for ax in ("composition", "force", "frequency")
        },
    }


def ext_case(ds, name, keep=None):
    d = ds if keep is None else E.subset_ds(ds, keep)
    return {
        "kind": "ext",
        "name": name,
        "ds": d,
        "y": d["y"],
        "X": d["X"],
        "axes": d["axes"],
        "restarts": ds["key"] in PARK_RESTARTS,
    }


def twin_case(gen, cv, design, rep, seed, P, se_pool):
    des = TWIN_DESIGNS[design]
    rows = [(k, F, f) for k in range(5) for F in des["forces"] for f in des["freqs"]]
    k = np.array([r[0] for r in rows])
    c = np.array([COMP_C[i] for i in k])
    pr = np.array([COMP_PR[i] for i in k])
    F = np.array([r[1] for r in rows])
    f = np.array([r[2] for r in rows])
    rng = np.random.default_rng([seed, rep, TWIN_GENS.index(gen), int(round(cv * 100))])
    spec = T.spec_mult(rng.standard_normal(5), cv)
    se = rng.choice(se_pool, size=len(rows))
    M = T.response(gen, P, c, F, f)
    y = M * spec[k] * (1.0 + se * rng.standard_normal(len(rows)))
    comp = np.array([A.COMP_ORDER[i] for i in k])
    return {
        "kind": "twin",
        "name": f"twin/{design}/{gen}/cv{int(round(cv * 100))}/s{seed}/r{rep}",
        "design": design,
        "gen": gen,
        "cv": cv,
        "seed": seed,
        "rep": rep,
        "c": c,
        "F": F,
        "f": f,
        "y": y,
        "X": np.column_stack([c, pr, F, f]).astype(float),
        "axes": {"composition": comp, "force": F, "frequency": f},
    }


def fold_predict(case, model, tr, te):
    """Out-of-fold mean for rows te of the case, trained on rows tr (case-local indices)."""
    tr, te = np.asarray(tr), np.asarray(te)
    if case["kind"] == "real":
        rows = case["rows"]
        yfull = A.df[case["target"]].values.astype(float)
        mu, _, _ = A.fold_predict(
            "gp" if model == "gp" else "physgp", rows[tr], rows[te], yfull
        )
        return mu
    if case["kind"] == "ext":
        mdl = model + ("_r4" if case["restarts"] else "")
        mu, _ = E.fold_predict(case["ds"], mdl, tr, te)
        return mu
    y, X = case["y"], case["X"]
    if model == "gp":
        btr, bte = np.zeros(len(tr)), np.zeros(len(te))
    else:
        xt = (case["c"][tr], case["F"][tr], case["f"][tr])
        p, _, scale, _ = A._fit(A.law5, tr, y, A.P0_5, A.BOUNDS_5, xt=xt)
        btr = A.law5(xt, *p) * scale
        bte = A.law5((case["c"][te], case["F"][te], case["f"][te]), *p) * scale
    xs = StandardScaler().fit(X[tr])
    g = A.gp().fit(xs.transform(X[tr]), y[tr] - btr)
    return bte + g.predict(xs.transform(X[te]))


def lolo(labels, idx):
    """Leave-one-level-out folds on rows idx: (train, test, level)."""
    idx = np.asarray(idx)
    g = np.asarray(labels)[idx]
    out = []
    for v in sorted(np.unique(g), key=lambda t: (str(type(t)), t)):
        m = g == v
        if m.any() and (~m).any():
            out.append((idx[~m], idx[m], v))
    return out


def unit_key(case, axis):
    h = hashlib.sha1()
    h.update(VERSION.encode())
    h.update(case["kind"].encode())
    h.update(np.ascontiguousarray(case["X"], dtype=float).tobytes())
    h.update(np.ascontiguousarray(case["y"], dtype=float).tobytes())
    h.update(str(axis).encode())
    h.update(str(list(map(str, case["axes"][axis]))).encode())
    h.update(str(case.get("restarts", False)).encode())
    if case["kind"] == "ext":
        h.update(case["ds"]["law"].encode())
    return h.hexdigest()[:20]


def audit_unit(case, axis):
    """Outer leave-one-level-out truth and the nested inner audit for one (case, axis)."""
    key = unit_key(case, axis)
    path = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    y = case["y"]
    lab = case["axes"][axis]
    n = len(y)
    outer = lolo(lab, np.arange(n))
    L = len(outer)
    mu = {m: np.full(n, np.nan) for m in ("gp", "lawgp")}
    folds = []
    for tr, te, v in outer:
        for m in mu:
            mu[m][te] = fold_predict(case, m, tr, te)
        inner = lolo(lab, tr)
        fo = {"level": str(v), "n_test": int(len(te)), "n_inner": len(inner)}
        if len(inner) >= 2:
            mi = {m: np.full(n, np.nan) for m in ("gp", "lawgp")}
            for itr, ite, _ in inner:
                for m in mi:
                    mi[m][ite] = fold_predict(case, m, itr, ite)
            ir2 = {m: float(r2_score(y[tr], mi[m][tr])) for m in mi}
            fo["inner_R2"] = ir2
            fo["inner_gain"] = ir2["lawgp"] - ir2["gp"]
            fo["pick"] = "lawgp" if ir2["lawgp"] > ir2["gp"] else "gp"
        else:
            fo["inner_R2"], fo["inner_gain"], fo["pick"] = None, None, None
        sse = {m: float(np.sum((y[te] - mu[m][te]) ** 2)) for m in mu}
        fo["outer_SSE"] = sse
        fo["fold_truth"] = "lawgp" if sse["lawgp"] < sse["gp"] else "gp"
        folds.append((fo, te))
    r2 = {m: float(r2_score(y, mu[m])) for m in mu}
    gain = r2["lawgp"] - r2["gp"]
    out = {
        "n": int(n),
        "n_levels": L,
        "outer_R2": r2,
        "outer_gain": gain,
        "truth": "helps"
        if gain > MARGIN
        else ("hurts" if gain < -MARGIN else "neutral"),
        "truth_sign": "helps" if gain > 0 else "hurts",
        "folds": [fo for fo, _ in folds],
    }
    auditable = all(fo["pick"] is not None for fo, _ in folds) and L >= 3
    out["auditable"] = bool(auditable)
    if auditable:
        picks = [fo["pick"] for fo, _ in folds]
        k = sum(p == "lawgp" for p in picks)
        out["n_pick_lawgp"] = int(k)
        out["verdict"] = "helps" if k > L - k else "hurts"
        out["mean_inner_gain"] = float(np.mean([fo["inner_gain"] for fo, _ in folds]))
        out["verdict_mean"] = "helps" if out["mean_inner_gain"] > 0 else "hurts"
        out["fold_hits"] = int(sum(fo["pick"] == fo["fold_truth"] for fo, _ in folds))
        sel = np.full(n, np.nan)
        for fo, te in folds:
            sel[te] = mu[fo["pick"]][te]
        orc = np.full(n, np.nan)
        for fo, te in folds:
            orc[te] = mu[fo["fold_truth"]][te]
        out["realized_R2"] = {
            "audit": float(r2_score(y, sel)),
            "always_gp": r2["gp"],
            "always_lawgp": r2["lawgp"],
            "oracle": float(r2_score(y, orc)),
        }
        verd_mu = mu[out["verdict"] == "helps" and "lawgp" or "gp"]
        out["realized_R2"]["axis_verdict"] = float(r2_score(y, verd_mu))
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def run_units(jobs, n_jobs):
    """jobs: list of (meta dict, case, axis) -> list of (meta, result)."""
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(audit_unit)(c, a) for _, c, a in jobs
    )
    return [(m, r) for (m, _, _), r in zip(jobs, res)]


# =============================================================================== job lists
def level_subsets(levels, k, cap, rng):
    combos = list(itertools.combinations(levels, k))
    if len(combos) <= cap:
        return combos
    pick = rng.choice(len(combos), size=cap, replace=False)
    return [combos[i] for i in sorted(pick)]


def build_jobs(args):
    jobs = []
    # real grid
    keep72 = np.array([i for i in range(A.N) if i not in DUP_IDS])
    for t in A.TARGETS:
        for tag, rows in (("n75", np.arange(A.N)), ("n72", keep72)):
            case = real_case(t, rows, tag)
            for ax in case["axes"]:
                jobs.append(
                    (
                        {
                            "family": "real",
                            "case": case["name"],
                            "target": t,
                            "tag": tag,
                            "axis": ax,
                            "subset": "full",
                            "k": None,
                        },
                        case,
                        ax,
                    )
                )
    # external, full and sub-sampled
    DS = E.load_datasets()
    rng = np.random.default_rng(args.seed)
    for key, ds in DS.items():
        if args.smoke and key not in ("SalamaFreq", "Li"):
            continue
        case = ext_case(ds, f"ext/{key}/full")
        for ax in ds["axes"]:
            jobs.append(
                (
                    {
                        "family": "external",
                        "case": case["name"],
                        "dataset": key,
                        "axis": ax,
                        "subset": "full",
                        "k": None,
                    },
                    case,
                    ax,
                )
            )
        for ax, lab in ds["axes"].items():
            levels = sorted(np.unique(lab).tolist())
            if len(levels) < 5:
                continue
            for k in (3, 5):
                if k >= len(levels):
                    continue
                for j, sub in enumerate(
                    level_subsets(levels, k, 2 if args.smoke else args.subsets, rng)
                ):
                    keep = np.where(np.isin(lab, sub))[0]
                    c2 = ext_case(ds, f"ext/{key}/{ax}{k}/{j}", keep)
                    jobs.append(
                        (
                            {
                                "family": "external_sub",
                                "case": c2["name"],
                                "dataset": key,
                                "axis": ax,
                                "subset": "|".join(map(str, sub)),
                                "k": k,
                            },
                            c2,
                            ax,
                        )
                    )
    # twin
    law_p = A.fit_law5(np.arange(A.N), A.df["rms_Voc"].values.astype(float))[1]
    P = {"law": [float(v) for v in law_p], "twomode": T.fit_twomode()["params"]}
    rec = pd.read_csv(os.path.join(RES, "a1_recordings.csv"))
    rec = rec[~rec.condition_id.isin(DUP_IDS)]
    se_pool = rec["rms_se_rel_pct"].dropna().values.astype(float) / 100.0
    for seed, reps, fam in (
        (args.seed, args.reps, "twin"),
        (args.seed_sens, args.reps_sens, "twin_sens"),
    ):
        for design in TWIN_DESIGNS:
            for gen in TWIN_GENS:
                for cv in TWIN_CV:
                    for rep in range(reps):
                        case = twin_case(gen, cv, design, rep, seed, P, se_pool)
                        for ax in case["axes"]:
                            jobs.append(
                                (
                                    {
                                        "family": fam,
                                        "case": case["name"],
                                        "design": design,
                                        "gen": gen,
                                        "cv": cv,
                                        "seed": seed,
                                        "rep": rep,
                                        "axis": ax,
                                        "subset": "full",
                                        "k": None,
                                    },
                                    case,
                                    ax,
                                )
                            )
    twin_meta = {
        "law_params_volt": P["law"],
        "twomode_params": P["twomode"],
        "se_pool_median": float(np.median(se_pool)),
        "se_pool_n": int(len(se_pool)),
    }
    return jobs, twin_meta


# =============================================================================== summaries
def acc_block(rows):
    """rows: list of unit dicts (auditable). Accuracy on non-neutral units, plus sign-only accuracy."""
    nn = [r for r in rows if r["truth"] != "neutral"]
    hit = sum(r["verdict"] == r["truth"] for r in nn)
    hit_s = sum(r["verdict"] == r["truth_sign"] for r in rows)
    hit_m = sum(r["verdict_mean"] == r["truth"] for r in nn)
    tp = sum(r["verdict"] == "helps" and r["truth"] == "helps" for r in nn)
    fp = sum(r["verdict"] == "helps" and r["truth"] == "hurts" for r in nn)
    fn = sum(r["verdict"] == "hurts" and r["truth"] == "helps" for r in nn)
    tn = sum(r["verdict"] == "hurts" and r["truth"] == "hurts" for r in nn)
    folds = sum(r["n_levels"] for r in rows)
    fh = sum(r["fold_hits"] for r in rows)
    reg = [r["realized_R2"]["oracle"] - r["realized_R2"]["audit"] for r in rows]
    best_fixed = [max(r["outer_R2"].values()) - r["realized_R2"]["audit"] for r in rows]
    return {
        "n_units": len(rows),
        "n_non_neutral": len(nn),
        "n_neutral": len(rows) - len(nn),
        "accuracy": hit / len(nn) if nn else None,
        "hits": int(hit),
        "accuracy_sign": hit_s / len(rows) if rows else None,
        "accuracy_mean_rule": hit_m / len(nn) if nn else None,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_truth_helps": int(tp + fn),
        "n_truth_hurts": int(fp + tn),
        "fold_accuracy": fh / folds if folds else None,
        "n_folds": int(folds),
        "median_regret_vs_oracle": float(np.median(reg)) if reg else None,
        "median_shortfall_vs_best_fixed": float(np.median(best_fixed))
        if best_fixed
        else None,
        # no-skill references: the majority class of the truth, and the mean of the two class recalls
        "accuracy_majority_baseline": max(tp + fn, fp + tn) / len(nn) if nn else None,
        "balanced_accuracy": float(
            np.mean(
                [x for x in (tp / (tp + fn) if tp + fn else None, tn / (tn + fp) if tn + fp else None) if x is not None]
            )
        )
        if nn
        else None,
    }


def gain_bin(g):
    a = abs(g)
    return "small" if a <= 0.1 else ("medium" if a <= 0.5 else "large")


def lev_bin(L):
    return "3" if L == 3 else ("4" if L == 4 else "5+")


def axis_choice(units_by_case):
    """units_by_case: {case: {axis: unit}} -> list of (case, picked, true)."""
    out = []
    for case, d in units_by_case.items():
        aud = {a: u for a, u in d.items() if u["auditable"]}
        if len(aud) < 2:
            continue
        pa = max(aud, key=lambda a: aud[a]["mean_inner_gain"])
        picked = pa if aud[pa]["mean_inner_gain"] > MARGIN else "none"
        ta = max(aud, key=lambda a: aud[a]["outer_gain"])
        true = ta if aud[ta]["outer_gain"] > MARGIN else "none"
        out.append(
            {
                "case": case,
                "picked": picked,
                "true": true,
                "axes": sorted(aud),
                "levels": {a: aud[a]["n_levels"] for a in aud},
            }
        )
    return out


def summarize(results, twin_meta, args):
    J = {"meta": {}, "units": [], "summary": {}, "index": {}}
    J["meta"] = {
        "script": "code/a12_axisrule.py",
        "version": VERSION,
        "margin_R2": MARGIN,
        "seed": args.seed,
        "seed_sens": args.seed_sens,
        "reps": args.reps,
        "reps_sens": args.reps_sens,
        "subsets_cap": args.subsets,
        "n_jobs": args.n_jobs,
        "smoke": bool(args.smoke),
        "rule": "per outer fold: inner leave-one-level-out on outer training rows along the same axis; "
        "pick LawGP if inner pooled R2 (global denominator on outer training rows) of LawGP > GP; "
        "axis verdict = majority of fold picks (ties -> GP)",
        "truth": f"outer pooled leave-one-level-out R2 gain LawGP - GP: helps > {MARGIN}, hurts < -{MARGIN}, else neutral",
        "twin": twin_meta,
        "twin_note": "synthetic grids from a6_twin.response surfaces with a6_twin.spec_mult specimen multipliers "
        "and a1 cycle-averaging noise; not the a6 waveform simulator (E1's waveform twin)",
        "truth_note": "truth = realized outer leave-one-level-out gain (LawGP - GP pooled R2) on the same data, "
        "for synthetic grids too; it is not the generator's truth (the known response surface is never used "
        "for scoring)",
    }
    flat = []
    for m, r in results:
        row = dict(m)
        row.update({k: v for k, v in r.items() if k != "folds"})
        row["fold_detail"] = r["folds"]
        flat.append(row)
    J["units"] = flat
    S = J["summary"]

    def sel(**kw):
        return [u for u in flat if all(u.get(k) == v for k, v in kw.items())]

    # real grid
    S["real"] = {}
    for t in A.TARGETS:
        for tag in ("n75", "n72"):
            S["real"].setdefault(t, {})[tag] = {
                u["axis"]: {
                    k: u.get(k)
                    for k in (
                        "n_levels",
                        "outer_R2",
                        "outer_gain",
                        "truth",
                        "auditable",
                        "n_pick_lawgp",
                        "verdict",
                        "mean_inner_gain",
                        "fold_hits",
                        "realized_R2",
                    )
                }
                for u in sel(family="real", target=t, tag=tag)
            }
    # paper reference: a2_physgp inner_cv picks on the same grid
    try:
        a2 = json.load(open(os.path.join(RES, "a2_physgp.json")))["selection_rule"]
        S["real_reference_a2"] = {
            t: {ax: a2[t][ax]["summary"]["inner_cv"]["n_pick_physgp"] for ax in a2[t]}
            for t in a2
        }
    except Exception as e:  # pragma: no cover
        S["real_reference_a2"] = {"error": str(e)}
    # external full
    S["external"] = {}
    for u in sel(family="external"):
        S["external"].setdefault(u["dataset"], {})[u["axis"]] = {
            k: u.get(k)
            for k in (
                "n",
                "n_levels",
                "outer_R2",
                "outer_gain",
                "truth",
                "auditable",
                "n_pick_lawgp",
                "verdict",
                "mean_inner_gain",
                "fold_hits",
                "realized_R2",
            )
        }
    # accuracy by number of levels
    groups = {
        "external_full": [u for u in sel(family="external") if u["auditable"]],
        "external_sub": [u for u in sel(family="external_sub") if u["auditable"]],
        "real_n75": [u for u in sel(family="real", tag="n75") if u["auditable"]],
        "real_n72": [u for u in sel(family="real", tag="n72") if u["auditable"]],
        "twin": [u for u in sel(family="twin") if u["auditable"]],
        "twin_sens": [u for u in sel(family="twin_sens") if u["auditable"]],
    }
    groups["external_all"] = groups["external_full"] + groups["external_sub"]
    S["by_levels"] = {}
    for gname, rows in groups.items():
        S["by_levels"][gname] = {"all": acc_block(rows)}
        S.setdefault("by_effect", {})[gname] = {
            b: acc_block([u for u in rows if u["truth"] != "neutral" and gain_bin(u["outer_gain"]) == b])
            for b in ("small", "medium", "large")
        }
        for b in ("small", "medium", "large"):
            rr = [u for u in rows if u["truth"] != "neutral" and gain_bin(u["outer_gain"]) == b]
            ag = np.abs([u["outer_gain"] for u in rr]) if rr else np.array([])
            S["by_effect"][gname][b]["abs_gain"] = (
                {"median": float(np.median(ag)), "p10": float(np.percentile(ag, 10)),
                 "p90": float(np.percentile(ag, 90)), "min": float(ag.min()), "max": float(ag.max())}
                if len(ag) else None)
            S["by_effect"][gname][b]["axis_counts"] = {
                ax: int(sum(u["axis"] == ax for u in rr)) for ax in sorted({u["axis"] for u in rr})}
        for b in ("3", "4", "5+"):
            rr = [u for u in rows if lev_bin(u["n_levels"]) == b]
            if rr:
                S["by_levels"][gname][b] = acc_block(rr)
    # external sub-samples per dataset and axis, k = 3 vs 5 vs full
    S["external_sub"] = {}
    for u in groups["external_sub"]:
        S["external_sub"].setdefault(u["dataset"], {}).setdefault(
            u["axis"], {}
        ).setdefault(str(u["k"]), []).append(u)
    for ds, d in S["external_sub"].items():
        for ax, dd in d.items():
            for k, rows in dd.items():
                dd[k] = acc_block(rows)
                dd[k]["truth_helps_share"] = float(
                    np.mean([r["truth"] == "helps" for r in rows])
                )
                dd[k]["median_outer_gain"] = float(np.median([r["outer_gain"] for r in rows]))
                dd[k]["mean_outer_gain"] = float(
                    np.mean([r["outer_gain"] for r in rows])
                )
    # twin by design / gen / cv / axis
    S["twin"] = {}
    for fam in ("twin", "twin_sens"):
        S[fam] = S.get(fam, {})
        for des in TWIN_DESIGNS:
            for gen in TWIN_GENS:
                for cv in TWIN_CV:
                    for ax in ("composition", "force", "frequency"):
                        rows = [
                            u
                            for u in groups[fam]
                            if u["design"] == des
                            and u["gen"] == gen
                            and u["cv"] == cv
                            and u["axis"] == ax
                        ]
                        if not rows:
                            continue
                        b = acc_block(rows)
                        b["truth_helps_share"] = float(
                            np.mean([r["truth"] == "helps" for r in rows])
                        )
                        b["median_outer_gain"] = float(np.median([r["outer_gain"] for r in rows]))
                        b["mean_outer_gain"] = float(
                            np.mean([r["outer_gain"] for r in rows])
                        )
                        b["n_levels"] = rows[0]["n_levels"]
                        S[fam].setdefault(des, {}).setdefault(gen, {}).setdefault(
                            f"cv{int(cv * 100)}", {}
                        )[ax] = b
                for ax in ("composition", "force", "frequency"):
                    rows = [
                        u for u in groups[fam] if u["design"] == des and u["axis"] == ax
                    ]
                    if rows:
                        b = acc_block(rows)
                        b["truth_helps_share"] = float(
                            np.mean([r["truth"] == "helps" for r in rows])
                        )
                        b["n_levels"] = rows[0]["n_levels"]
                        S[fam].setdefault(des, {}).setdefault("pooled", {})[ax] = b
    # inner against outer gain scale per twin design, generator and axis (why the argmax over axes misleads)
    S["twin_gain_scale"] = {}
    for des in TWIN_DESIGNS:
        for gen in TWIN_GENS + ["pooled"]:
            for ax in ("composition", "force", "frequency"):
                rows = [u for u in groups["twin"] if u["design"] == des and u["axis"] == ax
                        and (gen == "pooled" or u["gen"] == gen)]
                if rows:
                    S["twin_gain_scale"].setdefault(des, {}).setdefault(gen, {})[ax] = {
                        "median_inner_gain": float(np.median([r["mean_inner_gain"] for r in rows])),
                        "median_outer_gain": float(np.median([r["outer_gain"] for r in rows])),
                        "n": len(rows)}
    # axis choice confusion
    S["axis_choice"] = {}
    for gname, fams in (
        ("external", ("external",)),
        ("real_n75", ("real",)),
        ("twin", ("twin",)),
        ("twin_sens", ("twin_sens",)),
    ):
        byc = {}
        for u in flat:
            if u["family"] in fams and (gname != "real_n75" or u.get("tag") == "n75"):
                byc.setdefault(u["case"], {})[u["axis"]] = u
        ch = axis_choice(byc)
        labels = sorted({c["picked"] for c in ch} | {c["true"] for c in ch})
        conf = {
            t: {p: sum(c["true"] == t and c["picked"] == p for c in ch) for p in labels}
            for t in labels
        }
        S["axis_choice"][gname] = {
            "n_cases": len(ch),
            "n_correct": int(sum(c["picked"] == c["true"] for c in ch)),
            "accuracy": float(np.mean([c["picked"] == c["true"] for c in ch]))
            if ch
            else None,
            "labels": labels,
            "confusion_true_by_picked": conf,
            "cases": ch if gname != "twin" and gname != "twin_sens" else None,
        }
        if gname in ("twin", "twin_sens"):
            for des in TWIN_DESIGNS:
                cd = [c for c in ch if f"/{des}/" in c["case"]]
                labs = sorted({c["picked"] for c in cd} | {c["true"] for c in cd})
                S["axis_choice"][gname][des] = {
                    "n_cases": len(cd),
                    "n_correct": int(sum(c["picked"] == c["true"] for c in cd)),
                    "n_picked": {p: int(sum(c["picked"] == p for c in cd)) for p in labs},
                    "n_true": {t: int(sum(c["true"] == t for c in cd)) for t in labs},
                    "accuracy": float(np.mean([c["picked"] == c["true"] for c in cd]))
                    if cd
                    else None,
                    "labels": labs,
                    "confusion_true_by_picked": {
                        t: {
                            p: sum(c["true"] == t and c["picked"] == p for c in cd)
                            for p in labs
                        }
                        for t in labs
                    },
                }
    return J


# =============================================================================== output
def fnum(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0:
            s = s[1:]
        else:
            s = r"\ensuremath{-}" + s[1:]
    return s


def tnum(x, d=2):
    return fnum(x, d).replace(r"\ensuremath{-}", "$-$")


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}"


def frac(a, b):
    return f"{a}/{b}"


class Macros:
    def __init__(self, J):
        self.lines, self.J = [], J

    def add(self, name, val, key):
        assert name.isalpha(), name
        self.lines.append(rf"\newcommand{{\axr{name}}}{{{val}}}")
        self.J["index"]["axr" + name] = key


def write_outputs(J):
    os.makedirs(TAB, exist_ok=True)
    S = J["summary"]
    M = Macros(J)
    VW = {"helps": "helps", "hurts": "hurts", "neutral": "neutral"}
    TX = {
        "composition": "comp.",
        "force": "force",
        "frequency": "freq.",
        "preload": "preload",
    }

    # ---- real grid table
    real_miss = []
    L = [
        r"\begin{tabular}{l l r r r r c c c r r}",
        r"\toprule",
        r"Target & Axis & Levels & \GP & \LawGP & Gain & Truth & Picks & Verdict & Audit $R^2$ & a2 picks \\",
        r"\midrule",
    ]
    for t in A.TARGETS:
        for i, ax in enumerate(("composition", "force", "frequency")):
            u = S["real"][t]["n75"][ax]
            ref = (
                S["real_reference_a2"].get(t, {}).get(ax, "--")
                if isinstance(S["real_reference_a2"], dict)
                else "--"
            )
            L.append(
                f"{A.TARGET_TEX[t] if i == 0 else ''} & {TX[ax]} & {u['n_levels']} & {tnum(u['outer_R2']['gp'])} & "
                f"{tnum(u['outer_R2']['lawgp'])} & {tnum(u['outer_gain'])} & {VW[u['truth']]} & "
                f"{frac(u['n_pick_lawgp'], u['n_levels'])} & "
                f"{u['verdict'] + (r'$^{\dagger}$' if u['truth'] != 'neutral' and u['verdict'] != u['truth'] else '')} & "
                f"{tnum(u['realized_R2']['audit'])} & {ref}/{u['n_levels']} \\\\"
            )
            if u["truth"] != "neutral" and u["verdict"] != u["truth"]:
                u72 = S["real"][t]["n72"][ax]
                real_miss.append(
                    rf"$^{{\dagger}}$ Audit miss at $n=75$: {A.TARGET_TEX[t]} {TX[ax]} verdict {u['verdict']} "
                    rf"({u['n_pick_lawgp']}/{u['n_levels']} folds pick \LawGP) while \LawGP {u['truth']} "
                    rf"(gain {tnum(u['outer_gain'])}); at $n=72$ {u72['n_pick_lawgp']}/{u72['n_levels']} picks, "
                    rf"verdict {u72['verdict']}, "
                    + ("right." if u72["truth"] != "neutral" and u72["verdict"] == u72["truth"] else "still wrong.")
                )
            pre = f"Real{TGT_WORD[t]}{AX_WORD[ax]}"
            key = f"summary.real.{t}.n75.{ax}"
            M.add(pre + "Gain", fnum(u["outer_gain"]), key + ".outer_gain")
            M.add(pre + "Picks", str(u["n_pick_lawgp"]), key + ".n_pick_lawgp")
            M.add(pre + "Verdict", u["verdict"], key + ".verdict")
            M.add(pre + "Truth", u["truth"], key + ".truth")
            M.add(
                pre + "AuditRsq",
                fnum(u["realized_R2"]["audit"]),
                key + ".realized_R2.audit",
            )
            u72 = S["real"][t]["n72"][ax]
            M.add(
                pre + "GainSeventyTwo",
                fnum(u72["outer_gain"]),
                f"summary.real.{t}.n72.{ax}.outer_gain",
            )
            M.add(
                pre + "PicksSeventyTwo",
                str(u72["n_pick_lawgp"]),
                f"summary.real.{t}.n72.{ax}.n_pick_lawgp",
            )
            M.add(
                pre + "VerdictSeventyTwo",
                u72["verdict"],
                f"summary.real.{t}.n72.{ax}.verdict",
            )
        if t != A.TARGETS[-1]:
            L.append(r"\addlinespace")
    for note in real_miss:
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{11}}{{l}}{{\footnotesize {note}}} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_real.tex"), "w").write("\n".join(L) + "\n")

    # ---- external full table
    L = [
        r"\begin{tabular}{l l r r r r c c c r}",
        r"\toprule",
        r"Dataset & Axis & Levels & \GP & \LawGP & Gain & Truth & Picks & Verdict & Audit $R^2$ \\",
        r"\midrule",
    ]
    first = True
    for ds in E.load_datasets():
        if ds not in S["external"]:
            continue
        if not first:
            L.append(r"\addlinespace")
        first = False
        for i, (ax, u) in enumerate(S["external"][ds].items()):
            aud = u["auditable"]
            L.append(
                f"{E.DS_TEX[ds] if i == 0 else ''} & {TX[ax]} & {u['n_levels']} & {tnum(u['outer_R2']['gp'])} & "
                f"{tnum(u['outer_R2']['lawgp'])} & {tnum(u['outer_gain'])} & {VW[u['truth']]} & "
                + (
                    f"{frac(u['n_pick_lawgp'], u['n_levels'])} & {u['verdict']} & {tnum(u['realized_R2']['audit'])}"
                    if aud
                    else r"-- & not auditable & --"
                )
                + r" \\"
            )
            pre = f"Ext{DS_WORD[ds]}{AX_WORD[ax]}"
            key = f"summary.external.{ds}.{ax}"
            M.add(pre + "Gain", fnum(u["outer_gain"]), key + ".outer_gain")
            M.add(pre + "Truth", u["truth"], key + ".truth")
            M.add(
                pre + "Verdict",
                u["verdict"] if aud else "not auditable",
                key + ".verdict",
            )
            if aud:
                M.add(pre + "Picks", str(u["n_pick_lawgp"]), key + ".n_pick_lawgp")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_external.tex"), "w").write("\n".join(L) + "\n")

    # ---- accuracy by number of levels
    GN = {
        "real_n75": "Real grid ($n=75$)",
        "real_n72": "Real grid ($n=72$)",
        "external_full": "External, all levels",
        "external_sub": "External, sub-sampled",
        "twin": "Twin grids (seed 0)",
        "twin_sens": "Twin grids (seed 42)",
    }
    GW = {
        "real_n75": "Real",
        "real_n72": "RealSeventyTwo",
        "external_full": "ExtFull",
        "external_sub": "ExtSub",
        "external_all": "ExtAll",
        "twin": "Twin",
        "twin_sens": "TwinSens",
    }
    BW = {"3": "Three", "4": "Four", "5+": "FivePlus", "all": "All"}
    L = [
        r"\begin{tabular}{l l r r r r r r r r}",
        r"\toprule",
        r"Source & Levels & Units & Helps/hurts & Correct & Accuracy (\%) & Majority (\%) & Balanced (\%) & Folds correct (\%) & Regret \\",
        r"\midrule",
    ]
    for g in (
        "real_n75",
        "real_n72",
        "external_full",
        "external_sub",
        "twin",
        "twin_sens",
    ):
        if g not in S["by_levels"]:
            continue
        for i, b in enumerate(("3", "4", "5+", "all")):
            a = S["by_levels"][g].get(b)
            if a is None:
                continue
            L.append(
                f"{GN[g] if i == 0 else ''} & {b if b != 'all' else 'all'} & {a['n_units']} & "
                f"{a['n_truth_helps']}/{a['n_truth_hurts']} & {a['hits']}/{a['n_non_neutral']} & "
                f"{pct(a['accuracy'])} & {pct(a['accuracy_majority_baseline'])} & {pct(a['balanced_accuracy'])} & "
                f"{pct(a['fold_accuracy'])} & {tnum(a['median_regret_vs_oracle'])} \\\\"
            )
        L.append(r"\addlinespace")
    for g, d in S["by_levels"].items():
        for b, a in d.items():
            key = f"summary.by_levels.{g}.{b}"
            pre = f"Acc{GW[g]}{BW[b]}"
            M.add(pre, pct(a["accuracy"]), key + ".accuracy")
            M.add(pre + "Hits", str(a["hits"]), key + ".hits")
            M.add(pre + "Units", str(a["n_non_neutral"]), key + ".n_non_neutral")
            M.add(pre + "Folds", pct(a["fold_accuracy"]), key + ".fold_accuracy")
            M.add(pre + "Sign", pct(a["accuracy_sign"]), key + ".accuracy_sign")
            M.add(pre + "Majority", pct(a["accuracy_majority_baseline"]), key + ".accuracy_majority_baseline")
            M.add(pre + "Balanced", pct(a["balanced_accuracy"]), key + ".balanced_accuracy")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_levels.tex"), "w").write("\n".join(L) + "\n")

    # ---- accuracy by size of the outer effect |LawGP - GP|
    EN = {"small": r"small, $0.02 < |\Delta R^2| \le 0.1$", "medium": r"medium, $0.1 < |\Delta R^2| \le 0.5$",
          "large": r"large effects are detected, $|\Delta R^2| > 0.5$"}
    EW = {"small": "Small", "medium": "Medium", "large": "Large"}
    L = [r"\begin{tabular}{l l r r r r r l}", r"\toprule",
         r"Source & Outer effect & Units & Helps/hurts & Correct & Accuracy (\%) & $|\Delta R^2|$ median [p10, p90] & Axes \\",
         r"\midrule"]
    for g in ("real_n75", "external_full", "external_sub", "twin", "twin_sens"):
        for i, b in enumerate(("small", "medium", "large")):
            a = S["by_effect"][g][b]
            L.append(f"{GN[g] if i == 0 else ''} & {EN[b]} & {a['n_non_neutral']} & "
                     f"{a['n_truth_helps']}/{a['n_truth_hurts']} & {a['hits']}/{a['n_non_neutral']} & "
                     f"{pct(a['accuracy'])} & "
                     + (f"{tnum(a['abs_gain']['median'])} [{tnum(a['abs_gain']['p10'])}, {tnum(a['abs_gain']['p90'])}]"
                        if a["abs_gain"] else "--")
                     + " & " + (", ".join(f"{TX[x]} {v}" for x, v in a["axis_counts"].items()) or "--")
                     + " \\\\")
            if a["abs_gain"]:
                for q in ("median", "p10", "p90", "max"):
                    M.add(f"Eff{GW[g]}{EW[b]}Gain{q.capitalize().replace('P10', 'Low').replace('P90', 'High')}",
                          fnum(a["abs_gain"][q]), f"summary.by_effect.{g}.{b}.abs_gain.{q}")
                for x, v in a["axis_counts"].items():
                    M.add(f"Eff{GW[g]}{EW[b]}N{AX_WORD[x]}", str(v), f"summary.by_effect.{g}.{b}.axis_counts.{x}")
            M.add(f"Eff{GW[g]}{EW[b]}Acc", pct(a["accuracy"]), f"summary.by_effect.{g}.{b}.accuracy")
            M.add(f"Eff{GW[g]}{EW[b]}Units", str(a["n_non_neutral"]), f"summary.by_effect.{g}.{b}.n_non_neutral")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_effect.tex"), "w").write("\n".join(L) + "\n")

    # ---- external sub-samples: k = 3 vs 5 vs full per dataset/axis
    L = [
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Dataset & Axis & Levels kept & Subsets & LawGP helps (\%) & Median gain & Audit correct \\",
        r"\midrule",
    ]
    first = True
    for ds, d in S["external_sub"].items():
        if not first:
            L.append(r"\addlinespace")
        first = False
        for i, (ax, dd) in enumerate(d.items()):
            full = S["external"][ds][ax]
            rows = sorted(dd.items(), key=lambda kv: int(kv[0]))
            for j, (k, a) in enumerate(rows):
                L.append(
                    f"{E.DS_TEX[ds] if i == 0 and j == 0 else ''} & {TX[ax] if j == 0 else ''} & {k} & "
                    f"{a['n_units']} & {pct(a['truth_helps_share'])} & {tnum(a['median_outer_gain'])} & "
                    f"{a['hits']}/{a['n_non_neutral']} \\\\"
                )
                M.add(
                    f"Sub{DS_WORD[ds]}{AX_WORD[ax]}{ {'3': 'Three', '4': 'Four', '5': 'Five'}[str(k)] }Acc",
                    pct(a["accuracy"]),
                    f"summary.external_sub.{ds}.{ax}.{k}.accuracy",
                )
            ok = int(
                full["auditable"]
                and full["truth"] != "neutral"
                and full["verdict"] == full["truth"]
            )
            L.append(
                f" & & all ({full['n_levels']}) & 1 & {100 if full['truth'] == 'helps' else 0} & "
                f"{tnum(full['outer_gain'])} & "
                + (
                    f"{ok}/{int(full['truth'] != 'neutral')}"
                    if full["auditable"]
                    else "--"
                )
                + r" \\"
            )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_extsub.tex"), "w").write("\n".join(L) + "\n")

    # ---- twin table
    DW = {"F3": "FThree", "F5": "FFive", "f3": "fThree"}
    GEN_W = {"law": "True", "twomode": "Two"}
    DT = {
        "F3": r"3 forces, 5 freq.",
        "F5": r"5 forces, 5 freq.",
        "f3": r"3 forces, 3 freq.",
    }
    GT = {"law": "law-true", "twomode": "two-mode"}
    L = [
        r"\begin{tabular}{l l r l r r r r}",
        r"\toprule",
        r"Design & Surface & CV & Axis & Levels & LawGP helps (\%) & Median gain & Audit correct \\",
        r"\midrule",
    ]
    for des in TWIN_DESIGNS:
        if des not in S["twin"]:
            continue
        for gen in TWIN_GENS:
            for cv in TWIN_CV:
                cvk = f"cv{int(cv * 100)}"
                for i, ax in enumerate(("composition", "force", "frequency")):
                    a = S["twin"][des][gen][cvk][ax]
                    L.append(
                        f"{DT[des] if gen == TWIN_GENS[0] and cv == TWIN_CV[0] and i == 0 else ''} & "
                        f"{GT[gen] if cv == TWIN_CV[0] and i == 0 else ''} & "
                        f"{int(cv * 100) if i == 0 else ''} & {TX[ax]} & {a['n_levels']} & "
                        f"{pct(a['truth_helps_share'])} & {tnum(a['median_outer_gain'])} & "
                        f"{a['hits']}/{a['n_non_neutral']} \\\\"
                    )
        L.append(r"\addlinespace")
        for ax in ("composition", "force", "frequency"):
            a = S["twin"][des]["pooled"][ax]
            key = f"summary.twin.{des}.pooled.{ax}"
            M.add(
                f"Twin{DW[des]}{AX_WORD[ax]}Acc", pct(a["accuracy"]), key + ".accuracy"
            )
            M.add(
                f"Twin{DW[des]}{AX_WORD[ax]}Helps",
                pct(a["truth_helps_share"]),
                key + ".truth_helps_share",
            )
            if des in S.get("twin_sens", {}):
                b = S["twin_sens"][des]["pooled"][ax]
                M.add(
                    f"TwinSens{DW[des]}{AX_WORD[ax]}Acc",
                    pct(b["accuracy"]),
                    f"summary.twin_sens.{des}.pooled.{ax}.accuracy",
                )
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_twin.tex"), "w").write("\n".join(L) + "\n")

    # ---- axis choice confusion
    L = [
        r"\begin{tabular}{l l l r}",
        r"\toprule",
        r"Source & True axis & Picked axis: count & Correct \\",
        r"\midrule",
    ]
    CW = {
        "external": "Ext",
        "real_n75": "Real",
        "twin": "Twin",
        "twin_sens": "TwinSens",
    }
    CN = {
        "external": "External, all levels",
        "real_n75": r"Real grid ($n=75$)",
        "twin": "Twin (seed 0)",
        "twin_sens": "Twin (seed 42)",
    }
    DN = {"F3": r"3 forces, 5 freq. (our design)", "F5": r"5 forces, 5 freq.", "f3": r"3 forces, 3 freq."}
    CAX = ("composition", "force", "frequency", "none")
    for g in ("real_n75", "external", "twin", "twin_sens"):
        a = S["axis_choice"].get(g)
        if not a or not a["n_cases"]:
            continue
        if g in ("twin", "twin_sens"):
            # per design: rows = true axis, cells = picked axis (the pooled table hides the force bias)
            for des in TWIN_DESIGNS:
                b = a.get(des)
                if not b or not b["n_cases"]:
                    continue
                for i, t in enumerate(b["labels"]):
                    row = b["confusion_true_by_picked"][t]
                    if sum(row.values()) == 0:
                        continue
                    cells = ", ".join(f"{TX.get(p, p)} {v}" for p, v in row.items() if v)
                    L.append(f"{CN[g] + ', ' + DN[des] if i == 0 else ''} & {TX.get(t, t)} & {cells} & "
                             f"{row.get(t, 0)}/{sum(row.values())} \\\\")
                picks = ", ".join(f"{TX.get(p, p)} {v}" for p, v in b["n_picked"].items() if v)
                L.append(f" & all (picked: {picks}) & & {b['n_correct']}/{b['n_cases']} \\\\")
                L.append(r"\addlinespace")
                for p in CAX:
                    M.add(f"Choice{CW[g]}{DW[des]}Picked{AX_WORD.get(p, 'None')}", str(b["n_picked"].get(p, 0)),
                          f"summary.axis_choice.{g}.{des}.n_picked.{p}")
                    M.add(f"Choice{CW[g]}{DW[des]}True{AX_WORD.get(p, 'None')}", str(b["n_true"].get(p, 0)),
                          f"summary.axis_choice.{g}.{des}.n_true.{p}")
                M.add(f"Choice{CW[g]}{DW[des]}Correct", str(b["n_correct"]), f"summary.axis_choice.{g}.{des}.n_correct")
                M.add(f"Choice{CW[g]}{DW[des]}Cases", str(b["n_cases"]), f"summary.axis_choice.{g}.{des}.n_cases")
        for i, t in enumerate(a["labels"]):
            row = a["confusion_true_by_picked"][t]
            if sum(row.values()) == 0:
                continue
            cells = ", ".join(f"{TX.get(p, p)} {v}" for p, v in row.items() if v)
            L.append(
                f"{CN[g] if i == 0 else ''} & {TX.get(t, t)} & {cells} & {row.get(t, 0)}/{sum(row.values())} \\\\"
            )
        L.append(f" & all & & {a['n_correct']}/{a['n_cases']} \\\\")
        L.append(r"\addlinespace")
        M.add(
            f"Choice{CW[g]}Acc", pct(a["accuracy"]), f"summary.axis_choice.{g}.accuracy"
        )
        M.add(
            f"Choice{CW[g]}Correct",
            str(a["n_correct"]),
            f"summary.axis_choice.{g}.n_correct",
        )
        M.add(
            f"Choice{CW[g]}Cases", str(a["n_cases"]), f"summary.axis_choice.{g}.n_cases"
        )
        if g in ("twin", "twin_sens"):
            for des in TWIN_DESIGNS:
                if des in a:
                    M.add(
                        f"Choice{CW[g]}{DW[des]}Acc",
                        pct(a[des]["accuracy"]),
                        f"summary.axis_choice.{g}.{des}.accuracy",
                    )
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a12_choice.tex"), "w").write("\n".join(L) + "\n")

    for des in TWIN_DESIGNS:
        for gen in TWIN_GENS:
            for ax in ("composition", "force", "frequency"):
                g = S["twin_gain_scale"][des][gen][ax]
                key = f"summary.twin_gain_scale.{des}.{gen}.{ax}"
                pre = f"Scale{DW[des]}{GEN_W[gen]}{AX_WORD[ax]}"
                M.add(pre + "Inner", fnum(g["median_inner_gain"]), key + ".median_inner_gain")
                M.add(pre + "Outer", fnum(g["median_outer_gain"]), key + ".median_outer_gain")
    M.add("Margin", fnum(MARGIN), "meta.margin_R2")
    M.add("Reps", str(J["meta"]["reps"]), "meta.reps")
    M.add("RepsSens", str(J["meta"]["reps_sens"]), "meta.reps_sens")
    hdr = [
        "% numbers_a12.tex: generated by code/a12_axisrule.py (task E7); do not edit by hand.",
        "% Source of every value: results/a12_axisrule.json (key in its 'index').",
    ]
    open(os.path.join(RES, "numbers_a12.tex"), "w").write(
        "\n".join(hdr + M.lines) + "\n"
    )

    cols = [
        "family",
        "case",
        "dataset",
        "target",
        "tag",
        "design",
        "gen",
        "cv",
        "seed",
        "rep",
        "axis",
        "subset",
        "k",
        "n",
        "n_levels",
        "outer_gain",
        "truth",
        "auditable",
        "n_pick_lawgp",
        "verdict",
        "mean_inner_gain",
        "fold_hits",
    ]
    pd.DataFrame([{c: u.get(c) for c in cols} for u in J["units"]]).to_csv(
        os.path.join(RES, "a12_units.csv"), index=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-sens", type=int, default=10)
    ap.add_argument("--subsets", type=int, default=6)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-sens", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run: two external sets, 1 twin rep per cell",
    )
    args = ap.parse_args()
    if args.smoke:
        args.reps, args.reps_sens = 1, 1
    args.n_jobs = min(args.n_jobs, 4)
    jobs, twin_meta = build_jobs(args)
    # longest units first so the pool drains evenly
    jobs.sort(
        key=lambda j: -len(j[1]["y"])
        * len(np.unique(j[1]["axes"][j[2]])) ** 2
        * (5 if j[1].get("restarts") else 1)
    )
    print(f"[a12] {len(jobs)} units, n_jobs={args.n_jobs}", flush=True)
    results = run_units(jobs, args.n_jobs)
    J = summarize(results, twin_meta, args)
    J["meta"]["runtime_s"] = round(time.time() - T0, 1)
    write_outputs(J)
    with open(OUT_JSON, "w") as fh:
        json.dump(
            J,
            fh,
            indent=1,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    S = J["summary"]
    for g, d in S["by_levels"].items():
        print(
            f"[a12] {g}: "
            + ", ".join(f"{b}: {a['hits']}/{a['n_non_neutral']}" for b, a in d.items())
        )
    for g, a in S["axis_choice"].items():
        print(f"[a12] axis choice {g}: {a['n_correct']}/{a['n_cases']}")
    print(f"[a12] done in {time.time() - T0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
