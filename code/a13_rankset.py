"""E8 RANKSET: composition statements that survive specimen variability (analysis protocol 5, E8).

E3 (a8_specvar.py) showed that the full five-way composition ranking does not
survive a literature-calibrated specimen effect (P(ranking unchanged) 6 percent
at the upper CV). This script replaces the five-way ranking with weaker
statements and asks which of them survive, on the same re-draw machinery.

Re-draws. Imported from a8_specvar (not edited): specimen multipliers
m_bc = exp(s z_bc - s^2/2), s^2 = ln(1 + CV^2), z from numpy default_rng(42)
standard_normal((B, 5)), the same z for every CV; the per-composition value is
the force- and frequency-averaged target times m_bc (a8_specvar.comp_means,
multipliers). The replication variant (a new fabricated set, doubled log
variance) is reported beside it. CV = literature median and maximum study-level
CV from a8_specvar.literature() (4 and 24 percent). With B = 2000 the script
first reproduces the published a8 optimum survival (P(2 wt% best)) exactly and
stops if it does not.

Statements per draw (targets V_rms, V_pp, |V|max; n = 75 and n = 72):
  pairwise win matrix  W_ij = P(value_i > value_j);
  top-two set          P(the two best compositions are the observed two, in
                       any order), P(2 wt% among the two best), and the most
                       frequent top-two sets;
  near-optimality      P(value_2wt >= (1 - d) max_c value_c), d = 5 and 10
                       percent;
  beats the rest       P(2 wt% above PVDF, PVDF/BaTiO3 and 1 wt% jointly),
                       i.e. every composition except its observed runner-up.
A CV scan (0 to 60 percent) gives the CV at which each statement falls to
95 and 50 percent.

Hierarchical posterior versions. The Gibbs sampler of a8_specvar (partial
pooling on the log target, cell effects for the 15 force x frequency cells,
specimen effect s_c ~ N(0, ln(1+CV^2)) fixed, alpha_c ~ N(0, omega^2),
omega ~ half-Cauchy(1)) is repeated here with its draws kept (a8_specvar.gibbs
returns summaries only); the code is the a8 sampler line for line, and its
P(2 wt% best) on V_rms is checked against a8_specvar.json. The statements
are evaluated on exp(alpha_c) (composition effect without the specimen term).
The a8 hierarchy was run on V_rms only; V_pp and |V|max are an extension with
the same model.

Survival rule (fixed before the run): a statement survives at a CV when its
probability is at least 95 percent at both n = 75 and n = 72; 80 to 95 percent
is reported as 'likely', below 80 percent as 'does not survive'.

Round 2 (independent evidence on where CNT optima sit, and a specimen bound).
(A) Literature optimum. results/a13_literature_optimum.csv (hand-entered
    from the opened pages; 'where_read' and 'tier' per row) lists studies that
    vary CNT loading in PVDF-family piezoelectric composites. The optimum of
    study i is known only to lie between the geometric midpoints to its tested
    neighbours (interval censoring; unknown neighbours: best x/÷ 2; best at the
    highest loading: right-censored). Random-effects model log10 c*_i ~ N(mu,
    tau^2), mu flat, tau half-Cauchy(1 decade), grid posterior, predictive for a
    new study. Curvature: ln(y/y*) = -k (log2(c/c*))^2 per curve, log k pooled.
    A study whose best loading is its lowest nonzero loading is left-censored
    (open lower bound; Wu 2013). Shehata 2018 states 0.1 and 0.3 wt% CNT added to
    a 15 wt% PVDF solution, which is ambiguous; the 'shehata_converted' sets read
    them per solution (0.67 and 2.0 wt% of PVDF, lower bound 1.15). Basis of our
    own loadings: wt% of PVDF (as used) or wt% of total solids with 5-20 wt%
    BaTiO3 (not documented locally), converted by c / (1 - b - c); the percentile
    of our 2 wt% and the curve-prior posterior are reported for each basis.
(B) Informative prior. For the CNT compositions (1, 2, 3 wt%) the composition
    effect is alpha_c = a - k (log2 c - m)^2 + eta_c (eta ~ N(0, 0.10^2)), with
    (m, k) from (A); PVDF and PVDF/BaTiO3 stay flat. Data: cell-adjusted log
    composition means and their OLS covariance plus the specimen variance
    ln(1+CV^2). Posterior by importance weighting of prior (m, k) draws and the
    Gaussian conditional of alpha. Priors: meta sets, the two E4 external sweeps
    (Salama 1.95 wt%, Shi 5.08 wt%), shape only, flat.
(C) Specimen bound. Deviations of the composition means from a smooth
    composition factor (the fitted five-parameter law; a quadratic factor; a log
    quadratic on the BaTiO3-bearing compositions) minus the sampling variance
    give an implied specimen CV and chi-square upper bounds.

Run from the repository root:
    python code/a13_rankset.py
Options: --b (re-draws, default 2000 = a8), --iters/--burn/--chains (Gibbs),
--n-jobs (default 4), --n-prior/--n-post/--s-eta (round 2), --quick (smoke
test, writes to $A13_OUT if set).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2_common as A2  # noqa: E402
import a3_common as A3  # noqa: E402
import a8_specvar as A8  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

SEED = A8.SEED
from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
RES = Path(os.environ.get("A13_OUT", ROOT / "results"))
TAB = RES / "tables"
A8_JSON = ROOT / "results" / "a8_specvar.json"

NSPEC = A8.NSPEC
COMP_ORDER = A8.COMP_ORDER
BEST = A8.BEST  # 2 wt%
ROWS = A8.ROWS
TARGETS = A8.TARGETS
Y0 = A8.Y0
CV_SCAN = A8.CV_SCAN
DELTAS = {"5": 0.05, "10": 0.10}
SURVIVE, LIKELY = 0.95, 0.80
SHORT = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "BT",
    "PVDF+BaTiO3+%1CNT": "1CNT",
    "PVDF+BaTiO3+%2CNT": "2CNT",
    "PVDF+BaTiO3+%3CNT": "3CNT",
}
TEXNAME = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "PVDF/BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "1 wt\\%",
    "PVDF+BaTiO3+%2CNT": "2 wt\\%",
    "PVDF+BaTiO3+%3CNT": "3 wt\\%",
}
MACNAME = {
    "PVDF": "Pvdf",
    "PVDF+BaTiO3": "Bt",
    "PVDF+BaTiO3+%1CNT": "One",
    "PVDF+BaTiO3+%2CNT": "Two",
    "PVDF+BaTiO3+%3CNT": "Three",
}


# ---------------------------------------------------------------- statements
def statements(V, obs):
    """Statements on per-draw composition values V (draws x 5); obs = observed values."""
    order0 = np.argsort(-obs)
    top2_obs = frozenset(order0[:2].tolist())

    runner = [c for c in order0[:2] if c != BEST]
    others_but_runner = [c for c in range(NSPEC) if c != BEST and c not in runner]
    W = (V[:, :, None] > V[:, None, :]).mean(axis=0)
    order = np.argsort(-V, axis=1)
    t2 = np.sort(order[:, :2], axis=1)
    sets, cnt = np.unique(t2, axis=0, return_counts=True)
    k = np.argsort(-cnt)[:3]
    mx = V.max(axis=1)
    ratio_to_best_other = V[:, BEST] / np.max(np.delete(V, BEST, axis=1), axis=1)
    return {
        "observed_order": [COMP_ORDER[i] for i in order0],
        "observed_top2": sorted(COMP_ORDER[i] for i in top2_obs),
        "win": W.tolist(),
        "p_best_2wt": float(np.mean(order[:, 0] == BEST)),
        "p_top2_set": float(np.mean([frozenset(r) == top2_obs for r in t2.tolist()])),
        "p_2wt_in_top2": float(np.mean(np.any(order[:, :2] == BEST, axis=1))),
        "p_best_2or3": float(np.mean(np.isin(order[:, 0], [BEST, COMP_ORDER.index("PVDF+BaTiO3+%3CNT")]))),
        "top2_sets": [
            {"set": [COMP_ORDER[i] for i in sets[j]], "p": float(cnt[j] / len(V))}
            for j in k
        ],
        "p_within": {
            d: float(np.mean(V[:, BEST] >= (1 - dv) * mx)) for d, dv in DELTAS.items()
        },
        "beats_rest_set": [COMP_ORDER[c] for c in others_but_runner],
        "p_2wt_beats_rest": float(
            np.mean(np.all(V[:, [BEST]] > V[:, others_but_runner], axis=1))
        ),
        "ratio_2wt_to_best_other": A8.summarize(ratio_to_best_other),
    }


def scalar_keys(S):
    return {
        "p_best_2wt": S["p_best_2wt"],
        "p_top2_set": S["p_top2_set"],
        "p_2wt_in_top2": S["p_2wt_in_top2"],
        "p_best_2or3": S["p_best_2or3"],
        "p_within_5": S["p_within"]["5"],
        "p_within_10": S["p_within"]["10"],
        "p_2wt_beats_rest": S["p_2wt_beats_rest"],
    }


def redraw_block(Z, cv, rows, rep=False):
    out = {}
    for t in TARGETS:
        mu = A8.comp_means(Y0[t], rows)
        V = A8.multipliers(Z, cv, rep) * mu[None, :]
        out[t] = statements(V, mu)
    return out


def crossing_down(cvs, p, level):
    """Smallest CV at which p falls to `level` or below (linear interpolation);
    None if it never does, 0 if it is already at or below the level at CV = 0."""
    p = np.asarray(p)
    if p[0] <= level:
        return 0.0
    return A8.crossing(cvs, p, level)


# ---------------------------------------------------------------- hierarchy
def gibbs_draws(y, rows, tau2, A=1.0, iters=6000, burn=1000, chains=4, seed=SEED):
    """a8_specvar.gibbs line for line, returning the alpha draws and split R-hat."""
    rng = np.random.default_rng(seed)
    ly = np.log(y[rows])
    sp = A8.SPEC[rows]
    cell_id = pd.factorize(list(zip(A2.FRC[rows], A2.FRQ[rows])))[0]
    ncell = cell_id.max() + 1
    n = len(ly)
    n_c = np.bincount(sp, minlength=NSPEC)
    n_cell = np.bincount(cell_id, minlength=ncell)
    draws = []
    for ch in range(chains):
        u = rng.normal(0, 0.1, NSPEC)
        alpha = u.copy()
        om2 = 0.1
        a_aux = 1.0
        s2 = 0.1
        keep = []
        for it in range(iters):
            r = ly - u[sp]
            gm = np.bincount(cell_id, weights=r, minlength=ncell) / n_cell
            gam = gm + rng.standard_normal(ncell) * np.sqrt(s2 / n_cell)
            r = ly - gam[cell_id]
            sc = np.bincount(sp, weights=r, minlength=NSPEC)
            prec = n_c / s2 + 1.0 / (om2 + tau2)
            u = sc / s2 / prec + rng.standard_normal(NSPEC) / np.sqrt(prec)
            if tau2 > 0:
                w = om2 / (om2 + tau2)
                alpha = w * u + rng.standard_normal(NSPEC) * np.sqrt(w * tau2)
            else:
                alpha = u.copy()
            om2 = 1.0 / rng.gamma(
                (NSPEC + 1) / 2.0, 1.0 / (1.0 / a_aux + 0.5 * np.sum(alpha**2))
            )
            a_aux = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / A**2 + 1.0 / om2))
            res = ly - gam[cell_id] - u[sp]
            s2 = 1.0 / rng.gamma(n / 2.0, 1.0 / (0.5 * np.sum(res**2)))
            if it >= burn:
                keep.append(np.concatenate((alpha, u, [np.sqrt(om2), np.sqrt(s2)])))
        draws.append(np.array(keep))
    D = np.array(draws)
    half = D.shape[1] // 2
    S = np.concatenate([D[:, :half], D[:, half : 2 * half]], axis=0)
    W = S.var(axis=1, ddof=1).mean(axis=0)
    Bv = S.mean(axis=1).var(axis=0, ddof=1) * half
    rhat = np.sqrt(((half - 1) / half * W + Bv / half) / W)
    flat = D.reshape(-1, D.shape[2])
    return flat[:, :NSPEC], float(np.max(rhat[:NSPEC]))


def hier_job(tag, t, name, cv, iters, burn, chains):
    rows = ROWS[tag]
    al, rhat = gibbs_draws(
        Y0[t], rows, A8.sig2(cv), iters=iters, burn=burn, chains=chains
    )
    V = np.exp(al)  # composition effect on the ratio scale (cell effects cancel)
    obs = A8.comp_means(Y0[t], rows)
    S = statements(V, obs)
    S["rhat_max_alpha"] = rhat
    S["n_draws"] = int(len(al))
    S["cv"] = cv
    return (tag, t, name), S


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-prior", type=int, default=200000)
    ap.add_argument("--n-post", type=int, default=40000)
    ap.add_argument("--s-eta", type=float, default=0.10)
    args = ap.parse_args()
    if args.quick:
        args.b, args.iters, args.burn = 200, 600, 100
        args.n_prior, args.n_post = 20000, 4000
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    J = {
        "task": "E8 RANKSET",
        "seed": SEED,
        "script": "code/a13_rankset.py",
        "reuses": "a8_specvar.literature, comp_means, multipliers, sig2, crossing, summarize (imported); a8_specvar.gibbs repeated with draws kept",
        "survival_rule": f"survives if P >= {SURVIVE} at n=75 and n=72; likely if >= {LIKELY}; otherwise does not survive (fixed before the run)",
    }
    lit = A8.literature()
    CVS = {"median": lit["cv_median"], "upper": lit["cv_max"]}
    J["cv_used"] = CVS
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((args.b, NSPEC))
    J["draws"] = {
        "B": args.b,
        "stream": "numpy default_rng(42).standard_normal((B, 5)), as a8_specvar (same z for every CV)",
        "gibbs": {"iters": args.iters, "burn": args.burn, "chains": args.chains},
    }
    print(
        f"[a13] CV median {CVS['median']:.4f}, upper {CVS['upper']:.4f}, B={args.b}",
        flush=True,
    )

    # ---------------- (0) reproduce the published a8 optimum survival
    a8 = json.load(open(A8_JSON))
    check = {}
    for tag, rows in ROWS.items():
        for name, cv in CVS.items():
            R = redraw_block(Z, cv, rows)
            for t in TARGETS:
                mine = R[t]["p_best_2wt"]
                pub = a8["ranking"]["at_cv"][tag][name][t]["p_opt"]
                check[f"{tag}/{name}/{t}"] = {"a13": mine, "a8": pub}
    ok = all(abs(v["a13"] - v["a8"]) < 1e-12 for v in check.values())
    J["reproduce_a8"] = {"optimum_survival": check, "exact": ok}
    print(
        f"[a13] a8 reproduction ({'exact' if ok else 'MISMATCH'}): n75 rms median "
        f"{check['n75/median/rms_Voc']['a13']:.4f} (a8 {check['n75/median/rms_Voc']['a8']:.4f}), "
        f"upper {check['n75/upper/rms_Voc']['a13']:.4f} (a8 {check['n75/upper/rms_Voc']['a8']:.4f})",
        flush=True,
    )
    if args.b == a8["draws"]["B_rank"] and not ok:
        raise SystemExit("[a13] a8 optimum survival not reproduced; stopping")

    # ---------------- (1) re-draw statements at the two CVs
    boot = {}
    for tag, rows in ROWS.items():
        boot[tag] = {}
        for name, cv in CVS.items():
            boot[tag][name] = {
                "cv": cv,
                "single": redraw_block(Z, cv, rows),
                "replication": redraw_block(Z, cv, rows, rep=True),
            }
    J["redraw"] = boot

    # ---------------- (2) CV scan of the scalar statements (single variant)
    scan = {}
    flips = {}
    for tag, rows in ROWS.items():
        scan[tag], flips[tag] = {}, {}
        for t in TARGETS:
            mu = A8.comp_means(Y0[t], rows)
            per = [
                scalar_keys(statements(A8.multipliers(Z, cv) * mu[None, :], mu))
                for cv in CV_SCAN
            ]
            scan[tag][t] = {k: [p[k] for p in per] for k in per[0]}
            flips[tag][t] = {
                k: {
                    "cv_at_95": crossing_down(CV_SCAN, v, SURVIVE),
                    "cv_at_50": crossing_down(CV_SCAN, v, 0.5),
                }
                for k, v in scan[tag][t].items()
            }
    J["cv_scan"] = {"cv": CV_SCAN.tolist(), "p": scan}
    J["flip_cv"] = flips
    with open(RES / "a13_figdata_scan.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(scan["n75"][TARGETS[0]].keys())
        w.writerow(["rows", "target", "cv"] + keys)
        for tag in ROWS:
            for t in TARGETS:
                for i, cv in enumerate(CV_SCAN):
                    w.writerow(
                        [tag, t, f"{cv:.2f}"]
                        + [f"{scan[tag][t][k][i]:.5f}" for k in keys]
                    )

    # ---------------- (3) hierarchical posterior versions
    jobs = [
        (tag, t, name, cv)
        for tag in ROWS
        for t in TARGETS
        for name, cv in (
            ("none", 0.0),
            ("median", CVS["median"]),
            ("upper", CVS["upper"]),
        )
    ]
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(hier_job)(tag, t, name, cv, args.iters, args.burn, args.chains)
        for tag, t, name, cv in jobs
    )
    hier = {}
    for (tag, t, name), S in res:
        hier.setdefault(tag, {}).setdefault(name, {})[t] = S
    hcheck = {}
    for tag in ROWS:
        for name in ("none", "median", "upper"):
            hcheck[f"{tag}/{name}"] = {
                "a13": hier[tag][name]["rms_Voc"]["p_best_2wt"],
                "a8": a8["hierarchy"][tag][name]["gibbs"]["p_best_2wt"],
            }
    hok = all(abs(v["a13"] - v["a8"]) < 1e-12 for v in hcheck.values())
    J["reproduce_a8"]["hierarchy_p_best_2wt"] = hcheck
    J["reproduce_a8"]["hierarchy_exact"] = hok
    print(
        f"[a13] a8 hierarchy reproduction: {'exact' if hok else 'MISMATCH'} {hcheck}",
        flush=True,
    )
    J["hierarchy"] = hier
    J["hierarchy_model"] = (
        "a8_specvar hierarchy on the log target: log y_ck = gamma_cell(k) + alpha_c + s_c + e_ck; "
        "s_c ~ N(0, ln(1+CV^2)) fixed; alpha_c ~ N(0, omega^2), omega ~ half-Cauchy(1); statements on exp(alpha_c); "
        "V_rms reproduces a8, V_pp and |V|max are an extension"
    )

    # ---------------- (4) verdicts
    J["verdict"] = verdicts(J)

    # ---------------- (5) round 2: literature optimum, informative prior, specimen bound
    J["round2"] = round2(CVS, args)
    J["runtime_s"] = time.time() - t0
    json.dump(A3.to_plain(J), open(RES / "a13_rankset.json", "w"), indent=1)
    write_tex(J)
    write_tex_r2(J)
    for line in J["verdict"]["lines"] + J["round2"]["verdict_r2"]:
        print("[a13] " + line, flush=True)
    print(f"[a13] done in {J['runtime_s']:.0f}s", flush=True)


STMT_LABEL = {
    "p_best_2wt": "2 wt% is the best composition (a8 reference)",
    "p_2wt_in_top2": "2 wt% is among the two best compositions",
    "p_top2_set": "the observed top two are the top two (any order)",
    "p_within_5": "2 wt% is within 5 percent of the best composition",
    "p_within_10": "2 wt% is within 10 percent of the best composition",
    "p_2wt_beats_rest": "2 wt% beats every composition except its observed runner-up",
    "p_best_2or3": "the best composition is 2 or 3 wt% CNT",
}


def classify(p75, p72):
    lo = min(p75, p72)
    return (
        "survives"
        if lo >= SURVIVE
        else ("likely" if lo >= LIKELY else "does not survive")
    )


def verdicts(J):
    out = {"table": {}, "lines": []}
    for method in ("redraw", "hierarchy"):
        out["table"][method] = {}
        for name in ("median", "upper"):
            out["table"][method][name] = {}
            for t in TARGETS:
                g = lambda tag: (  # noqa: E731
                    J["redraw"][tag][name]["single"][t]
                    if method == "redraw"
                    else J["hierarchy"][tag][name][t]
                )
                a, b = scalar_keys(g("n75")), scalar_keys(g("n72"))
                out["table"][method][name][t] = {
                    k: {"n75": a[k], "n72": b[k], "verdict": classify(a[k], b[k])}
                    for k in a
                }
    for method in ("redraw", "hierarchy"):
        for name in ("median", "upper"):
            for k, lab in STMT_LABEL.items():
                v = [out["table"][method][name][t][k]["verdict"] for t in TARGETS]
                ps = [
                    min(
                        out["table"][method][name][t][k]["n75"],
                        out["table"][method][name][t][k]["n72"],
                    )
                    for t in TARGETS
                ]
                out["lines"].append(
                    f"{method:9s} CV {name:6s} | {lab}: "
                    + ", ".join(
                        f"{t} {vv} ({100 * p:.0f}%)" for t, vv, p in zip(TARGETS, v, ps)
                    )
                )
    # pairwise statements "i beats j" (observed direction at n = 75) that survive
    out["pairs_surviving"] = {}
    for method in ("redraw", "hierarchy"):
        out["pairs_surviving"][method] = {}
        for name in ("none", "median", "upper"):
            if method == "redraw" and name == "none":
                continue
            out["pairs_surviving"][method][name] = {}
            for t in TARGETS:
                mu = A8.comp_means(Y0[t], ROWS["n75"])
                g = lambda tag: np.array(  # noqa: E731
                    J["redraw"][tag][name]["single"][t]["win"]
                    if method == "redraw"
                    else J["hierarchy"][tag][name][t]["win"]
                )
                W75, W72 = g("n75"), g("n72")
                keep = []
                for i, j in combinations(range(NSPEC), 2):
                    a, b = (i, j) if mu[i] > mu[j] else (j, i)
                    lo = min(W75[a, b], W72[a, b])
                    if lo >= SURVIVE:
                        keep.append(
                            {
                                "pair": f"{SHORT[COMP_ORDER[a]]} > {SHORT[COMP_ORDER[b]]}",
                                "p_min": float(lo),
                            }
                        )
                out["pairs_surviving"][method][name][t] = keep
                out["lines"].append(
                    f"{method:9s} CV {name:6s} | surviving pairs {t}: "
                    + (
                        ", ".join(
                            f"{d['pair']} ({100 * d['p_min']:.0f}%)" for d in keep
                        )
                        or "none"
                    )
                )
    return out


# ---------------------------------------------------------------- tex
def p100(x):
    return "--" if x is None else f"{100 * x:.0f}"


def write_tex(J):
    TAB.mkdir(parents=True, exist_ok=True)
    L = [
        "% numbers_a13.tex, written by code/a13_rankset.py (E8 RANKSET). Do not edit."
    ]

    def mac(name, val, comment):
        L.append(f"\\newcommand{{\\rks{name}}}{{{val}}}  % {comment}")

    mac("NDraws", J["draws"]["B"], "specimen re-draws (a8 stream)")
    mac("CvMed", p100(J["cv_used"]["median"]), "percent, literature median CV")
    mac("CvUp", p100(J["cv_used"]["upper"]), "percent, literature upper CV")
    mac("SurviveThr", p100(SURVIVE), "percent, survival threshold at n=75 and n=72")
    SK = {
        "p_best_2wt": "Best",
        "p_2wt_in_top2": "InTopTwo",
        "p_top2_set": "TopTwoSet",
        "p_within_5": "WithinFive",
        "p_within_10": "WithinTen",
        "p_2wt_beats_rest": "BeatsRest",
        "p_best_2or3": "BestTwoOrThree",
    }
    CVN = {"median": "Med", "upper": "Up"}
    NN = {"n75": "", "n72": "NSeventyTwo"}
    for tag in ROWS:
        for name in ("median", "upper"):
            for t in TARGETS:
                tm = A2.TARGET_MACRO[t]
                for var, suf in (("single", ""), ("replication", "Rep")):
                    S = scalar_keys(J["redraw"][tag][name][var][t])
                    for k, s in SK.items():
                        mac(
                            f"{s}{CVN[name]}{tm}{suf}{NN[tag]}",
                            p100(S[k]),
                            f"percent, re-draw {var}, {tag}, CV {name}, {k}",
                        )
                H = scalar_keys(J["hierarchy"][tag][name][t])
                for k, s in SK.items():
                    mac(
                        f"{s}{CVN[name]}{tm}Hier{NN[tag]}",
                        p100(H[k]),
                        f"percent, hierarchical posterior, {tag}, CV {name}, {k}",
                    )
                # pairwise wins, each unordered pair as P(observed winner of the pair wins)
                for src, suf in (("redraw", ""), ("hier", "Hier")):
                    S = (
                        J["redraw"][tag][name]["single"][t]
                        if src == "redraw"
                        else J["hierarchy"][tag][name][t]
                    )
                    W = np.array(S["win"])
                    mu = A8.comp_means(Y0[t], ROWS[tag])
                    for i, j in combinations(range(NSPEC), 2):
                        a, b = (i, j) if mu[i] > mu[j] else (j, i)
                        mac(
                            f"Win{MACNAME[COMP_ORDER[a]]}{MACNAME[COMP_ORDER[b]]}{CVN[name]}{tm}{suf}{NN[tag]}",
                            p100(W[a, b]),
                            f"percent, P({SHORT[COMP_ORDER[a]]} > {SHORT[COMP_ORDER[b]]}), {src}, {tag}, CV {name}",
                        )
    for tag in ROWS:
        for t in TARGETS:
            for k, s in SK.items():
                fc = J["flip_cv"][tag][t][k]
                for lev, ln in (("cv_at_95", "NinetyFive"), ("cv_at_50", "Half")):
                    x = fc[lev]
                    mac(
                        f"Cv{ln}{s}{A2.TARGET_MACRO[t]}{NN[tag]}",
                        "$>$60" if x is None else p100(x),
                        f"percent CV at which {k} falls to {lev[-2:]} percent, {tag}",
                    )
    mac(
        "ReproExact",
        "yes" if J["reproduce_a8"]["exact"] else "no",
        "a8 optimum survival reproduced exactly",
    )
    mac(
        "ReproHierExact",
        "yes" if J["reproduce_a8"]["hierarchy_exact"] else "no",
        "a8 hierarchy P(best) reproduced exactly",
    )
    (RES / "numbers_a13.tex").write_text("\n".join(L) + "\n")

    # statements table: target x n, statements at median / upper, re-draw and hierarchy
    tl = {
        "rms_Voc": r"$V_\mathrm{rms}$",
        "Vpp": r"$V_\mathrm{pp}$",
        "Vmax": r"$|V|_\mathrm{max}$",
    }
    cols = [
        "p_best_2wt",
        "p_2wt_in_top2",
        "p_top2_set",
        "p_within_5",
        "p_within_10",
        "p_2wt_beats_rest",
    ]
    head = (
        "\\toprule\n & & \\multicolumn{2}{c}{2 wt\\% best} & \\multicolumn{2}{c}{2 wt\\% in top two} & "
        "\\multicolumn{2}{c}{Top-two set} & \\multicolumn{2}{c}{Within 5\\%} & \\multicolumn{2}{c}{Within 10\\%} & "
        "\\multicolumn{2}{c}{Beats all but runner-up} \\\\\n"
        + "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(6))
        + "\nTarget & $n$ & "
        + " & ".join(["Med. & Upper"] * 6)
        + " \\\\\n\\midrule\n"
    )
    for method, fn in (
        ("redraw", "tab_a13_statements.tex"),
        ("hierarchy", "tab_a13_statements_hier.tex"),
    ):
        rows = []
        for tag, nlab in (("n75", "75"), ("n72", "72")):
            for t in TARGETS:
                cells = []
                for k in cols:
                    for name in ("median", "upper"):
                        S = (
                            J["redraw"][tag][name]["single"][t]
                            if method == "redraw"
                            else J["hierarchy"][tag][name][t]
                        )
                        cells.append(p100(scalar_keys(S)[k]))
                rows.append(f"{tl[t]} & {nlab} & " + " & ".join(cells) + " \\\\")
        model = "re-draw (a8 specimen re-draw, specimen effect only)" if method == "redraw" else "hierarchical (a8 Gibbs, exchangeable composition prior)"
        (TAB / fn).write_text(f"% model: {model}\n" + head + "\n".join(rows) + "\n\\bottomrule\n")

    # pairwise tables for V_rms: row beats column, median | upper
    order = [COMP_ORDER.index(c) for c in COMP_ORDER]
    for tag, suf in (("n75", ""), ("n72", "_n72")):
        for method, msuf in (("redraw", ""), ("hierarchy", "_hier")):
            Ws = {}
            for name in ("median", "upper"):
                S = (
                    J["redraw"][tag][name]["single"]["rms_Voc"]
                    if method == "redraw"
                    else J["hierarchy"][tag][name]["rms_Voc"]
                )
                Ws[name] = np.array(S["win"])
            short = [TEXNAME[COMP_ORDER[c]] for c in order]
            body = (
                "\\toprule\n & \\multicolumn{5}{c}{CV median} & \\multicolumn{5}{c}{CV upper} \\\\\n"
                "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
                "Row beats column & " + " & ".join(short + short) + " \\\\\n\\midrule\n"
            )
            lines = []
            for i in order:
                cells = []
                for name in ("median", "upper"):
                    cells += ["--" if i == j else p100(Ws[name][i, j]) for j in order]
                lines.append(
                    f"{TEXNAME[COMP_ORDER[i]]} & " + " & ".join(cells) + " \\\\"
                )
            model = "re-draw (a8 specimen re-draw)" if method == "redraw" else "hierarchical (a8 Gibbs, exchangeable prior)"
            (TAB / f"tab_a13_pairwise{msuf}{suf}.tex").write_text(
                f"% model: {model}; V_rms, n = {tag[1:]}\n" + body + "\n".join(lines) + "\n\\bottomrule\n"
            )


# ================================================================ round 2
LIT_OPT = ROOT / "results" / "a13_literature_optimum.csv"
CNT_LEVELS = np.array([1.0, 2.0, 3.0])  # CNT-bearing compositions of the grid
CNT_IDX = [COMP_ORDER.index(f"PVDF+BaTiO3+%{i}CNT") for i in (1, 2, 3)]
LOG2_10 = np.log2(10.0)
# Shehata 2018 adds 0.1 and 0.3 wt% CNT to a 15 wt% PVDF solution (PMC6403798 Sec. 2.2);
# read per solution these are 0.667 and 2.0 wt% of PVDF, so the right-censored optimum
# is at least the geometric midpoint sqrt(0.667 * 2.0) = 1.155 wt% of PVDF.
SHEHATA_PVDF_LO = float(np.sqrt(0.1 / 0.15 * 0.3 / 0.15))
# Basis of our own loadings: the source study's wt% basis is not documented locally.
# (a) wt% of PVDF (as used); (b) wt% of total solids with BaTiO3 at b of the solids,
# b in 5-20 wt% (BaTiO3 fraction not documented locally), converted to wt% of PVDF
# as c / (1 - b - c).
BASES = {"pvdf": None, "solids_bt5": 0.05, "solids_bt10": 0.10, "solids_bt15": 0.15, "solids_bt20": 0.20}


def to_pvdf_basis(c_pct, b):
    if b is None:
        return np.asarray(c_pct, float)
    c = np.asarray(c_pct, float) / 100.0
    return 100.0 * c / (1.0 - b - c)
SCREENED_OUT = [
    ("10.1007/s11664-024-11463-5", "Verma 2024 J. Electron. Mater.: the KNNLTS content is varied, not the MWCNT content"),
    ("10.1016/j.ceramint.2022.02.047", "Shoorangiz 2022 Ceram. Int.: optimum CNT loading not stated in the abstract, full text not reachable"),
    ("10.1016/j.sna.2013.01.007", "Liu 2013 Sens. Actuators A: abstract gives no loading optimum, full text not reachable"),
    ("10.3390/nano8060420", "Wu 2018 Nanomaterials: a single CNT loading"),
    ("10.3390/nano8121021", "Wang 2018 Nanomaterials: a single MWCNT loading (0.2 wt%)"),
    ("10.1016/j.matdes.2021.109785", "Eun 2021 Mater. Des.: the abstract reports beta fraction and strength only, full text blocked"),
    ("10.3390/cryst13121626", "Crystals 2023 PVDF-TrFE/MWCNT nanowires: loading choice from unpublished 'stability' results"),
    ("10.1021/acsanm.5c00006", "ACS ANM 2025 BaTiO3/f-MWCNT hybrid: the hybrid filler ratio and loading vary together"),
    ("10.1007/s10853-025-11872-9", "Koç 2025 J. Mater. Sci.: source of our own grid, never part of the prior"),
]


def lit_rows():
    rows = list(csv.DictReader(open(LIT_OPT)))
    for r in rows:
        r["lo"] = float(r["lower_wt_pct"]) if r["lower_wt_pct"] else 0.0  # 0 = left-censored
        r["hi"] = float(r["upper_wt_pct"]) if r["upper_wt_pct"] else np.inf
        r["best"] = float(r["best_wt_pct"])
        pts = []
        for kv in r["curve_points"].split(";") if r["curve_points"] else []:
            c, v = kv.split(":")
            pts.append((float(c), float(v)))
        r["pts"] = pts
    return rows


def meta_sets(rows):
    prim = [r for r in rows if r["use_primary"] == "1"]
    ext = [r for r in rows if r["tier"] == "E"]
    series = [r for r in rows if r["study"] == "Attaoui2025" and r["series"].startswith("BT")]
    return {
        "primary": prim,
        "independent": [r for r in prim if r["independent_of_our_group"] == "1"],
        "full_text": [r for r in prim if r["tier"] == "A"],
        "all": [r for r in rows if r["use_all"] == "1"],
        "primary_series": [r for r in prim if r["study"] != "Attaoui2025"] + series,
        "primary_plus_external": prim + ext,
        "shehata_converted": [
            dict(r, lo=SHEHATA_PVDF_LO) if r["study"] == "Shehata2018" else r for r in prim
        ],
        "full_text_shehata_converted": [
            dict(r, lo=SHEHATA_PVDF_LO) if r["study"] == "Shehata2018" else r
            for r in prim
            if r["tier"] == "A"
        ],
    }


def meta_fit(rs, seed=SEED, ndraw=200000):
    """Random-effects model of log10 optimum location, interval-censored by the tested grid:
    log10 c*_i ~ N(mu, tau^2), c*_i observed only as lying in [lo_i, hi_i]; mu flat,
    tau half-Cauchy(1 decade); grid posterior; predictive draws for a new study."""
    from scipy.stats import norm

    L = np.array([np.log10(r["lo"]) if r["lo"] > 0 else -np.inf for r in rs])
    U = np.array([np.log10(r["hi"]) if np.isfinite(r["hi"]) else np.inf for r in rs])
    mu = np.linspace(-3.0, 2.0, 1001)
    tau = np.linspace(0.005, 2.5, 500)
    M, T = np.meshgrid(mu, tau, indexing="ij")
    ll = np.zeros_like(M)
    for lo, hi in zip(L, U):
        p = norm.cdf((hi - M) / T) - norm.cdf((lo - M) / T)
        ll += np.log(np.clip(p, 1e-300, None))
    lp = ll + np.log(1.0 / (1.0 + T**2))
    w = np.exp(lp - lp.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    k = rng.choice(w.size, size=ndraw, p=w.ravel())
    mu_d, tau_d = M.ravel()[k], T.ravel()[k]
    pred = mu_d + tau_d * rng.standard_normal(ndraw)  # log10 wt% of a new study
    c = 10**pred
    i_ml = np.unravel_index(np.argmax(ll), ll.shape)
    pct3 = lambda v: [float(x) for x in np.percentile(v, [2.5, 50, 97.5])]  # noqa: E731
    return {
        "n_studies": len(rs),
        "studies": [r["study"] + (f"/{r['series']}" if r["series"] else "") for r in rs],
        "mu_log10": pct3(mu_d),
        "tau_log10": pct3(tau_d),
        "optimum_median_wt_pct_ci": [float(10**x) for x in pct3(mu_d)],
        "ml_mu_log10": float(mu[i_ml[0]]),
        "ml_tau_log10": float(tau[i_ml[1]]),
        "predictive_wt_pct": pct3(c),
        "p_pred_in_1p5_2p5": float(np.mean((c >= 1.5) & (c <= 2.5))),
        "p_pred_ge_1p5": float(np.mean(c >= 1.5)),
        "p_pred_le_1": float(np.mean(c <= 1.0)),
        "percentile_of_2wt": float(np.mean(c <= 2.0)),
        "_pred_log10": pred,
    }


def curvature(pts, best, side="all"):
    """k of ln(y/y*) = -k (log2(c/c*))^2 by least squares through the optimum (c > 0 only)."""
    y_best = dict(pts).get(best)
    if y_best is None:
        return None
    x, r = [], []
    for c, v in pts:
        if c <= 0 or c == best:
            continue
        if side == "up" and c < best:
            continue
        x.append(np.log2(c / best))
        r.append(-np.log(v / y_best))
    if not x:
        return None
    x, r = np.array(x), np.array(r)
    return float(np.sum(r * x**2) / np.sum(x**4))


def comp_effects(y, rows):
    """Cell-adjusted composition effects on log y and their covariance (OLS, 15 cell
    effects shared by the compositions; one cell dummy dropped so the five composition
    effects carry a common arbitrary offset)."""
    ly = np.log(y[rows])
    sp = A8.SPEC[rows]
    cell = pd.factorize(list(zip(A2.FRC[rows], A2.FRQ[rows])))[0]
    Xc = np.eye(cell.max() + 1)[cell][:, 1:]
    X = np.column_stack([np.eye(NSPEC)[sp], Xc])
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ ly)
    res = ly - X @ beta
    dof = len(ly) - X.shape[1]
    s2 = float(res @ res / dof)
    cov = s2 * np.linalg.inv(XtX)[:NSPEC, :NSPEC]
    return beta[:NSPEC], cov, s2


def curve_posterior(yhat, cov, tau2, m2, kk, s_eta, n_post, rng, flat=False, levels=CNT_LEVELS):
    """Posterior of the five composition effects alpha under the curve prior.

    alpha_PVDF, alpha_BT flat; for the CNT compositions c = 1, 2, 3:
    alpha_c = a - k (log2 c - m)^2 + eta_c, a flat, eta ~ N(0, s_eta^2);
    (m, k) drawn from the literature prior (m2 in log2 wt%); data yhat ~ N(alpha, cov + tau2 I).
    Flat components use variance BIG^2 (BIG = 10 on the log scale). Importance weights on
    the (m, k) draws, resampling, then the Gaussian conditional of alpha given (m, k)."""
    BIG2 = 100.0
    S0 = np.zeros((NSPEC, NSPEC))
    for c in range(NSPEC):
        if c not in CNT_IDX:
            S0[c, c] = BIG2
    for i in CNT_IDX:
        for j in CNT_IDX:
            S0[i, j] = BIG2 + (s_eta**2 if i == j else 0.0)
    Sig = cov + tau2 * np.eye(NSPEC)
    A = np.linalg.inv(S0 + Sig)
    if flat:
        mu0 = np.zeros((1, NSPEC))
        idx = np.zeros(n_post, dtype=int)
        ess = float("nan")
    else:
        f = -kk[:, None] * (np.log2(levels)[None, :] - m2[:, None]) ** 2
        mu0 = np.zeros((len(m2), NSPEC))
        mu0[:, CNT_IDX] = f
        r = yhat[None, :] - mu0
        q = np.einsum("ij,jk,ik->i", r, A, r)
        lw = -0.5 * q
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ess = float(1.0 / np.sum(w**2))
        idx = rng.choice(len(m2), size=n_post, p=w)
    mpost = mu0[idx] + (S0 @ A @ (yhat[None, :] - mu0[idx]).T).T
    C = S0 - S0 @ A @ S0
    C = 0.5 * (C + C.T)
    Lc = np.linalg.cholesky(C + 1e-12 * np.eye(NSPEC))
    al = mpost + rng.standard_normal((n_post, NSPEC)) @ Lc.T
    return al, idx, ess


def al_statements(al):
    V = np.exp(al - al.max(axis=1, keepdims=True))
    order = np.argsort(-al, axis=1)
    rel = V[:, BEST]
    return {
        "p_best_2wt": float(np.mean(order[:, 0] == BEST)),
        "p_2wt_in_top2": float(np.mean(np.any(order[:, :2] == BEST, axis=1))),
        "p_within_5": float(np.mean(rel >= 0.95)),
        "p_within_10": float(np.mean(rel >= 0.90)),
        "p_2wt_best_of_cnt": float(np.mean(np.argmax(al[:, CNT_IDX], axis=1) == 1)),
        "p_best_2or3": float(np.mean(np.isin(order[:, 0], CNT_IDX[1:]))),
    }


def spec_bound(y, rows, key):
    """Specimen-effect bound from smoothness across compositions (see round-2 doc)."""
    from scipy.stats import chi2

    yhat, cov, s2 = comp_effects(y, rows)
    v_s = float(np.mean(np.diag(cov) - (np.sum(cov) - np.trace(cov)) / (NSPEC * (NSPEC - 1))))
    cnt = np.array([A2.CNT[rows][A8.SPEC[rows] == c][0] for c in range(NSPEC)])
    out = {"sampling_var_per_composition": v_s, "sigma_record": float(np.sqrt(s2))}
    fits = {}
    # (a) law composition factor a0 + a1 c + a2 c^2 on the linear scale, five compositions
    g = np.exp(yhat - yhat.mean())
    Xq = np.column_stack([np.ones(NSPEC), cnt, cnt**2])
    b = np.linalg.lstsq(Xq, g, rcond=None)[0]
    fits["law_quadratic_linear_scale"] = (np.log(g / (Xq @ b)), 2)
    # (b) quadratic on the log scale, five compositions
    bl = np.linalg.lstsq(Xq, yhat, rcond=None)[0]
    fits["quadratic_log_scale"] = (yhat - Xq @ bl, 2)
    # (c) quadratic on the log scale, the four BaTiO3-bearing compositions only
    keep = np.array([c != COMP_ORDER.index("PVDF") for c in range(NSPEC)])
    bl4 = np.linalg.lstsq(Xq[keep], yhat[keep], rcond=None)[0]
    fits["quadratic_log_scale_bt_only"] = (yhat[keep] - Xq[keep] @ bl4, 1)
    # (d) the paper's fitted five-parameter law (a2_common.fit_law5 on all rows), V_rms only
    if key == "rms_Voc":
        pred, pv, info = A2.fit_law5(rows, y)
        lr = np.log(y[rows] / pred(rows))
        rc = np.array([lr[A8.SPEC[rows] == c].mean() for c in range(NSPEC)])
        fits["paper_law5_fit"] = (rc - rc.mean(), 2)
    for name, (r, dof) in fits.items():
        rss = float(np.sum((r - r.mean()) ** 2)) if name == "paper_law5_fit" else float(np.sum(r**2))
        s2_tot = rss / dof
        s2_spec = max(0.0, s2_tot - v_s)
        ub = {}
        for lev in (0.5, 0.9, 0.95):
            s2u = max(0.0, rss / chi2.ppf(1 - lev, dof) - v_s)
            ub[f"{int(lev * 100)}"] = float(np.sqrt(np.expm1(s2u)))
        p_cons = {
            f"cv{int(round(100 * cv))}": float(chi2.sf(rss / (A8.sig2(cv) + v_s), dof))
            for cv in (0.0, 0.0374, 0.2381)
        }
        out[name] = {
            "dof": dof,
            "p_consistent_with_cv": p_cons,
            "residuals_log": [float(x) for x in r],
            "rss": rss,
            "cv_point": float(np.sqrt(np.expm1(s2_spec))),
            "cv_upper": ub,
            "cv_total_misfit": float(np.sqrt(np.expm1(s2_tot))),
        }
    return out


def round2(CVS, args):
    rng = np.random.default_rng(SEED)
    rows = lit_rows()
    sets = meta_sets(rows)
    R = {
        "literature_csv": "results/a13_literature_optimum.csv",
        "screened_out": [{"doi": d, "reason": s} for d, s in SCREENED_OUT],
        "loading_note": "loadings are wt% as reported (relative to the polymer where the paper says so); two papers giving wt% of the spinning solution were converted to wt% of PVDF",
    }
    meta = {name: meta_fit(rs, ndraw=args.n_prior) for name, rs in sets.items()}
    # curvature (shape of the drop around the optimum)
    kst = {}
    for r in rows:
        if r["pts"] and r["tier"] in ("A", "E") and r["series"] != "study" and r["study"] != "Li2020":
            key = r["study"] + (f"/{r['series']}" if r["series"] and r["series"] != "external" else "")
            kst[key] = {
                "k_all": curvature(r["pts"], r["best"], "all"),
                "k_up": curvature(r["pts"], r["best"], "up"),
                "rel_drop_first_above": float(r["rel_drop_first_above"]) if r["rel_drop_first_above"] else None,
                "external": r["tier"] == "E",
            }
    # study-level log k (Attaoui series averaged in the log)
    def study_k(which, ext):
        by = {}
        for key, v in kst.items():
            if v["external"] != ext or v[which] is None or v[which] <= 0:
                continue
            by.setdefault(key.split("/")[0], []).append(np.log(v[which]))
        return {s: float(np.mean(v)) for s, v in by.items()}

    lk = study_k("k_all", False)
    lk_up = study_k("k_up", False)
    lk_ext = study_k("k_all", True)
    ours = {}
    for t in TARGETS:
        mu = A8.comp_means(Y0[t], ROWS["n75"])
        pts = [(c, mu[i]) for c, i in zip(CNT_LEVELS, CNT_IDX)]
        ours[t] = {
            "k_all": curvature(pts, 2.0, "all"),
            "k_up": curvature(pts, 2.0, "up"),
            "rel_drop_2_to_3": float(mu[CNT_IDX[2]] / mu[CNT_IDX[1]] - 1),
        }
    lkv = np.array(list(lk.values()))
    R["curvature"] = {
        "model": "ln(y/y*) = -k (log2(c/c*))^2 through the observed optimum, c > 0",
        "per_series": kst,
        "study_log_k_all": lk,
        "study_log_k_up": lk_up,
        "external_log_k_all": lk_ext,
        "k_all_median_literature": float(np.exp(np.median(lkv))),
        "log_k_mean": float(lkv.mean()),
        "log_k_sd": float(lkv.std(ddof=1)),
        "drop_per_doubling_above_literature": {
            s: float(1 - np.exp(-np.exp(v))) for s, v in lk_up.items()
        },
        "ours_n75": ours,
    }
    # predictive draws of k (Student-t on log k, n-1 df)
    n_k = len(lkv)
    t_draw = rng.standard_t(n_k - 1, size=args.n_prior)
    k_lit = np.exp(lkv.mean() + lkv.std(ddof=1) * np.sqrt(1 + 1 / n_k) * t_draw)

    tau_prim = meta["primary"]["tau_log10"][1]
    priors = {}
    for name in ("primary", "independent", "full_text", "all", "primary_plus_external", "shehata_converted"):
        priors[f"meta_{name}"] = (meta[name]["_pred_log10"] * LOG2_10, k_lit)
    for ext, c_ext in (("Salama2024", 1.95), ("Shi2025", 5.08)):
        m2 = (np.log10(c_ext) + tau_prim * rng.standard_normal(args.n_prior)) * LOG2_10
        kx = np.exp(lk_ext[ext] + lkv.std(ddof=1) * rng.standard_normal(args.n_prior))
        priors[f"external_{ext}"] = (m2, kx)
    # location-free control: optimum anywhere in 0.01-10 wt% (log-uniform), literature k
    priors["shape_only"] = (
        rng.uniform(np.log2(0.01), np.log2(10.0), args.n_prior),
        k_lit,
    )
    R["prior_definitions"] = {
        "meta_*": "optimum location m from the random-effects predictive of that study set; curvature k from the literature curves (Xu2024, Song2022, Attaoui2025; Student-t on log k)",
        "external_Salama2024": "m ~ N(log10 1.95, tau_meta^2) (E4 law argmax), k from the Salama curve with the literature log-k spread",
        "external_Shi2025": "m ~ N(log10 5.08, tau_meta^2) (E4 law argmax), k from the Shi curve with the literature log-k spread",
        "shape_only": "m log-uniform on 0.01-10 wt%, literature k: tests what the curve shape alone adds",
        "flat": "no curve: the five composition effects are flat (data only)",
        "s_eta": f"curve misfit sd {args.s_eta} on the log scale (sensitivity 0.05 and 0.20 with meta_primary)",
    }
    for name, (m2, _) in priors.items():
        c = 2.0 ** m2
        priors_p = float(np.mean((c >= 1.5) & (c <= 2.5)))
        R.setdefault("prior_p_opt_in_1p5_2p5", {})[name] = priors_p

    post = {}
    for tag, rws in ROWS.items():
        post[tag] = {}
        for t in TARGETS:
            yhat, cov, _ = comp_effects(Y0[t], rws)
            post[tag][t] = {}
            for cvn, cv in (("none", 0.0), ("median", CVS["median"]), ("upper", CVS["upper"])):
                tau2 = A8.sig2(cv)
                D = {}
                al, _, _ = curve_posterior(yhat, cov, tau2, None, None, 10.0, args.n_post, rng, flat=True)
                D["flat"] = al_statements(al)
                for name, (m2, kk) in priors.items():
                    s_list = [args.s_eta]
                    if name == "meta_primary":
                        s_list = [args.s_eta, 0.05, 0.20]
                    for s_eta in s_list:
                        al, idx, ess = curve_posterior(yhat, cov, tau2, m2, kk, s_eta, args.n_post, rng)
                        S = al_statements(al)
                        cpost = 2.0 ** m2[idx]
                        S["ess"] = ess
                        S["p_opt_in_1p5_2p5"] = float(np.mean((cpost >= 1.5) & (cpost <= 2.5)))
                        S["p_opt_in_1p5_3p5"] = float(np.mean((cpost >= 1.5) & (cpost <= 3.5)))
                        S["p_opt_ge_1p5"] = float(np.mean(cpost >= 1.5))
                        S["opt_wt_pct_post"] = [float(x) for x in np.percentile(cpost, [2.5, 50, 97.5])]
                        lab = name if s_eta == args.s_eta else f"{name}_seta{s_eta:.2f}"
                        D[lab] = S
                post[tag][t][cvn] = D
    R["posterior"] = post

    # basis sensitivity of our own loadings (item 1 of the RA fix round)
    basis = {}
    for bn, b in BASES.items():
        lev = to_pvdf_basis(CNT_LEVELS, b)
        two = float(lev[1])
        B = {"levels_wt_pct_of_pvdf": lev.tolist(), "two_wt_pct_of_pvdf": two, "percentile_of_2wt": {}, "p_pred_in_pm25pct_of_2wt": {}}
        for mn in ("primary", "independent", "full_text", "shehata_converted", "full_text_shehata_converted"):
            c = 10 ** meta[mn]["_pred_log10"]
            B["percentile_of_2wt"][mn] = float(np.mean(c <= two))
            B["p_pred_in_pm25pct_of_2wt"][mn] = float(np.mean((c >= 0.75 * two) & (c <= 1.25 * two)))
        B["posterior_rms"] = {}
        for tag, rws in ROWS.items():
            yhat, cov, _ = comp_effects(Y0["rms_Voc"], rws)
            for cvn, cv in (("median", CVS["median"]), ("upper", CVS["upper"])):
                for pr in ("meta_primary", "meta_full_text", "meta_shehata_converted"):
                    m2, kk = priors[pr]
                    al, idx, ess = curve_posterior(yhat, cov, A8.sig2(cv), m2, kk, args.s_eta, args.n_post, rng, levels=lev)
                    S = al_statements(al)
                    S["ess"] = ess
                    B["posterior_rms"][f"{tag}/{cvn}/{pr}"] = S
        basis[bn] = B
    R["basis_sensitivity"] = {
        "definition": "our loadings read as wt% of PVDF (pvdf) or as wt% of total solids with BaTiO3 at 5-20 wt% of the solids (BaTiO3 fraction not documented locally), converted to wt% of PVDF by c / (1 - b - c); the literature optima stay as recorded in the csv",
        "bases": basis,
    }
    R["model_names"] = {
        "re-draw": "a8 specimen re-draw of the observed composition means (specimen effect only, recording noise ignored); round 1, key redraw",
        "hierarchical (exchangeable)": "a8 Gibbs hierarchy, alpha_c ~ N(0, omega^2), half-Cauchy omega; round 1, key hierarchy; the paper's primary numbers",
        "hierarchical (flat)": "round-2 Gaussian model with flat composition effects (recording noise and specimen variance); key round2.posterior.*.flat",
        "hierarchical (curve prior)": "round-2 Gaussian model with the literature curve prior on the CNT compositions; key round2.posterior.*.meta_* / external_* / shape_only",
    }
    R["specimen_bound"] = {
        tag: {t: spec_bound(Y0[t], rws, t) for t in TARGETS} for tag, rws in ROWS.items()
    }
    for m in meta.values():
        m.pop("_pred_log10")
    R["meta"] = meta
    R["verdict_r2"] = verdict_r2(R, CVS)
    return R


def verdict_r2(R, CVS):
    P = R["posterior"]
    lines = []
    m = R["meta"]["primary"]
    lines.append(
        f"meta ({m['n_studies']} studies): typical CNT optimum {m['optimum_median_wt_pct_ci'][1]:.2f} wt% "
        f"[{m['optimum_median_wt_pct_ci'][0]:.2f}, {m['optimum_median_wt_pct_ci'][2]:.2f}], between-study sd "
        f"{m['tau_log10'][1]:.2f} decades; a new study's optimum lies in 1.5-2.5 wt% with P {m['p_pred_in_1p5_2p5']:.2f}; "
        f"2 wt% is at the {100 * m['percentile_of_2wt']:.0f}th percentile of the predictive"
    )
    for cvn in ("median", "upper"):
        for pr in ("flat", "meta_primary", "meta_independent", "external_Salama2024", "external_Shi2025", "shape_only"):
            a, b = P["n75"]["rms_Voc"][cvn][pr], P["n72"]["rms_Voc"][cvn][pr]
            lines.append(
                f"V_rms CV {cvn:6s} prior {pr:20s}: P(2 wt% best) {a['p_best_2wt']:.2f}/{b['p_best_2wt']:.2f}, "
                f"in top two {a['p_2wt_in_top2']:.2f}/{b['p_2wt_in_top2']:.2f}, within 10% {a['p_within_10']:.2f}/{b['p_within_10']:.2f}, "
                f"best is 2 or 3 wt% {a['p_best_2or3']:.2f}/{b['p_best_2or3']:.2f}"
                + (f", c* in 1.5-3.5 wt% {a['p_opt_in_1p5_3p5']:.2f}/{b['p_opt_in_1p5_3p5']:.2f}" if "p_opt_in_1p5_3p5" in a else "")
                + " (n75/n72)"
            )
    sb = R["specimen_bound"]["n75"]["rms_Voc"]
    for k in ("paper_law5_fit", "law_quadratic_linear_scale", "quadratic_log_scale", "quadratic_log_scale_bt_only"):
        lines.append(
            f"specimen bound V_rms n75 {k}: CV point {100 * sb[k]['cv_point']:.1f}%, upper 50/90/95% "
            + "/".join(f"{100 * sb[k]['cv_upper'][q]:.1f}" for q in ("50", "90", "95"))
            + f"% (literature {100 * CVS['median']:.0f}% / {100 * CVS['upper']:.0f}%); chi-square P(misfit | smooth truth, CV 0/4/24%) "
            + "/".join(f"{v:.3f}" for v in sb[k]["p_consistent_with_cv"].values())
        )
    return lines


def write_tex_r2(J):
    R = J["round2"]
    L = ["% ---- round 2 (literature optimum, informative prior, specimen bound)"]

    def mac(name, val, comment):
        L.append(f"\\newcommand{{\\rks{name}}}{{{val}}}  % {comment}")

    def f2(x, d=2):
        return "--" if x is None else f"{x:.{d}f}"

    SETN = {"primary": "", "independent": "Indep", "full_text": "FullText", "all": "All", "primary_series": "Series", "primary_plus_external": "PlusExt", "shehata_converted": "ShehataConv", "full_text_shehata_converted": "FullTextShehataConv"}
    for k, nm in SETN.items():
        m = R["meta"][k]
        mac(f"MetaN{nm}", m["n_studies"], f"studies in the meta set {k}")
        mac(f"MetaOpt{nm}", f2(m["optimum_median_wt_pct_ci"][1]), f"wt%, typical optimum (10^mu), set {k}")
        mac(f"MetaOptLo{nm}", f2(m["optimum_median_wt_pct_ci"][0]), f"wt%, 95% CrI low, set {k}")
        mac(f"MetaOptHi{nm}", f2(m["optimum_median_wt_pct_ci"][2]), f"wt%, 95% CrI high, set {k}")
        mac(f"MetaTau{nm}", f2(m["tau_log10"][1]), f"decades, between-study sd of log10 optimum, set {k}")
        mac(f"MetaPredLo{nm}", f2(m["predictive_wt_pct"][0]), f"wt%, predictive 2.5%, set {k}")
        mac(f"MetaPredHi{nm}", f2(m["predictive_wt_pct"][2]), f"wt%, predictive 97.5%, set {k}")
        mac(f"MetaPInBand{nm}", p100(m["p_pred_in_1p5_2p5"]), f"percent, P(new optimum in 1.5-2.5 wt%), set {k}")
        mac(f"MetaPctTwo{nm}", p100(m["percentile_of_2wt"]), f"percentile of 2 wt% in the predictive, set {k}")
    C = R["curvature"]
    mac("KLit", f2(C["k_all_median_literature"]), "curvature k (ln units per squared doubling), literature median")
    mac("KOursRms", f2(C["ours_n75"]["rms_Voc"]["k_all"]), "curvature k of our V_rms composition means, n75")
    mac("DropOursRms", p100(-C["ours_n75"]["rms_Voc"]["rel_drop_2_to_3"]), "percent drop 2 -> 3 wt%, V_rms n75")
    for s, v in C["drop_per_doubling_above_literature"].items():
        mac(f"DropPerDoubling{''.join(ch for ch in s if ch.isalpha())}", p100(v), f"percent drop per doubling above the optimum, {s}")
    PR = {"flat": "Flat", "meta_primary": "Meta", "meta_independent": "MetaIndep", "meta_full_text": "MetaFullText", "meta_all": "MetaAll", "meta_primary_plus_external": "MetaPlusExt", "external_Salama2024": "Salama", "external_Shi2025": "Shi", "shape_only": "ShapeOnly", "meta_primary_seta0.05": "MetaEtaLo", "meta_primary_seta0.20": "MetaEtaHi", "meta_shehata_converted": "MetaShehataConv"}
    SK = {"p_best_2wt": "Best", "p_2wt_in_top2": "InTopTwo", "p_within_10": "WithinTen", "p_within_5": "WithinFive", "p_best_2or3": "BestTwoOrThree", "p_opt_in_1p5_2p5": "OptBand", "p_opt_in_1p5_3p5": "OptBandWide", "p_opt_ge_1p5": "OptAboveOneHalf"}
    CVN = {"none": "Zero", "median": "Med", "upper": "Up"}
    NN = {"n75": "", "n72": "NSeventyTwo"}
    for tag in ROWS:
        for t in TARGETS:
            for cvn in CVN:
                for pr, pn in PR.items():
                    S = R["posterior"][tag][t][cvn][pr]
                    for k, kn in SK.items():
                        if k in S:
                            mac(f"Pr{kn}{pn}{CVN[cvn]}{A2.TARGET_MACRO[t]}{NN[tag]}", p100(S[k]), f"percent, prior {pr}, CV {cvn}, {t}, {tag}, {k}")
    for tag in ROWS:
        for t in TARGETS:
            sb = R["specimen_bound"][tag][t]
            for k, kn in (("paper_law5_fit", "Law"), ("law_quadratic_linear_scale", "LawQuad"), ("quadratic_log_scale", "LogQuad"), ("quadratic_log_scale_bt_only", "LogQuadBt")):
                if k not in sb:
                    continue
                mac(f"SpecCv{kn}{A2.TARGET_MACRO[t]}{NN[tag]}", f2(100 * sb[k]["cv_point"], 1), f"percent, implied specimen CV (point), {k}, {t}, {tag}")
                for cvk, cvn2 in (("cv0", "Zero"), ("cv4", "Med"), ("cv24", "Up")):
                    mac(f"SpecPCons{cvn2}{kn}{A2.TARGET_MACRO[t]}{NN[tag]}", f2(sb[k]["p_consistent_with_cv"][cvk], 3), f"chi-square P of the misfit if the truth is smooth and the specimen CV is {cvk}, {k}, {t}, {tag}")
                for q, qn in (("50", "Fifty"), ("90", "Ninety"), ("95", "NinetyFive")):
                    mac(f"SpecCvUp{qn}{kn}{A2.TARGET_MACRO[t]}{NN[tag]}", f2(100 * sb[k]["cv_upper"][q], 1), f"percent, {q}% upper bound of the specimen CV, {k}, {t}, {tag}")
    BN = {"pvdf": "Pvdf", "solids_bt5": "SolidsFive", "solids_bt10": "SolidsTen", "solids_bt15": "SolidsFifteen", "solids_bt20": "SolidsTwenty"}
    MSN = {"primary": "", "independent": "Indep", "full_text": "FullText", "shehata_converted": "ShehataConv", "full_text_shehata_converted": "FullTextShehataConv"}
    PB = {"meta_primary": "Meta", "meta_full_text": "MetaFullText", "meta_shehata_converted": "MetaShehataConv"}
    for bn, B in R["basis_sensitivity"]["bases"].items():
        mac(f"BasisTwo{BN[bn]}", f2(B["two_wt_pct_of_pvdf"]), f"wt% of PVDF equivalent to our 2 wt% under basis {bn}")
        for mn, mnn in MSN.items():
            mac(f"BasisPctTwo{mnn}{BN[bn]}", p100(B["percentile_of_2wt"][mn]), f"percentile of our 2 wt% in the predictive of set {mn}, basis {bn}")
        for key, S in B["posterior_rms"].items():
            tag, cvn, pr = key.split("/")
            for k, kn in (("p_best_2wt", "Best"), ("p_2wt_in_top2", "InTopTwo"), ("p_within_10", "WithinTen"), ("p_best_2or3", "BestTwoOrThree")):
                mac(f"BasisPr{kn}{PB[pr]}{CVN[cvn]}{BN[bn]}{NN[tag]}", p100(S[k]), f"percent, hierarchical (curve prior {pr}), V_rms, CV {cvn}, {tag}, basis {bn}, {k}")
    with open(RES / "numbers_a13.tex", "a") as f:
        f.write("\n".join(L) + "\n")

    # table: literature studies
    rows = lit_rows()
    lines = []
    for r in rows:
        if r["series"] in ("BT10", "BT15", "BT20"):
            continue
        use = "primary" if r["use_primary"] == "1" else ("sens." if r["use_all"] == "1" or r["tier"] == "E" else "not used")
        hi = "$\\infty$)" if not np.isfinite(r["hi"]) else f"{r['hi']:.2g}]"
        lo = "(0" if r["lo"] <= 0 else f"[{r['lo']:.2g}"
        mat = r["matrix"].split(" (")[0].replace("_", "\\_").replace("BaTiO3", "BaTiO$_3$")
        fil = r["filler"].split(" (")[0].replace("BaTiO3", "BaTiO$_3$")
        lines.append(
            f"{r['first_author']} {r['year']} & {mat} & {fil} & {r['best_wt_pct']} & {lo}, {hi} & {r['tier']} & {use} \\\\"
        )
    body = (
        "\\toprule\nStudy & Matrix & Filler & Best (wt\\%) & Interval & Tier & Use \\\\\n\\midrule\n"
        + "\n".join(lines).replace("%", "\\%").replace("\\\\%", "\\%")
        + "\n\\bottomrule\n"
    )
    (TAB / "tab_a13_litopt.tex").write_text(body)

    # table: prior variants, V_rms, n75 | n72, CV median | upper
    order = ["flat", "shape_only", "meta_primary", "meta_independent", "meta_full_text", "meta_all", "meta_primary_plus_external", "external_Salama2024", "external_Shi2025"]
    lab = {"flat": "Flat (data only)", "shape_only": "Curve shape only", "meta_primary": "Meta (primary)", "meta_independent": "Meta (independent)", "meta_full_text": "Meta (full text)", "meta_all": "Meta (all)", "meta_primary_plus_external": "Meta + Salama + Shi", "external_Salama2024": "Salama 1.95 wt\\%", "external_Shi2025": "Shi 5.08 wt\\%"}
    ln = []
    for pr in order:
        cells = []
        for cvn in ("median", "upper"):
            for tag in ("n75", "n72"):
                S = R["posterior"][tag]["rms_Voc"][cvn][pr]
                cells += [p100(S["p_best_2wt"]), p100(S["p_2wt_in_top2"]), p100(S["p_within_10"])]
        ln.append(f"{lab[pr]} & " + " & ".join(cells) + " \\\\")
    body = (
        "\\toprule\n & \\multicolumn{6}{c}{CV median} & \\multicolumn{6}{c}{CV upper} \\\\\n"
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
        " & \\multicolumn{3}{c}{$n = 75$} & \\multicolumn{3}{c}{$n = 72$} & \\multicolumn{3}{c}{$n = 75$} & \\multicolumn{3}{c}{$n = 72$} \\\\\n"
        "Prior & Best & Top 2 & 10\\% & Best & Top 2 & 10\\% & Best & Top 2 & 10\\% & Best & Top 2 & 10\\% \\\\\n\\midrule\n"
        + "\n".join(ln)
        + "\n\\bottomrule\n"
    )
    (TAB / "tab_a13_prior.tex").write_text("% model: hierarchical (round-2 Gaussian model: flat or literature curve prior), V_rms\n" + body)

    # table: specimen bound
    tl = {"rms_Voc": r"$V_\mathrm{rms}$", "Vpp": r"$V_\mathrm{pp}$", "Vmax": r"$|V|_\mathrm{max}$"}
    ln = []
    for tag, nl in (("n75", "75"), ("n72", "72")):
        for t in TARGETS:
            sb = R["specimen_bound"][tag][t]
            cells = []
            for k in ("paper_law5_fit", "law_quadratic_linear_scale", "quadratic_log_scale_bt_only"):
                if k in sb:
                    u = 100 * sb[k]["cv_upper"]["95"]
                    cells += [f2(100 * sb[k]["cv_point"], 1), "unbounded (4 points, 3 parameters)" if u > 1000 else f2(u, 1)]
                else:
                    cells += ["--", "--"]
            ln.append(f"{tl[t]} & {nl} & " + " & ".join(cells) + " \\\\")
    body = (
        "\\toprule\n & & \\multicolumn{2}{c}{Paper law} & \\multicolumn{2}{c}{Quadratic factor} & \\multicolumn{2}{c}{Log quadratic, BaTiO$_3$ only} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Target & $n$ & CV & 95\\% bound & CV & 95\\% bound & CV & 95\\% bound \\\\\n\\midrule\n"
        + "\n".join(ln)
        + "\n\\bottomrule\n"
    )
    (TAB / "tab_a13_specbound.tex").write_text(body)

    # table: meta-analysis sets with the basis sensitivity of our 2 wt%
    BS = R["basis_sensitivity"]["bases"]
    msets = [("primary", "Primary"), ("independent", "Independent (no Ko\\c{c} 2024)"), ("full_text", "Full text"), ("shehata_converted", "Primary, Shehata converted"), ("full_text_shehata_converted", "Full text, Shehata converted"), ("all", "All (with Li 2020)"), ("primary_series", "Primary, Attaoui series"), ("primary_plus_external", "Primary + Salama + Shi")]
    ln = []
    for k, labm in msets:
        m = R["meta"][k]
        pct = lambda b: p100(BS[b]["percentile_of_2wt"][k]) if k in BS[b]["percentile_of_2wt"] else "--"  # noqa: E731
        ln.append(
            f"{labm} & {m['n_studies']} & {f2(m['optimum_median_wt_pct_ci'][1])} [{f2(m['optimum_median_wt_pct_ci'][0])}, {f2(m['optimum_median_wt_pct_ci'][2])}] & {f2(m['tau_log10'][1])} & "
            f"[{f2(m['predictive_wt_pct'][0])}, {f2(m['predictive_wt_pct'][2])}] & {p100(m['p_pred_in_1p5_2p5'])} & {pct('pvdf')} & {pct('solids_bt5')} & {pct('solids_bt20')} \\\\"
        )
    body = (
        "\\toprule\n & & & & & & \\multicolumn{3}{c}{Percentile of our 2 wt\\%} \\\\\n\\cmidrule(lr){7-9}\n"
        f"Set & $n$ & Typical optimum (wt\\%) & $\\tau$ (dec.) & Predictive 95\\% & $P$(1.5--2.5) & PVDF basis & Solids, 5\\% BT ({f2(BS['solids_bt5']['two_wt_pct_of_pvdf'])}) & Solids, 20\\% BT ({f2(BS['solids_bt20']['two_wt_pct_of_pvdf'])}) \\\\\n\\midrule\n"
        + "\n".join(ln)
        + "\n\\bottomrule\n"
    )
    (TAB / "tab_a13_meta.tex").write_text(body)

    # table: every V_rms probability by model, n75/n72 side by side
    Jr, Jh = J["redraw"], J["hierarchy"]
    stm = [("p_best_2wt", "2 wt\\% best"), ("p_2wt_in_top2", "2 wt\\% in top two"), ("p_within_10", "Within 10\\% of best"), ("p_within_5", "Within 5\\% of best"), ("p_best_2or3", "Best is 2 or 3 wt\\%")]
    ln = []
    for k, labs in stm:
        cells = []
        for cvn in ("median", "upper"):
            vals = [
                [scalar_keys(Jr[tag][cvn]["single"]["rms_Voc"])[k] for tag in ("n75", "n72")],
                [scalar_keys(Jh[tag][cvn]["rms_Voc"])[k] for tag in ("n75", "n72")],
                [R["posterior"][tag]["rms_Voc"][cvn]["flat"][k] for tag in ("n75", "n72")],
                [R["posterior"][tag]["rms_Voc"][cvn]["meta_primary"][k] for tag in ("n75", "n72")],
            ]
            cells += [f"{p100(a)}/{p100(b)}" for a, b in vals]
        ln.append(f"{labs} & " + " & ".join(cells) + " \\\\")
    body = (
        "\\toprule\n & \\multicolumn{4}{c}{CV median} & \\multicolumn{4}{c}{CV upper} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "Statement ($n = 75$/$72$) & Re-draw & Hier. exch. & Hier. flat & Hier. curve & Re-draw & Hier. exch. & Hier. flat & Hier. curve \\\\\n\\midrule\n"
        + "\n".join(ln)
        + "\n\\bottomrule\n"
    )
    (TAB / "tab_a13_models.tex").write_text(body)



if __name__ == "__main__":
    main()
