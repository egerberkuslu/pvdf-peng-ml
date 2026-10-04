"""E3 SPECVAR: literature-calibrated specimen-variability sensitivity (analysis protocol E3).

The grid has one fabricated specimen per composition, so composition and
specimen are confounded (project notes). This script asks how far the
composition ranking, the 2 wt% optimum and the headline numbers of the paper
depend on that single specimen, using a device-to-device coefficient of
variation (CV) taken from published electrospun PVDF-family nanogenerators
(results/a8_literature_cv.csv, hand-entered from the opened papers).

(1) Literature CV. Study-level CV = median CV over the rows of a study; the
    median and the maximum (upper bound) over studies calibrate the simulation,
    the 75th percentile is reported beside them.
(2) Specimen re-draws. Draw b multiplies all 15 recordings of specimen c (all
    three targets) by m_bc = exp(s z_bc - s^2/2), s^2 = ln(1 + CV^2), z_bc
    standard normal from numpy default_rng(42), the same z for every CV
    (common random numbers). This is the multiplicative specimen-level
    bootstrap of analysis protocol E3: under a flat prior on the log specimen
    effect, the true composition mean given the observed one is the observed
    mean times such a multiplier. A replication variant (a newly fabricated
    set compared with the observed set) doubles the log variance and is
    reported for the ranking and the optimum only.
    Per draw the script recomputes, with the code of the paper:
      ranking and optimum: force- and frequency-averaged target per
        composition (all B draws, all CVs of the scan, three targets);
      law optimum c* = -a1/(2 a2) of the five-parameter law
        (a2_common.fit_law5 on all rows, all B draws);
      LOO R2 of the plain GP (a2_common.fold_predict 'gp', benchLooRmsGp);
      held-out-force within-level R2 of LawGP and GP
        (a2_common.fold_predict 'physgp'/'gp' on a2_common.group_folds,
         physForce*RmsPhysGP);
      within-grid jackknife+ hits (a3_common.fit_predict 'gp' and jk_bounds,
        nested LOO as a3_conformal._nested_outer; covWithinJk*RmsHits);
      held-out-level jackknife+ hits of the plain GP on the three axes
        (a3_common.fit_predict 'gp' and jk_bounds as
         a3_conformal._group_level; covGroupGp*RmsHits).
    Refits are expensive (one GP fit is about 55 ms; the nested jackknife+
    needs 2775 fits per draw), so the refit headlines use the first B_x draws
    of the same stream (nested subsets); the counts are in the JSON.
    The within-grid nested jackknife+ fits one model per unordered pair
    {i, j}: the model trained without i and j is the same object for outer j
    / inner i and for outer i / inner j (sorted training rows, deterministic
    optimizer start), so this halves the work without changing any number.
(3) Hierarchical composition effect on log V_rms (numpy Gibbs sampler):
      y_ck = gamma_cell(k) + u_c + e_ck,  e ~ N(0, sigma^2),
      u_c = alpha_c + s_c,  s_c ~ N(0, tau^2) with tau^2 = ln(1 + CV^2) fixed,
      alpha_c ~ N(0, omega^2),  omega ~ half-Cauchy(A),  gamma flat,
      sigma^2 Jeffreys.
    gamma_cell are the 15 force x frequency cell effects shared by the
    compositions; alpha_c is the composition effect. P(2 wt% best) is the
    posterior probability that alpha of 2 wt% is the largest. A closed-form
    empirical-Bayes normal hierarchy is reported as a check.
(4) Every headline block is repeated without the scaled copies 52, 54, 73
    (n = 72, duplicated-recordings note in the README).

(5) Reading (analysis protocol section 9, E3). The median study-level CV (4 percent)
    is an optimistic bound: it comes from studies with unspecified replicate
    type, spreads that look like repeat measurements, or non-electrospun
    devices. He 2025 (24 percent) is the only direct evidence from separately
    fabricated electrospun devices. The re-draw treats the observed composition
    means as exact, so every re-draw probability is printed with the
    hierarchical probability beside it (same question, posterior of the
    hierarchy of (3) fitted to each target; replication variant = posterior
    predictive of a newly fabricated set). Intervals from 25 or fewer draws are
    printed as ranges (min, max), not percentile intervals; probabilities that
    round to 100 or 0 are printed as >99 and <1; the empirical-Bayes closed form
    at omega_EB = 0 is reported as not identifiable.

Run from the repository root:
    python code/a8_specvar.py
Options: --b-rank, --b-loo, --b-group, --b-within, --b-dup (draw counts),
--quick (smoke test), --tables-only (rebuild the hierarchical counterparts,
summary, macros and tables from a8_specvar.json without the re-draw refits),
--n-jobs (at most 4; one BLAS thread per worker).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # BLAS threads, set before numpy loads

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a2_common as A2  # noqa: E402
import a3_common as A3  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

SEED = 42
from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
RES = Path(
    os.environ.get("A8_OUT", ROOT / "results")
)  # override only for smoke tests
TAB = RES / "tables"
LIT_CSV = Path(
    os.environ.get("A8_LIT_CSV", ROOT / "results" / "a8_literature_cv.csv")
)

N = A2.N
COMP = A2.COMP
COMP_ORDER = A2.COMP_ORDER
SPEC = np.array([COMP_ORDER.index(c) for c in COMP])  # specimen index 0..4
NSPEC = len(COMP_ORDER)
BEST = COMP_ORDER.index("PVDF+BaTiO3+%2CNT")
DUP_IDS = [52, 54, 73]
KEEP72 = np.array([i for i in range(N) if i not in DUP_IDS])
ROWS = {"n75": np.arange(N), "n72": KEEP72}
TARGETS = ["rms_Voc", "Vpp", "Vmax"]
Y0 = {t: A2.df[t].values.astype(float) for t in TARGETS}
AXES = ["composition", "force", "frequency"]
ALPHAS = {"90": 0.10, "95": 0.05}
CV_SCAN = np.round(np.arange(0.0, 0.6001, 0.01), 2)
MAX_JOBS = 4
RANGE_MAX_DRAWS = 25  # intervals from this many draws or fewer are ranges (min, max)


def sig2(cv):
    return float(np.log1p(cv**2))


def multipliers(z, cv, rep=False):
    """Lognormal mean-one multipliers with the given CV (rep doubles log variance)."""
    s2 = sig2(cv) * (2.0 if rep else 1.0)
    return np.exp(np.sqrt(s2) * z - s2 / 2.0)


# ---------------------------------------------------------------- (1) literature
SEARCH_LOG = (
    "2026-09-26: second brain (research-cache) searches gave no device-dispersion values; web searches and "
    "OpenAlex full-text phrase searches located candidates; Europe PMC queries returned 1541 open-access "
    "PVDF-family piezoelectric full texts, grepped for replicate-device wording and mean +- SD voltages; the "
    "supplementary files of 203 primary-research papers among them were grepped as well; Nature-family Source "
    "Data were read where available; Elsevier, Wiley and ACS pages outside PMC were not reachable (HTTP 403). "
    "Most papers report a single device or cycle-to-cycle spread only; five studies gave traceable replicate "
    "dispersion, of which one (He 2025) states separately fabricated electrospun devices."
)


def literature():
    rows = list(csv.DictReader(open(LIT_CSV)))
    for r in rows:
        r["cv"] = float(r["cv"])
    studies = {}
    for r in rows:
        studies.setdefault(r["doi"], []).append(r)
    per = []
    for doi, rr in studies.items():
        cvs = [r["cv"] for r in rr]
        per.append(
            {
                "doi": doi,
                "first_author": rr[0]["first_author"],
                "year": rr[0]["year"],
                "material": rr[0]["material"],
                "n_rows": len(rr),
                "n_devices": ";".join(sorted({r["n_devices"] for r in rr})),
                "cv_study": float(np.median(cvs)),
                "cv_min": float(min(cvs)),
                "cv_max": float(max(cvs)),
                "replicate_type": rr[0]["replicate_type"],
                "electrospun": rr[0]["electrospun"],
            }
        )
    c = np.array([p["cv_study"] for p in per])

    def subset(pred):
        sub = {}
        for r in rows:
            if pred(r):
                sub.setdefault(r["doi"], []).append(r["cv"])
        v = np.array([np.median(x) for x in sub.values()])
        return {
            "n_studies": len(v),
            "cv_study": [float(x) for x in v],
            "cv_median": float(np.median(v)) if len(v) else None,
            "cv_max": float(np.max(v)) if len(v) else None,
        }

    subsets = {
        "electrospun_only": subset(lambda r: r["electrospun"].startswith("yes")),
        "explicitly_separate_devices": subset(
            lambda r: "independent" in r["replicate_type"]
        ),
        "row_level_cv_max": float(max(r["cv"] for r in rows)),
    }
    return {
        "subsets": subsets,
        "search_log": SEARCH_LOG,
        "n_rows": len(rows),
        "n_studies": len(per),
        "studies": per,
        "cv_median": float(np.median(c)),
        "cv_p75": float(np.percentile(c, 75)),
        "cv_max": float(np.max(c)),
        "cv_min": float(np.min(c)),
        "rule": "study-level CV = median over the rows of a study; median, 75th percentile and maximum over studies; the simulation uses the median (typical) and the maximum (upper bound)",
    }


# ---------------------------------------------------------------- (2a) ranking / optimum
def comp_means(y, rows):
    return np.array([y[rows][SPEC[rows] == c].mean() for c in range(NSPEC)])


def rank_block(Z, cvs, rows, rep=False):
    """P(full ranking unchanged), P(2 wt% still best) per CV for every target."""
    out = {}
    for t in TARGETS:
        mu = comp_means(Y0[t], rows)
        order0 = np.argsort(-mu)
        best0 = int(order0[0])
        pr, po, pw = [], [], []
        for cv in cvs:
            m = multipliers(Z, cv, rep)  # (B, 5)
            mm = m * mu[None, :]
            order = np.argsort(-mm, axis=1)
            pr.append(float(np.mean(np.all(order == order0[None, :], axis=1))))
            po.append(float(np.mean(order[:, 0] == best0)))
            pw.append(np.bincount(order[:, 0], minlength=NSPEC) / len(Z))
        out[t] = {
            "observed_means": mu.tolist(),
            "observed_order": [COMP_ORDER[i] for i in order0],
            "observed_best": COMP_ORDER[best0],
            "p_rank": pr,
            "p_opt": po,
            "p_winner": np.array(pw).tolist(),
        }
    return out


def crossing(cvs, p, level=0.5):
    """Smallest CV at which p falls to `level` (linear interpolation), or None."""
    p = np.asarray(p)
    for k in range(1, len(p)):
        if p[k - 1] > level >= p[k]:
            x0, x1, y0, y1 = cvs[k - 1], cvs[k], p[k - 1], p[k]
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return None


def law_opt(y, rows):
    _, pv, info = A2.fit_law5(rows, y)
    a0, a1, a2 = pv[:3]
    cstar = float(-a1 / (2 * a2)) if a2 < 0 else float("nan")
    levels = np.array([0.0, 1.0, 2.0, 3.0])
    g = a0 + a1 * levels + a2 * levels**2
    return cstar, float(levels[np.argmax(g)]), bool(info["ok"])


# ---------------------------------------------------------------- (2b) refit headline units
def unit_loo_force(y, rows):
    """LOO R2 of plain GP and held-out-force per-level/pooled R2 of LawGP and GP."""
    mu = np.full(N, np.nan)
    for tr, te in A2.loo_folds(rows):
        m, _, _ = A2.fold_predict("gp", tr, te, y)
        mu[te] = m
    out = {"loo_r2_gp": float(r2_score(y[rows], mu[rows]))}
    for model in ("physgp", "gp"):
        muf = np.full(N, np.nan)
        for tr, te, lvl in A2.group_folds("force", rows):
            m, _, _ = A2.fold_predict(model, tr, te, y)
            muf[te] = m
            out[f"force_{int(lvl)}N_r2_{model}"] = float(r2_score(y[te], m))
        out[f"force_pooled_r2_{model}"] = float(r2_score(y[rows], muf[rows]))
    return out


def unit_within_pairs(y, rows, j):
    """Pair models {i, j}, i < j (row order), trained on rows minus {i, j}."""
    rows = np.asarray(rows)
    pos = int(np.where(rows == j)[0][0])
    res = []
    for i in rows[:pos]:
        tr = rows[(rows != i) & (rows != j)]
        mu, _ = A3.fit_predict("gp", tr, [i, j], y)
        res.append((int(i), int(j), float(mu[0]), float(mu[1])))
    return res


def within_hits(y, rows, pairs):
    """Jackknife+ within-grid hits from the pair predictions (a3_conformal.within_grid)."""
    idx = {int(r): k for k, r in enumerate(rows)}
    n = len(rows)
    P = np.full((n, n), np.nan)  # P[a, b] = prediction at row b of model without {a, b}
    for i, j, mi, mj in pairs:
        P[idx[j], idx[i]] = mi
        P[idx[i], idx[j]] = mj
    hits = {a: 0 for a in ALPHAS}
    for bj, j in enumerate(rows):
        others = [k for k in range(n) if k != bj]
        r_abs = np.array([abs(y[rows[k]] - P[bj, k]) for k in others])
        mu_at_j = np.array([P[k, bj] for k in others])
        for a, alpha in ALPHAS.items():
            lo, hi = A3.jk_bounds(mu_at_j, r_abs, alpha)
            hits[a] += int(lo <= y[j] <= hi)
    return hits


def unit_group_level(y, rows, axis, v):
    """Held-out-level jackknife+ hits of the plain GP (a3_conformal._group_level)."""
    g = A3.GROUPS[axis]
    rows = np.asarray(rows)
    tr = rows[g[rows] != v]
    te = rows[g[rows] == v]
    resid = np.empty(len(tr))
    preds = np.empty((len(tr), len(te)))
    for r, j in enumerate(tr):
        trj = tr[tr != j]
        mu, _ = A3.fit_predict("gp", trj, np.concatenate(([j], te)), y)
        resid[r] = abs(y[j] - mu[0])
        preds[r] = mu[1:]
    hits = {}
    for a, alpha in ALPHAS.items():
        h = 0
        for k, tt in enumerate(te):
            lo, hi = A3.jk_bounds(preds[:, k], resid, alpha)
            h += int(lo <= y[tt] <= hi)
        hits[a] = h
    return axis, hits, len(te)


# ---------------------------------------------------------------- (3) hierarchy
def gibbs(y, rows, tau2, A=1.0, iters=6000, burn=1000, chains=4, seed=SEED, ref=None):
    """Gibbs sampler of the partial-pooling hierarchy on log V_rms (see module doc).

    ref: observed composition order (indices, best first) of the re-draw; if given, the
    posterior P(order unchanged), P(observed best stays best) and their replication
    counterparts (posterior predictive of a new set: alpha + s_new + mean noise, drawn from
    a separate stream seed + 1) are added. The sampler stream is not changed by ref.
    """
    rng = np.random.default_rng(seed)
    ly = np.log(y[rows])
    sp = SPEC[rows]
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
            # cell effects (flat prior)
            r = ly - u[sp]
            gm = np.bincount(cell_id, weights=r, minlength=ncell) / n_cell
            gam = gm + rng.standard_normal(ncell) * np.sqrt(s2 / n_cell)
            # u_c = alpha_c + s_c, prior N(0, om2 + tau2)
            r = ly - gam[cell_id]
            sc = np.bincount(sp, weights=r, minlength=NSPEC)
            prec = n_c / s2 + 1.0 / (om2 + tau2)
            u = sc / s2 / prec + rng.standard_normal(NSPEC) / np.sqrt(prec)
            # alpha | u
            if tau2 > 0:
                w = om2 / (om2 + tau2)
                alpha = w * u + rng.standard_normal(NSPEC) * np.sqrt(w * tau2)
            else:
                alpha = u.copy()
            # omega^2 | alpha, half-Cauchy(A) via inverse-gamma auxiliary
            om2 = 1.0 / rng.gamma(
                (NSPEC + 1) / 2.0, 1.0 / (1.0 / a_aux + 0.5 * np.sum(alpha**2))
            )
            a_aux = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / A**2 + 1.0 / om2))
            # sigma^2 | rest (Jeffreys)
            res = ly - gam[cell_id] - u[sp]
            s2 = 1.0 / rng.gamma(n / 2.0, 1.0 / (0.5 * np.sum(res**2)))
            if it >= burn:
                keep.append(np.concatenate((alpha, u, [np.sqrt(om2), np.sqrt(s2)])))
        draws.append(np.array(keep))
    D = np.array(draws)  # chains x draws x (5 + 5 + 2)
    # split R-hat on alpha
    half = D.shape[1] // 2
    S = np.concatenate([D[:, :half], D[:, half : 2 * half]], axis=0)
    W = S.var(axis=1, ddof=1).mean(axis=0)
    Bv = S.mean(axis=1).var(axis=0, ddof=1) * half
    rhat = np.sqrt(((half - 1) / half * W + Bv / half) / W)
    flat = D.reshape(-1, D.shape[2])
    al = flat[:, :NSPEC]
    best = np.argmax(al, axis=1)
    contrast = al[:, BEST][:, None] - al  # 2 wt% minus each composition
    extra = {}
    if ref is not None:
        ref = np.asarray(ref)
        rng2 = np.random.default_rng(seed + 1)
        n_c = np.bincount(SPEC[rows], minlength=NSPEC)
        sig = flat[:, 2 * NSPEC + 1][:, None]
        new = (
            al
            + rng2.standard_normal(al.shape) * np.sqrt(tau2)
            + rng2.standard_normal(al.shape) * sig / np.sqrt(n_c)[None, :]
        )
        for nm, v in (("", al), ("_replication", new)):
            o = np.argsort(-v, axis=1)
            extra["p_rank" + nm] = float(np.mean(np.all(o == ref[None, :], axis=1)))
            extra["p_opt" + nm] = float(np.mean(o[:, 0] == ref[0]))
    return extra | {
        "tau": float(np.sqrt(tau2)),
        "half_cauchy_scale": A,
        "chains": chains,
        "iters_kept_per_chain": int(D.shape[1]),
        "rhat_max_alpha": float(np.max(rhat[:NSPEC])),
        "p_best": (np.bincount(best, minlength=NSPEC) / len(best)).tolist(),
        "p_best_2wt": float(np.mean(best == BEST)),
        "alpha_mean": al.mean(axis=0).tolist(),
        "alpha_ci95": np.percentile(al, [2.5, 97.5], axis=0).T.tolist(),
        "ratio_2wt_vs_each_median": np.exp(np.median(contrast, axis=0)).tolist(),
        "ratio_2wt_vs_each_ci95": np.exp(
            np.percentile(contrast, [2.5, 97.5], axis=0)
        ).T.tolist(),
        "p_2wt_above_each": np.mean(contrast > 0, axis=0).tolist(),
        "omega_median": float(np.median(flat[:, 2 * NSPEC])),
        "sigma_median": float(np.median(flat[:, 2 * NSPEC + 1])),
    }


def closed_form(y, rows, tau2, nmc=200000, seed=SEED):
    """Empirical-Bayes normal hierarchy: sigma^2 from the cell + composition fit,
    omega^2 by moments, then alpha | data Gaussian; P(best) by Monte Carlo."""
    ly = np.log(y[rows])
    sp = SPEC[rows]
    cell_id = pd.factorize(list(zip(A2.FRC[rows], A2.FRQ[rows])))[0]
    Xd = np.column_stack([np.eye(cell_id.max() + 1)[cell_id], np.eye(NSPEC)[sp][:, 1:]])
    beta, *_ = np.linalg.lstsq(Xd, ly, rcond=None)
    res = ly - Xd @ beta
    dof = len(ly) - np.linalg.matrix_rank(Xd)
    s2 = float(res @ res / dof)
    # composition means adjusted for cells (balanced when n = 75)
    u_hat = np.concatenate(([0.0], beta[-(NSPEC - 1) :]))
    u_hat = u_hat - u_hat.mean()
    n_c = np.bincount(sp, minlength=NSPEC)
    v_e = s2 / n_c
    om2 = max(float(np.var(u_hat, ddof=1) - np.mean(v_e) - tau2), 0.0)
    v = v_e + tau2
    if om2 > 0:
        w = om2 / (om2 + v)
        m = w * u_hat
        sd = np.sqrt(w * v)
    else:
        m = np.zeros(NSPEC)
        sd = np.zeros(NSPEC)
    rng = np.random.default_rng(seed)
    al = m[None, :] + rng.standard_normal((nmc, NSPEC)) * sd[None, :]
    best = np.argmax(al, axis=1) if om2 > 0 else np.full(nmc, -1)
    return {
        "tau": float(np.sqrt(tau2)),
        "sigma": float(np.sqrt(s2)),
        "omega_eb": float(np.sqrt(om2)),
        "p_best_2wt": float(np.mean(best == BEST)),
        "alpha_mean": m.tolist(),
        "alpha_sd": sd.tolist(),
        "note": "omega^2 = 0 means complete pooling (no composition effect detectable); P(best) is then undefined and reported as 0",
    }


# ---------------------------------------------------------------- summaries
def pct(v, q=(2.5, 50, 97.5)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, q)] if len(v) else [None] * len(q)


def summarize(vals):
    v = np.asarray(vals, float)
    lo, med, hi = pct(v)
    return {
        "n_draws": int(np.isfinite(v).sum()),
        "median": med,
        "ci95": [lo, hi],
        "mean": float(np.nanmean(v)),
    }


# ---------------------------------------------------------------- section 9 additions
def direct_study(lit):
    """The study with separately fabricated electrospun devices (He 2025)."""
    d = [
        x
        for x in lit["studies"]
        if x["electrospun"].startswith("yes")
        and x["replicate_type"] == "independently fabricated devices"
    ]
    assert len(d) == 1, d
    return d[0]


def interval_kind(n):
    return "range" if n <= RANGE_MAX_DRAWS else "percentile"


def annotate(J, n_jobs=MAX_JOBS):
    """analysis protocol section 9 (E3): hierarchical counterparts of every re-draw probability,
    ranges for few-draw intervals, identifiability of the closed form, and the reading."""
    CVS = J["cv_used"]
    settings = {"none": 0.0, **CVS}
    keys, jobs = [], []
    for tag, rows in ROWS.items():
        for t in TARGETS:
            ref = [
                COMP_ORDER.index(c)
                for c in J["ranking"]["observed"][tag][t]["observed_order"]
            ]
            for name, cv in settings.items():
                keys.append((tag, t, name, cv))
                jobs.append(delayed(gibbs)(Y0[t], rows, sig2(cv), ref=ref))
    outs = Parallel(n_jobs=n_jobs)(jobs)
    HC = {}
    for (tag, t, name, cv), g in zip(keys, outs):
        HC.setdefault(tag, {}).setdefault(t, {})[name] = {
            "cv": cv,
            "p_rank": g["p_rank"],
            "p_opt": g["p_opt"],
            "p_rank_replication": g["p_rank_replication"],
            "p_opt_replication": g["p_opt_replication"],
            "p_best": g["p_best"],
            "rhat_max_alpha": g["rhat_max_alpha"],
        }
        if t == "rms_Voc" and name in ("none", "median", "upper"):
            # same sampler, same stream: must reproduce the stored V_rms hierarchy
            assert g["p_best"] == J["hierarchy"][tag][name]["gibbs"]["p_best"], (tag, name)
    J["hierarchy_counterparts"] = {
        "definition": "hierarchy of (3) fitted to log of each target (V_rms, V_pp, |V|_max) at the specimen CV of each setting; p_rank = posterior P(order of alpha equals the observed order of the re-draw), p_opt = posterior P(observed best has the largest alpha); *_replication = posterior predictive of a newly fabricated set, alpha + s_new + mean noise (s_new ~ N(0, ln(1 + CV^2)), noise ~ N(0, sigma^2 / n_c)), drawn from default_rng(43); 4 chains x 5000 kept draws; the V_rms rows at none/median/upper reproduce hierarchy.*.gibbs.p_best exactly",
        "values": HC,
    }
    for tag in ROWS:
        for name in ("median", "upper"):
            H = J["headlines"][tag][name]
            if "loo_force" in H:
                for m, v in H["loo_force"].items():
                    vals = [d[m] for d in H["draws_loo_force"]]
                    v["range"] = [float(np.min(vals)), float(np.max(vals))]
                    v["interval"] = interval_kind(len(vals))
            if "group_hits" in H:
                for ax in AXES:
                    for h in ALPHAS:
                        vals = [d[ax][h] for d in H["draws_group_hits"]]
                        H["group_hits"][ax][h]["range"] = [min(vals), max(vals)]
                        H["group_hits"][ax][h]["interval"] = interval_kind(len(vals))
            if "within_hits" in H:
                for h in ALPHAS:
                    vals = [d[h] for d in H["draws_within_hits"]]
                    H["within_hits"][h]["range"] = [min(vals), max(vals)]
                    H["within_hits"][h]["interval"] = interval_kind(len(vals))
    for tag in ROWS:
        for name in ("none", "median", "upper"):
            cf = J["hierarchy"][tag][name]["closed_form"]
            cf["identifiable"] = bool(cf["omega_eb"] > 0)
            if not cf["identifiable"]:
                cf["p_best_2wt_reading"] = (
                    "not identifiable: omega_EB = 0 (the moment estimate of the composition "
                    "variance is zero once the specimen variance is subtracted), so the closed "
                    "form pools completely and P(2 wt% best) is undefined, not a probability of 0"
                )
    lit = J["literature"]
    he = direct_study(lit)
    J["summary"] = {
        "cv_median": lit["cv_median"],
        "cv_median_reading": (
            f"optimistic bound: the median study-level CV ({100 * lit['cv_median']:.0f} percent) comes from "
            "studies with unspecified replicate type (Lang 2016, Qu 2026), a spread that looks like repeat "
            "measurement (Li 2023: seven 'independent samples' that differ only in the second decimal) or "
            "non-electrospun devices (Ge 2024, commercial screen-printed film)"
        ),
        "cv_direct": he["cv_study"],
        "cv_direct_reading": (
            f"{he['first_author']} {he['year']} ({100 * he['cv_study']:.0f} percent, {he['n_devices']} devices) is the only "
            "direct evidence from separately fabricated electrospun devices; it is the upper setting of the simulation"
        ),
        "redraw_reading": "the specimen re-draw treats the observed composition means as exact (it perturbs them by the specimen factor only, not by the within-specimen noise of the 15 recordings); every re-draw probability is printed with the hierarchical probability beside it (hierarchy_counterparts)",
        "few_draw_intervals": f"intervals from {RANGE_MAX_DRAWS} or fewer draws (held-out-level jackknife+ {J['draws']['B_group_coverage']} draws, within-grid jackknife+ {J['draws']['B_within_coverage']}, n = 72 coverage {J['draws']['B_dup_coverage']}) are ranges (min, max), not percentile intervals",
        "rounding": "probabilities that round to 100 percent are printed as >99 and those that round to 0 as <1",
        "closed_form_upper": "the empirical-Bayes closed form at the upper CV is not identifiable (omega_EB = 0); the Gibbs posterior is the reported value",
        "no_hierarchical_counterpart": "P(LawGP pooled held-out-force R2 > GP) compares two models on re-drawn data; it is not a composition probability and has no hierarchical counterpart",
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-rank", type=int, default=2000)
    ap.add_argument("--b-loo", type=int, default=150)
    ap.add_argument("--b-group", type=int, default=25)
    ap.add_argument("--b-within", type=int, default=6)
    ap.add_argument("--b-dup", type=int, default=50)
    ap.add_argument("--b-dup-cov", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=MAX_JOBS)
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args()
    args.n_jobs = MAX_JOBS if args.n_jobs < 1 else min(args.n_jobs, MAX_JOBS)
    if args.tables_only:
        J = json.load(open(RES / "a8_specvar.json"))
        annotate(J, args.n_jobs)
        write_tex(J)
        json.dump(A3.to_plain(J), open(RES / "a8_specvar.json", "w"), indent=1)
        print("[a8] tables, macros and summary rebuilt from a8_specvar.json", flush=True)
        return
    if args.quick:
        (
            args.b_rank,
            args.b_loo,
            args.b_group,
            args.b_within,
            args.b_dup,
            args.b_dup_cov,
        ) = (200, 4, 1, 1, 2, 1)
    t0 = time.time()
    J = {
        "task": "E3 SPECVAR",
        "seed": SEED,
        "script": "code/a8_specvar.py",
    }

    lit = literature()
    J["literature"] = lit
    CVS = {"median": lit["cv_median"], "upper": lit["cv_max"], "p75": lit["cv_p75"]}
    J["cv_used"] = CVS
    print(
        f"[a8] literature: {lit['n_studies']} studies, CV median {CVS['median']:.3f}, p75 {CVS['p75']:.3f}, max {CVS['upper']:.3f}",
        flush=True,
    )

    rng = np.random.default_rng(SEED)
    Zall = rng.standard_normal((args.b_rank, NSPEC))
    J["draws"] = {
        "B_rank": args.b_rank,
        "B_loo_force": args.b_loo,
        "B_group_coverage": args.b_group,
        "B_within_coverage": args.b_within,
        "B_dup_loo_force": args.b_dup,
        "B_dup_coverage": args.b_dup_cov,
        "stream": "numpy default_rng(42).standard_normal((B_rank, 5)); refit headlines use the first B_x rows (nested subsets); same z for every CV",
    }

    # ---------------- (2a) ranking and optimum, CV scan
    scan = {}
    for tag, rows in ROWS.items():
        scan[tag] = {
            "single": rank_block(Zall, CV_SCAN, rows),
            "replication": rank_block(Zall, CV_SCAN, rows, rep=True),
        }
    at_cv = {}
    for tag, rows in ROWS.items():
        at_cv[tag] = {}
        for name, cv in CVS.items():
            b = rank_block(Zall, [cv], rows)
            br = rank_block(Zall, [cv], rows, rep=True)
            at_cv[tag][name] = {
                t: {
                    "cv": cv,
                    "p_rank": b[t]["p_rank"][0],
                    "p_opt": b[t]["p_opt"][0],
                    "p_winner": dict(zip(COMP_ORDER, b[t]["p_winner"][0])),
                    "p_rank_replication": br[t]["p_rank"][0],
                    "p_opt_replication": br[t]["p_opt"][0],
                }
                for t in TARGETS
            }
    flips = {}
    for tag in ROWS:
        flips[tag] = {}
        for t in TARGETS:
            s, r = scan[tag]["single"][t], scan[tag]["replication"][t]
            flips[tag][t] = {
                "cv_opt_half": crossing(CV_SCAN, s["p_opt"]),
                "cv_rank_half": crossing(CV_SCAN, s["p_rank"]),
                "cv_opt_half_replication": crossing(CV_SCAN, r["p_opt"]),
                "cv_rank_half_replication": crossing(CV_SCAN, r["p_rank"]),
                "p_opt_at_cv_060": s["p_opt"][-1],
            }
    J["ranking"] = {
        "definition": "force- and frequency-averaged target per composition (mean of the recordings of the specimen); ranking survives if all five positions are unchanged, optimum survives if the observed best composition stays best",
        "observed": {
            tag: {
                t: {
                    k: scan[tag]["single"][t][k]
                    for k in ("observed_means", "observed_order", "observed_best")
                }
                for t in TARGETS
            }
            for tag in ROWS
        },
        "at_cv": at_cv,
        "flip_cv": flips,
        "cv_scan": CV_SCAN.tolist(),
    }
    with open(RES / "a8_figdata_scan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["rows", "target", "variant", "cv", "p_rank", "p_opt"]
            + [f"p_best_{COMP_ORDER[c]}" for c in range(NSPEC)]
        )
        for tag in ROWS:
            for var in ("single", "replication"):
                for t in TARGETS:
                    s = scan[tag][var][t]
                    for k, cv in enumerate(CV_SCAN):
                        w.writerow(
                            [
                                tag,
                                t,
                                var,
                                f"{cv:.2f}",
                                f"{s['p_rank'][k]:.5f}",
                                f"{s['p_opt'][k]:.5f}",
                            ]
                            + [f"{x:.5f}" for x in s["p_winner"][k]]
                        )
    print(
        f"[a8] ranking done {time.time()-t0:.0f}s; n75 rms flip CV opt {flips['n75']['rms_Voc']['cv_opt_half']}",
        flush=True,
    )

    # law optimum c* per draw (all B draws), V_rms
    y = Y0["rms_Voc"]
    law = {}
    for tag, rows in ROWS.items():
        base = law_opt(y, rows)
        law[tag] = {"observed": {"c_star": base[0], "best_level": base[1]}}
        for name in ("median", "upper"):
            M = multipliers(Zall, CVS[name])
            res = Parallel(n_jobs=args.n_jobs, batch_size=64)(
                delayed(law_opt)(y * M[b][SPEC], rows) for b in range(len(Zall))
            )
            cs = np.array([r[0] for r in res])
            bl = np.array([r[1] for r in res])
            law[tag][name] = {
                "c_star": summarize(cs),
                "p_c_star_concave": float(np.mean(np.isfinite(cs))),
                "p_best_fabricated_level_is_2": float(np.mean(bl == 2.0)),
                "p_c_star_in_1p5_2p5": float(np.mean((cs > 1.5) & (cs <= 2.5))),
            }
    J["law_optimum"] = law
    print(f"[a8] law optimum done {time.time()-t0:.0f}s", flush=True)

    # ---------------- (2b) refit headlines
    def mult_rows(b, cv):
        return multipliers(Zall[b], cv)[SPEC]

    head = {}
    # baseline (CV = 0) sanity check against the published numbers
    jobs = []
    for tag, rows in ROWS.items():
        jobs.append(("lf", tag, "base", -1, None, delayed(unit_loo_force)(y, rows)))
    for name in ("median", "upper"):
        for b in range(args.b_loo):
            jobs.append(
                (
                    "lf",
                    "n75",
                    name,
                    b,
                    None,
                    delayed(unit_loo_force)(y * mult_rows(b, CVS[name]), ROWS["n75"]),
                )
            )
        for b in range(args.b_dup):
            jobs.append(
                (
                    "lf",
                    "n72",
                    name,
                    b,
                    None,
                    delayed(unit_loo_force)(y * mult_rows(b, CVS[name]), ROWS["n72"]),
                )
            )

    # coverage units
    def cov_jobs(tag, name, b, rows, yy):
        out = []
        for ax in AXES:
            for v in np.unique(A3.GROUPS[ax][rows]):
                out.append(
                    (
                        "grp",
                        tag,
                        name,
                        b,
                        None,
                        delayed(unit_group_level)(yy, rows, ax, v),
                    )
                )
        return out

    def within_jobs(tag, name, b, rows, yy):
        return [
            ("wth", tag, name, b, None, delayed(unit_within_pairs)(yy, rows, int(j)))
            for j in rows[1:]
        ]

    for tag, rows in ROWS.items():
        jobs += cov_jobs(tag, "base", -1, rows, y)
        jobs += within_jobs(tag, "base", -1, rows, y)
    for name in ("median", "upper"):
        for b in range(args.b_group):
            jobs += cov_jobs("n75", name, b, ROWS["n75"], y * mult_rows(b, CVS[name]))
        for b in range(args.b_within):
            jobs += within_jobs(
                "n75", name, b, ROWS["n75"], y * mult_rows(b, CVS[name])
            )
        for b in range(args.b_dup_cov):
            jobs += cov_jobs("n72", name, b, ROWS["n72"], y * mult_rows(b, CVS[name]))
            jobs += within_jobs(
                "n72", name, b, ROWS["n72"], y * mult_rows(b, CVS[name])
            )
    # largest units first for load balance
    order = sorted(
        range(len(jobs)), key=lambda k: {"wth": 1, "grp": 0, "lf": 2}[jobs[k][0]]
    )
    print(f"[a8] refit units: {len(jobs)}", flush=True)
    outs = Parallel(n_jobs=args.n_jobs, batch_size=1, verbose=0)(
        jobs[k][5] for k in order
    )
    results = [None] * len(jobs)
    for k, o in zip(order, outs):
        results[k] = o
    print(f"[a8] refits done {time.time()-t0:.0f}s", flush=True)

    # assemble
    lf = {}
    grp = {}
    wth = {}
    for (kind, tag, name, b, _, _), o in zip(jobs, results):
        key = (tag, name, b)
        if kind == "lf":
            lf[key] = o
        elif kind == "grp":
            ax, hits, nte = o
            d = grp.setdefault(key, {a: {h: 0 for h in ALPHAS} for a in AXES})
            for h in ALPHAS:
                d[ax][h] += hits[h]
        else:
            wth.setdefault(key, []).extend(o)
    wh = {
        key: within_hits(
            y if key[1] == "base" else y * mult_rows(key[2], CVS[key[1]]),
            ROWS[key[0]],
            pairs,
        )
        for key, pairs in wth.items()
    }

    for tag in ROWS:
        head[tag] = {
            "baseline": {
                "loo_force": lf[(tag, "base", -1)],
                "group_hits": grp[(tag, "base", -1)],
                "within_hits": wh[(tag, "base", -1)],
                "n": int(len(ROWS[tag])),
            }
        }
        for name in ("median", "upper"):
            H = {"cv": CVS[name]}
            keys = sorted(k for k in lf if k[0] == tag and k[1] == name)
            if keys:
                metrics = lf[keys[0]].keys()
                H["loo_force"] = {
                    m: summarize([lf[k][m] for k in keys]) for m in metrics
                }
                H["p_lawgp_beats_gp_on_force_pooled"] = float(
                    np.mean(
                        [
                            lf[k]["force_pooled_r2_physgp"]
                            > lf[k]["force_pooled_r2_gp"]
                            for k in keys
                        ]
                    )
                )
                H["p_force_1N_lawgp_negative"] = float(
                    np.mean([lf[k]["force_1N_r2_physgp"] < 0 for k in keys])
                )
                H["draws_loo_force"] = [lf[k] for k in keys]
            gk = sorted(k for k in grp if k[0] == tag and k[1] == name)
            if gk:
                H["group_hits"] = {
                    ax: {h: summarize([grp[k][ax][h] for k in gk]) for h in ALPHAS}
                    for ax in AXES
                }
                H["draws_group_hits"] = [grp[k] for k in gk]
            wk = sorted(k for k in wh if k[0] == tag and k[1] == name)
            if wk:
                H["within_hits"] = {
                    h: summarize([wh[k][h] for k in wk]) for h in ALPHAS
                }
                H["draws_within_hits"] = [wh[k] for k in wk]
            head[tag][name] = H
    J["headlines"] = head
    J["headline_definitions"] = {
        "loo_r2_gp": "within-grid LOO R2 of the plain GP on V_rms (benchLooRmsGp; a2_common.fold_predict 'gp')",
        "force_<L>N_r2_physgp": "within-level R2 at held-out force L of LawGP (physForce<L>NRmsPhysGP; a2_common.fold_predict 'physgp' on group_folds('force'))",
        "force_pooled_r2_<model>": "pooled held-out-force R2 over all rows (global mean)",
        "within_hits": "within-grid jackknife+ hits of the plain GP on V_rms out of n (covWithinJk<level>RmsHits)",
        "group_hits": "held-out-level jackknife+ hits of the plain GP on V_rms per axis out of n (covGroupGp<Axis><level>RmsHits)",
    }
    print(f"[a8] baseline n75: {head['n75']['baseline']}", flush=True)

    # ---------------- (3) hierarchy
    hier = {}
    for tag, rows in ROWS.items():
        hier[tag] = {}
        for name, cv in (
            ("none", 0.0),
            ("median", CVS["median"]),
            ("upper", CVS["upper"]),
        ):
            hier[tag][name] = {
                "cv": cv,
                "gibbs": gibbs(y, rows, sig2(cv)),
                "closed_form": closed_form(y, rows, sig2(cv)),
            }
        hier[tag]["prior_sensitivity_median_cv"] = {
            str(A): gibbs(y, rows, sig2(CVS["median"]), A=A)["p_best_2wt"]
            for A in (0.25, 2.5)
        }
    J["hierarchy"] = hier
    J[
        "hierarchy_model"
    ] = "log V_rms_ck = gamma_cell(k) + alpha_c + s_c + e_ck; s_c ~ N(0, ln(1+CV^2)) fixed; alpha_c ~ N(0, omega^2), omega ~ half-Cauchy(1); gamma flat (15 force x frequency cells); sigma^2 Jeffreys; 4 chains x 5000 kept draws (1000 burn-in); prior sensitivity with half-Cauchy scale 0.25 and 2.5"
    with open(RES / "a8_figdata_hier.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rows",
                "cv_setting",
                "cv",
                "composition",
                "alpha_mean",
                "alpha_lo95",
                "alpha_hi95",
                "p_best",
            ]
        )
        for tag in ROWS:
            for name in ("none", "median", "upper"):
                g = hier[tag][name]["gibbs"]
                for c in range(NSPEC):
                    w.writerow(
                        [
                            tag,
                            name,
                            f"{hier[tag][name]['cv']:.4f}",
                            COMP_ORDER[c],
                            f"{g['alpha_mean'][c]:.5f}",
                            f"{g['alpha_ci95'][c][0]:.5f}",
                            f"{g['alpha_ci95'][c][1]:.5f}",
                            f"{g['p_best'][c]:.5f}",
                        ]
                    )
    print(
        f"[a8] hierarchy done {time.time()-t0:.0f}s; P(2wt best) median CV {hier['n75']['median']['gibbs']['p_best_2wt']:.3f}",
        flush=True,
    )

    # per-draw headline csv
    with open(RES / "a8_figdata_draws.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = None
        for tag in ROWS:
            for name in ("median", "upper"):
                for b, d in enumerate(head[tag][name].get("draws_loo_force", [])):
                    if cols is None:
                        cols = list(d.keys())
                        w.writerow(
                            ["rows", "cv_setting", "draw"]
                            + cols
                            + [f"m_{c}" for c in range(NSPEC)]
                        )
                    w.writerow(
                        [tag, name, b]
                        + [f"{d[c]:.6f}" for c in cols]
                        + [f"{x:.6f}" for x in multipliers(Zall[b], CVS[name])]
                    )

    J["runtime_s"] = time.time() - t0
    J = json.loads(json.dumps(A3.to_plain(J)))  # same key types as a --tables-only rebuild
    annotate(J, args.n_jobs)
    write_tex(J)
    json.dump(A3.to_plain(J), open(RES / "a8_specvar.json", "w"), indent=1)
    print(f"[a8] done in {J['runtime_s']:.0f}s", flush=True)


# ---------------------------------------------------------------- tex
def f2(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    return s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s


def p100(x):
    return "--" if x is None else f"{100 * x:.0f}"


def pp(x):
    """Probability in percent; >99 and <1 where the rounding would print 100 or 0."""
    if x is None or not np.isfinite(x):
        return "--"
    r = round(100 * x)
    if r >= 100:
        return r"\ensuremath{>}99"
    return r"\ensuremath{<}1" if r <= 0 else f"{r:.0f}"


def lohi(s):
    """Interval of a re-draw summary: (lo, hi, kind, comment) with ranges for few draws."""
    if s.get("interval") == "range":
        return s["range"][0], s["range"][1], "range", f"range over {s['n_draws']} draws"
    return s["ci95"][0], s["ci95"][1], "percentile", f"percentile over {s['n_draws']} draws"


def write_tex(J):
    TAB.mkdir(parents=True, exist_ok=True)
    lit = J["literature"]
    SUM = J["summary"]
    HC = J["hierarchy_counterparts"]["values"]
    he = direct_study(lit)
    L = []
    NOTES = {}

    def mac(name, val, comment):
        L.append(f"\\newcommand{{\\spv{name}}}{{{val}}}  % {comment}")

    def table(fname, body, notes):
        NOTES[fname] = notes
        head = "".join(f"% note: {n}\n" for n in notes)
        (TAB / fname).write_text(head + body)
        (TAB / fname.replace(".tex", "_note.tex")).write_text(" ".join(notes) + "\n")

    L.append(
        "% numbers_a8.tex, written by code/a8_specvar.py (E3 SPECVAR). Do not edit."
    )
    mac(
        "LitNStudies",
        lit["n_studies"],
        "studies with device-to-device voltage dispersion",
    )
    mac("LitNRows", lit["n_rows"], "rows of a8_literature_cv.csv")
    mac(
        "LitCvMedian",
        p100(lit["cv_median"]),
        "percent, median study-level CV; " + SUM["cv_median_reading"],
    )
    mac(
        "LitCvPSeventyFive",
        p100(lit["cv_p75"]),
        "percent, 75th percentile study-level CV",
    )
    mac(
        "LitCvMax",
        p100(lit["cv_max"]),
        "percent, maximum study-level CV (upper setting); " + SUM["cv_direct_reading"],
    )
    mac(
        "LitCvDirect",
        p100(he["cv_study"]),
        f"percent, {he['first_author']} {he['year']}: only direct evidence from separately fabricated electrospun devices",
    )
    mac("LitDirectNDevices", he["n_devices"], "devices in that study")
    mac("LitCvMin", p100(lit["cv_min"]), "percent, minimum study-level CV")
    mac(
        "LitCvRowMax",
        p100(lit["subsets"]["row_level_cv_max"]),
        "percent, largest single-condition CV in the table",
    )
    mac(
        "LitNElectrospun",
        lit["subsets"]["electrospun_only"]["n_studies"],
        "studies on electrospun material",
    )
    B = J["draws"]
    mac("NDraws", B["B_rank"], "specimen re-draws for ranking and optimum")
    mac("NDrawsLoo", B["B_loo_force"], "re-draws with LOO and held-out-force refits")
    mac(
        "NDrawsGroup",
        B["B_group_coverage"],
        "re-draws with held-out-level jackknife+ refits",
    )
    mac(
        "NDrawsWithin",
        B["B_within_coverage"],
        "re-draws with within-grid nested jackknife+ refits",
    )
    mac("NDrawsDup", B["B_dup_loo_force"], "n=72 re-draws with LOO and force refits")
    mac("NDrawsDupCov", B["B_dup_coverage"], "n=72 re-draws with coverage refits")
    R = J["ranking"]
    tagm = {"n75": "", "n72": "Dup"}
    namem = {"median": "Med", "upper": "Up", "p75": "PSeventyFive"}
    tm = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}
    for tag in ("n75", "n72"):
        for name in ("median", "upper", "p75"):
            for t in TARGETS:
                a = R["at_cv"][tag][name][t]
                hc = HC[tag][t][name]
                pre = f"{tagm[tag]}"
                cvr = (
                    "CV median = optimistic bound"
                    if name == "median"
                    else "CV upper = He 2025, direct evidence"
                    if name == "upper"
                    else "CV 75th percentile"
                )
                for key, stem, what in (
                    ("p_rank", "RankSurvive", "P(full ranking unchanged)"),
                    ("p_opt", "OptSurvive", "P(observed best stays best)"),
                    ("p_opt_replication", "OptSurviveRep", "replication variant of P(best stays best)"),
                    ("p_rank_replication", "RankSurviveRep", "replication variant of P(ranking unchanged)"),
                ):
                    mac(
                        f"{pre}{stem}{namem[name]}{tm[t]}",
                        pp(a[key]),
                        f"percent, re-draw (composition means treated as exact), {what}, {tag}, {cvr}",
                    )
                    mac(
                        f"{pre}{stem}{namem[name]}{tm[t]}Hier",
                        pp(hc[key]),
                        f"percent, hierarchical posterior counterpart, {what}, {tag}, {cvr}",
                    )
        for t in TARGETS:
            fl = R["flip_cv"][tag][t]
            pre = tagm[tag]
            mac(
                f"{pre}FlipCvOpt{tm[t]}",
                p100(fl["cv_opt_half"]),
                f"percent, CV at which P(optimum survives) = 0.5, {tag}",
            )
            mac(
                f"{pre}FlipCvRank{tm[t]}",
                p100(fl["cv_rank_half"]),
                f"percent, CV at which P(ranking survives) = 0.5, {tag}",
            )
            mac(
                f"{pre}FlipCvOptRep{tm[t]}",
                p100(fl["cv_opt_half_replication"]),
                f"percent, replication variant, {tag}",
            )
    for tag in ("n75", "n72"):
        lo = J["law_optimum"][tag]
        pre = tagm[tag]
        mac(
            f"{pre}LawCstarObs",
            f2(lo["observed"]["c_star"]),
            f"wt%, law c* on observed data, {tag}",
        )
        for name in ("median", "upper"):
            s = lo[name]
            mac(
                f"{pre}LawCstar{namem[name]}",
                f2(s["c_star"]["median"]),
                f"wt%, median c* over re-draws, {tag}",
            )
            mac(
                f"{pre}LawCstar{namem[name]}Lo",
                f2(s["c_star"]["ci95"][0]),
                f"wt%, 2.5 percentile over {s['c_star']['n_draws']} draws",
            )
            mac(
                f"{pre}LawCstar{namem[name]}Hi",
                f2(s["c_star"]["ci95"][1]),
                f"wt%, 97.5 percentile over {s['c_star']['n_draws']} draws",
            )
            mac(
                f"{pre}LawLevelTwo{namem[name]}",
                pp(s["p_best_fabricated_level_is_2"]),
                "percent, re-draw (means treated as exact), law factor largest at 2 wt% among 0-3 wt%",
            )
            mac(
                f"{pre}LawLevelTwo{namem[name]}Hier",
                pp(HC[tag]["rms_Voc"][name]["p_best"][BEST]),
                "percent, hierarchical counterpart: posterior P(2 wt% has the largest V_rms effect)",
            )
    H = J["headlines"]
    for tag in ("n75", "n72"):
        pre = tagm[tag]
        base = H[tag]["baseline"]
        mac(
            f"{pre}BaseLoo",
            f2(base["loo_force"]["loo_r2_gp"], 3),
            f"baseline LOO R2 plain GP, {tag}",
        )
        for L_ in (1, 2, 3):
            w = {1: "One", 2: "Two", 3: "Three"}[L_]
            mac(
                f"{pre}BaseForce{w}N",
                f2(base["loo_force"][f"force_{L_}N_r2_physgp"]),
                f"baseline held-out force {L_} N R2 LawGP, {tag}",
            )
        mac(
            f"{pre}BaseWithinHits",
            base["within_hits"]["95"],
            f"baseline within-grid jk+ 95 hits, {tag}",
        )
        for ax in AXES:
            for h in ALPHAS:
                mac(
                    f"{pre}BaseGroup{ax.capitalize()}{'Ninety' if h == '90' else 'NinetyFive'}Hits",
                    base["group_hits"][ax][h],
                    f"baseline group-wise jk+ hits, {tag}",
                )
        for name in ("median", "upper"):
            h = H[tag][name]
            nm = namem[name]
            if "loo_force" in h:
                s = h["loo_force"]["loo_r2_gp"]
                mac(
                    f"{pre}Loo{nm}",
                    f2(s["median"], 3),
                    f"median LOO R2 GP over re-draws, {tag}, CV {name}",
                )
                lo_, hi_, _, how = lohi(s)
                mac(f"{pre}Loo{nm}Lo", f2(lo_, 3), f"low end, {how}")
                mac(f"{pre}Loo{nm}Hi", f2(hi_, 3), f"high end, {how}")
                for L_ in (1, 2, 3):
                    w = {1: "One", 2: "Two", 3: "Three"}[L_]
                    s = h["loo_force"][f"force_{L_}N_r2_physgp"]
                    mac(
                        f"{pre}Force{w}N{nm}",
                        f2(s["median"]),
                        f"median held-out {L_} N R2 LawGP, {tag}, CV {name}",
                    )
                    lo_, hi_, _, how = lohi(s)
                    mac(f"{pre}Force{w}N{nm}Lo", f2(lo_), f"low end, {how}")
                    mac(f"{pre}Force{w}N{nm}Hi", f2(hi_), f"high end, {how}")
                mac(
                    f"{pre}LawBeatsGpForce{nm}",
                    pp(h["p_lawgp_beats_gp_on_force_pooled"]),
                    f"percent, P(LawGP pooled held-out-force R2 > GP), {tag}; model comparison, no hierarchical counterpart",
                )
            if "within_hits" in h:
                s = h["within_hits"]["95"]
                mac(
                    f"{pre}WithinHits{nm}",
                    f"{s['median']:.0f}",
                    f"median within-grid jk+ 95 hits, {tag}, CV {name}",
                )
                lo_, hi_, _, how = lohi(s)
                mac(f"{pre}WithinHits{nm}Lo", f"{lo_:.0f}", f"min, {how}")
                mac(f"{pre}WithinHits{nm}Hi", f"{hi_:.0f}", f"max, {how}")
            if "group_hits" in h:
                for ax in AXES:
                    for lv in ALPHAS:
                        s = h["group_hits"][ax][lv]
                        nmL = "Ninety" if lv == "90" else "NinetyFive"
                        mac(
                            f"{pre}Group{ax.capitalize()}{nmL}Hits{nm}",
                            f"{s['median']:.0f}",
                            f"median group-wise jk+ hits, {tag}, CV {name}",
                        )
                        lo_, hi_, _, how = lohi(s)
                        mac(
                            f"{pre}Group{ax.capitalize()}{nmL}Hits{nm}Lo",
                            f"{lo_:.0f}",
                            f"min, {how}",
                        )
                        mac(
                            f"{pre}Group{ax.capitalize()}{nmL}Hits{nm}Hi",
                            f"{hi_:.0f}",
                            f"max, {how}",
                        )
    HI = J["hierarchy"]
    for tag in ("n75", "n72"):
        pre = tagm[tag]
        for name, nm in (("none", "NoSpec"), ("median", "Med"), ("upper", "Up")):
            g = HI[tag][name]["gibbs"]
            mac(
                f"{pre}HierPBest{nm}",
                pp(g["p_best_2wt"]),
                f"percent, posterior P(2 wt% best), {tag}, specimen CV {name}",
            )
            mac(
                f"{pre}HierRhat{nm}",
                f2(g["rhat_max_alpha"], 3),
                "max split R-hat of composition effects",
            )
            cf = HI[tag][name]["closed_form"]
            if cf["identifiable"]:
                mac(
                    f"{pre}HierCf{nm}",
                    pp(cf["p_best_2wt"]),
                    "percent, empirical-Bayes closed form P(2 wt% best)",
                )
            else:
                mac(
                    f"{pre}HierCf{nm}",
                    "not identifiable",
                    f"empirical-Bayes closed form, omega_EB = {cf['omega_eb']:g}: complete pooling, P(2 wt% best) undefined (not 0)",
                )
            k3 = COMP_ORDER.index("PVDF+BaTiO3+%3CNT")
            mac(
                f"{pre}HierRatioThree{nm}",
                f2(g["ratio_2wt_vs_each_median"][k3]),
                "posterior median ratio of 2 wt% to 3 wt% composition effect",
            )
            mac(
                f"{pre}HierRatioThree{nm}Lo",
                f2(g["ratio_2wt_vs_each_ci95"][k3][0]),
                "2.5 percentile",
            )
            mac(
                f"{pre}HierRatioThree{nm}Hi",
                f2(g["ratio_2wt_vs_each_ci95"][k3][1]),
                "97.5 percentile",
            )
            mac(
                f"{pre}HierPAboveThree{nm}",
                pp(g["p_2wt_above_each"][k3]),
                "percent, P(2 wt% effect > 3 wt% effect)",
            )
    (RES / "numbers_a8.tex").write_text("\n".join(L) + "\n")

    # --- tables (booktabs bodies)
    rows = []
    for s in sorted(lit["studies"], key=lambda s: (s["year"], s["first_author"])):
        mat = s["material"].split(" (")[0].split(",")[0].replace("%", "\\%")
        rows.append(
            f"{s['first_author']} ({s['year']}) & {mat} & {s['n_devices']} & {s['replicate_type']} & "
            f"{p100(s['cv_min'])}--{p100(s['cv_max'])} & {p100(s['cv_study'])} \\\\"
        )
    body = (
        "\\toprule\nStudy & Material & $n$ & Replicates & CV range (\\%) & CV (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
    )
    body += f"\n\\midrule\nMedian / 75th pct. / max & & & & & {p100(lit['cv_median'])} / {p100(lit['cv_p75'])} / {p100(lit['cv_max'])} \\\\\n\\bottomrule\n"
    table(
        "tab_a8_literature.tex",
        body,
        [
            f"The median study-level CV ({p100(lit['cv_median'])}\\%) is an optimistic bound: it comes from studies "
            "with unspecified replicate type (Lang 2016, Qu 2026), a spread that looks like repeat measurement "
            "(Li 2023) or non-electrospun devices (Ge 2024).",
            f"{he['first_author']} ({he['year']}, {p100(he['cv_study'])}\\%) is the only direct evidence from "
            "separately fabricated electrospun devices.",
        ],
    )

    lab = {
        "rms_Voc": r"$V_\mathrm{rms}$",
        "Vpp": r"$V_\mathrm{pp}$",
        "Vmax": r"$|V|_\mathrm{max}$",
    }
    rows = []
    for tag, tl in (("n75", "75"), ("n72", "72")):
        for t in TARGETS:
            a = R["at_cv"][tag]
            fl = R["flip_cv"][tag][t]

            def both(name, key):
                return f"{pp(a[name][t][key])} ({pp(HC[tag][t][name][key])})"

            rows.append(
                f"{lab[t]} & {tl} & {both('median', 'p_rank')} & {both('upper', 'p_rank')} & "
                f"{both('median', 'p_opt')} & {both('upper', 'p_opt')} & "
                f"{both('median', 'p_opt_replication')} & {both('upper', 'p_opt_replication')} & "
                f"{p100(fl['cv_rank_half'])} & {p100(fl['cv_opt_half'])} \\\\"
            )
    body = (
        "\\toprule\n & & \\multicolumn{2}{c}{Ranking (\\%)} & \\multicolumn{2}{c}{Optimum (\\%)} & \\multicolumn{2}{c}{Opt., replication (\\%)} & \\multicolumn{2}{c}{CV at 50\\% (\\%)} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
        "Target & $n$ & Med. & Upper & Med. & Upper & Med. & Upper & Ranking & Optimum \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
    )
    table(
        "tab_a8_survival.tex",
        body,
        [
            "Re-draw probability (hierarchical posterior probability in parentheses). The re-draw treats the "
            "observed composition means as exact and perturbs them by the specimen factor only; the hierarchy "
            "also carries the within-specimen noise of the 15 recordings.",
            f"Med.: median study-level CV ({p100(lit['cv_median'])}\\%), an optimistic bound; Upper: "
            f"{he['first_author']} {he['year']} ({p100(he['cv_study'])}\\%), the only direct evidence from separately "
            "fabricated electrospun devices.",
            "$>$99 and $<$1: rounds to 100 or 0 percent. CV at 50\\%: re-draw only.",
        ],
    )

    def cell(s, d=2, integer=False):
        lo_, hi_, kind, _ = lohi(s)
        fmt = (lambda x: f"{x:.0f}") if integer else (lambda x: f2(x, d))
        med = fmt(s["median"])
        if kind == "range":
            return f"{med} ({fmt(lo_)}--{fmt(hi_)})"
        return f"{med} [{fmt(lo_)}, {fmt(hi_)}]"

    rows = []
    for tag, tl in (("n75", "75"), ("n72", "72")):
        base = H[tag]["baseline"]
        nn = base["n"]
        med, up = H[tag]["median"], H[tag]["upper"]
        rows.append(
            f"LOO $R^2$, GP & {tl} & {f2(base['loo_force']['loo_r2_gp'], 3)} & {cell(med['loo_force']['loo_r2_gp'], 3)} & {cell(up['loo_force']['loo_r2_gp'], 3)} \\\\"
        )
        for L_ in (1, 2, 3):
            k = f"force_{L_}N_r2_physgp"
            rows.append(
                f"Held-out {L_} N $R^2$, LawGP & {tl} & {f2(base['loo_force'][k])} & {cell(med['loo_force'][k])} & {cell(up['loo_force'][k])} \\\\"
            )
        if "within_hits" in med:
            rows.append(
                f"Within-grid JK+ 95\\% hits (of {nn}) & {tl} & {base['within_hits']['95']} & {cell(med['within_hits']['95'], integer=True)} & {cell(up['within_hits']['95'], integer=True)} \\\\"
            )
        if "group_hits" in med:
            for ax in AXES:
                nax = nn
                for lv in ALPHAS:
                    rows.append(
                        f"Held-out {ax} JK+ {lv}\\% hits (of {nax}) & {tl} & {base['group_hits'][ax][lv]} & {cell(med['group_hits'][ax][lv], integer=True)} & {cell(up['group_hits'][ax][lv], integer=True)} \\\\"
                    )
        if tag == "n75":
            rows.append("\\midrule")
    body = (
        "\\toprule\nHeadline & $n$ & Observed & CV median & CV upper \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
    )
    B = J["draws"]
    table(
        "tab_a8_headlines.tex",
        body,
        [
            "Median over specimen re-draws; square brackets: 2.5 to 97.5 percentile over "
            f"{B['B_loo_force']} ($n = 75$) or {B['B_dup_loo_force']} ($n = 72$) draws; parentheses: range (min to max) "
            f"over {B['B_within_coverage']} (within-grid), {B['B_group_coverage']} (held-out level, $n = 75$) or "
            f"{B['B_dup_coverage']} ($n = 72$) draws, too few for a percentile interval.",
            "The re-draw treats the observed composition means as exact. CV median: optimistic bound; "
            f"CV upper: {he['first_author']} {he['year']}, the only direct evidence from separately fabricated electrospun devices.",
        ],
    )

    rows = []
    short = {c: A2.COMP_TEX[c] for c in COMP_ORDER}
    for c in range(NSPEC):
        cells = []
        for name in ("none", "median", "upper"):
            g = HI["n75"][name]["gibbs"]
            cells.append(
                f"{f2(g['alpha_mean'][c])} [{f2(g['alpha_ci95'][c][0])}, {f2(g['alpha_ci95'][c][1])}] & {pp(g['p_best'][c])}"
            )
        rows.append(f"{short[COMP_ORDER[c]]} & " + " & ".join(cells) + " \\\\")
    g72 = [HI["n72"][nm]["gibbs"]["p_best_2wt"] for nm in ("none", "median", "upper")]
    rows.append("\\midrule")
    rows.append(
        "$P$(2 wt\\% best), $n = 72$ & & "
        + " & & ".join(pp(x) for x in g72)
        + " \\\\"
    )
    body = (
        "\\toprule\n & \\multicolumn{2}{c}{No specimen effect} & \\multicolumn{2}{c}{CV median} & \\multicolumn{2}{c}{CV upper} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Composition & $\\alpha_c$ [95\\% CrI] & $P$(best) & $\\alpha_c$ [95\\% CrI] & $P$(best) & $\\alpha_c$ [95\\% CrI] & $P$(best) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
    )
    cfu = HI["n75"]["upper"]["closed_form"]
    table(
        "tab_a8_hierarchy.tex",
        body,
        [
            "Gibbs posterior of the composition effect $\\alpha_c$ on $\\log V_\\mathrm{rms}$ with the specimen "
            "variance fixed at $\\ln(1 + \\mathrm{CV}^2)$; $P$(best) in percent.",
            f"CV median ({p100(lit['cv_median'])}\\%): optimistic bound; CV upper ({p100(he['cv_study'])}\\%, "
            f"{he['first_author']} {he['year']}): the only direct evidence from separately fabricated electrospun devices.",
        ]
        + (
            []
            if cfu["identifiable"]
            else [
                "At CV upper the empirical-Bayes closed-form check is not identifiable "
                "($\\omega_\\mathrm{EB} = 0$, complete pooling); the Gibbs values are the reported ones."
            ]
        ),
    )
    J["table_notes"] = NOTES


if __name__ == "__main__":
    main()
