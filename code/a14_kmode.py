#!/usr/bin/env python3
"""E9 KMODE: the law's frequency factor as a sum of K Lorentzian modes, K chosen by BIC.

One command regenerates everything:
    python code/a14_kmode.py [--kmax 5] [--n-jobs 4] [--seed 0] [--smoke]
Inputs: data/targets_design.parquet (through a2_common), the digitized external sweeps
(through a9_external.load_datasets), the reference scores a2_physgp.json and a9_external.json.
Outputs (all under results/):
    a14_kmode.json                  every statistic, with an index of macro -> json key
    a14_figdata_folds.csv           per training fold: dataset, target, scheme, level, BIC per K, chosen K
    a14_figdata_oof.csv             out-of-fold predictions of LawGP-1 and LawGP-K
    tables/tab_a14_bic.tex          full-data BIC per K and the chosen K per dataset
    tables/tab_a14_curves.tex       per-curve K (each sweep alone) on the external resonance data
    tables/tab_a14_scores.tex       GP, LawGP-1, LawGP-K per dataset and held-out scheme (+ fold K counts)
    tables/tab_a14_scores_n72.tex   the same on our grid without the three duplicated recordings
    tables/tab_a14_levels.tex       our grid, V_rms, per held-out level
    numbers_a14.tex                 \\kmd* macros

The K-mode law. The frequency factor L(f; f0, gamma) of the five-parameter law becomes
    L_K(f) = L(f; f01, g1) + sum_{k=2..K} w_k L(f; f0k, gk),  w_k >= 0,
so that on our grid
    m(x) = (a0 + a1 c + a2 c^2) * F * L_K(f)
and on the external resonance sweeps (a9 law form 'cL', no force factor)
    m(x) = (a0 + a1 c + a2 c^2) * L_K(f).
The first mode carries weight one because the bracket already carries the amplitude, so K modes
add 3 (K - 1) parameters. K = 1 is not reimplemented: it is a2_common.fit_law5 on our grid and
a9_external.fit_law on the external data, so LawGP-1 here is the published LawGP bit for bit.
K >= 2 is fitted on y / std(y_train) (a2_common scaling, tolerance a2_common.TOL_TIGHT) from a
deterministic multistart: the best (K - 1)-mode solution plus one new mode at five positions and
two widths, and (K = 2) the a9 two-Lorentzian position pairs; every extra mode lives in the same
(f0, gamma) box as the first (a2_common box on our grid, the a9 per-dataset box outside).
BIC = n log(RSS / n) + k log n on the training rows (a9_external.bic), k = free bracket terms
+ 2 + 3 (K - 1). K is chosen inside every training fold (never on the full data before scoring);
the full-data choice is reported separately as the descriptive answer per dataset.

LawGP-K is the plain GP (a2_common.gp) on the residuals of the fold's BIC-chosen K-mode law;
when a fold picks K = 1 its prediction is exactly LawGP-1's. The GP reference numbers are read
from a2_physgp.json (our grid, one optimizer start) and a9_external.json (Salama one start,
Park five starts = a2_common.gp(4), as in the a9 headline tables). Park is a continuous
loudspeaker sweep digitized at 5 Hz steps; peak heights between samples are not in the data
(park2017.provenance.json), so a narrow mode can be under-resolved and K is a lower bound there.
Shi (one frequency) and the Salama impulse test (two frequencies) have no frequency factor to
generalize; they are listed with K not identifiable.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_common as A2  # noqa: E402
import a9_external as A9  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = A2.ROOT
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
SCRIPT = "code/a14_kmode.py"
DUP_IDS = [52, 54, 73]  # a2_physgp.py, duplicated-recordings note in the README
KEEP72 = np.array([i for i in range(A2.N) if i not in DUP_IDS])
PARK = ("ParkG", "ParkT")
EXT_KEYS = (
    "SalamaFreq",
    "ParkG",
    "ParkT",
)  # a9 law form 'cL': the frequency factor exists
EXT_NA = {
    "SalamaForce": "two frequency levels (10 and 15 Hz): the law uses one level ratio, no Lorentzian",
    "Shi": "one frequency (0.2 Hz): no frequency factor",
}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# =============================================================================== problems
class Grid:
    """Our 75-record grid, one target."""

    name = "Grid"

    def __init__(self, target):
        self.target = target
        self.y = A2.df[target].values.astype(float)
        self.c, self.F, self.f = A2.CNT, A2.FRC, A2.FRQ
        self.mult = A2.FRC
        self.X = A2.X
        self.fb = (A2.BOUNDS_5[0][3], A2.BOUNDS_5[1][3])
        self.gb = (A2.BOUNDS_5[0][4], A2.BOUNDS_5[1][4])
        self.restarts = 0

    def bracket_setup(self, tr):
        lo, hi = list(A2.BOUNDS_5[0][:3]), list(A2.BOUNDS_5[1][:3])
        return [0, 1, 2], lo, hi

    def fit1(self, tr):
        """K = 1: a2_common.fit_law5, the published path. Returns (pred, scaled free params, scale)."""
        m, pv, info = A2.fit_law5(tr, self.y)
        return m, np.asarray(info["p_scaled"], float), info["scale"]

    def lawgp1(self, tr, te):
        mu, sd, _ = A2.fold_predict("physgp", tr, te, self.y)
        return mu, sd


class Ext:
    """An a9 external dataset with law form 'cL'."""

    def __init__(self, ds):
        self.ds = ds
        self.name = ds["key"]
        self.y = ds["y"]
        self.c, self.F, self.f = ds["c"], ds["F"], ds["f"]
        self.mult = np.ones(ds["n"])
        self.X = ds["X"]
        self.fb, self.gb = ds["f_bounds"], ds["g_bounds"]
        self.restarts = A2.RESTARTS_CHECK if self.name in PARK else 0

    def bracket_setup(self, tr):
        nc = len(np.unique(self.c[tr]))
        free = [0, 1, 2][: min(3, nc)]
        return free, [0.0, -1e6, -1e6][: len(free)], [1e6, 1e6, 1e6][: len(free)]

    def fit1(self, tr):
        pred, pv, info = A9.fit_law(self.ds, tr)
        s = info["scale"]
        free, _, _ = self.bracket_setup(tr)
        q = np.concatenate([pv[free] / s, pv[3:5]])
        return pred, q, s

    def lawgp1(self, tr, te):
        return A9.fold_predict(
            self.ds, "lawgp_r4" if self.restarts else "lawgp", tr, te
        )


# =============================================================================== K-mode law
def lk(f, q_modes):
    """L_K(f): q_modes = [f01, g1, w2, f02, g2, ...]."""
    out = A2.lorentz(f, q_modes[0], q_modes[1])
    for j in range(2, len(q_modes), 3):
        out = out + q_modes[j] * A2.lorentz(f, q_modes[j + 1], q_modes[j + 2])
    return out


def kmode_eval(P, q, nb, idx):
    """Scaled K-mode law at rows idx; q = free bracket terms (nb of them) + mode params."""
    b = np.zeros(3)
    b[:nb] = q[:nb]
    c = P.c[idx]
    return (b[0] + b[1] * c + b[2] * c**2) * P.mult[idx] * lk(P.f[idx], q[nb:])


def fit_ladder(P, tr, kmax):
    """Fit K = 1..kmax on rows tr. Returns list of dicts (one per K) with pred(idx) in data units."""
    tr = np.asarray(tr)
    n = len(tr)
    free, blo, bhi = P.bracket_setup(tr)
    nb = len(free)
    pred1, q1, scale = P.fit1(tr)
    ys = P.y[tr] / scale
    out = []
    r1 = P.y[tr] - pred1(tr)
    rss = float(np.sum(r1**2))
    out.append(
        {
            "K": 1,
            "k": nb + 2,
            "rss": rss,
            "bic": A9.bic(rss, n, nb + 2),
            "q": q1,
            "pred": pred1,
            "ok": True,
        }
    )
    fr = P.f[tr]
    R = float(fr.max() - fr.min()) or 1.0
    pos = [float(fr.min() + t * R) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    kw = {"ftol": A2.TOL_TIGHT, "xtol": A2.TOL_TIGHT, "gtol": A2.TOL_TIGHT}
    for K in range(2, kmax + 1):
        k = nb + 2 + 3 * (K - 1)
        lo = blo + [P.fb[0], P.gb[0]] + [0.0, P.fb[0], P.gb[0]] * (K - 1)
        hi = bhi + [P.fb[1], P.gb[1]] + [100.0, P.fb[1], P.gb[1]] * (K - 1)
        prev = out[-1]["q"]
        if (
            prev is None or n <= k + 1
        ):  # too few rows for K modes (or K - 1 already failed)
            out.append(
                {
                    "K": K,
                    "k": k,
                    "rss": float("nan"),
                    "bic": float("nan"),
                    "q": None,
                    "pred": None,
                    "ok": False,
                }
            )
            continue
        starts = []
        for p0 in pos:
            for gw in (0.1 * R, 0.3 * R):
                starts.append(np.concatenate([prev, [0.5, p0, gw]]))
        if K == 2:  # a9 two-Lorentzian position pairs, bracket from K = 1
            grid = np.linspace(fr.min(), fr.max(), 7)[1:-1]
            for i in range(len(grid)):
                for j in range(i + 1, len(grid)):
                    starts.append(
                        np.concatenate(
                            [q1[:nb], [grid[i], 0.1 * R, 1.0, grid[j], 0.1 * R]]
                        )
                    )
        best = None
        if n <= k + 1:
            out.append(
                {
                    "K": K,
                    "k": k,
                    "rss": float("nan"),
                    "bic": float("nan"),
                    "q": None,
                    "pred": None,
                    "ok": False,
                }
            )
            continue
        for st in starts:
            st = np.clip(st, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
            try:
                q, _ = curve_fit(
                    lambda _, *qq: kmode_eval(P, np.asarray(qq), nb, tr),
                    None,
                    ys,
                    p0=st,
                    bounds=(lo, hi),
                    maxfev=20000,
                    **kw,
                )
            except Exception:
                continue
            sse = float(np.sum((kmode_eval(P, q, nb, tr) - ys) ** 2))
            if best is None or sse < best[1]:
                best = (q, sse)
        if best is None:
            out.append(
                {
                    "K": K,
                    "k": k,
                    "rss": float("nan"),
                    "bic": float("nan"),
                    "q": None,
                    "pred": None,
                    "ok": False,
                }
            )
            continue
        q = best[0]
        # nesting guard: the (K-1)-mode solution is feasible (w_K = 0), keep it if the search did worse
        rss = best[1] * scale**2
        if rss > out[-1]["rss"]:
            q = np.concatenate([prev, [0.0, pos[2], 0.3 * R]])
            rss = out[-1]["rss"]

        def pred(idx, q=q):
            return kmode_eval(P, q, nb, np.asarray(idx)) * scale

        out.append(
            {
                "K": K,
                "k": k,
                "rss": rss,
                "bic": A9.bic(rss, n, k),
                "q": q,
                "pred": pred,
                "ok": True,
            }
        )
    return out, nb, scale


def choose(ladder):
    ok = [r for r in ladder if r["ok"] and np.isfinite(r["bic"])]
    return min(ok, key=lambda r: r["bic"])


def ladder_json(ladder, nb, scale, n):
    rows = []
    for r in ladder:
        d = {"K": r["K"], "k": r["k"], "RSS": r["rss"], "BIC": r["bic"], "ok": r["ok"]}
        if r["q"] is not None:
            q = np.asarray(r["q"], float)
            d["bracket_volt_scale"] = (q[:nb] * scale).tolist()
            modes = [{"f0": float(q[nb]), "gamma": float(q[nb + 1]), "w": 1.0}]
            for j in range(nb + 2, len(q), 3):
                modes.append(
                    {"f0": float(q[j + 1]), "gamma": float(q[j + 2]), "w": float(q[j])}
                )
            d["modes"] = modes
        rows.append(d)
    return rows


# =============================================================================== fold job
def fold_job(P, tr, te, kmax):
    """One outer fold: ladder on the training rows, BIC choice, LawGP-1 and LawGP-K predictions."""
    tr, te = np.asarray(tr), np.asarray(te)
    ladder, nb, scale = fit_ladder(P, tr, kmax)
    ch = choose(ladder)
    mu1, sd1 = P.lawgp1(tr, te)
    if ch["K"] == 1:
        muK, sdK = mu1, sd1
    else:
        btr, bte = ch["pred"](tr), ch["pred"](te)
        xs = StandardScaler().fit(P.X[tr])
        g = A2.gp(P.restarts).fit(xs.transform(P.X[tr]), P.y[tr] - btr)
        m, s = g.predict(xs.transform(P.X[te]), return_std=True)
        muK, sdK = bte + m, s
    return {
        "te": te,
        "mu1": mu1,
        "muK": muK,
        "lawK": ch["pred"](te),
        "K": ch["K"],
        "bic": [r["bic"] for r in ladder],
    }


def folds_for(P, scheme, keep=None):
    if isinstance(P, Grid):
        if scheme == "loo":
            return [(tr, te, str(int(te[0]))) for tr, te in A2.loo_folds(keep)]
        return [
            (tr, te, A2.level_key(scheme, v))
            for tr, te, v in A2.group_folds(scheme, idx=keep)
        ]
    return [(tr, te, str(v)) for tr, te, v in A9.folds(P.ds, scheme)]


def score(y, rows, mu, levels=None):
    out = {
        "R2": float(r2_score(y[rows], mu[rows])),
        "MAE": float(mean_absolute_error(y[rows], mu[rows])),
    }
    if levels:
        per = {}
        for lv, te in levels:
            per[lv] = {
                "n": int(len(te)),
                "R2": float(r2_score(y[te], mu[te]))
                if len(te) > 1 and np.var(y[te]) > 0
                else None,
                "MAE": float(mean_absolute_error(y[te], mu[te])),
            }
        out["per_level"] = per
        v = [p["R2"] for p in per.values() if p["R2"] is not None]
        out["mean_within_level_R2"] = float(np.mean(v)) if v else None
    return out


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--kmax", type=int, default=5, help="largest number of Lorentzian modes"
    )
    ap.add_argument(
        "--n-jobs", type=int, default=4, help="joblib workers (a6_twin runs beside)"
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="recorded; every fit is deterministic"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="V_rms held-out frequency and SalamaFreq only",
    )
    ap.add_argument("--no-manifest", action="store_true")
    a = ap.parse_args()
    np.random.seed(a.seed)
    os.makedirs(TAB, exist_ok=True)

    ref2 = json.load(open(os.path.join(RES, "a2_physgp.json")))
    ref9 = json.load(open(os.path.join(RES, "a9_external.json")))
    DS = A9.load_datasets()

    targets = ["rms_Voc"] if a.smoke else list(A2.TARGETS)
    grid_schemes = (
        ["frequency"] if a.smoke else ["loo", "composition", "force", "frequency"]
    )
    ext_keys = ["SalamaFreq"] if a.smoke else list(EXT_KEYS)

    J = {
        "meta": {
            "task": "E9",
            "script": SCRIPT,
            "seed": a.seed,
            "kmax": a.kmax,
            "n_jobs": a.n_jobs,
            "smoke": a.smoke,
            "law_tolerance": A2.TOL_TIGHT,
            "bic": "n log(RSS/n) + k log n on the training rows (a9_external.bic)",
            "k_count": "free bracket terms + 2 + 3 (K - 1)",
            "gp_restarts": {
                "Grid": 0,
                "SalamaFreq": 0,
                "ParkG": A2.RESTARTS_CHECK,
                "ParkT": A2.RESTARTS_CHECK,
            },
            "park_caveat": "Park sweeps digitized at 5 Hz steps; peak heights between samples are not in the csv "
            "(park2017.provenance.json digitization_error.stated.fig4a_note), so K is a lower bound there",
            "date": time.strftime("%Y-%m-%d"),
        },
        "full_data": {},
        "per_curve": {},
        "schemes": {},
        "not_applicable": EXT_NA,
        "reproduction": {},
        "index": {},
    }

    # ------------------------------------------------------------ full-data K per dataset
    problems = {}
    for t in targets:
        problems[f"Grid:{t}"] = Grid(t)
    for key in ext_keys:
        problems[key] = Ext(DS[key])
    for name, P in problems.items():
        rows = np.arange(len(P.y))
        lad, nb, sc = fit_ladder(P, rows, a.kmax)
        ch = choose(lad)
        sst = float(np.sum((P.y - P.y.mean()) ** 2))
        J["full_data"][name] = {
            "n": int(len(rows)),
            "ladder": ladder_json(lad, nb, sc, len(rows)),
            "K_bic": ch["K"],
            "R2_in_sample": {
                str(r["K"]): (1 - r["rss"] / sst if r["ok"] else None) for r in lad
            },
            "delta_BIC_vs_K1": {
                str(r["K"]): (r["bic"] - lad[0]["bic"] if r["ok"] else None)
                for r in lad
            },
        }
        if isinstance(P, Grid):
            lad72, nb72, sc72 = fit_ladder(P, KEEP72, a.kmax)
            J["full_data"][name]["n72"] = {
                "K_bic": choose(lad72)["K"],
                "delta_BIC_vs_K1": {
                    str(r["K"]): (r["bic"] - lad72[0]["bic"] if r["ok"] else None)
                    for r in lad72
                },
            }
        log(
            f"full data {name}: K_bic = {ch['K']}, dBIC = {J['full_data'][name]['delta_BIC_vs_K1']}"
        )

    # ------------------------------------------------------------ per-curve K (external sweeps)
    for key in ext_keys:
        ds = DS[key]
        cur = pd.Series(ds["curve"])
        rows = []
        for cv in cur.unique():
            idx = np.where((cur == cv).values)[0]
            sub = dict(ds)
            sub.update(
                c=np.zeros(len(idx)),
                F=ds["F"][idx],
                f=ds["f"][idx],
                y=ds["y"][idx],
                X=ds["X"][idx],
                n=len(idx),
            )
            P = Ext(sub)
            lad, nb, sc = fit_ladder(P, np.arange(len(idx)), a.kmax)
            ch = choose(lad)
            sst = float(np.sum((P.y - P.y.mean()) ** 2))
            rows.append(
                {
                    "curve": cv,
                    "n": int(len(idx)),
                    "K_bic": ch["K"],
                    "BIC": {str(r["K"]): r["bic"] for r in lad},
                    "R2": {
                        str(r["K"]): (1 - r["rss"] / sst if r["ok"] else None)
                        for r in lad
                    },
                    "modes_chosen": ladder_json([ch], nb, sc, len(idx))[0].get("modes"),
                }
            )
        J["per_curve"][key] = {
            "curves": rows,
            "K_counts": {
                str(k): int(sum(r["K_bic"] == k for r in rows))
                for k in range(1, a.kmax + 1)
            },
            "median_R2_K1": float(np.median([r["R2"]["1"] for r in rows])),
            "median_R2_Kbic": float(
                np.median([r["R2"][str(r["K_bic"])] for r in rows])
            ),
        }
        log(f"per curve {key}: K counts {J['per_curve'][key]['K_counts']}")

    # ------------------------------------------------------------ held-out schemes
    fold_rows, oof_rows = [], []
    variants = [(f"Grid:{t}", sc, "n75", None) for t in targets for sc in grid_schemes]
    if not a.smoke:
        variants += [
            (f"Grid:{t}", sc, "n72", KEEP72) for t in targets for sc in grid_schemes
        ]
    for key in ext_keys:
        for sc in ["loo"] + list(DS[key]["axes"]):
            if a.smoke and sc == "loo":
                continue
            variants.append((key, sc, "all", None))
    jobs = []
    for name, sc, sub, keep in variants:
        P = problems[name]
        for tr, te, lv in folds_for(P, sc, keep):
            jobs.append((name, sc, sub, lv, tr, te))
    log(f"{len(jobs)} outer folds on {a.n_jobs} workers")
    res = Parallel(n_jobs=a.n_jobs, verbose=0)(
        delayed(fold_job)(problems[name], tr, te, a.kmax)
        for name, sc, sub, lv, tr, te in jobs
    )
    log("folds done")
    acc = {}
    for (name, sc, sub, lv, tr, te), r in zip(jobs, res):
        acc.setdefault((name, sc, sub), []).append((lv, r))
        fold_rows.append(
            {
                "dataset": name,
                "scheme": sc,
                "subset": sub,
                "level": lv,
                "n_train": len(tr),
                "K_chosen": r["K"],
                **{f"BIC_K{k + 1}": b for k, b in enumerate(r["bic"])},
            }
        )
    for (name, sc, sub), lst in acc.items():
        P = problems[name]
        y = P.y
        n = len(y)
        mu1, muK, lawK = (np.full(n, np.nan) for _ in range(3))
        levels = []
        for lv, r in lst:
            mu1[r["te"]], muK[r["te"]], lawK[r["te"]] = r["mu1"], r["muK"], r["lawK"]
            levels.append((lv, r["te"]))
        rows = np.where(np.isfinite(mu1))[0]
        lvls = None if sc == "loo" else levels
        Ks = [r["K"] for _, r in lst]
        out = {
            "n": int(len(rows)),
            "n_folds": len(lst),
            "K_counts": {
                str(k): int(sum(x == k for x in Ks)) for k in range(1, a.kmax + 1)
            },
            "lawgp1": score(y, rows, mu1, lvls),
            "lawgpK": score(y, rows, muK, lvls),
            "lawK": score(y, rows, lawK, lvls),
        }
        if sc != "loo":
            out["K_by_level"] = {lv: r["K"] for lv, r in lst}
        # reference numbers and exact-reproduction check
        if name.startswith("Grid:"):
            t = name.split(":")[1]
            if sub == "n75":
                ref = ref2["pooled"][t][sc]
                out["gp"] = {
                    "R2": ref["gp"]["R2"],
                    "MAE": ref["gp"]["MAE"],
                    "source": f"a2_physgp.json pooled.{t}.{sc}.gp",
                }
                out["lawgp1_reference"] = {
                    "R2": ref["physgp"]["R2"],
                    "source": f"a2_physgp.json pooled.{t}.{sc}.physgp",
                }
                if sc != "loo":
                    pl = ref2["per_level"][t][sc]
                    out["gp"]["per_level"] = {
                        k: {"R2": v["R2"], "MAE": v["MAE"]} for k, v in pl["gp"].items()
                    }
                    out["lawgp1_reference"]["per_level_R2"] = {
                        k: v["R2"] for k, v in pl["physgp"].items()
                    }
            else:
                D = ref2["sensitivity_excluding_duplicates"]
                ref = D["loo"][t] if sc == "loo" else D["pooled"][t][sc]
                out["gp"] = {
                    "R2": ref["gp"]["R2"],
                    "MAE": ref["gp"]["MAE"],
                    "source": "a2_physgp.json sensitivity_excluding_duplicates",
                }
                out["lawgp1_reference"] = {
                    "R2": ref["physgp"]["R2"],
                    "source": "a2_physgp.json sensitivity_excluding_duplicates",
                }
                if sc != "loo":
                    out["gp"]["per_level"] = {
                        k: {"R2": v["R2"]}
                        for k, v in D["per_level"][t][sc]["gp"].items()
                    }
        else:
            sch = ref9["datasets"][name]["schemes"][sc]
            gk = "gp_r4" if name in PARK else "gp"
            lk_ = "lawgp_r4" if name in PARK else "lawgp"
            out["gp"] = {
                "R2": sch[gk]["R2"],
                "MAE": sch[gk]["MAE"],
                "source": f"a9_external.json datasets.{name}.schemes.{sc}.{gk}",
            }
            if "per_level" in sch[gk]:
                out["gp"]["per_level"] = sch[gk]["per_level"]
            out["lawgp1_reference"] = {
                "R2": sch[lk_]["R2"],
                "source": f"a9_external.json datasets.{name}.schemes.{sc}.{lk_}",
            }
        out["lawgp1_minus_reference_R2"] = (
            out["lawgp1"]["R2"] - out["lawgp1_reference"]["R2"]
        )
        out["lawgpK_minus_lawgp1_R2"] = out["lawgpK"]["R2"] - out["lawgp1"]["R2"]
        out["lawgpK_minus_gp_R2"] = out["lawgpK"]["R2"] - out["gp"]["R2"]
        out["lawgp1_minus_gp_R2"] = out["lawgp1"]["R2"] - out["gp"]["R2"]
        J["schemes"].setdefault(name, {}).setdefault(sub, {})[sc] = out
        for i in rows:
            oof_rows.append(
                {
                    "dataset": name,
                    "scheme": sc,
                    "subset": sub,
                    "row": int(i),
                    "y": y[i],
                    "lawgp1": mu1[i],
                    "lawgpK": muK[i],
                    "lawK": lawK[i],
                }
            )
        log(
            f"{name} {sub} {sc}: K {out['K_counts']}, GP {out['gp']['R2']:.3f}, LawGP-1 {out['lawgp1']['R2']:.3f} "
            f"(ref {out['lawgp1_reference']['R2']:.3f}), LawGP-K {out['lawgpK']['R2']:.3f}"
        )

    diffs = [
        abs(v["lawgp1_minus_reference_R2"])
        for d in J["schemes"].values()
        for s in d.values()
        for v in s.values()
    ]
    J["reproduction"] = {
        "max_abs_R2_diff_lawgp1_vs_published": float(max(diffs)),
        "n_checks": len(diffs),
        "exact": bool(max(diffs) == 0.0),
    }
    log(
        f"reproduction: max |dR2| = {J['reproduction']['max_abs_R2_diff_lawgp1_vs_published']:.3g} over {len(diffs)} checks"
    )

    J["verdict"] = verdict(J)
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(RES, "a14_figdata_folds.csv"), index=False
    )
    pd.DataFrame(oof_rows).to_csv(os.path.join(RES, "a14_figdata_oof.csv"), index=False)
    write_tables(J)
    write_macros(J)
    with open(os.path.join(RES, "a14_kmode.json"), "w") as fh:
        json.dump(
            J,
            fh,
            indent=1,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    pass  # run manifest omitted in the public version
    log("done")


# =============================================================================== verdict
def verdict(J):
    V = {"full_data_K": {k: v["K_bic"] for k, v in J["full_data"].items()}}
    V["full_data_K_n72"] = {
        k: v["n72"]["K_bic"] for k, v in J["full_data"].items() if "n72" in v
    }
    V["per_curve_K_counts"] = {k: v["K_counts"] for k, v in J["per_curve"].items()}
    helps, hurts = [], []
    for name, d in J["schemes"].items():
        for sub, s in d.items():
            for sc, v in s.items():
                dK = v["lawgpK_minus_lawgp1_R2"]
                rec = {
                    "dataset": name,
                    "subset": sub,
                    "scheme": sc,
                    "gp": v["gp"]["R2"],
                    "lawgp1": v["lawgp1"]["R2"],
                    "lawgpK": v["lawgpK"]["R2"],
                    "K_counts": v["K_counts"],
                }
                if dK > 0.01:
                    rec["beats_gp"] = v["lawgpK"]["R2"] > v["gp"]["R2"]
                    rec["lawgp1_beat_gp"] = v["lawgp1"]["R2"] > v["gp"]["R2"]
                    helps.append(rec)
                elif dK < -0.01:
                    hurts.append(rec)
    V["lawgpK_better_than_lawgp1_by_0.01"] = helps
    V["lawgpK_worse_than_lawgp1_by_0.01"] = hurts
    V["new_wins_over_gp"] = [
        h for h in helps if h["beats_gp"] and not h["lawgp1_beat_gp"]
    ]
    return V


# =============================================================================== formatting
fnum, tnum, sig = A9.fnum, A9.tnum, A9.sig
DS_TEX = {
    "Grid:rms_Voc": r"This grid, \RMS",
    "Grid:Vpp": r"This grid, \Vpp",
    "Grid:Vmax": r"This grid, \Vmax",
    "SalamaFreq": "Salama 2024, frequency",
    "ParkG": r"Park 2017, G/PVDF/G (5\,Hz samples)",
    "ParkT": r"Park 2017, P(VDF-TrFE) (5\,Hz samples)",
}
DS_MAC = {
    "Grid:rms_Voc": "GridRms",
    "Grid:Vpp": "GridVpp",
    "Grid:Vmax": "GridVmax",
    "SalamaFreq": "SalamaFreq",
    "ParkG": "ParkG",
    "ParkT": "ParkT",
}
SC_TEX = {
    "loo": "LOO",
    "composition": "held-out composition",
    "force": "held-out force",
    "frequency": "held-out frequency",
    "preload": "held-out preload",
}
SC_MAC = {
    "loo": "Loo",
    "composition": "Comp",
    "force": "Force",
    "frequency": "Freq",
    "preload": "Preload",
}
K_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def kcounts(d):
    return "/".join(str(d[k]) for k in sorted(d, key=int))


def write_tables(J):
    kmax = J["meta"]["kmax"]
    # BIC per K
    L = [
        r"\begin{tabular}{l r " + "r " * (kmax - 1) + "c r r}",
        r"\toprule",
    ]
    L.append(
        "Dataset & $n$ & "
        + " & ".join(rf"$\Delta$BIC, $K={k}$" for k in range(2, kmax + 1))
        + r" & BIC $K$ & $R^2$, $K=1$ & $R^2$, BIC $K$ \\"
    )
    L.append(r"\midrule")
    for name, v in J["full_data"].items():
        L.append(
            f"{DS_TEX[name]} & {v['n']} & "
            + " & ".join(
                fnum(v["delta_BIC_vs_K1"][str(k)], 1) for k in range(2, kmax + 1)
            )
            + f" & {v['K_bic']} & {fnum(v['R2_in_sample']['1'], 3)} & {fnum(v['R2_in_sample'][str(v['K_bic'])], 3)}"
            + r" \\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a14_bic.tex"), "w").write("\n".join(L) + "\n")

    # per curve
    L = [
        r"\begin{tabular}{l r c r r}",
        r"\toprule",
        r"Dataset & Curves & Curves with BIC $K = "
        + "/".join(str(k) for k in range(1, kmax + 1))
        + r"$ & Median $R^2$, $K=1$ & Median $R^2$, BIC $K$ \\",
        r"\midrule",
    ]
    for name, v in J["per_curve"].items():
        L.append(
            f"{DS_TEX[name]} & {len(v['curves'])} & {kcounts(v['K_counts'])} & {fnum(v['median_R2_K1'])} & {fnum(v['median_R2_Kbic'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a14_curves.tex"), "w").write("\n".join(L) + "\n")

    # scores
    for sub, fname in (
        ("main", "tab_a14_scores.tex"),
        ("n72", "tab_a14_scores_n72.tex"),
    ):
        L = [
            r"\begin{tabular}{l l r r r r c}",
            r"\toprule",
            r"Dataset & Scheme & \GP & \LawGP, $K=1$ & \LawGP, BIC $K$ & Law alone, BIC $K$ & Folds with $K = "
            + "/".join(str(k) for k in range(1, kmax + 1))
            + r"$ \\",
            r"\midrule",
        ]
        any_row = False
        for name, d in J["schemes"].items():
            key = (
                ("n75" if name.startswith("Grid") else "all")
                if sub == "main"
                else "n72"
            )
            if key not in d:
                continue
            first = True
            for sc, v in d[key].items():
                any_row = True
                L.append(
                    f"{DS_TEX[name] if first else ''} & {SC_TEX[sc]} & {fnum(v['gp']['R2'])} & {fnum(v['lawgp1']['R2'])} & "
                    f"{fnum(v['lawgpK']['R2'])} & {fnum(v['lawK']['R2'])} & {kcounts(v['K_counts'])} \\\\"
                )
                first = False
        L += [r"\bottomrule", r"\end{tabular}"]
        if any_row:
            open(os.path.join(TAB, fname), "w").write("\n".join(L) + "\n")

    # per level, V_rms on our grid
    d = J["schemes"].get("Grid:rms_Voc", {}).get("n75", {})
    L = [
        r"\begin{tabular}{l l r r r c}",
        r"\toprule",
        r"Held-out axis & Level & \GP & \LawGP, $K=1$ & \LawGP, BIC $K$ & Fold $K$ \\",
        r"\midrule",
    ]
    for sc in ("composition", "force", "frequency"):
        if sc not in d:
            continue
        v = d[sc]
        first = True
        for lv in v["lawgp1"]["per_level"]:
            ax_lv = lv if sc == "composition" else lv
            gpr = v["gp"].get("per_level", {}).get(lv, {}).get("R2")
            lab = A2.level_tex(sc, lv if sc == "composition" else float(lv))
            L.append(
                f"{SC_TEX[sc] if first else ''} & {lab} & {fnum(gpr)} & {fnum(v['lawgp1']['per_level'][ax_lv]['R2'])} & "
                f"{fnum(v['lawgpK']['per_level'][ax_lv]['R2'])} & {v['K_by_level'][lv]} \\\\"
            )
            first = False
    L += [r"\bottomrule", r"\end{tabular}"]
    if d:
        open(os.path.join(TAB, "tab_a14_levels.tex"), "w").write("\n".join(L) + "\n")


def write_macros(J):
    M = {}
    idx = {}

    def put(name, val, key):
        M[name] = val
        idx[name] = key

    kmax = J["meta"]["kmax"]
    put("kmdKmax", str(kmax), "meta.kmax")
    for name, v in J["full_data"].items():
        m = DS_MAC[name]
        put(f"kmd{m}K", str(v["K_bic"]), f"full_data.{name}.K_bic")
        for k in range(2, kmax + 1):
            put(
                f"kmd{m}DeltaBicK{K_WORD[k]}",
                fnum(v["delta_BIC_vs_K1"][str(k)], 1),
                f"full_data.{name}.delta_BIC_vs_K1.{k}",
            )
        for k in range(1, kmax + 1):
            put(
                f"kmd{m}InRsqK{K_WORD[k]}",
                fnum(v["R2_in_sample"][str(k)], 3),
                f"full_data.{name}.R2_in_sample.{k}",
            )
        if "n72" in v:
            put(
                f"kmd{m}KSeventyTwo",
                str(v["n72"]["K_bic"]),
                f"full_data.{name}.n72.K_bic",
            )
    for name, v in J["per_curve"].items():
        m = DS_MAC[name]
        put(f"kmd{m}Curves", str(len(v["curves"])), f"per_curve.{name}.curves")
        for k, c in v["K_counts"].items():
            put(
                f"kmd{m}CurvesK{K_WORD[int(k)]}",
                str(c),
                f"per_curve.{name}.K_counts.{k}",
            )
        put(
            f"kmd{m}CurveRsqKOne",
            fnum(v["median_R2_K1"]),
            f"per_curve.{name}.median_R2_K1",
        )
        put(
            f"kmd{m}CurveRsqKBic",
            fnum(v["median_R2_Kbic"]),
            f"per_curve.{name}.median_R2_Kbic",
        )
    for name, d in J["schemes"].items():
        m = DS_MAC[name]
        for sub, s in d.items():
            tag = "SeventyTwo" if sub == "n72" else ""
            for sc, v in s.items():
                base = f"kmd{m}{SC_MAC[sc]}{tag}"
                key = f"schemes.{name}.{sub}.{sc}"
                put(base + "GPRsq", fnum(v["gp"]["R2"]), key + ".gp.R2")
                put(base + "LawGPOneRsq", fnum(v["lawgp1"]["R2"]), key + ".lawgp1.R2")
                put(base + "LawGPKRsq", fnum(v["lawgpK"]["R2"]), key + ".lawgpK.R2")
                put(base + "LawKRsq", fnum(v["lawK"]["R2"]), key + ".lawK.R2")
                put(
                    base + "DeltaKRsq",
                    fnum(v["lawgpK_minus_lawgp1_R2"]),
                    key + ".lawgpK_minus_lawgp1_R2",
                )
                for k, c in v["K_counts"].items():
                    put(
                        base + f"FoldsK{K_WORD[int(k)]}", str(c), key + f".K_counts.{k}"
                    )
                put(base + "Folds", str(v["n_folds"]), key + ".n_folds")
    r = J["reproduction"]
    put(
        "kmdReproMaxDiff",
        sig(r["max_abs_R2_diff_lawgp1_vs_published"])
        if r["max_abs_R2_diff_lawgp1_vs_published"]
        else "0",
        "reproduction.max_abs_R2_diff_lawgp1_vs_published",
    )
    put("kmdReproChecks", str(r["n_checks"]), "reproduction.n_checks")
    put(
        "kmdNewWinsOverGP",
        str(len(J["verdict"]["new_wins_over_gp"])),
        "verdict.new_wins_over_gp",
    )
    put(
        "kmdHelpsCount",
        str(len(J["verdict"]["lawgpK_better_than_lawgp1_by_0.01"])),
        "verdict.lawgpK_better_than_lawgp1_by_0.01",
    )
    put(
        "kmdHurtsCount",
        str(len(J["verdict"]["lawgpK_worse_than_lawgp1_by_0.01"])),
        "verdict.lawgpK_worse_than_lawgp1_by_0.01",
    )
    J["index"] = idx
    L = [
        "% numbers_a14.tex: generated by code/a14_kmode.py (task E9); do not edit by hand.",
        "% Source of every value: results/a14_kmode.json (key in its 'index').",
        "% K = number of Lorentzian modes in the law's frequency factor, chosen by BIC; R2 values are pooled out-of-fold.",
    ]
    for k in sorted(M):
        L.append(rf"\newcommand{{\{k}}}{{{M[k]}}}")
    open(os.path.join(RES, "numbers_a14.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
