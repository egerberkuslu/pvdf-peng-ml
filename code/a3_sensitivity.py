"""A3 sensitivity block without the three duplicated recordings (duplicated-recordings note in the README).

Rows 52, 54 and 73 of targets_design.parquet are scaled copies of other
recordings. Everything here is refitted fresh on the remaining 72 rows (never
read from the cache):
  1. within-grid LOO R2 of ARD-GP, random forest, extra-trees and XGBoost with
     the canonical benchmark factories (regression/reg_common.py), also rerun
     on all 75 rows with the same code so the two columns differ only in data;
  2. within-grid nested-LOO raw GP, jackknife+ and sigma-scaled jackknife+
     coverage (same construction as a3_conformal.within_grid);
  3. held-out force level group-wise jackknife+ for the plain GP and PhysGP on
     V_rms (same construction as a3_conformal.groupwise).
"""

import numpy as np
from joblib import Parallel, delayed

import a3_common as C

EXCLUDED = [52, 54, 73]
KEEP = np.array([i for i in range(C.N) if i not in EXCLUDED])
BENCH_MODELS = ["ARD-GP", "RandomForest", "ExtraTrees", "XGBoost"]


def _nested_outer(tgt, j):
    y = C.yv(tgt)
    rest = KEEP[KEEP != j]
    mu_j, sd_j = C.fit_predict("gp", rest, [j], y)
    mem = np.empty((len(rest), 4))
    for r, i in enumerate(rest):
        tr = rest[rest != i]
        mu, sd = C.fit_predict("gp", tr, [i, j], y)
        mem[r] = (abs(y[i] - mu[0]), sd[0], mu[1], sd[1])
    return tgt, j, float(mu_j[0]), float(sd_j[0]), mem


def _force_level(model, v):
    tgt = "rms_Voc"
    y = C.yv(tgt)
    g = C.GROUPS["force"]
    tr = KEEP[g[KEEP] != v]
    te = KEEP[g[KEEP] == v]
    resid = np.empty(len(tr))
    preds = np.empty((len(tr), len(te)))
    for r, j in enumerate(tr):
        trj = tr[tr != j]
        mu, _ = C.fit_predict(model, trj, np.concatenate(([j], te)), y)
        resid[r] = abs(y[j] - mu[0])
        preds[r] = mu[1:]
    return model, v, te, resid, preds


def bench_loo():
    """LOO R2 on 75 and on 72 rows with the canonical factories."""
    import sys

    sys.path[:0] = [str(C.ROOT / "regression"), str(C.ROOT)]
    import reg_common
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    facs = reg_common.model_factories()
    out = {}
    for t in C.TARGETS:
        y = C.yv(t)
        out[t] = {}
        for m in BENCH_MODELS:
            rec = {}
            for tag, rows in (("n75", np.arange(C.N)), ("n72", KEEP)):
                p = cross_val_predict(
                    facs[m](), C.X[rows], y[rows], cv=LeaveOneOut(), n_jobs=-1
                )
                rec[tag] = {
                    "loo_r2": float(r2_score(y[rows], p)),
                    "loo_mae": float(mean_absolute_error(y[rows], p)),
                    "n": int(len(rows)),
                }
            out[t][m] = rec
    return out


def run():
    tasks = [delayed(_nested_outer)(t, j) for t in C.TARGETS for j in KEEP]
    tasks += [
        delayed(_force_level)(m, v)
        for m in ("gp", "physgp")
        for v in np.unique(C.GROUPS["force"])
    ]
    out = Parallel(n_jobs=-1, batch_size=1)(tasks)
    nested = {t: {} for t in C.TARGETS}
    force = []
    for o in out:
        if len(o) == 5 and isinstance(o[0], str) and o[0] in C.TARGETS:
            t, j, mu, sd, mem = o
            nested[t][j] = (mu, sd, mem)
        else:
            force.append(o)

    within = {}
    for t in C.TARGETS:
        y = C.yv(t)
        fam = {
            f: {a: {"hits": [], "w": []} for a in C.LEVELS}
            for f in ("raw_gp", "jackknife_plus", "scaled_jackknife_plus")
        }
        for j in KEEP:
            mu_j, sd_j, mem = nested[t][j]
            r_abs, sd_i, mu_at_j, sd_at_j = mem.T
            r_norm = r_abs / np.maximum(sd_i, 1e-12)
            for a, alpha in C.LEVELS.items():
                ivs = {
                    "raw_gp": (mu_j - C.Z[a] * sd_j, mu_j + C.Z[a] * sd_j),
                    "jackknife_plus": C.jk_bounds(mu_at_j, r_abs, alpha),
                    "scaled_jackknife_plus": C.jk_bounds(
                        mu_at_j, r_norm * sd_at_j, alpha
                    ),
                }
                for f, (lo, hi) in ivs.items():
                    fam[f][a]["hits"].append(bool(lo <= y[j] <= hi))
                    fam[f][a]["w"].append(hi - lo)
        within[t] = {
            f: {a: C.cov_record(fam[f][a]["hits"], fam[f][a]["w"]) for a in C.LEVELS}
            for f in fam
        }
        within[t]["mcnemar_raw_vs_jackknife_plus"] = {}
        for a in C.LEVELS:
            b, c, p = C.mcnemar_exact(
                fam["raw_gp"][a]["hits"], fam["jackknife_plus"][a]["hits"]
            )
            within[t]["mcnemar_raw_vs_jackknife_plus"][a] = {
                "jk_only_hits_b": b,
                "raw_only_hits_c": c,
                "exact_two_sided_p": p,
            }

    y = C.yv("rms_Voc")
    gforce = {}
    for m in ("gp", "physgp"):
        acc = {a: {"hits": [], "w": []} for a in C.LEVELS}
        for mm, v, te, resid, preds in force:
            if mm != m:
                continue
            for k, tt in enumerate(te):
                for a, alpha in C.LEVELS.items():
                    lo, hi = C.jk_bounds(preds[:, k], resid, alpha)
                    acc[a]["hits"].append(bool(lo <= y[tt] <= hi))
                    acc[a]["w"].append(hi - lo)
        gforce[m] = {a: C.cov_record(acc[a]["hits"], acc[a]["w"]) for a in C.LEVELS}

    return {
        "excluded_row_ids": EXCLUDED,
        "excluded_reason": "scaled copies of other recordings, duplicated-recordings note in the README",
        "n": int(len(KEEP)),
        "benchmark_loo": bench_loo(),
        "within_grid": within,
        "held_out_force_rms_jackknife_plus": gforce,
        "fits": "fresh on every run, never cached",
    }
