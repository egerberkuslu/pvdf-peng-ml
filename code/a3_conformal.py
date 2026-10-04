"""A3 coverage computations.

within_grid(): nested LOO for the plain GP on V_rms, V_pp, |V|_max. For each
held-out condition j the model trained on the other 74 gives the raw
+-z sigma interval, and 74 inner models (each also missing one training point
i) give the jackknife+ and sigma-scaled jackknife+ intervals. This is the
construction of legacy calibrated_conformal.py, repeated here because the
McNemar comparison needs per-point hits for every target (legacy stored them
for V_rms only).

groupwise(): legacy groupwise_conformal.py construction extended to all three
targets and to PhysGP. For a held-out level v of an axis, the training set is
every condition not at v. Jackknife+ residuals come from leave-one-out models
inside that training set (for PhysGP the law is refitted inside every one of
those inner training sets), and the same inner models predict the held-out
level. The outer model trained on the whole training set gives the raw
+-z sigma interval at the held-out level.
"""

import hashlib
import inspect

import numpy as np
from joblib import Parallel, delayed

import a3_common as C


def _nested_outer(tgt, j):
    y = C.yv(tgt)
    rest = np.delete(np.arange(C.N), j)
    mu_j, sd_j = C.fit_predict("gp", rest, [j], y)
    mem = np.empty((len(rest), 4))
    for r, i in enumerate(rest):
        tr = rest[rest != i]
        mu, sd = C.fit_predict("gp", tr, [i, j], y)
        mem[r] = (abs(y[i] - mu[0]), sd[0], mu[1], sd[1])
    return tgt, j, float(mu_j[0]), float(sd_j[0]), mem


def _group_level(tgt, model, axis, v):
    y = C.yv(tgt)
    g = C.GROUPS[axis]
    tr = np.where(g != v)[0]
    te = np.where(g == v)[0]
    mu_o, sd_o = C.fit_predict(model, tr, te, y)
    resid = np.empty(len(tr))
    sd_i = np.empty(len(tr))
    preds = np.empty((len(tr), len(te)))
    sds = np.empty((len(tr), len(te)))
    for r, j in enumerate(tr):
        trj = tr[tr != j]
        mu, sd = C.fit_predict(model, trj, np.concatenate(([j], te)), y)
        resid[r] = abs(y[j] - mu[0])
        sd_i[r] = sd[0]
        preds[r] = mu[1:]
        sds[r] = sd[1:]
    return dict(
        tgt=tgt,
        model=model,
        axis=axis,
        level=v,
        tr=tr,
        te=te,
        mu_outer=mu_o,
        sd_outer=sd_o,
        resid=resid,
        sd_i=sd_i,
        preds=preds,
        sds=sds,
    )


def _code_hash():
    """Hash of the code that determines the fits (cache key)."""
    parts = [C.gp, C.law, C.fit_law, C.fit_predict, _nested_outer, _group_level]
    src = "".join(inspect.getsource(f) for f in parts) + repr((C.P0, C.BOUNDS, C.FEAT))
    src += hashlib.sha256(C.DATA.read_bytes()).hexdigest()
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def compute_raw(use_cache=False):
    """Run (or load) all model fits. Returns (nested, group) raw arrays."""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    cfile = C.CACHE / f"a3_fits_{_code_hash()}.npz"
    if use_cache and cfile.exists():
        z = np.load(cfile, allow_pickle=True)
        return z["nested"].item(), list(z["group"])
    tasks = [delayed(_nested_outer)(t, j) for t in C.TARGETS for j in range(C.N)]
    tasks += [
        delayed(_group_level)(t, m, a, v)
        for t in C.TARGETS
        for m in ("gp", "physgp")
        for a in C.AXES
        for v in np.unique(C.GROUPS[a])
    ]
    out = Parallel(n_jobs=-1, batch_size=1, verbose=0)(tasks)
    nested = {t: {} for t in C.TARGETS}
    group = []
    for o in out:
        if isinstance(o, tuple):
            t, j, mu, sd, mem = o
            nested[t][j] = (mu, sd, mem)
        else:
            group.append(o)
    np.savez_compressed(
        cfile,
        nested=np.array(nested, dtype=object),
        group=np.array(group, dtype=object),
    )
    return nested, group


def within_grid(nested):
    """Coverage, width and per-point intervals within the grid (nested LOO)."""
    res, pts = {}, []
    for t in C.TARGETS:
        y = C.yv(t)
        fam = {
            f: {a: {"hits": [], "w": []} for a in C.LEVELS}
            for f in ("raw_gp", "jackknife_plus", "scaled_jackknife_plus")
        }
        for j in range(C.N):
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
                    hit = bool(lo <= y[j] <= hi)
                    fam[f][a]["hits"].append(hit)
                    fam[f][a]["w"].append(hi - lo)
                    pts.append(
                        dict(
                            condition_id=j,
                            target=t,
                            model="gp",
                            split_scheme="within_grid_nested_loo",
                            held_out_level="",
                            interval=f,
                            nominal=int(a),
                            y_true=y[j],
                            y_pred=mu_j if f == "raw_gp" else float(np.mean(mu_at_j)),
                            lower=lo,
                            upper=hi,
                            hit=hit,
                        )
                    )
        res[t] = {
            f: {a: C.cov_record(fam[f][a]["hits"], fam[f][a]["w"]) for a in C.LEVELS}
            for f in fam
        }
        # McNemar raw vs jackknife+ on the same 75 conditions
        res[t]["mcnemar_raw_vs_jackknife_plus"] = {}
        for a in C.LEVELS:
            b, c, p = C.mcnemar_exact(
                fam["raw_gp"][a]["hits"], fam["jackknife_plus"][a]["hits"]
            )
            res[t]["mcnemar_raw_vs_jackknife_plus"][a] = {
                "jk_only_hits_b": b,
                "raw_only_hits_c": c,
                "discordant": b + c,
                "exact_two_sided_p": p,
                "test": "exact McNemar (binomial on discordant pairs, two-sided)",
            }
        res[t]["hits_vectors"] = {
            f: {a: [bool(h) for h in fam[f][a]["hits"]] for a in C.LEVELS} for f in fam
        }
    return res, pts


def groupwise(group):
    """Held-out-level coverage for gp/physgp x axes x targets."""
    res, pts = {}, []
    for t in C.TARGETS:
        res[t] = {}
        y = C.yv(t)
        for m in ("gp", "physgp"):
            res[t][m] = {}
            for ax in C.AXES:
                blocks = [
                    b
                    for b in group
                    if b["tgt"] == t and b["model"] == m and b["axis"] == ax
                ]
                acc = {
                    f: {a: {"hits": [], "w": []} for a in C.LEVELS}
                    for f in ("raw", "jackknife_plus", "scaled_jackknife_plus")
                }
                per_level = {}
                for b in sorted(blocks, key=lambda b: str(b["level"])):
                    lvl_acc = {a: {"hits": [], "w": []} for a in C.LEVELS}
                    r_norm = b["resid"] / np.maximum(b["sd_i"], 1e-12)
                    for k, tt in enumerate(b["te"]):
                        for a, alpha in C.LEVELS.items():
                            mu_o, sd_o = b["mu_outer"][k], b["sd_outer"][k]
                            ivs = {
                                "raw": (mu_o - C.Z[a] * sd_o, mu_o + C.Z[a] * sd_o),
                                "jackknife_plus": C.jk_bounds(
                                    b["preds"][:, k], b["resid"], alpha
                                ),
                                "scaled_jackknife_plus": C.jk_bounds(
                                    b["preds"][:, k], r_norm * b["sds"][:, k], alpha
                                ),
                            }
                            for f, (lo, hi) in ivs.items():
                                hit = bool(lo <= y[tt] <= hi)
                                acc[f][a]["hits"].append(hit)
                                acc[f][a]["w"].append(hi - lo)
                                if f == "jackknife_plus":
                                    lvl_acc[a]["hits"].append(hit)
                                    lvl_acc[a]["w"].append(hi - lo)
                                pts.append(
                                    dict(
                                        condition_id=int(tt),
                                        target=t,
                                        model=m,
                                        split_scheme=f"held_out_{ax}",
                                        held_out_level=str(b["level"]),
                                        interval=f,
                                        nominal=int(a),
                                        y_true=y[tt],
                                        y_pred=mu_o
                                        if f == "raw"
                                        else float(np.mean(b["preds"][:, k])),
                                        lower=lo,
                                        upper=hi,
                                        hit=hit,
                                    )
                                )
                    per_level[str(b["level"])] = {
                        a: C.cov_record(lvl_acc[a]["hits"], lvl_acc[a]["w"])
                        for a in C.LEVELS
                    }
                res[t][m][ax] = {
                    f: {
                        a: C.cov_record(acc[f][a]["hits"], acc[f][a]["w"])
                        for a in C.LEVELS
                    }
                    for f in acc
                }
                res[t][m][ax]["jackknife_plus_per_level"] = per_level
    return res, pts
