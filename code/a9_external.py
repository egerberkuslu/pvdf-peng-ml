#!/usr/bin/env python3
"""E4 EXTERNAL: the paper's protocol on nanogenerator data digitized from other groups' papers.

One command regenerates everything from the digitized csv files:
    python code/a9_external.py
Inputs (built by data/external_peng/digitize/digitize_*.py from open-access PDFs):
    data/external_peng/{salama2024,shi2025,li2022,park2017}.csv  (+ .provenance.json)
Outputs (all under results/):
    a9_external.json                 every statistic, with an index of macro -> json key
    tables/tab_a9_studies.tex        studies, axes, points, digitization error
    tables/tab_a9_law.tex            adapted law per dataset, parameters with bootstrap CIs
    tables/tab_a9_force.tex          force factor: exponent, intercept, model choice per curve set
    tables/tab_a9_freq.tex           frequency factor: single Lorentzian against alternatives
    tables/tab_a9_scores.tex         plain GP, LawGP and law alone, R2 per scheme, jackknife+ hits
    tables/tab_a9_restarts.tex       the same R2 with the GP optimizer restarted (a2_common gp(4))
    tables/tab_a9_sens_salama.tex    PVDF/GO impulse scores without the cells whose two figure readings differ > 25 %
    a9_provenance.csv, a9_figdata_{force,freq,oof,scores}.csv
    numbers_a9.tex                   \\ext* macros

Code path. The plain GP is a2_common.gp() (ARD Matern-5/2 + white noise, normalize_y, four
standardized inputs); LawGP is the same GP on the residuals of the law refitted inside every
training fold, the law fitted on y/std(y_train) with the tolerance of a2_common (1e-15) as in
a2_common.fit_law5. The four inputs mirror the paper's x = (c, is_pristine, F, f); an axis a study
does not vary is a constant column (standardized to zero). The law keeps the paper's form
(a0 + a1 c + a2 c^2) * F * L(f; f0, gamma) with the factors a study cannot identify removed:
force-only studies drop L, frequency-only studies drop F, a two-level composition drops a2,
and two frequency levels (Salama impulse test) replace L by one level ratio r. Because the
external frequency grids differ from the paper's, the Lorentzian bounds are set from each
study's grid (f0 within half a range outside the grid, gamma from 2 % to 5 ranges) and the
Lorentzian fit is started from five f0 values (best least-squares optimum kept).
The level ratio's high frequency fhi and the Lorentzian bounds f_bounds and g_bounds are set once
per dataset from the full frequency grid (inputs only, never the targets), so every training fold
fits the law inside the same box.
Composition coding follows the paper's x = (c, is_pristine). Salama: c = GO wt%, is_pristine = 1 at
c = 0. Shi: c = MWCNTs@BaTiO3 wt%; pristine PVDF-TrFE is c = 0, is_pristine = 1, and PB-5 (5 wt%
BaTiO3, no MWCNT) is c = 0, is_pristine = 0, exactly like PVDF/BaTiO3 in the paper's grid, so the law
cannot tell PB-5 from the pristine film and the GP separates them through is_pristine. Li: c = 1 for
TOS-BaTiO3, 0 for BaTiO3. Park: c = preload mass in kg.
Park scores in the tables and headline macros use the GP optimizer restarted from five starts
(a2_common.gp(4)); the one-start numbers stay in tab_a9_restarts.tex and in *OneStart* macros.
Jackknife+ (Barber et al. 2021, a3_common.jk_bounds) at nominal 95 % within each study's grid:
for every condition j the models trained without j and without one more condition i give the
interval; with more than 120 conditions (the traced spectra) the inner LOO is replaced by
ten-fold CV+ (as for the public sets in the paper).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small GP matrices: one BLAS thread per worker is ~15x faster

import hashlib
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_common as A2  # noqa: E402
import a3_common as A3  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 42
B_BOOT = 1000
ALPHA = 0.05
JK_LOO_MAX = 120
CV_K = 10

ROOT = A2.ROOT
EXT = os.path.join(ROOT, "data", "external_peng")
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
CACHE = os.path.join(RES, "a9_cache")
for d in (TAB, CACHE):
    os.makedirs(d, exist_ok=True)


# =============================================================================== datasets
def load_datasets():
    sal = pd.read_csv(os.path.join(EXT, "salama2024.csv"))
    shi = pd.read_csv(os.path.join(EXT, "shi2025.csv"))
    li = pd.read_csv(os.path.join(EXT, "li2022.csv"))
    park = pd.read_csv(os.path.join(EXT, "park2017.csv"))
    out = {}

    s = sal[sal.subset == "impulse"].reset_index(drop=True)
    out["SalamaForce"] = dict(
        study="salama2024",
        label="Salama 2024, impulse test",
        material="electrospun PVDF/GO nanofibers",
        c=s.filler_wt_pct.values,
        flag=(s.filler_wt_pct.values == 0).astype(float),
        F=s.force.values,
        f=s.freq_Hz.values,
        y=s.voltage.values,
        unit="V/g",
        law="cF_r",
        axes={
            "composition": s.filler_wt_pct.values,
            "force": s.force.values,
            "frequency": s.freq_Hz.values,
        },
        curve=[f"{c:g} wt% GO, {f:g} Hz" for c, f in zip(s.filler_wt_pct, s.freq_Hz)],
        axis_label={
            "composition": "GO loading",
            "force": "force",
            "frequency": "frequency",
        },
    )
    s = sal[sal.subset == "frequency"].reset_index(drop=True)
    out["SalamaFreq"] = dict(
        study="salama2024",
        label="Salama 2024, frequency test",
        material="electrospun PVDF/GO nanofibers",
        c=s.filler_wt_pct.values,
        flag=np.zeros(len(s)),
        F=s.force.values,
        f=s.freq_Hz.values,
        y=s.voltage.values,
        unit="V/g",
        law="cL",
        axes={"composition": s.filler_wt_pct.values, "frequency": s.freq_Hz.values},
        curve=[f"{c:g} wt% GO" for c in s.filler_wt_pct],
        axis_label={"composition": "GO loading", "frequency": "frequency"},
    )
    out["Shi"] = dict(
        study="shi2025",
        label="Shi 2025",
        material="PVDF-TrFE with BaTiO3 or MWCNTs@BaTiO3 films",
        c=shi.filler_wt_pct.values,
        flag=(shi.composition.values == "PVDF-TrFE").astype(float),
        F=shi.force.values,
        f=shi.freq_Hz.values,
        y=shi.voltage.values,
        unit="V",
        law="cF",
        axes={"composition": shi.composition.values, "force": shi.force.values},
        curve=list(shi.composition.values),
        axis_label={"composition": "composition", "force": "force"},
    )
    m = (li.filler.values == "TOS-BTO").astype(float)
    out["Li"] = dict(
        study="li2022",
        label="Li 2022",
        material="screen-printed PVDF with 20 wt% BaTiO3 or TOS-BaTiO3",
        c=m,
        flag=np.zeros(len(li)),
        F=li.force.values,
        f=li.freq_Hz.values,
        y=li.voltage.values,
        unit="V (pp)",
        law="mF",
        axes={"composition": li.composition.values, "force": li.force.values},
        curve=list(li.composition.values),
        axis_label={"composition": "device", "force": "pressure"},
    )
    for key, dev in (("ParkG", "G/PVDF/G"), ("ParkT", "P(VDF-TrFE)")):
        p = park[park.device == dev].reset_index(drop=True)
        f = p.freq_Hz.values
        edges = np.linspace(f.min(), f.max(), 6)
        band = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, 4)
        out[key] = dict(
            study="park2017",
            label=f"Park 2017, {dev}",
            material=f"stretched {dev} film, acoustic excitation",
            c=p.preload_mass_g.values / 1000.0,
            flag=np.zeros(len(p)),
            F=np.ones(len(p)),
            f=f,
            y=p.voltage.values,
            unit="V (pp)",
            law="cL",
            axes={"preload": p.preload_mass_g.values, "frequency": band},
            band_edges=edges.tolist(),
            curve=[f"{int(g)} g" for g in p.preload_mass_g],
            axis_label={"preload": "preload mass", "frequency": "frequency band"},
        )
    for k, d in out.items():
        d["X"] = np.column_stack([d["c"], d["flag"], d["F"], d["f"]]).astype(float)
        d["n"] = len(d["y"])
        d["key"] = k
        fr = d["f"]
        R = float(fr.max() - fr.min()) or 1.0
        d["f_bounds"] = (max(0.0, float(fr.min()) - 0.5 * R), float(fr.max()) + 0.5 * R)
        d["g_bounds"] = (0.02 * R, 5.0 * R)
    return out


# =============================================================================== law
def law_eval(kind, p, c, F, f, fhi=None):
    if kind == "mF":
        return (p[0] + p[1] * c) * F
    br = p[0] + p[1] * c + p[2] * c**2
    if kind == "cF":
        return br * F
    if kind == "mF":
        return (p[0] + p[1] * c) * F
    if kind == "cF_r":
        return br * F * np.where(f == fhi, p[3], 1.0)
    if kind == "cL":
        return br * A2.lorentz(f, p[3], p[4])
    raise ValueError(kind)


PNAMES = {
    "cF": ["a0", "a1", "a2"],
    "mF": ["a0", "a1"],
    "cF_r": ["a0", "a1", "a2", "r"],
    "cL": ["a0", "a1", "a2", "f0", "gamma"],
}


def fit_law(ds, tr):
    """Law on rows tr, fitted on y/std(y[tr]) (a2_common scaling and tolerance).
    Returns (predict(idx) in data units, full params in data units, info)."""
    kind = ds["law"]
    tr = np.asarray(tr)
    c, F, f, y = ds["c"][tr], ds["F"][tr], ds["f"][tr], ds["y"][tr]
    scale = float(np.std(y)) or 1.0
    ys = y / scale
    fhi = float(np.max(ds["f"])) if kind == "cF_r" else None
    nc = len(np.unique(c))
    names = PNAMES[kind]
    full0 = np.zeros(len(names))
    free = list(range(len(names)))
    if kind in ("cF", "cF_r", "cL"):
        if nc < 3:
            free.remove(2)
        if nc < 2:
            free.remove(1)
    if kind == "mF" and nc < 2:
        free.remove(1)
    if kind == "cF_r":
        full0[3] = 1.0
        if len(np.unique(f)) < 2:
            free.remove(3)
    lo = {
        "a0": 0.0,
        "a1": -1e6,
        "a2": -1e6,
        "r": 0.0,
        "f0": ds["f_bounds"][0],
        "gamma": ds["g_bounds"][0],
    }
    hi = {
        "a0": 1e6,
        "a1": 1e6,
        "a2": 1e6,
        "r": 100.0,
        "f0": ds["f_bounds"][1],
        "gamma": ds["g_bounds"][1],
    }

    def fn(_, *q):
        p = full0.copy()
        p[free] = q
        return law_eval(kind, p, c, F, f, fhi)

    base = F if kind != "cL" else np.ones_like(F)
    a0s = (
        float(np.median(ys / np.where(base == 0, 1, base))) if np.all(base > 0) else 1.0
    )
    starts = []
    if kind == "cL":
        span = ds["f_bounds"]
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            f0s = float(f.min() + q * (f.max() - f.min()))
            g0 = float(0.5 * (f.max() - f.min())) or 1.0
            L = A2.lorentz(f, f0s, g0)
            st = full0.copy()
            st[0] = max(float(np.median(ys / np.maximum(L, 1e-3))), 1e-3)
            st[3], st[4] = np.clip(f0s, *span), np.clip(g0, *ds["g_bounds"])
            starts.append(st)
    else:
        st = full0.copy()
        st[0] = max(a0s, 1e-6)
        starts.append(st)
    best = None
    kw = {"ftol": A2.TOL_TIGHT, "xtol": A2.TOL_TIGHT, "gtol": A2.TOL_TIGHT}
    for st in starts:
        try:
            q, _ = curve_fit(
                fn,
                None,
                ys,
                p0=st[free],
                bounds=([lo[names[i]] for i in free], [hi[names[i]] for i in free]),
                maxfev=20000,
                **kw,
            )
            sse = float(np.sum((fn(None, *q) - ys) ** 2))
            if best is None or sse < best[1]:
                best = (q, sse)
        except Exception:
            continue
    ok = best is not None
    p = full0.copy()
    if ok:
        p[free] = best[0]
    else:
        p[free] = starts[0][free]
    pv = p.copy()
    lin = {"cF": [0, 1, 2], "mF": [0, 1], "cF_r": [0, 1, 2], "cL": [0, 1, 2]}[kind]
    pv[lin] *= scale
    at_bound = []
    for i in free:
        nm = names[i]
        if nm in ("f0", "gamma", "r") and (
            np.isclose(p[i], lo[nm], rtol=1e-6, atol=1e-9)
            or np.isclose(p[i], hi[nm], rtol=1e-6, atol=1e-9)
        ):
            at_bound.append(nm)
        if nm == "a0" and p[i] <= 1e-9:
            at_bound.append("a0")

    def pred(idx):
        idx = np.asarray(idx)
        return law_eval(kind, pv, ds["c"][idx], ds["F"][idx], ds["f"][idx], fhi)

    return (
        pred,
        pv,
        {
            "ok": ok,
            "free": [names[i] for i in free],
            "at_bound": at_bound,
            "scale": scale,
        },
    )


# =============================================================================== fold predictors
def fold_predict(ds, model, tr, te):
    tr, te = np.asarray(tr), np.asarray(te)
    y = ds["y"]
    restarts = 0
    if model.endswith("_r4"):  # optimizer-restart sensitivity, a2_common.RESTARTS_CHECK starts
        model, restarts = model[:-3], A2.RESTARTS_CHECK
    if model == "gp":
        btr, bte = np.zeros(len(tr)), np.zeros(len(te))
    else:
        m, pv, info = fit_law(ds, tr)
        btr, bte = m(tr), m(te)
        if model == "law":
            return bte, np.full(len(te), np.nan)
    xs = StandardScaler().fit(ds["X"][tr])
    g = A2.gp(restarts).fit(xs.transform(ds["X"][tr]), y[tr] - btr)
    mu, sd = g.predict(xs.transform(ds["X"][te]), return_std=True)
    return bte + mu, sd


MODELS = ("gp", "lawgp", "law")
SENS_MODELS = ("gp_r4", "lawgp_r4")  # same models with the GP optimizer restarted from 4 extra starts
MODEL_TEX = {"gp": r"\GP", "lawgp": r"\LawGP", "law": "law"}
MODEL_MAC = {"gp": "GP", "lawgp": "LawGP", "law": "Law", "gp_r4": "GPRestarts", "lawgp_r4": "LawGPRestarts"}


def folds(ds, scheme):
    n = ds["n"]
    idx = np.arange(n)
    if scheme == "loo":
        return [(np.delete(idx, i), idx[[i]], i) for i in range(n)]
    g = ds["axes"][scheme]
    out = []
    for v in sorted(np.unique(g), key=lambda t: (str(type(t)), t)):
        msk = g == v
        if msk.any() and (~msk).any():
            out.append((idx[~msk], idx[msk], v))
    return out


def run_scheme(ds, scheme, model):
    fs = folds(ds, scheme)
    res = Parallel(n_jobs=-1)(
        delayed(fold_predict)(ds, model, tr, te) for tr, te, _ in fs
    )
    mu = np.full(ds["n"], np.nan)
    sd = np.full(ds["n"], np.nan)
    lev = np.empty(ds["n"], dtype=object)
    for (tr, te, v), (m, s) in zip(fs, res):
        mu[te], sd[te], lev[te] = m, s, v
    y = ds["y"]
    out = {
        "R2": float(r2_score(y, mu)),
        "MAE": float(mean_absolute_error(y, mu)),
        "n_folds": len(fs),
    }
    if scheme != "loo":
        per = {}
        for tr, te, v in fs:
            r2 = (
                float(r2_score(y[te], mu[te]))
                if len(te) > 1 and np.var(y[te]) > 0
                else None
            )
            per[str(v)] = {
                "n": int(len(te)),
                "R2": r2,
                "MAE": float(mean_absolute_error(y[te], mu[te])),
            }
        out["per_level"] = per
        vals = [p["R2"] for p in per.values() if p["R2"] is not None]
        out["mean_within_level_R2"] = float(np.mean(vals)) if vals else None
        out["n_levels"] = len(fs)
    return out, mu, sd


# =============================================================================== jackknife+
def nested_one(ds, model, j):
    n = ds["n"]
    rest = np.delete(np.arange(n), j)
    if n <= JK_LOO_MAX:
        groups = [[i] for i in rest]
        kind = "jackknife+"
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(rest)
        groups = [perm[k::CV_K] for k in range(CV_K)]
        kind = "CV+"
    preds, resid = [], []
    for g in groups:
        g = np.asarray(g)
        tr = np.setdiff1d(rest, g)
        mu, _ = fold_predict(ds, model, tr, np.concatenate([g, [j]]))
        r = np.abs(ds["y"][g] - mu[:-1])
        preds += [mu[-1]] * len(g)
        resid += list(r)
    return j, np.array(preds), np.array(resid), kind


def jackknife(ds, model):
    h = hashlib.sha1()
    h.update(ds["X"].tobytes())
    h.update(ds["y"].tobytes())
    h.update(f"{model}|{ds['law']}|{JK_LOO_MAX}|{CV_K}|v1".encode())
    path = os.path.join(CACHE, f"jk_{ds['key']}_{model}_{h.hexdigest()[:12]}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            res = pickle.load(fh)
    else:
        res = Parallel(n_jobs=-1)(
            delayed(nested_one)(ds, model, j) for j in range(ds["n"])
        )
        with open(path, "wb") as fh:
            pickle.dump(res, fh)
    hits, widths = [], []
    for j, preds, resid, kind in sorted(res, key=lambda t: t[0]):
        lo, hi = A3.jk_bounds(preds, resid, ALPHA)
        hits.append(lo <= ds["y"][j] <= hi)
        widths.append(hi - lo)
    k, n = int(np.sum(hits)), len(hits)
    return {
        "method": res[0][3],
        "nominal": 1 - ALPHA,
        "hits": k,
        "n": n,
        "coverage": k / n,
        "clopper_pearson_95": A3.clopper(k, n),
        "mean_width": float(np.mean(widths)),
        "mean_width_over_sd": float(np.mean(widths) / np.std(ds["y"])),
    }


# =============================================================================== bootstrap law CIs
def boot_one(ds, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ds["n"], ds["n"])
    _, pv, info = fit_law(ds, idx)
    return pv if info["ok"] else None


def law_full(ds):
    _, pv, info = fit_law(ds, np.arange(ds["n"]))
    seeds = np.random.SeedSequence(SEED).spawn(B_BOOT)
    draws = Parallel(n_jobs=-1)(
        delayed(boot_one)(ds, int(s.generate_state(1)[0])) for s in seeds
    )
    draws = np.array([d for d in draws if d is not None])
    names = PNAMES[ds["law"]]
    out = {
        "params": dict(zip(names, map(float, pv))),
        "free": info["free"],
        "at_bound": info["at_bound"],
        "n_boot_ok": int(len(draws)),
        "ci95": {},
    }
    for i, nm in enumerate(names):
        if nm in info["free"]:
            out["ci95"][nm] = [
                float(np.quantile(draws[:, i], 0.025)),
                float(np.quantile(draws[:, i], 0.975)),
            ]
    pred, _, _ = fit_law(ds, np.arange(ds["n"]))
    yhat = pred(np.arange(ds["n"]))
    out["R2_in_sample"] = float(r2_score(ds["y"], yhat))
    if ds["law"] == "cL":
        f0, g = pv[3], pv[4]
        out["peak_inside_grid"] = bool(ds["f"].min() <= f0 <= ds["f"].max())
        out["Q_halfpower"] = float(f0 / (0.6436 * g)) if g > 0 else None
    if ds["law"] in ("cF", "cF_r"):
        cc = np.linspace(ds["c"].min(), ds["c"].max(), 301)
        br = pv[0] + pv[1] * cc + pv[2] * cc**2
        out["bracket_argmax_c"] = float(cc[np.argmax(br)])
        out["bracket_interior_max"] = bool(
            pv[2] < 0 and ds["c"].min() < cc[np.argmax(br)] < ds["c"].max()
        )
        # a1, a2 sign stability over the bootstrap
        if "a2" in info["free"]:
            out["a2_negative_share"] = float(np.mean(draws[:, 2] < 0))
    return out, yhat


# =============================================================================== per-curve factor tests
def aicc(rss, n, k):
    rss = max(rss, 1e-300)
    a = n * np.log(rss / n) + 2 * k
    return float(a + (2 * k * (k + 1)) / (n - k - 1)) if n - k - 1 > 0 else float("nan")


def bic(rss, n, k):
    """Model choice uses BIC: with 4 to 10 points per curve the AICc correction (24 for three
    parameters at n = 5) dominates the fit term; AICc is kept in the json for reference."""
    rss = max(rss, 1e-300)
    return float(n * np.log(rss / n) + k * np.log(n)) if np.isfinite(rss) else float("nan")


def force_tests(ds):
    """Per curve (composition x frequency): through-origin line, affine line, power law."""
    rows = []
    curves = pd.Series(ds["curve"])
    for cv in curves.unique():
        m = (curves == cv).values
        F, y = ds["F"][m], ds["y"][m]
        n = len(F)
        if n < 4:
            continue
        s0 = float(np.sum(F * y) / np.sum(F * F))
        rss0 = float(np.sum((y - s0 * F) ** 2))
        X1 = np.column_stack([F, np.ones(n)])
        beta, res, *_ = np.linalg.lstsq(X1, y, rcond=None)
        rss1 = float(np.sum((y - X1 @ beta) ** 2))
        s2 = rss1 / (n - 2)
        cov = s2 * np.linalg.inv(X1.T @ X1)
        tq = stats.t.ppf(0.975, n - 2)
        b_ci = [
            float(beta[1] - tq * np.sqrt(cov[1, 1])),
            float(beta[1] + tq * np.sqrt(cov[1, 1])),
        ]
        b_p = (
            float(2 * stats.t.sf(abs(beta[1]) / np.sqrt(cov[1, 1]), n - 2))
            if cov[1, 1] > 0
            else float("nan")
        )
        try:
            (k_, p_), pc = curve_fit(
                lambda x, k, p: k * x**p,
                F,
                y,
                p0=[max(s0, 1e-9), 1.0],
                bounds=([0, 0.05], [np.inf, 5]),
                maxfev=20000,
            )
            rss2 = float(np.sum((y - k_ * F**p_) ** 2))
            sep = float(np.sqrt(pc[1, 1])) if np.isfinite(pc[1, 1]) else float("nan")
            p_ci = [float(p_ - tq * sep), float(p_ + tq * sep)]
        except Exception:
            p_, rss2, p_ci = float("nan"), float("nan"), [float("nan")] * 2
        sst = float(np.sum((y - y.mean()) ** 2))
        a = {
            "origin": aicc(rss0, n, 1),
            "affine": aicc(rss1, n, 2),
            "power": aicc(rss2, n, 2),
        }
        bb = {"origin": bic(rss0, n, 1), "affine": bic(rss1, n, 2), "power": bic(rss2, n, 2)}
        rows.append(
            {
                "curve": cv,
                "n": n,
                "slope_origin": s0,
                "R2_origin": 1 - rss0 / sst,
                "slope_affine": float(beta[0]),
                "intercept": float(beta[1]),
                "intercept_ci95": b_ci,
                "intercept_p": b_p,
                "R2_affine": 1 - rss1 / sst,
                "exponent": float(p_),
                "exponent_ci95": p_ci,
                "R2_power": 1 - rss2 / sst if np.isfinite(rss2) else None,
                "AICc": a,
                "best_AICc": min(
                    a, key=lambda k: a[k] if np.isfinite(a[k]) else np.inf
                ),
                "BIC": bb,
                "best_BIC": min(bb, key=lambda k: bb[k] if np.isfinite(bb[k]) else np.inf),
                "F_range": [float(F.min()), float(F.max())],
            }
        )
    ex = np.array([r["exponent"] for r in rows])
    cover1 = [r["exponent_ci95"][0] <= 1 <= r["exponent_ci95"][1] for r in rows]
    summ = {
        "n_curves": len(rows),
        "exponent_median": float(np.nanmedian(ex)),
        "exponent_min": float(np.nanmin(ex)),
        "exponent_max": float(np.nanmax(ex)),
        "n_exponent_ci_contains_1": int(np.sum(cover1)),
        "n_intercept_significant": int(np.sum([r["intercept_p"] < 0.05 for r in rows])),
        "n_intercept_positive_significant": int(
            np.sum([r["intercept_p"] < 0.05 and r["intercept"] > 0 for r in rows])
        ),
        "n_best_origin": int(np.sum([r["best_BIC"] == "origin" for r in rows])),
        "n_best_affine": int(np.sum([r["best_BIC"] == "affine" for r in rows])),
        "n_best_power": int(np.sum([r["best_BIC"] == "power" for r in rows])),
        "median_R2_origin": float(np.median([r["R2_origin"] for r in rows])),
    }
    return {"curves": rows, "summary": summ}


def lorentz_fit(f, y, fb, gb):
    best = None
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        f0s = float(f.min() + q * (f.max() - f.min()))
        g0 = float(np.clip(0.3 * (f.max() - f.min()), *gb))
        try:
            p, _ = curve_fit(
                lambda x, A, f0, g: A * A2.lorentz(x, f0, g),
                f,
                y,
                p0=[max(y.max(), 1e-9), f0s, g0],
                bounds=([0, fb[0], gb[0]], [np.inf, fb[1], gb[1]]),
                maxfev=20000,
            )
            rss = float(np.sum((y - p[0] * A2.lorentz(f, p[1], p[2])) ** 2))
            if best is None or rss < best[1]:
                best = (p, rss)
        except Exception:
            pass
    return best


def freq_tests(ds):
    """Per curve: single Lorentzian (3 par) against a first-order saturating high-pass (2 par) and a
    sum of two Lorentzians (6 par); local maxima; shared (f0, gamma) across curves against per-curve.
    """
    fb, gb = ds["f_bounds"], ds["g_bounds"]
    rows = []
    curves = pd.Series(ds["curve"])
    rss_shared_parts = []
    for cv in curves.unique():
        m = (curves == cv).values
        f, y = ds["f"][m], ds["y"][m]
        o = np.argsort(f)
        f, y = f[o], y[o]
        n = len(f)
        sst = float(np.sum((y - y.mean()) ** 2))
        L1 = lorentz_fit(f, y, fb, gb)
        r = {"curve": cv, "n": n, "f_range": [float(f.min()), float(f.max())]}
        if L1 is not None:
            p, rss = L1
            r.update(
                {
                    "A": float(p[0]),
                    "f0": float(p[1]),
                    "gamma": float(p[2]),
                    "R2_lorentz": 1 - rss / sst,
                    "f0_at_bound": bool(
                        np.isclose(p[1], fb[0]) or np.isclose(p[1], fb[1])
                    ),
                    "f0_inside_grid": bool(f.min() <= p[1] <= f.max()),
                    "AICc_lorentz": aicc(rss, n, 3),
                    "BIC_lorentz": bic(rss, n, 3),
                }
            )
        try:
            (A_, fc), _ = curve_fit(
                lambda x, A, fc: A * (x / fc) / np.sqrt(1 + (x / fc) ** 2),
                f,
                y,
                p0=[y.max(), np.median(f)],
                bounds=([0, 1e-3], [np.inf, 1e4]),
                maxfev=20000,
            )
            rss_hp = float(
                np.sum((y - A_ * (f / fc) / np.sqrt(1 + (f / fc) ** 2)) ** 2)
            )
            r.update(
                {
                    "R2_highpass": 1 - rss_hp / sst,
                    "fc_highpass": float(fc),
                    "AICc_highpass": aicc(rss_hp, n, 2),
                    "BIC_highpass": bic(rss_hp, n, 2),
                }
            )
        except Exception:
            pass
        if n >= 12:
            best2 = None
            grid = np.linspace(f.min(), f.max(), 7)[1:-1]
            for i in range(len(grid)):
                for k in range(i + 1, len(grid)):
                    g0 = 0.1 * (f.max() - f.min())
                    try:
                        p2, _ = curve_fit(
                            lambda x, A1, a, g1, A2_, b, g2: A1 * A2.lorentz(x, a, g1)
                            + A2_ * A2.lorentz(x, b, g2),
                            f,
                            y,
                            p0=[y.max(), grid[i], g0, y.max(), grid[k], g0],
                            bounds=(
                                [0, fb[0], gb[0], 0, fb[0], gb[0]],
                                [np.inf, fb[1], gb[1], np.inf, fb[1], gb[1]],
                            ),
                            maxfev=20000,
                        )
                        rss2 = float(
                            np.sum(
                                (
                                    y
                                    - (
                                        p2[0] * A2.lorentz(f, p2[1], p2[2])
                                        + p2[3] * A2.lorentz(f, p2[4], p2[5])
                                    )
                                )
                                ** 2
                            )
                        )
                        if best2 is None or rss2 < best2[1]:
                            best2 = (p2, rss2)
                    except Exception:
                        pass
            if best2 is not None:
                r.update(
                    {
                        "R2_two_lorentz": 1 - best2[1] / sst,
                        "AICc_two_lorentz": aicc(best2[1], n, 6),
                        "BIC_two_lorentz": bic(best2[1], n, 6),
                        "two_lorentz_f0": sorted(
                            [float(best2[0][1]), float(best2[0][4])]
                        ),
                    }
                )
            pk, _ = find_peaks(
                np.concatenate([[0.0], y, [0.0]]), prominence=0.1 * y.max()
            )
            r["n_local_maxima_prom10"] = int(len(pk))
        cands = {k[5:]: r[k] for k in r if k.startswith("AICc_") and np.isfinite(r[k])}
        r["best_AICc"] = min(cands, key=cands.get) if cands else None
        cands = {k[4:]: r[k] for k in r if k.startswith("BIC_") and np.isfinite(r[k])}
        r["best_BIC"] = min(cands, key=cands.get) if cands else None
        rows.append(r)
    # shared (f0, gamma) with per-curve amplitude against per-curve (A, f0, gamma)
    cs = curves.unique()
    cid = np.array([list(cs).index(v) for v in curves])
    f_all, y_all = ds["f"], ds["y"]

    def shared(x, *q):
        A = np.array(q[:-2])
        return A[cid] * A2.lorentz(f_all, q[-2], q[-1])

    best = None
    for q0 in (0.1, 0.3, 0.5, 0.7, 0.9):
        f0s = float(f_all.min() + q0 * (f_all.max() - f_all.min()))
        p0 = [max(y_all[cid == i].max(), 1e-9) for i in range(len(cs))] + [
            f0s,
            float(np.clip(0.3 * (f_all.max() - f_all.min()), *gb)),
        ]
        try:
            p, _ = curve_fit(
                shared,
                None,
                y_all,
                p0=p0,
                bounds=(
                    [0] * len(cs) + [fb[0], gb[0]],
                    [np.inf] * len(cs) + [fb[1], gb[1]],
                ),
                maxfev=40000,
            )
            rss = float(np.sum((y_all - shared(None, *p)) ** 2))
            if best is None or rss < best[1]:
                best = (p, rss)
        except Exception:
            pass
    rss_sep = float(
        np.sum(
            [
                (1 - r["R2_lorentz"])
                * np.sum((y_all[cid == i] - y_all[cid == i].mean()) ** 2)
                for i, r in enumerate(rows)
            ]
        )
    )
    n = len(y_all)
    shared_out = {
        "f0": float(best[0][-2]),
        "gamma": float(best[0][-1]),
        "AICc_shared": aicc(best[1], n, len(cs) + 2),
        "AICc_per_curve": aicc(rss_sep, n, 3 * len(cs)),
        "BIC_shared": bic(best[1], n, len(cs) + 2),
        "BIC_per_curve": bic(rss_sep, n, 3 * len(cs)),
        "R2_shared": 1 - best[1] / float(np.sum((y_all - y_all.mean()) ** 2)),
        "R2_per_curve": 1 - rss_sep / float(np.sum((y_all - y_all.mean()) ** 2)),
    }
    shared_out["delta_BIC_shared_minus_per_curve"] = shared_out["BIC_shared"] - shared_out["BIC_per_curve"]
    shared_out["delta_AICc_shared_minus_per_curve"] = (
        shared_out["AICc_shared"] - shared_out["AICc_per_curve"]
    )
    f0s = [r["f0"] for r in rows]
    summ = {
        "n_curves": len(rows),
        "median_R2_lorentz": float(np.median([r["R2_lorentz"] for r in rows])),
        "min_R2_lorentz": float(np.min([r["R2_lorentz"] for r in rows])),
        "f0_min": float(np.min(f0s)),
        "f0_max": float(np.max(f0s)),
        "n_f0_inside_grid": int(np.sum([r["f0_inside_grid"] for r in rows])),
        "n_best_lorentz": int(np.sum([r["best_BIC"] == "lorentz" for r in rows])),
        "n_best_highpass": int(np.sum([r["best_BIC"] == "highpass" for r in rows])),
        "n_best_two_lorentz": int(
            np.sum([r["best_BIC"] == "two_lorentz" for r in rows])
        ),
        "n_multi_peak": int(
            np.sum([r.get("n_local_maxima_prom10", 1) > 1 for r in rows])
        ),
        "shared_f0": shared_out,
    }
    return {"curves": rows, "summary": summ}


# =============================================================================== formatting
def fnum(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0:
            s = s[1:]
        else:
            return r"\ensuremath{-}" + s[1:]
    return s


def tnum(x, d=2):
    s = fnum(x, d)
    return s.replace(r"\ensuremath{-}", "$-$")


def sig(x, n=3):
    """Significant-figure formatting for parameters of very different scales."""
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    d = max(0, n - 1 - e)
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        m = x / 10**e
        return rf"\num{{{m:.{n-1}f}e{e}}}"
    return tnum(x, d)


def main():
    DS = load_datasets()
    J = {
        "meta": {
            "task": "E4",
            "script": "code/a9_external.py",
            "seed": SEED,
            "B_boot": B_BOOT,
            "alpha": ALPHA,
            "jk_loo_max_n": JK_LOO_MAX,
            "cv_plus_folds": CV_K,
            "gp": "a2_common.gp() on StandardScaler inputs (c, is_pristine, F, f)",
            "law_tolerance": A2.TOL_TIGHT,
            "date": time.strftime("%Y-%m-%d"),
        },
        "datasets": {},
        "index": {},
    }
    prov = {
        s: json.load(open(os.path.join(EXT, f"{s}.provenance.json")))
        for s in ("salama2024", "shi2025", "li2022", "park2017")
    }
    oof_rows, score_rows, force_fig, freq_fig = [], [], [], []
    ref = json.load(open(os.path.join(RES, "a2_physgp.json")))["pooled"]["rms_Voc"]
    J["paper_grid_reference_Vrms"] = {
        sc: {
            "gp": ref[sc]["gp"]["R2"],
            "lawgp": ref[sc]["physgp"]["R2"],
            "law": ref[sc]["law"]["R2"] if "law" in ref[sc] else None,
        }
        for sc in ("loo", "composition", "force", "frequency")
    }
    for key, ds in DS.items():
        print(f"[{time.time() - T0:6.1f}s] {key} n={ds['n']}", flush=True)
        R = {
            "label": ds["label"],
            "study": ds["study"],
            "material": ds["material"],
            "n": ds["n"],
            "unit": ds["unit"],
            "law_form": ds["law"],
            "axes": {
                a: sorted(
                    map(lambda v: v.item() if hasattr(v, "item") else v, np.unique(v)),
                    key=str,
                )
                for a, v in ds["axes"].items()
            },
        }
        if "band_edges" in ds:
            R["frequency_band_edges_Hz"] = ds["band_edges"]
        law, yhat = law_full(ds)
        R["law"] = law
        if "force" in ds["axes"]:
            R["force_factor"] = force_tests(ds)
        if ds["law"] == "cL":
            R["frequency_factor"] = freq_tests(ds)
        R["schemes"] = {}
        for sc in ["loo"] + list(ds["axes"]):
            R["schemes"][sc] = {}
            for mdl in MODELS + SENS_MODELS:
                out, mu, sd = run_scheme(ds, sc, mdl)
                R["schemes"][sc][mdl] = out
                score_rows.append(
                    {
                        "dataset": key,
                        "scheme": sc,
                        "model": mdl,
                        "R2": out["R2"],
                        "MAE": out["MAE"],
                        "mean_within_level_R2": out.get("mean_within_level_R2"),
                    }
                )
                for i in range(ds["n"]):
                    oof_rows.append(
                        {
                            "dataset": key,
                            "scheme": sc,
                            "model": mdl,
                            "row": i,
                            "curve": ds["curve"][i],
                            "c": ds["c"][i],
                            "F": ds["F"][i],
                            "f": ds["f"][i],
                            "y": ds["y"][i],
                            "mu": mu[i],
                            "sd": sd[i],
                        }
                    )
            R["schemes"][sc]["lawgp_r4_minus_gp_r4_R2"] = R["schemes"][sc]["lawgp_r4"]["R2"] - R["schemes"][sc]["gp_r4"]["R2"]
            R["schemes"][sc]["lawgp_minus_gp_R2"] = (
                R["schemes"][sc]["lawgp"]["R2"] - R["schemes"][sc]["gp"]["R2"]
            )
        print(f"[{time.time() - T0:6.1f}s] {key} jackknife+", flush=True)
        R["jackknife_plus"] = {m: jackknife(ds, m) for m in ("gp", "lawgp")}
        J["datasets"][key] = R
        # figure data: observed vs full-data law
        for i in range(ds["n"]):
            rec = {
                "dataset": key,
                "curve": ds["curve"][i],
                "c": ds["c"][i],
                "F": ds["F"][i],
                "f": ds["f"][i],
                "y": ds["y"][i],
                "law_full": yhat[i],
            }
            (force_fig if "force" in ds["axes"] else freq_fig).append(rec)
        if ds["law"] == "cL":
            for r in R["frequency_factor"]["curves"]:
                ff = np.linspace(r["f_range"][0], r["f_range"][1], 121)
                for x in ff:
                    freq_fig.append(
                        {
                            "dataset": key,
                            "curve": r["curve"],
                            "c": None,
                            "F": None,
                            "f": float(x),
                            "y": None,
                            "law_full": None,
                            "lorentz_per_curve": r["A"]
                            * float(A2.lorentz(x, r["f0"], r["gamma"])),
                        }
                    )

    # --------------------------------------------------------------- verdicts
    V = {}
    fc = {
        k: J["datasets"][k]["force_factor"]["summary"]
        for k in J["datasets"]
        if "force_factor" in J["datasets"][k]
    }
    qc = {
        k: J["datasets"][k]["frequency_factor"]["summary"]
        for k in J["datasets"]
        if "frequency_factor" in J["datasets"][k]
    }
    V["force_factor"] = {
        k: {
            "exponent_median": v["exponent_median"],
            "ci_contains_1": f"{v['n_exponent_ci_contains_1']}/{v['n_curves']}",
            "best_origin_or_affine": v["n_best_origin"] + v["n_best_affine"],
            "best_power": v["n_best_power"],
            "n_curves": v["n_curves"],
        }
        for k, v in fc.items()
    }
    V["frequency_factor"] = {
        k: {
            "median_R2_single_lorentz": v["median_R2_lorentz"],
            "f0_range": [v["f0_min"], v["f0_max"]],
            "n_best_lorentz": v["n_best_lorentz"],
            "n_best_highpass": v["n_best_highpass"],
            "n_best_two_lorentz": v["n_best_two_lorentz"],
            "n_multi_peak": v["n_multi_peak"],
            "shared_f0_delta_BIC": v["shared_f0"]["delta_BIC_shared_minus_per_curve"],
            "n_curves": v["n_curves"],
        }
        for k, v in qc.items()
    }
    V["lawgp_vs_gp_restarts"] = {k: {sc: J["datasets"][k]["schemes"][sc]["lawgp_r4_minus_gp_r4_R2"]
                                     for sc in J["datasets"][k]["schemes"]} for k in J["datasets"]}
    V["lawgp_vs_gp"] = {
        k: {
            sc: J["datasets"][k]["schemes"][sc]["lawgp_minus_gp_R2"]
            for sc in J["datasets"][k]["schemes"]
        }
        for k in J["datasets"]
    }
    J["verdicts"] = V

    # --------------------------------------------------------------- csv outputs
    pd.DataFrame(oof_rows).to_csv(os.path.join(RES, "a9_figdata_oof.csv"), index=False)
    pd.DataFrame(score_rows).to_csv(
        os.path.join(RES, "a9_figdata_scores.csv"), index=False
    )
    pd.DataFrame(force_fig).to_csv(
        os.path.join(RES, "a9_figdata_force.csv"), index=False
    )
    pd.DataFrame(freq_fig).to_csv(os.path.join(RES, "a9_figdata_freq.csv"), index=False)
    prow = []
    for s, p in prov.items():
        de = p["digitization_error"]
        rep = de.get("repeat_pass", {})
        prow.append(
            {
                "study": s,
                "doi": p["doi"],
                "citation": p["citation"],
                "license": p["license"].split(" (")[0],
                "figures": "; ".join(p["figures"]) if "figures" in p else p["figure"],
                "n_rows": p["n_rows"],
                "quantity": p["quantity"],
                "repeat_median_abs": rep.get(
                    "median_abs_V_per_g",
                    rep.get(
                        "median_abs_V", (rep.get("G/PVDF/G") or {}).get("median_abs_V")
                    ),
                ),
                "provenance_json": f"data/external_peng/{s}.provenance.json",
                "csv": f"data/external_peng/{s}.csv",
            }
        )
    pd.DataFrame(prow).to_csv(os.path.join(RES, "a9_provenance.csv"), index=False)
    J["provenance"] = prow
    J["digitization_error"] = {s: p["digitization_error"] for s, p in prov.items()}

    finalize(J, DS, prov)
    write_tables(J, DS, prov)
    write_macros(J, prov)
    with open(os.path.join(RES, "a9_external.json"), "w") as fh:
        json.dump(
            J,
            fh,
            indent=1,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    print(f"[{time.time() - T0:6.1f}s] done")


# =============================================================================== tables
PARK = ("ParkG", "ParkT")
DS_TEX = {
    "SalamaForce": r"Salama 2024, impulse",
    "SalamaFreq": r"Salama 2024, frequency",
    "Shi": "Shi 2025",
    "Li": "Li 2022",
    "ParkG": r"Park 2017, G/PVDF/G",
    "ParkT": r"Park 2017, P(VDF-TrFE)",
}
SC_TEX = {
    "loo": "LOO",
    "composition": "comp.",
    "force": "force",
    "frequency": "freq.",
    "preload": "preload",
}
LAW_TEX = {
    "cF": r"$(a_0+a_1c+a_2c^2)F$",
    "mF": r"$(a_0+a_1m)P$",
    "cF_r": r"$(a_0+a_1c+a_2c^2)F\,r^{[f=15]}$",
    "cL": r"$(a_0+a_1c+a_2c^2)L(f)$",
}


def write_tables(J, DS, prov):
    D = J["datasets"]
    # studies
    L = [
        r"\begin{tabular}{l l l r l}",
        r"\toprule",
        r"Dataset & Device & Axes (levels) & $n$ & Digitization check \\",
        r"\midrule",
    ]
    sde = prov["salama2024"]["digitization_error"]
    tx_imp = [t["rel_err"] for t in sde["text_check"] if t["condition"][2] in (10.0, 15.0)]
    tx_frq = [t["rel_err"] for t in sde["text_check"] if t["condition"][2] < 10.0]
    chk = {
        "SalamaForce": f"text max {tnum(100 * max(tx_imp), 1)}\\,\\%, cross-figure median "
        f"{tnum(100 * sde['cross_figure']['median_rel'], 1)}\\,\\%",
        "SalamaFreq": f"text max {tnum(100 * max(tx_frq), 1)}\\,\\%",
        "Shi": "second figure median "
        + tnum(100 * prov["shi2025"]["digitization_error"]["second_figure_PBC5"]["median_rel"], 1)
        + r"\,\%",
        "Li": "repeat median "
        + tnum(prov["li2022"]["digitization_error"]["repeat_pass"]["median_abs_V"], 3)
        + r"\,V",
        "ParkG": "repeat median "
        + tnum(
            prov["park2017"]["digitization_error"]["repeat_pass"]["G/PVDF/G"][
                "median_abs_V"
            ],
            3,
        )
        + r"\,V",
        "ParkT": "repeat median "
        + tnum(
            prov["park2017"]["digitization_error"]["repeat_pass"]["P(VDF-TrFE)"][
                "median_abs_V"
            ],
            3,
        )
        + r"\,V",
    }
    dev = {
        "SalamaForce": "electrospun PVDF/GO",
        "SalamaFreq": "electrospun PVDF/GO",
        "Shi": "PVDF-TrFE/BTO, MWCNT@BTO",
        "Li": "printed PVDF/BTO",
        "ParkG": "stretched PVDF film",
        "ParkT": "stretched P(VDF-TrFE) film",
    }
    for k, R in D.items():
        ax = ", ".join(f"{a} ({len(v)})" for a, v in R["axes"].items())
        L.append(f"{DS_TEX[k]} & {dev[k]} & {ax} & {R['n']} & {chk[k]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a9_studies.tex"), "w").write("\n".join(L) + "\n")

    # law parameters
    L = [
        r"\begin{tabular}{l l l l l l r}",
        r"\toprule",
        r"Dataset & Law & $a_0$ & $a_1$ & $a_2$ & $f_0$, $\gamma$ or $r$ & $R^2$ \\",
        r"\midrule",
    ]
    for k, R in D.items():
        lw = R["law"]
        cells = []
        for nm in ("a0", "a1", "a2"):
            if nm in lw["at_bound"]:
                cells.append("0 (bound)")
            elif nm in lw["free"]:
                lo, hi = lw["ci95"][nm]
                cells.append(f"{sig(lw['params'][nm])} [{sig(lo)}, {sig(hi)}]")
            else:
                cells.append("--")
        if "f0" in lw["free"]:
            ex = (
                f"{sig(lw['params']['f0'])} [{sig(lw['ci95']['f0'][0])}, {sig(lw['ci95']['f0'][1])}] Hz; "
                f"{sig(lw['params']['gamma'])} Hz"
            )
            if {"f0", "gamma"} & set(lw["at_bound"]):
                ex += " (bound)"
        elif "r" in lw["free"]:
            ex = f"$r$ = {sig(lw['params']['r'])} [{sig(lw['ci95']['r'][0])}, {sig(lw['ci95']['r'][1])}]"
        else:
            ex = "--"
        L.append(
            f"{DS_TEX[k]} & {LAW_TEX[R['law_form']]} & "
            + " & ".join(cells)
            + f" & {ex} & {tnum(lw['R2_in_sample'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a9_law.tex"), "w").write("\n".join(L) + "\n")

    # force factor
    L = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Dataset & Curves & Exponent $p$ median (range) & $p$ CI $\ni 1$ & Intercept $\neq 0$ & Best BIC: line / power & Median $R^2$ through origin \\",
        r"\midrule",
    ]
    for k, R in D.items():
        if "force_factor" not in R:
            continue
        s = R["force_factor"]["summary"]
        L.append(
            f"{DS_TEX[k]} & {s['n_curves']} & {tnum(s['exponent_median'])} ({tnum(s['exponent_min'])} to {tnum(s['exponent_max'])}) & "
            f"{s['n_exponent_ci_contains_1']}/{s['n_curves']} & {s['n_intercept_significant']}/{s['n_curves']} & "
            f"{s['n_best_origin'] + s['n_best_affine']} / {s['n_best_power']} & {tnum(s['median_R2_origin'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a9_force.tex"), "w").write("\n".join(L) + "\n")

    # frequency factor (Park rows: all statistics from the 5 Hz samples of the traced curves)
    L = [
        r"\begin{tabular}{l r r r r r r r r r}",
        r"\toprule",
        r"Dataset & Curves & Median $R^2$, one Lorentzian & $f_0$ range (Hz) & $f_0$ inside grid & $f_0$ at bound & "
        r"Best BIC: one peak / other & Best AICc: one peak / other & Multi-peak curves & $\Delta$BIC shared $f_0$ \\",
        r"\midrule",
    ]
    for k, R in D.items():
        if "frequency_factor" not in R:
            continue
        s = R["frequency_factor"]["summary"]
        cur = R["frequency_factor"]["curves"]
        other = s["n_best_highpass"] + s["n_best_two_lorentz"]
        a_l = sum(c["best_AICc"] == "lorentz" for c in cur)
        a_o = sum(c["best_AICc"] in ("highpass", "two_lorentz") for c in cur)
        nb = sum(bool(c["f0_at_bound"]) for c in cur)
        lab = DS_TEX[k] + (r" (5\,Hz samples)" if k in PARK else "")
        L.append(
            f"{lab} & {s['n_curves']} & {tnum(s['median_R2_lorentz'])} & {tnum(s['f0_min'], 1)} to {tnum(s['f0_max'], 1)} & "
            f"{s['n_f0_inside_grid']}/{s['n_curves']} & {nb}/{s['n_curves']} & {s['n_best_lorentz']} / {other} & {a_l} / {a_o} & "
            f"{s['n_multi_peak']}/{s['n_curves']} & {tnum(s['shared_f0']['delta_BIC_shared_minus_per_curve'], 1)} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a9_freq.tex"), "w").write("\n".join(L) + "\n")

    # scores
    L = [
        r"\begin{tabular}{l l r r r r r r r}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Pooled $R^2$} & & \multicolumn{2}{c}{Jackknife+ hits (95\,\%)} & \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){7-8}",
        r"Dataset & Scheme & \GP & \LawGP & Law & Levels & \GP & \LawGP & $n$ \\",
        r"\midrule",
    ]
    for k, R in D.items():
        first = True
        for sc, S in R["schemes"].items():
            rs = "_r4" if k in PARK else ""  # Park: five-start GP (the one-start optimum is degenerate)
            g, lg, lw = S["gp" + rs]["R2"], S["lawgp" + rs]["R2"], S["law"]["R2"]
            best = max(g, lg)
            cg = (r"\textbf{" + tnum(g) + "}") if g == best else tnum(g)
            cl = (r"\textbf{" + tnum(lg) + "}") if lg == best else tnum(lg)
            lev = S["gp"].get("n_levels", "--")
            jk = R["jackknife_plus"]
            j1 = f"{jk['gp']['hits']}/{jk['gp']['n']}" if first else ""
            j2 = f"{jk['lawgp']['hits']}/{jk['lawgp']['n']}" if first else ""
            nn = str(R["n"]) if first else ""
            L.append(
                f"{(DS_TEX[k] + (r'$^{\dagger}$' if k in PARK else '')) if first else ''} & {SC_TEX[sc]} & {cg} & {cl} & {tnum(lw)} & {lev} & {j1} & {j2} & {nn} \\\\"
            )
            first = False
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "tab_a9_scores.tex"), "w").write("\n".join(L) + "\n")

    # optimizer-restart sensitivity
    L = [r"\begin{tabular}{l l r r r r}", r"\toprule",
         r" & & \multicolumn{2}{c}{\GP} & \multicolumn{2}{c}{\LawGP} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"Dataset & Scheme & one start & five starts & one start & five starts \\", r"\midrule"]
    for k, R in D.items():
        first = True
        for sc, S in R["schemes"].items():
            L.append(f"{DS_TEX[k] if first else ''} & {SC_TEX[sc]} & {tnum(S['gp']['R2'])} & {tnum(S['gp_r4']['R2'])} & "
                     f"{tnum(S['lawgp']['R2'])} & {tnum(S['lawgp_r4']['R2'])} \\\\")
            first = False
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "tab_a9_restarts.tex"), "w").write("\n".join(L) + "\n")

    # PVDF/GO impulse test without the cells whose two figure readings differ by more than 25 %
    sd = J.get("sensitivity_salama_drop_over25")
    if sd:
        S0 = D["SalamaForce"]["schemes"]
        L = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r" & \multicolumn{3}{c}{All %d conditions} & \multicolumn{3}{c}{%d conditions (%d dropped)} \\" % (
                 D["SalamaForce"]["n"], sd["n"], sd["n_dropped"]),
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"Scheme & \GP & \LawGP & Law & \GP & \LawGP & Law \\", r"\midrule"]
        for sc, S in sd["schemes"].items():
            L.append(f"{SC_TEX[sc]} & {tnum(S0[sc]['gp']['R2'])} & {tnum(S0[sc]['lawgp']['R2'])} & {tnum(S0[sc]['law']['R2'])} & "
                     f"{tnum(S['gp']['R2'])} & {tnum(S['lawgp']['R2'])} & {tnum(S['law']['R2'])} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "tab_a9_sens_salama.tex"), "w").write("\n".join(L) + "\n")


# =============================================================================== macros
def write_macros(J, prov):
    M, idx = [], {}

    def mac(name, val, key, d=2, raw=False):
        v = val if raw else fnum(val, d)
        M.append(f"\\newcommand{{\\ext{name}}}{{{v}}}")
        idx[f"ext{name}"] = key

    D = J["datasets"]
    mac("NStudies", str(len(prov)), "provenance", raw=True)
    mac("NDatasets", str(len(D)), "datasets", raw=True)
    mac("NPoints", str(sum(R["n"] for R in D.values())), "datasets.*.n", raw=True)
    SCM = {
        "loo": "Loo",
        "composition": "Comp",
        "force": "Force",
        "frequency": "Freq",
        "preload": "Preload",
    }
    for k, R in D.items():
        mac(f"{k}N", str(R["n"]), f"datasets.{k}.n", raw=True)
        for sc, S in R["schemes"].items():
            for m in MODELS + SENS_MODELS:
                if k in PARK and m in ("gp", "lawgp"):
                    # Park: the headline GP numbers are the five-start ones; one start kept as OneStart
                    mac(f"{k}{SCM[sc]}{MODEL_MAC[m]}Rsq", S[m + "_r4"]["R2"], f"datasets.{k}.schemes.{sc}.{m}_r4.R2")
                    mac(f"{k}{SCM[sc]}{MODEL_MAC[m]}OneStartRsq", S[m]["R2"], f"datasets.{k}.schemes.{sc}.{m}.R2")
                    continue
                mac(
                    f"{k}{SCM[sc]}{MODEL_MAC[m]}Rsq",
                    S[m]["R2"],
                    f"datasets.{k}.schemes.{sc}.{m}.R2",
                )
            dk_ = "lawgp_r4_minus_gp_r4_R2" if k in PARK else "lawgp_minus_gp_R2"
            mac(
                f"{k}{SCM[sc]}DeltaRsq",
                S[dk_],
                f"datasets.{k}.schemes.{sc}.{dk_}",
            )
            if sc != "loo":
                mac(
                    f"{k}{SCM[sc]}Levels",
                    str(S["gp"]["n_levels"]),
                    f"datasets.{k}.schemes.{sc}.gp.n_levels",
                    raw=True,
                )
        for m in ("gp", "lawgp"):
            jk = R["jackknife_plus"][m]
            mac(
                f"{k}Jk{MODEL_MAC[m]}Hits",
                f"{jk['hits']}/{jk['n']}",
                f"datasets.{k}.jackknife_plus.{m}.hits",
                raw=True,
            )
            mac(
                f"{k}Jk{MODEL_MAC[m]}Cov",
                100 * jk["coverage"],
                f"datasets.{k}.jackknife_plus.{m}.coverage",
                d=1,
            )
            mac(
                f"{k}Jk{MODEL_MAC[m]}WidthSd",
                jk["mean_width_over_sd"],
                f"datasets.{k}.jackknife_plus.{m}.mean_width_over_sd",
            )
        mac(
            f"{k}JkMethod",
            R["jackknife_plus"]["gp"]["method"].replace("+", "\\texttt{+}"),
            f"datasets.{k}.jackknife_plus.gp.method",
            raw=True,
        )
        lw = R["law"]
        mac(f"{k}LawRsq", lw["R2_in_sample"], f"datasets.{k}.law.R2_in_sample")
        for nm in lw["free"]:
            nmm = {
                "a0": "Aa",
                "a1": "Ab",
                "a2": "Ac",
                "f0": "Fo",
                "gamma": "Gamma",
                "r": "Ratio",
            }[nm]
            mac(
                f"{k}Law{nmm}",
                lw["params"][nm],
                f"datasets.{k}.law.params.{nm}",
                d=3 if nm in ("f0", "gamma", "r") else 4,
            )
            mac(
                f"{k}Law{nmm}Lo",
                lw["ci95"][nm][0],
                f"datasets.{k}.law.ci95.{nm}",
                d=3 if nm in ("f0", "gamma", "r") else 4,
            )
            mac(
                f"{k}Law{nmm}Hi",
                lw["ci95"][nm][1],
                f"datasets.{k}.law.ci95.{nm}",
                d=3 if nm in ("f0", "gamma", "r") else 4,
            )
        if "bracket_argmax_c" in lw:
            mac(
                f"{k}LawBracketArgmax",
                lw["bracket_argmax_c"],
                f"datasets.{k}.law.bracket_argmax_c",
            )
        if "force_factor" in R:
            s = R["force_factor"]["summary"]
            mac(
                f"{k}ForceCurves",
                str(s["n_curves"]),
                f"datasets.{k}.force_factor.summary.n_curves",
                raw=True,
            )
            mac(
                f"{k}ExpMedian",
                s["exponent_median"],
                f"datasets.{k}.force_factor.summary.exponent_median",
            )
            mac(
                f"{k}ExpMin",
                s["exponent_min"],
                f"datasets.{k}.force_factor.summary.exponent_min",
            )
            mac(
                f"{k}ExpMax",
                s["exponent_max"],
                f"datasets.{k}.force_factor.summary.exponent_max",
            )
            mac(
                f"{k}ExpCiOne",
                str(s["n_exponent_ci_contains_1"]),
                f"datasets.{k}.force_factor.summary.n_exponent_ci_contains_1",
                raw=True,
            )
            mac(
                f"{k}InterceptSig",
                str(s["n_intercept_significant"]),
                f"datasets.{k}.force_factor.summary.n_intercept_significant",
                raw=True,
            )
            mac(
                f"{k}BestPower",
                str(s["n_best_power"]),
                f"datasets.{k}.force_factor.summary.n_best_power",
                raw=True,
            )
            mac(
                f"{k}BestLine",
                str(s["n_best_origin"] + s["n_best_affine"]),
                f"datasets.{k}.force_factor.summary.n_best_origin",
                raw=True,
            )
            mac(
                f"{k}OriginRsq",
                s["median_R2_origin"],
                f"datasets.{k}.force_factor.summary.median_R2_origin",
            )
        if "frequency_factor" in R:
            s = R["frequency_factor"]["summary"]
            mac(
                f"{k}FreqCurves",
                str(s["n_curves"]),
                f"datasets.{k}.frequency_factor.summary.n_curves",
                raw=True,
            )
            mac(
                f"{k}LorentzRsq",
                s["median_R2_lorentz"],
                f"datasets.{k}.frequency_factor.summary.median_R2_lorentz",
            )
            mac(
                f"{k}LorentzRsqMin",
                s["min_R2_lorentz"],
                f"datasets.{k}.frequency_factor.summary.min_R2_lorentz",
            )
            mac(
                f"{k}FoMin",
                s["f0_min"],
                f"datasets.{k}.frequency_factor.summary.f0_min",
                d=1,
            )
            mac(f"{k}FoInGrid", str(s["n_f0_inside_grid"]), f"datasets.{k}.frequency_factor.summary.n_f0_inside_grid", raw=True)
            mac(
                f"{k}FoMax",
                s["f0_max"],
                f"datasets.{k}.frequency_factor.summary.f0_max",
                d=1,
            )
            mac(
                f"{k}BestLorentz",
                str(s["n_best_lorentz"]),
                f"datasets.{k}.frequency_factor.summary.n_best_lorentz",
                raw=True,
            )
            mac(
                f"{k}BestHighpass",
                str(s["n_best_highpass"]),
                f"datasets.{k}.frequency_factor.summary.n_best_highpass",
                raw=True,
            )
            mac(
                f"{k}BestTwoLorentz",
                str(s["n_best_two_lorentz"]),
                f"datasets.{k}.frequency_factor.summary.n_best_two_lorentz",
                raw=True,
            )
            mac(
                f"{k}MultiPeak",
                str(s["n_multi_peak"]),
                f"datasets.{k}.frequency_factor.summary.n_multi_peak",
                raw=True,
            )
            mac(
                f"{k}SharedFoDeltaBIC",
                s["shared_f0"]["delta_BIC_shared_minus_per_curve"],
                f"datasets.{k}.frequency_factor.summary.shared_f0.delta_BIC_shared_minus_per_curve",
                d=1,
            )
    # digitization checks
    sd = prov["salama2024"]["digitization_error"]
    mac(
        "SalamaTextMaxRelErr",
        100 * max(t["rel_err"] for t in sd["text_check"]),
        "digitization_error.salama2024.text_check",
        d=1,
    )
    mac(
        "SalamaTextN",
        str(len(sd["text_check"])),
        "digitization_error.salama2024.text_check",
        raw=True,
    )
    mac(
        "SalamaCrossMedRelErr",
        100 * sd["cross_figure"]["median_rel"],
        "digitization_error.salama2024.cross_figure.median_rel",
        d=1,
    )
    mac(
        "SalamaCrossN",
        str(sd["cross_figure"]["n"]),
        "digitization_error.salama2024.cross_figure.n",
        raw=True,
    )
    mac(
        "SalamaCrossWithinFive",
        str(sd["cross_figure"]["n_rel_below_5pct"]),
        "digitization_error.salama2024.cross_figure.n_rel_below_5pct",
        raw=True,
    )
    sh = prov["shi2025"]["digitization_error"]
    mac(
        "ShiSecondFigMedRelErr",
        100 * sh["second_figure_PBC5"]["median_rel"],
        "digitization_error.shi2025.second_figure_PBC5.median_rel",
        d=1,
    )
    mac(
        "ShiSlopeDigitized",
        sh["slope_check"]["digitized_mV_per_N"],
        "digitization_error.shi2025.slope_check",
        d=1,
    )
    mac(
        "ShiSlopeStated",
        sh["slope_check"]["stated_mV_per_N"],
        "digitization_error.shi2025.slope_check",
        d=1,
    )
    lf = prov["li2022"]["digitization_error"]["fit_check"]["20% TOS-BTO/PVDF"]
    mac(
        "LiSlopeDigitized",
        1000 * lf["digitized_slope_V_per_kPa"],
        "digitization_error.li2022.fit_check",
        d=1,
    )
    mac(
        "LiSlopeStated",
        1000 * lf["stated_slope_V_per_kPa"],
        "digitization_error.li2022.fit_check",
        d=1,
    )
    pe = prov["park2017"]["digitization_error"]["repeat_pass"]
    mac(
        "ParkRepeatMedV",
        pe["G/PVDF/G"]["median_abs_V"],
        "digitization_error.park2017.repeat_pass",
        d=3,
    )
    ref = J["paper_grid_reference_Vrms"]
    for sc, v in ref.items():
        mac(f"Paper{SCM[sc]}GPRsq", v["gp"], f"paper_grid_reference_Vrms.{sc}.gp")
        mac(
            f"Paper{SCM[sc]}LawGPRsq",
            v["lawgp"],
            f"paper_grid_reference_Vrms.{sc}.lawgp",
        )
    # composition-axis gain of LawGP over the plain GP (five-start GP everywhere)
    for k, v in J.get("composition_gain_r4", {}).items():
        mac(f"CompGain{k}", v["gain"], f"composition_gain_r4.{k}.gain")
    V = J.get("verdicts", {})
    fq = D["SalamaFreq"]["frequency_factor"]["curves"]
    mac("SalamaFreqBestBICLorentz", str(sum(c["best_BIC"] == "lorentz" for c in fq)),
        "datasets.SalamaFreq.frequency_factor.curves.best_BIC", raw=True)
    mac("SalamaFreqBestAICcHighpass", str(sum(c["best_AICc"] == "highpass" for c in fq)),
        "datasets.SalamaFreq.frequency_factor.curves.best_AICc", raw=True)
    mac("SalamaFreqFoAtBound", str(sum(bool(c["f0_at_bound"]) for c in fq)),
        "datasets.SalamaFreq.frequency_factor.curves.f0_at_bound", raw=True)
    fb = [c["f0"] for c in fq if c["f0_at_bound"]]
    if fb:
        mac("SalamaFreqFoBoundHz", max(fb), "datasets.SalamaFreq.frequency_factor.curves.f0 (at bound)", d=0)
    mac("SalamaFreqResonanceNote", "five frequencies from 1 to 5\\,Hz cannot separate a resonance peak from a "
        "high-pass rise", "verdicts.salama_frequency_axis", raw=True)
    pu = prov["park2017"]["digitization_error"]["peak_undersampling"]
    mac("ParkPeakShortfallMedian", 100 * pu["median_sampled_shortfall_rel"],
        "digitization_error.park2017.peak_undersampling.median_sampled_shortfall_rel", d=0)
    mac("ParkPeakShortfallMax", 100 * pu["max_sampled_shortfall_rel"],
        "digitization_error.park2017.peak_undersampling.max_sampled_shortfall_rel", d=0)
    mac("ParkTracedShortfallMax", 100 * max(abs(u["traced_shortfall_rel"]) for u in pu["per_curve"]),
        "digitization_error.park2017.peak_undersampling.per_curve.traced_shortfall_rel", d=0)
    mac("ParkSamplingNote", "from 5\\,Hz samples of the traced curves, which under-record narrow resonance peaks",
        "verdicts.park_sampling", raw=True)
    mac("ParkRestartNote", "optimizer restarted from five starts; with one start the plain \\GP falls into a "
        "near-interpolating optimum on the G/PVDF/G spectra", "verdicts.park_restarts", raw=True)
    sd = J.get("sensitivity_salama_drop_over25")
    if sd:
        mac("SalamaDropN", str(sd["n"]), "sensitivity_salama_drop_over25.n", raw=True)
        mac("SalamaDropNDropped", str(sd["n_dropped"]), "sensitivity_salama_drop_over25.n_dropped", raw=True)
        for sc, S in sd["schemes"].items():
            for m in MODELS + SENS_MODELS:
                mac(f"SalamaDrop{SCM[sc]}{MODEL_MAC[m]}Rsq", S[m]["R2"], f"sensitivity_salama_drop_over25.schemes.{sc}.{m}.R2")
        mac("SalamaDropExpMedian", sd["force_factor"]["exponent_median"],
            "sensitivity_salama_drop_over25.force_factor.exponent_median")
        mac("SalamaDropExpCiOne", str(sd["force_factor"]["n_exponent_ci_contains_1"]),
            "sensitivity_salama_drop_over25.force_factor.n_exponent_ci_contains_1", raw=True)
        mac("SalamaDropLawBracketArgmax", sd["law"]["bracket_argmax_c"], "sensitivity_salama_drop_over25.law.bracket_argmax_c")
    M.sort()
    head = [
        "% numbers_a9.tex: generated by code/a9_external.py (task E4); do not edit by hand.",
        "% Source of every value: results/a9_external.json (key in its 'index').",
        "% Units follow each study: Salama V/g, Shi V, Li V peak-to-peak (pressure kPa), Park V peak-to-peak.",
    ]
    open(os.path.join(RES, "numbers_a9.tex"), "w").write("\n".join(head + M) + "\n")
    J["index"] = idx


# =============================================================================== finalize (summaries, sensitivity)
def subset_ds(ds, keep):
    keep = np.asarray(keep)
    d = dict(ds)
    for f in ("c", "flag", "F", "f", "y", "X"):
        d[f] = ds[f][keep]
    d["axes"] = {a: np.asarray(v)[keep] for a, v in ds["axes"].items()}
    d["curve"] = [ds["curve"][i] for i in keep]
    d["n"] = int(len(keep))
    return d  # f_bounds, g_bounds kept from the full grid


def salama_drop_sensitivity(DS, prov):
    """PVDF/GO impulse test with every cell whose Fig. 10a/b and Fig. 10e/f readings differ by more than
    25 % removed; same schemes and models as the main run (no jackknife+)."""
    ds = DS["SalamaForce"]
    bad = [tuple(q["condition"]) for q in prov["salama2024"]["digitization_error"]["cross_figure_over_25pct"]["pairs"]]
    drop = [i for i in range(ds["n"]) if any(np.isclose(ds["c"][i], b[0]) and np.isclose(ds["F"][i], b[1])
                                             and np.isclose(ds["f"][i], b[2]) for b in bad)]
    keep = [i for i in range(ds["n"]) if i not in drop]
    d2 = subset_ds(ds, keep)
    d2["key"] = "SalamaForceDrop"
    out = {"dropped_conditions": [list(b) for b in bad], "n_dropped": len(drop), "n": d2["n"], "schemes": {}}
    for sc in ["loo"] + list(d2["axes"]):
        out["schemes"][sc] = {}
        for mdl in MODELS + SENS_MODELS:
            r, _, _ = run_scheme(d2, sc, mdl)
            out["schemes"][sc][mdl] = {"R2": r["R2"], "MAE": r["MAE"]}
    law, _ = law_full(d2)
    out["law"] = {"params": law["params"], "ci95": law["ci95"], "R2_in_sample": law["R2_in_sample"],
                  "bracket_argmax_c": law.get("bracket_argmax_c")}
    out["force_factor"] = force_tests(d2)["summary"]
    return out


def finalize(J, DS, prov):
    D = J["datasets"]
    J["summary"] = {"n_studies": len(prov), "n_datasets": len(D), "n_points": int(sum(R["n"] for R in D.values()))}
    # held-out composition (preload for Park), five-start GP for every dataset
    gain = {}
    for k, R in D.items():
        sc = "preload" if "preload" in R["schemes"] else "composition"
        S = R["schemes"][sc]
        gain[k] = {"scheme": sc, "gp_r4": S["gp_r4"]["R2"], "lawgp_r4": S["lawgp_r4"]["R2"],
                   "gain": S["lawgp_r4"]["R2"] - S["gp_r4"]["R2"], "n_levels": S["gp"]["n_levels"]}
    J["composition_gain_r4"] = gain
    g = {k: (0.0 if abs(v["gain"]) < 0.005 else round(v["gain"], 2)) for k, v in gain.items()}
    J.setdefault("verdicts", {})["composition_axis"] = (
        f"LawGP helps on the held-out composition only for the electrospun PVDF/GO data "
        f"(+{g['SalamaForce']:.2f} impulse, +{g['SalamaFreq']:.2f} frequency, five-start GP); the gain is about zero "
        f"on PVDF-TrFE ({g['Shi']:+.2f}) and printed PVDF ({g['Li']:+.2f}, two devices) and strongly negative on the "
        f"held-out preload of the stretched films ({g['ParkG']:+.2f} G/PVDF/G, {g['ParkT']:+.2f} P(VDF-TrFE)).")
    fq = D["SalamaFreq"]["frequency_factor"]["curves"]
    J["verdicts"]["salama_frequency_axis"] = (
        f"On 1 to 5 Hz the six curves rise and level off: BIC prefers one Lorentzian on "
        f"{sum(c['best_BIC'] == 'lorentz' for c in fq)}/6, AICc the first-order high-pass rise on "
        f"{sum(c['best_AICc'] == 'highpass' for c in fq)}/6, and f0 sits at its upper bound on "
        f"{sum(bool(c['f0_at_bound']) for c in fq)}/6; five frequencies from 1 to 5 Hz cannot separate a "
        f"resonance from a high-pass rise.")
    pu = prov["park2017"]["digitization_error"]["peak_undersampling"]
    J["verdicts"]["park_sampling"] = (
        f"Park statistics come from 5 Hz samples of the traced curves; the sampled maxima fall short of the printed "
        f"resonance peaks by a median {100 * pu['median_sampled_shortfall_rel']:.0f} % and at most "
        f"{100 * pu['max_sampled_shortfall_rel']:.0f} %.")
    J["verdicts"]["park_restarts"] = (
        "Park GP and LawGP scores are reported with the optimizer restarted from five starts (a2_common.gp(4)); "
        "with one start the plain GP falls into a near-interpolating optimum on the G/PVDF/G spectra "
        f"(LOO R2 {D['ParkG']['schemes']['loo']['gp']['R2']:.2f} against {D['ParkG']['schemes']['loo']['gp_r4']['R2']:.2f}). "
        "The Park jackknife+ intervals use the one-start models.")
    if "sensitivity_salama_drop_over25" not in J and DS is not None:
        print("[finalize] Salama drop sensitivity", flush=True)
        J["sensitivity_salama_drop_over25"] = salama_drop_sensitivity(DS, prov)


def tables_from_json():
    """Rebuild tables and macros from a9_external.json; only the cheap finalize step is (re)computed
    when its keys are missing (Salama drop sensitivity), everything else is read back."""
    path = os.path.join(RES, "a9_external.json")
    J = json.load(open(path))
    prov = {s: json.load(open(os.path.join(EXT, f"{s}.provenance.json"))) for s in
            ("salama2024", "shi2025", "li2022", "park2017")}
    finalize(J, load_datasets(), prov)
    write_tables(J, None, prov)
    write_macros(J, prov)
    with open(path, "w") as fh:
        json.dump(J, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        tables_from_json()
    else:
        main()
