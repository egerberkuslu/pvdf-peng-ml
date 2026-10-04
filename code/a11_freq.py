#!/usr/bin/env python3
r"""E6 FREQ (analysis protocol section 5, prefix a11, macros frq): frequency-axis rescue.

One command from the repository root:
    python code/a11_freq.py

Held-out frequency on V_rms (leave-one-nominal-level-out, the paper's folds; the
folds are always formed on the nominal label) for the plain GP, LawGP, ProductGP
and the Lorentzian-spectrum ProductGP of a10, plus transfer and twin-informed
variants. Four parts:

(i)  Measured tap rate as the frequency input. a1_recordings.csv tap_rate_used_Hz
     (period from the waveform autocorrelation, else the nominal period; timing
     only, never the amplitude target) replaces the nominal label in every input
     that carries frequency (column 4 of the GP inputs and f in the law). Nothing
     else changes: same models, kernels, seeds, folds. Variants
       nominal         the label (reproduces a10 exactly; checked)
       measured        tap_rate_used_Hz for every recording
       measured_agree  tap_rate_used_Hz where it agrees with the independent
                       spectral estimate spectral_rate_Hz within 10 percent,
                       otherwise the nominal label (sensitivity for unreliable rates)
(ii) Interior levels (10, 15, 20 Hz: interpolation in f) and edge levels (5, 25 Hz:
     extrapolation) are reported separately: per-level within-level R2 and MAE,
     band mean of the within-level R2, and band-pooled R2 over the held-out
     predictions of the band's rows.
(iii) Transfer. The frequency shape s(f) is fitted once on the external Salama et
     al. frequency sweep (data/external_peng/salama2024.csv, subset
     'frequency': six PVDF/GO films, 1 to 5 Hz, 0.3 N) with one amplitude per film
     and a shared shape, in log space: a Lorentzian L(f; f0, gamma) and a power law
     f^b. On our grid the prior mean is (a0 + a1 c + a2 c^2) F s(f), with only the
     amplitude (a0, a1, a2) refitted per fold by linear least squares, then the plain
     GP of A2/A5 on the residual ('tr_*_gp'), or the mean alone ('tr_*_mean').
     The twin's frequency shape is the A2 law fitted to the full real grid, i.e. it
     has seen the held-out level; it is run as 'oracle_shape' (f0, gamma from all
     rows of the row set, amplitude refitted per fold) and is a leaky diagnostic
     upper bound, never a result. Fitted in-fold, the twin shape is LawGP itself.
(iv) Twin-informed multi-fidelity GP. The low-fidelity model is the expected V_rms of
     the E1 twin generator (a6_twin.response, a6_twin.h_factor imported, the main
     block is not run): sqrt(E V_rms^2) = M(c, F, f) (r / f)^kappa h(c) with the
     law-true M re-fitted on the training fold only (a2_common.fit_law5), kappa the
     training-fold slope of log(y / M) on log(r / f), r the frequency input variant
     (r = f for 'nominal'), h(c) the twin's capacitance factor at sigma_t = 0.3 ms.
       mf_ar1  y = rho lo(x) + GP(x)   (Kennedy-O'Hagan AR(1), rho by least squares)
       mf_nar  GP on (x, lo(x))        (nonlinear autoregressive multi-fidelity GP)
     The twin's full-grid law is not used, because it contains the held-out level.

Every headline is repeated without recordings 52, 54, 73 (n = 72, duplicated-recordings note in the README).
Nothing is selected on outer test folds. Seeds: a5_common.fold_predict seed 0 for the
A5 reference models (as in a10), a10 SEED 42 for the Lorentzian-spectrum ProductGP,
42 for the new GPs' optimizer restarts.

Outputs (results/): a11_freq.json, a11_predictions.csv,
a11_figdata_levels.csv, numbers_a11.tex, tables/tab_a11_*.tex, cache a11_cache/.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
for _p in (str(ROOT), str(ROOT / "code"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(HERE), str(ROOT / "code"), str(ROOT)]
    + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel as CK  # noqa: E402
from sklearn.gaussian_process.kernels import Matern, WhiteKernel  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

import a2_common as A2  # noqa: E402
import a3_common as C3  # noqa: E402
import a5_common as A5  # noqa: E402
import a10_methods as M10  # noqa: E402  (predict_lorspec, Cache, jsonable; main not run)
import a6_twin as TW  # noqa: E402  (response, h_factor, SIGMA_T; main not run)

SEED = 42
RESULTS = ROOT / "results"
SALAMA = ROOT / "data" / "external_peng" / "salama2024.csv"
DUP_ROWS = [52, 54, 73]
ALL_ROWS = np.arange(A2.N)
KEEP = np.array([i for i in range(A2.N) if i not in DUP_ROWS])
ROWSETS = {"n75": ALL_ROWS, "n72": KEEP}
TARGET = "rms_Voc"
Y = A2.df[TARGET].values.astype(float)

FRQ_NOM = A2.df["freq_Hz"].values.astype(float).copy()  # fold labels, never patched
LEVELS = [5, 10, 15, 20, 25]
INTERIOR = [10, 15, 20]
EDGE = [5, 25]
AGREE_TOL = 0.10


# ====================================================================== frequency inputs
def rate_table():
    rec = pd.read_csv(RESULTS / "a1_recordings.csv").sort_values("condition_id")
    assert (rec.condition_id.values == np.arange(A2.N)).all()
    assert (rec.freq_Hz.values.astype(float) == FRQ_NOM).all()
    assert (rec.force_N.values.astype(float) == A2.FRC).all()
    tap = rec.tap_rate_used_Hz.values.astype(float)
    spec = rec.spectral_rate_Hz.values.astype(float)
    missing = ~np.isfinite(tap)
    tap_f = np.where(missing, FRQ_NOM, tap)  # missing measured rate -> nominal label
    rel = np.abs(spec - tap_f) / tap_f
    agree = np.isfinite(spec) & (rel <= AGREE_TOL) & ~missing
    rows = []
    for i in range(A2.N):
        rows.append(
            {
                "condition_id": i,
                "composition": rec.composition.values[i],
                "force_N": int(rec.force_N.values[i]),
                "freq_nominal_Hz": FRQ_NOM[i],
                "tap_rate_used_Hz": tap[i],
                "spectral_rate_Hz": spec[i],
                "period_source": rec.period_source.values[i],
                "rel_dev_tap_vs_nominal": float(tap_f[i] / FRQ_NOM[i] - 1),
                "rel_dev_spectral_vs_tap": float(rel[i]),
                "timing_estimates_agree": bool(agree[i]),
                "flag_weak_periodicity": bool(rec.flag_weak_periodicity.values[i]),
                "missing_tap_rate": bool(missing[i]),
                "duplicate_copy_of": int(rec.duplicate_copy_of.values[i]),
            }
        )
    inputs = {
        "nominal": FRQ_NOM.copy(),
        "measured": tap_f,
        "measured_agree": np.where(agree, tap_f, FRQ_NOM),
    }
    return inputs, pd.DataFrame(rows)


FIN, RATES = rate_table()
INPUTS = ["nominal", "measured", "measured_agree"]


def set_freq(mode):
    """Patch the frequency input in place in every module the models read from."""
    v = FIN[mode]
    for arr in (A2.X[:, 3], A2.FRQ, A5.X[:, 3], A5.FRQ):
        arr[:] = v
    assert M10.X is A2.X


# ====================================================================== (iii) Salama shape
def salama_shapes():
    d = pd.read_csv(SALAMA)
    d = d[d.subset == "frequency"].copy()
    comps = sorted(d.composition.unique())
    k = np.array([comps.index(c) for c in d.composition])
    f = d.freq_Hz.values.astype(float)
    ly = np.log(d.voltage.values.astype(float))
    nk = len(comps)

    def res_lor(p):
        return p[:nk][k] + np.log(A2.lorentz(f, p[nk], p[nk + 1])) - ly

    p0 = np.r_[[ly[k == j].max() for j in range(nk)], 8.0, 10.0]
    lo = np.r_[[-np.inf] * nk, 0.5, 0.1]
    hi = np.r_[[np.inf] * nk, 200.0, 400.0]
    r = least_squares(res_lor, p0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    f0, gam = float(r.x[nk]), float(r.x[nk + 1])
    rmse_lor = float(np.sqrt(np.mean(r.fun**2)))
    # power law: ly = alpha_k + b log f (linear least squares)
    D = np.column_stack([np.eye(nk)[k], np.log(f)])
    coef, *_ = np.linalg.lstsq(D, ly, rcond=None)
    b = float(coef[-1])
    rmse_pow = float(np.sqrt(np.mean((D @ coef - ly) ** 2)))
    return {
        "source": str(SALAMA.relative_to(ROOT)),
        "subset": "frequency (Fig. 11a), 0.3 N, V/g",
        "n_points": int(len(d)),
        "n_films": nk,
        "freq_range_Hz": [float(f.min()), float(f.max())],
        "lorentz_f0_Hz": f0,
        "lorentz_gamma_Hz": gam,
        "lorentz_at_bound": bool(
            np.isclose(f0, lo[nk])
            or np.isclose(f0, hi[nk])
            or np.isclose(gam, lo[nk + 1])
            or np.isclose(gam, hi[nk + 1])
        ),
        "lorentz_log_rmse": rmse_lor,
        "power_b": b,
        "power_log_rmse": rmse_pow,
        "note": "shape fitted in log space with one free amplitude per film; the sweep spans 1 to 5 Hz,"
        " so on our 5 to 25 Hz grid the shape is used outside the range it was fitted on",
    }


SAL = salama_shapes()


def shape(kind, f, extra=None):
    if kind == "tr_lor":
        return A2.lorentz(f, SAL["lorentz_f0_Hz"], SAL["lorentz_gamma_Hz"])
    if kind == "tr_pow":
        return (f / 10.0) ** SAL["power_b"]
    if kind == "oracle_shape":
        return A2.lorentz(f, extra[0], extra[1])
    raise ValueError(kind)


def fit_amp(tr, s):
    """(a0 + a1 c + a2 c^2) F s(f) by linear least squares on the training rows."""
    c, F = A2.CNT, A2.FRC
    D = np.column_stack([F * s, c * F * s, c**2 * F * s])
    a, *_ = np.linalg.lstsq(D[tr], Y[tr], rcond=None)
    return D @ a, a


# ====================================================================== (iv) twin low fidelity
def twin_lowfid(tr, r):
    """Expected V_rms of the E1 twin generator with the law re-fitted on the fold."""
    f_lab = FRQ_NOM
    # law on the nominal label (the twin's M uses f), whatever the patched input is
    tr = np.asarray(tr)
    p, _, scale, ok = A2._fit(
        A2.law5, tr, Y, A2.P0_5, A2.BOUNDS_5, xt=(A2.CNT[tr], A2.FRC[tr], f_lab[tr])
    )
    info = {"ok": ok}
    P = {"law": [float(v) for v in A2.to_volt(p, scale)]}
    Mx = TW.response("law", P, A2.CNT, A2.FRC, f_lab)
    h = np.array(
        [TW.h_factor(A2.CNT[i], A2.X[i, 1] > 0, TW.SIGMA_T) for i in range(A2.N)]
    )
    lr = np.log(r / f_lab)
    base = Mx * h
    if np.ptp(lr[tr]) > 1e-9:
        z = np.log(Y[tr] / base[tr])
        kappa = float(np.polyfit(lr[tr], z, 1)[0])
    else:
        kappa = 0.0
    return base * np.exp(kappa * lr), {
        "law": P["law"],
        "kappa": kappa,
        "ok": bool(info["ok"]),
    }


def gp_plain():
    return A5.gp_legacy()


# ====================================================================== jobs
REF = ["gp", "lawgp", "prod", "lorspec"]
TRANSFER = ["tr_lor_mean", "tr_lor_gp", "tr_pow_mean", "tr_pow_gp"]
ORACLE = ["oracle_shape_gp"]
MF = ["mf_ar1", "mf_nar"]
MODELS = REF + TRANSFER + ORACLE + MF
LABEL = {
    "gp": r"\GP",
    "lawgp": r"\LawGP",
    "prod": r"\Prod",
    "lorspec": r"Lor.\ spectrum",
    "tr_lor_mean": r"Salama Lor.\ mean",
    "tr_lor_gp": r"Salama Lor.\ + GP",
    "tr_pow_mean": r"Salama power mean",
    "tr_pow_gp": r"Salama power + GP",
    "oracle_shape_gp": r"Oracle shape + GP$^\dagger$",
    "mf_ar1": r"Twin MF AR(1)",
    "mf_nar": r"Twin MF NAR",
}
MAC = {
    "gp": "Gp",
    "lawgp": "Law",
    "prod": "Prod",
    "lorspec": "LorSpec",
    "tr_lor_mean": "SalLorMean",
    "tr_lor_gp": "SalLorGp",
    "tr_pow_mean": "SalPowMean",
    "tr_pow_gp": "SalPowGp",
    "oracle_shape_gp": "Oracle",
    "mf_ar1": "MfAr",
    "mf_nar": "MfNar",
}
IN_MAC = {"nominal": "Nom", "measured": "Meas", "measured_agree": "Agree"}
IN_TEX = {
    "nominal": "nominal label",
    "measured": "measured rate",
    "measured_agree": "measured if agreeing",
}


def folds(rows):
    return [(rows[FRQ_NOM[rows] != v], rows[FRQ_NOM[rows] == v], v) for v in LEVELS]


def job(model, inp, rs, fi):
    rows = ROWSETS[rs]
    tr, te, lvl = folds(rows)[fi]
    mu, info = predict(model, inp, rows, tr, te)
    return (model, inp, rs, fi), mu, info


def predict(model, inp, rows, tr, te, groups=None):
    """Out-of-fold prediction of `model` with frequency input `inp` (rows: the row set)."""
    warnings.filterwarnings("ignore")
    set_freq(inp)
    y = Y
    f_in = FIN[inp]
    info = {}
    if model in PEAK:
        return predict_peak(model, tr, te, groups)
    if model == "gp":
        mu, _ = A5.fold_predict("gp", tr, te, y)
    elif model == "lawgp":
        mu, _ = A5.fold_predict("physgp", tr, te, y)
    elif model == "prod":
        mu, _ = A5.fold_predict("m1", tr, te, y, seed=0)
    elif model == "lorspec":
        mu, info = M10.predict_lorspec(tr, te, y)
    elif model in TRANSFER or model in ORACLE:
        kind = model.rsplit("_", 1)[0]
        extra = None
        if kind == "oracle_shape":
            _, pv, _ = A2.fit_law5(rows, y)  # all rows of the row set: leaky by design
            extra = (float(pv[3]), float(pv[4]))
            info["f0_gamma_all_rows"] = list(extra)
        mean, a = fit_amp(tr, shape(kind, f_in, extra))
        info["amp"] = a.tolist()
        if model.endswith("_mean"):
            mu = mean[te]
        else:
            xs = StandardScaler().fit(A2.X[tr])
            g = gp_plain().fit(xs.transform(A2.X[tr]), y[tr] - mean[tr])
            mu = mean[te] + g.predict(xs.transform(A2.X[te]))
    elif model in MF:
        lo, info = twin_lowfid(tr, f_in)
        xs = StandardScaler().fit(A2.X[tr])
        if model == "mf_ar1":
            rho = float(np.sum(lo[tr] * y[tr]) / np.sum(lo[tr] ** 2))
            info["rho"] = rho
            g = gp_plain().fit(xs.transform(A2.X[tr]), y[tr] - rho * lo[tr])
            mu = rho * lo[te] + g.predict(xs.transform(A2.X[te]))
        else:
            ls = StandardScaler().fit(lo[tr, None])

            def aug(idx):
                return np.column_stack(
                    [xs.transform(A2.X[idx]), ls.transform(lo[idx, None])]
                )

            k = CK(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0] * 5, nu=2.5, length_scale_bounds=(1e-2, 1e3)
            ) + WhiteKernel(1e-3, (1e-6, 1e1))
            g = GaussianProcessRegressor(
                kernel=k,
                normalize_y=True,
                alpha=1e-10,
                n_restarts_optimizer=4,
                random_state=SEED,
            ).fit(aug(tr), y[tr])
            mu = g.predict(aug(te))
            info["kernel"] = str(g.kernel_)
    else:
        raise ValueError(model)
    return np.asarray(mu, float).ravel(), M10.jsonable(info)


# ====================================================================== metrics
def lvl_metrics(y, mu):
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, mu)),
        "MAE": float(mean_absolute_error(y, mu)),
        "bias": float(np.mean(mu - y)),
        "y_mean": float(np.mean(y)),
    }


def band_metrics(rows, mu, per_level, band):
    idx = rows[np.isin(FRQ_NOM[rows], band)]
    w = [per_level[str(v)]["R2"] for v in band]
    return {
        "levels": band,
        "n": int(len(idx)),
        "pooled_R2": float(r2_score(Y[idx], mu[idx])),
        "pooled_MAE": float(mean_absolute_error(Y[idx], mu[idx])),
        "mean_within_level_R2": float(np.mean(w)),
        "min_within_level_R2": float(np.min(w)),
        "n_levels_R2_positive": int(np.sum(np.array(w) > 0)),
        "all_levels_positive": bool(np.all(np.array(w) > 0)),
    }


def evaluate(store, rowsets):
    res, PRED, FIG = {}, [], []
    for rs in rowsets:
        rows = ROWSETS[rs]
        F = folds(rows)
        res[rs] = {}
        for inp in INPUTS:
            res[rs][inp] = {}
            for m in MODELS:
                mu = np.full(A2.N, np.nan)
                infos = {}
                for fi, (tr, te, v) in enumerate(F):
                    a, inf = store[(m, inp, rs, fi)]
                    mu[te] = a
                    infos[str(v)] = inf
                pl = {str(v): lvl_metrics(Y[te], mu[te]) for tr, te, v in F}
                r = {
                    "pooled_R2": float(r2_score(Y[rows], mu[rows])),
                    "pooled_MAE": float(mean_absolute_error(Y[rows], mu[rows])),
                    "mean_within_level_R2": float(np.mean([pl[k]["R2"] for k in pl])),
                    "per_level": pl,
                    "interior": band_metrics(rows, mu, pl, INTERIOR),
                    "edge": band_metrics(rows, mu, pl, EDGE),
                }
                if m in MF or m == "oracle_shape_gp" or m.startswith("tr_"):
                    r["fold_info"] = infos
                res[rs][inp][m] = r
                for k, v in pl.items():
                    FIG.append(
                        {
                            "rowset": rs,
                            "freq_input": inp,
                            "model": m,
                            "level_Hz": int(k),
                            "band": "interior" if int(k) in INTERIOR else "edge",
                            "R2_within": v["R2"],
                            "MAE": v["MAE"],
                            "bias": v["bias"],
                            "n": v["n"],
                        }
                    )
                for i in rows:
                    PRED.append(
                        {
                            "rowset": rs,
                            "freq_input": inp,
                            "model": m,
                            "condition_id": int(i),
                            "held_out_level_Hz": int(FRQ_NOM[i]),
                            "freq_input_Hz": float(FIN[inp][i]),
                            "y_true": float(Y[i]),
                            "y_pred": float(mu[i]),
                        }
                    )
    return res, PRED, FIG


def a10_check(res):
    """Nominal-input reference models must reproduce a10 per-level numbers."""
    J = json.loads((RESULTS / "a10_methods.json").read_text())["new_models"]
    out = {}
    for m in REF:
        d = []
        for rs in ("n75", "n72"):
            ref = J["per_level"][rs]["frequency"][m]
            mine = res[rs]["nominal"][m]["per_level"]
            d += [abs(ref[k]["R2"] - mine[k]["R2"]) for k in ref]
            d.append(
                abs(
                    J["pooled"][rs]["frequency"][m]["R2"]
                    - res[rs]["nominal"][m]["pooled_R2"]
                )
            )
        out[m] = float(max(d))
    return out


# ====================================================================== verdict
def verdict(res):
    V = {}
    for rs in res:
        v = {}
        for band in ("interior", "edge"):
            pos_pooled, pos_all = [], []
            for inp in INPUTS:
                for m in MODELS:
                    if m == "oracle_shape_gp":
                        continue
                    b = res[rs][inp][m][band]
                    if b["pooled_R2"] > 0:
                        pos_pooled.append(f"{m}/{inp}")
                    if b["all_levels_positive"]:
                        pos_all.append(f"{m}/{inp}")
            best = max(
                ((m, inp) for inp in INPUTS for m in MODELS if m != "oracle_shape_gp"),
                key=lambda t: res[rs][t[1]][t[0]][band]["mean_within_level_R2"],
            )
            v[band] = {
                "positive_band_pooled_R2": pos_pooled,
                "every_level_within_R2_positive": pos_all,
                "best_mean_within_level": {
                    "model": best[0],
                    "input": best[1],
                    "mean_within_level_R2": res[rs][best[1]][best[0]][band][
                        "mean_within_level_R2"
                    ],
                    "pooled_R2": res[rs][best[1]][best[0]][band]["pooled_R2"],
                },
            }
        # effect of the measured rate alone on the four reference models
        v["measured_minus_nominal_pooled_R2"] = {
            m: res[rs]["measured"][m]["pooled_R2"] - res[rs]["nominal"][m]["pooled_R2"]
            for m in REF
            if m in MODELS
        }
        v["oracle_shape_edge_mean_within"] = res[rs]["nominal"]["oracle_shape_gp"][
            "edge"
        ]["mean_within_level_R2"]
        V[rs] = v
    return V


# ====================================================================== outputs
MACROS = []


def mac(name, val, key):
    assert name.isalpha(), name
    MACROS.append((name, str(val), key))


def f2(x):
    return C3.fnum(x, 2)


def build_macros(res, J):
    mac("frqAgreeTolPct", f"{int(AGREE_TOL * 100)}", "rates.agree_tolerance")
    R = RATES
    mac(
        "frqNRateDisagree",
        str(int((~R.timing_estimates_agree).sum())),
        "rates.n_disagree",
    )
    mac("frqNRateMissing", str(int(R.missing_tap_rate.sum())), "rates.n_missing")
    mac(
        "frqNRateOffTen",
        str(int((R.rel_dev_tap_vs_nominal.abs() > 0.10).sum())),
        "rates.n_tap_off_nominal_gt10pct",
    )
    mac("frqSalFzero", C3.fnum(SAL["lorentz_f0_Hz"], 1), "salama_shape.lorentz_f0_Hz")
    mac("frqSalGamma", C3.fnum(SAL["lorentz_gamma_Hz"], 1), "salama_shape.lorentz_gamma_Hz")
    mac("frqSalPowB", f2(SAL["power_b"]), "salama_shape.power_b")
    for rs in res:
        rsm = "" if rs == "n75" else "Excl"
        for inp in INPUTS:
            for m in MODELS:
                r = res[rs][inp][m]
                base = f"frq{rsm}{IN_MAC[inp]}{MAC[m]}"
                key = f"results.{rs}.{inp}.{m}"
                mac(base + "Rsq", f2(r["pooled_R2"]), key + ".pooled_R2")
                mac(
                    base + "Within",
                    f2(r["mean_within_level_R2"]),
                    key + ".mean_within_level_R2",
                )
                for band, bm in (("interior", "Int"), ("edge", "Edge")):
                    mac(
                        base + bm + "Rsq",
                        f2(r[band]["pooled_R2"]),
                        f"{key}.{band}.pooled_R2",
                    )
                    mac(
                        base + bm + "Within",
                        f2(r[band]["mean_within_level_R2"]),
                        f"{key}.{band}.mean_within_level_R2",
                    )
                for k, v in r["per_level"].items():
                    w = A2.NUM_WORD[int(k)]
                    mac(
                        base + "Freq" + w + "Rsq",
                        f2(v["R2"]),
                        f"{key}.per_level.{k}.R2",
                    )
                    mac(
                        base + "Freq" + w + "Mae",
                        C3.fnum(v["MAE"], 3),
                        f"{key}.per_level.{k}.MAE",
                    )


def write_tables(out, res):
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    T = {}
    head_models = MODELS
    # per-level table: rows model x input, columns levels + bands (n75 and n72)
    for rs, suf in (("n75", ""), ("n72", "_n72")):
        L = [
            r"\footnotesize\setlength{\tabcolsep}{2pt}%",
            r"\begin{tabular}{@{}llccccccc@{}}",
            r"\toprule",
            r"Model & $f$ input & \SI{5}{\hertz} & \SI{10}{\hertz} & \SI{15}{\hertz} & \SI{20}{\hertz} & \SI{25}{\hertz} & Interior & Edge \\",
            r"\midrule",
        ]
        for m in head_models:
            for inp in INPUTS:
                r = res[rs][inp][m]
                cells = [f2(r["per_level"][str(v)]["R2"]) for v in LEVELS]
                cells += [
                    f2(r["interior"]["mean_within_level_R2"]),
                    f2(r["edge"]["mean_within_level_R2"]),
                ]
                L.append(
                    f"{LABEL[m] if inp == 'nominal' else ''} & {IN_TEX[inp]} & "
                    + " & ".join(cells)
                    + r" \\"
                )
            if m != head_models[-1]:
                L.append(r"\addlinespace[1pt]")
        L += [r"\bottomrule", r"\end{tabular}"]
        name = f"tab_a11_levels{suf}.tex"
        (tab / name).write_text("\n".join(L) + "\n")
        T[
            name
        ] = f"within-level R2 of held-out frequency per level and band mean ({rs}); dagger = leaky oracle"
        # band table: pooled R2 by band
        L = [
            r"\footnotesize\setlength{\tabcolsep}{3pt}%",
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            r" & \multicolumn{3}{c}{Interior (10--20 Hz) pooled $R^2$} & \multicolumn{3}{c}{Edge (5, 25 Hz) pooled $R^2$} \\",
            r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
            r"Model & nominal & measured & agree & nominal & measured & agree \\",
            r"\midrule",
        ]
        for m in head_models:
            cells = [f2(res[rs][inp][m]["interior"]["pooled_R2"]) for inp in INPUTS]
            cells += [f2(res[rs][inp][m]["edge"]["pooled_R2"]) for inp in INPUTS]
            L.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        name = f"tab_a11_bands{suf}.tex"
        (tab / name).write_text("\n".join(L) + "\n")
        T[name] = f"band-pooled R2 of held-out frequency by frequency input ({rs})"
    # rates table: recordings whose measured rate is off nominal or unreliable
    R = RATES
    sel = R[(R.rel_dev_tap_vs_nominal.abs() > 0.10) | ~R.timing_estimates_agree]
    L = [
        r"\footnotesize\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabular}{@{}rlccccc@{}}",
        r"\toprule",
        r"ID & Composition & $F$ (N) & $f$ label & Tap rate & Spectral & Agree \\",
        r"\midrule",
    ]
    for _, r in sel.iterrows():
        L.append(
            f"{r.condition_id} & {A2.COMP_TEX[r.composition]} & {r.force_N} & {int(r.freq_nominal_Hz)} & "
            f"{C3.fnum(r.tap_rate_used_Hz, 1)} & {C3.fnum(r.spectral_rate_Hz, 1)} & "
            f"{'yes' if r.timing_estimates_agree else 'no'} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    (tab / "tab_a11_rates.tex").write_text("\n".join(L) + "\n")
    T[
        "tab_a11_rates.tex"
    ] = "recordings with tap rate >10% off nominal or tap and spectral rates disagreeing"
    return T


# ====================================================================== round 2
# A  folds on the measured tap rate (bands centred on the nominal levels, edges at
#    the midpoints 7.5/12.5/17.5/22.5 Hz; sensitivity: five equal-count quantile bands)
# B  clean subset: recordings whose tap rate is within 10 percent of the label, minus
#    the three copies, nominal folds
# C  per-cycle continuum: instantaneous tap rate per cycle, leave-one-rate-band-out
#    with recording-level grouping
# D  peak-shape prior: law mean with (f0, gamma) from the training fold only, and a
#    hierarchical law with f0, gamma partially pooled across compositions (a10 fit_hier),
#    pooling strength chosen by inner leave-one-group-out folds of the training rows
BAND_EDGES = [7.5, 12.5, 17.5, 22.5]
PEAK = ["law_mean", "hier_mean", "hier_gp"]
R2_REF = REF
CLEAN = np.array(
    [
        i
        for i in range(A2.N)
        if abs(FIN["measured"][i] / FRQ_NOM[i] - 1) <= AGREE_TOL and i not in DUP_ROWS
    ]
)
ROWSETS["clean"] = CLEAN
QLAB = [1, 2, 3, 4, 5]
Q_WORD = {1: "QOne", 2: "QTwo", 3: "QThree", 4: "QFour", 5: "QFive"}


def band_of(r):
    return np.array(LEVELS, float)[np.searchsorted(BAND_EDGES, r, side="right")]


def qbands(rs):
    """Five equal-count bands of the measured rate over the rows of the row set."""
    rows = ROWSETS[rs]
    r = FIN["measured"][rows]
    rk = pd.Series(r).rank(method="first").values
    q = np.ceil(rk / len(rows) * 5).astype(int)
    g = np.full(A2.N, np.nan)
    g[rows] = q
    return g


def groups_for(scheme, rs):
    if scheme in ("nom", "clean"):
        return FRQ_NOM.copy()
    if scheme == "band":
        return band_of(FIN["measured"])
    if scheme == "bandq":
        return qbands(rs)
    raise ValueError(scheme)


def labels_for(scheme):
    return QLAB if scheme == "bandq" else LEVELS


def folds_g(rows, G, labs):
    return [(rows[G[rows] != v], rows[G[rows] == v], v) for v in labs if (G[rows] == v).any()]


def predict_peak(model, tr, te, groups):
    """Law with (f0, gamma) from the training fold; hierarchical variants pool per composition."""
    tr, te = np.asarray(tr), np.asarray(te)
    if model == "law_mean":
        m, pv, info = A2.fit_law5(tr, Y)
        return m(te), {"f0": float(pv[3]), "gamma": float(pv[4]), "ok": bool(info["ok"])}
    g = groups[tr]
    inner = [(tr[g != v], tr[g == v]) for v in np.unique(g) if (g != v).any()]

    def one(a, b, kap):
        if model == "hier_gp":
            return M10.hier_fit_predict(a, b, Y, kap)[0]
        return M10.fit_hier(a, Y, kap)[0](b)

    sse = [sum(float(np.sum((Y[b] - one(a, b, k)) ** 2)) for a, b in inner) for k in M10.KAPPAS]
    kap = M10.KAPPAS[int(np.argmin(sse))]
    if model == "hier_gp":
        mu, info = M10.hier_fit_predict(tr, te, Y, kap)
    else:
        m, info = M10.fit_hier(tr, Y, kap)
        mu = m(te)
    info = dict(info)
    info["inner_sse"] = sse
    return mu, info


def job2(model, inp, scheme, rs, fi):
    rows = ROWSETS[rs]
    G = groups_for(scheme, rs)
    tr, te, lab = folds_g(rows, G, labels_for(scheme))[fi]
    mu, info = predict(model, inp, rows, tr, te, G)
    return (model, inp, scheme, rs, fi), mu, info


# A/B/D run plan: (scheme, rowsets, inputs, models)
PLAN2 = [
    ("band", ["n75", "n72"], ["measured", "nominal"], R2_REF + PEAK),
    ("bandq", ["n75", "n72"], ["measured"], R2_REF),
    ("clean", ["clean"], ["nominal", "measured"], R2_REF + PEAK),
    ("nom", ["n75", "n72"], ["nominal", "measured"], PEAK),
]
SCH_WORD = {"band": "Band", "bandq": "Bandq", "clean": "Clean", "nom": "Nomfold"}
SCH_TEX = {
    "band": "measured-rate bands",
    "bandq": "measured-rate quantiles",
    "clean": "clean subset, label folds",
    "nom": "label folds (round 1)",
}
PEAK_LABEL = {
    "law_mean": r"Law mean",
    "hier_mean": r"Hier.\ law mean",
    "hier_gp": r"Hier.\ LawGP",
}
PEAK_MAC = {"law_mean": "LawMean", "hier_mean": "HierMean", "hier_gp": "HierGp"}
LABEL.update(PEAK_LABEL)
MAC.update(PEAK_MAC)


def r2_or_none(y, mu):
    return float(r2_score(y, mu)) if len(y) > 1 else None


def eval2(store2):
    res = {}
    for scheme, rsets, inps, models in PLAN2:
        res[scheme] = {}
        labs = labels_for(scheme)
        for rs in rsets:
            rows = ROWSETS[rs]
            G = groups_for(scheme, rs)
            F = folds_g(rows, G, labs)
            res[scheme][rs] = {}
            for inp in inps:
                res[scheme][rs][inp] = {}
                for m in models:
                    mu = np.full(A2.N, np.nan)
                    infos = {}
                    for fi, (tr, te, v) in enumerate(F):
                        a, inf = store2[(m, inp, scheme, rs, fi)]
                        mu[te] = a
                        infos[str(v)] = inf
                    pl = {}
                    for tr, te, v in F:
                        d = lvl_metrics(Y[te], mu[te]) if len(te) > 1 else {"n": 1, "R2": None, "MAE": float(abs(mu[te] - Y[te])[0])}
                        d["rate_range_Hz"] = [float(FIN["measured"][te].min()), float(FIN["measured"][te].max())]
                        d["n_test_label_seen_in_train"] = int(np.isin(FRQ_NOM[te], FRQ_NOM[tr]).sum())
                        pl[str(v)] = d
                    r = {
                        "n": int(len(rows)),
                        "pooled_R2": float(r2_score(Y[rows], mu[rows])),
                        "pooled_MAE": float(mean_absolute_error(Y[rows], mu[rows])),
                        "per_level": pl,
                    }
                    for band, bl in (("interior", labs[1:4]), ("edge", [labs[0], labs[4]])):
                        idx = rows[np.isin(G[rows], bl)]
                        w = [pl[str(v)]["R2"] for v in bl if str(v) in pl and pl[str(v)]["R2"] is not None]
                        r[band] = {
                            "levels": list(bl),
                            "n": int(len(idx)),
                            "pooled_R2": r2_or_none(Y[idx], mu[idx]),
                            "mean_within_level_R2": float(np.mean(w)) if w else None,
                            "all_levels_positive": bool(len(w) == len(bl) and all(x > 0 for x in w)),
                        }
                    if m in PEAK:
                        r["fold_info"] = infos
                    res[scheme][rs][inp][m] = r
    return res


# ---------------------------------------------------------------- (C) per-cycle continuum
C_MODELS = ["gp", "lawgp", "law_mean", "prod"]


def cycle_data():
    cyc = pd.read_csv(RESULTS / "a1_cycles.csv")
    rec = pd.read_csv(RESULTS / "a1_recordings.csv").set_index("condition_id")
    long = pd.read_parquet(ROOT / "data" / "long.parquet").sort_values(["series_id", "time"])
    waves = {int(i): g.voc.values for i, g in long.groupby("series_id")}
    out = []
    n_all = n_full = 0
    for cid, g in cyc.groupby("condition_id"):
        g = g.sort_values("cycle")
        x = waves[int(cid)] - np.median(waves[int(cid)])
        per = float(rec.loc[cid, "period_used_samples"])
        pk = np.array([s + int(np.argmax(np.abs(x[s:e]))) for s, e in zip(g.start_sample, g.end_sample)])
        d = np.diff(pk).astype(float)
        ok = (d > 0.5 * per) & (d < 1.5 * per)
        for k, row in enumerate(g.itertuples()):
            n_all += 1
            if not row.full_window:
                continue
            n_full += 1
            iv = []
            if k > 0 and ok[k - 1]:
                iv.append(d[k - 1])
            if k < len(g) - 1 and ok[k]:
                iv.append(d[k])
            if not iv:
                continue
            out.append(
                {
                    "condition_id": int(cid),
                    "cycle": int(row.cycle),
                    "cnt_pct": A2.CNT[cid],
                    "is_pristine": A2.X[cid, 1],
                    "force_N": A2.FRC[cid],
                    "freq_nominal_Hz": FRQ_NOM[cid],
                    "rate_inst_Hz": 1000.0 / float(np.mean(iv)),
                    "rms": float(row.rms),
                }
            )
    D = pd.DataFrame(out)
    D["band"] = band_of(D.rate_inst_Hz.values)
    gm = D.groupby("condition_id").rate_inst_Hz
    rel = gm.std() / gm.mean()
    meta = {
        "n_cycles_all": n_all,
        "n_full_window": n_full,
        "n_with_rate": int(len(D)),
        "peak_rule": "argmax |v - median| inside each a1 cycle window; interval kept if within 0.5 to 1.5"
        " periods; cycle rate = 1 kHz / mean of its valid adjacent intervals",
        "within_recording_rate_cv_median": float(rel.median()),
        "recording_mean_rate_vs_tap_rate_corr": float(
            np.corrcoef(gm.mean().values, FIN["measured"][gm.mean().index.values])[0, 1]
        ),
    }
    return D, meta


CYC, CYC_META = cycle_data()
CX = CYC[["cnt_pct", "is_pristine", "force_N", "rate_inst_Hz"]].values.astype(float)
CY = CYC.rms.values.astype(float)
CREC = CYC.condition_id.values
C_ROWS = {
    "n75": np.arange(len(CYC)),
    "n72": np.where(~np.isin(CREC, DUP_ROWS))[0],
}


def cyc_folds(scheme, rs):
    """Leave-one-band-out on cycles; training excludes every recording with a test cycle."""
    rows = C_ROWS[rs]
    G = CYC.band.values if scheme == "cyc_band" else CYC.freq_nominal_Hz.values
    F = []
    for v in LEVELS:
        te = rows[G[rows] == v]
        if len(te) == 0:
            continue
        tr = rows[(G[rows] != v) & ~np.isin(CREC[rows], np.unique(CREC[te]))]
        F.append((tr, te, v))
    return F


def cyc_job(model, scheme, rs, fi):
    warnings.filterwarnings("ignore")
    tr, te, v = cyc_folds(scheme, rs)[fi]
    xs = StandardScaler().fit(CX[tr])
    info = {"n_train": int(len(tr)), "n_test": int(len(te))}
    base_tr = np.zeros(len(tr))
    base_te = np.zeros(len(te))
    if model in ("lawgp", "law_mean"):
        xt = (CX[:, 0], CX[:, 2], CX[:, 3])
        p, _, sc, ok = A2._fit(
            A2.law5, np.arange(len(tr)), CY[tr], A2.P0_5, A2.BOUNDS_5,
            xt=(xt[0][tr], xt[1][tr], xt[2][tr]),
        )
        base_tr = A2.law5((xt[0][tr], xt[1][tr], xt[2][tr]), *p) * sc
        base_te = A2.law5((xt[0][te], xt[1][te], xt[2][te]), *p) * sc
        info.update({"f0": float(p[3]), "gamma": float(p[4]), "ok": bool(ok)})
        if model == "law_mean":
            return (model, scheme, rs, fi), base_te, info
    g = A5.gp_struct("product", 0) if model == "prod" else A5.gp_legacy()
    g.fit(xs.transform(CX[tr]), CY[tr] - base_tr)
    return (model, scheme, rs, fi), base_te + g.predict(xs.transform(CX[te])), info


def eval_cyc(storec):
    res = {}
    for scheme in ("cyc_band", "cyc_nom"):
        res[scheme] = {}
        for rs in C_ROWS:
            F = cyc_folds(scheme, rs)
            res[scheme][rs] = {}
            for m in C_MODELS:
                pl, allte, allmu = {}, [], []
                rec_y, rec_mu, rec_lvl = [], [], []
                for fi, (tr, te, v) in enumerate(F):
                    mu, info = storec[(m, scheme, rs, fi)]
                    d = lvl_metrics(CY[te], mu)
                    ry = pd.Series(CY[te]).groupby(CREC[te]).mean()
                    rm = pd.Series(mu).groupby(CREC[te]).mean()
                    d["recording_R2"] = r2_or_none(ry.values, rm.values)
                    d["n_recordings"] = int(len(ry))
                    d["n_train_cycles"] = info["n_train"]
                    d["rate_range_Hz"] = [float(CX[te, 3].min()), float(CX[te, 3].max())]
                    if "f0" in info:
                        d["train_f0"] = info["f0"]
                        d["train_gamma"] = info["gamma"]
                    pl[str(v)] = d
                    allte.append(te)
                    allmu.append(mu)
                    rec_y += ry.values.tolist()
                    rec_mu += rm.values.tolist()
                    rec_lvl += [v] * len(ry)
                te = np.concatenate(allte)
                mu = np.concatenate(allmu)
                r = {
                    "n_cycles": int(len(te)),
                    "pooled_R2": float(r2_score(CY[te], mu)),
                    "pooled_recording_R2": float(r2_score(rec_y, rec_mu)),
                    "per_level": pl,
                }
                for band, bl in (("interior", INTERIOR), ("edge", EDGE)):
                    w = [pl[str(v)]["R2"] for v in bl if str(v) in pl]
                    wr = [pl[str(v)]["recording_R2"] for v in bl if str(v) in pl and pl[str(v)]["recording_R2"] is not None]
                    r[band] = {
                        "mean_within_level_R2": float(np.mean(w)),
                        "mean_within_level_recording_R2": float(np.mean(wr)) if wr else None,
                        "all_levels_positive": bool(len(w) == len(bl) and all(x > 0 for x in w)),
                    }
                res[scheme][rs][m] = r
    # rate coverage: cycles per band and how many come from recordings with another label
    cov = {}
    for v in LEVELS:
        s = CYC[CYC.band == v]
        cov[str(v)] = {
            "n_cycles": int(len(s)),
            "n_recordings": int(s.condition_id.nunique()),
            "n_cycles_other_label": int((s.freq_nominal_Hz != v).sum()),
            "rate_min": float(s.rate_inst_Hz.min()) if len(s) else None,
            "rate_max": float(s.rate_inst_Hz.max()) if len(s) else None,
        }
    res["coverage"] = cov
    res["meta"] = CYC_META
    res["gaps_Hz"] = gaps(CYC.rate_inst_Hz.values)
    return res


def gaps(r, min_gap=1.0):
    s = np.sort(r)
    d = np.diff(s)
    return [[float(s[i]), float(s[i + 1])] for i in np.where(d >= min_gap)[0]]


# ---------------------------------------------------------------- round 2 driver
def run_round2(cache, n_jobs):
    T0 = time.time()
    store2, storec, jobs = {}, {}, []
    for scheme, rsets, inps, models in PLAN2:
        for rs in rsets:
            nf = len(folds_g(ROWSETS[rs], groups_for(scheme, rs), labels_for(scheme)))
            for inp in inps:
                for m in models:
                    tag = f"r2_{scheme}_{m}_{inp}_{rs}"
                    if cache.has(tag):
                        store2.update(cache.get(tag))
                        continue
                    jobs += [delayed(job2)(m, inp, scheme, rs, fi) for fi in range(nf)]
    for scheme in ("cyc_band", "cyc_nom"):
        for rs in C_ROWS:
            for m in C_MODELS:
                tag = f"r2c_{scheme}_{m}_{rs}"
                if cache.has(tag):
                    storec.update(cache.get(tag))
                    continue
                jobs += [delayed(cyc_job)(m, scheme, rs, fi) for fi in range(len(cyc_folds(scheme, rs)))]
    print(f"[a11/r2] {len(jobs)} fold jobs to run", flush=True)
    if jobs:
        out = Parallel(n_jobs=n_jobs)(jobs)
        b2, bc = {}, {}
        for key, mu, info in out:
            if len(key) == 5:
                b2.setdefault(f"r2_{key[2]}_{key[0]}_{key[1]}_{key[3]}", {})[key] = (mu, M10.jsonable(info))
            else:
                bc.setdefault(f"r2c_{key[1]}_{key[0]}_{key[2]}", {})[key] = (mu, M10.jsonable(info))
        for tag, d in b2.items():
            cache.put(tag, d)
            store2.update(d)
        for tag, d in bc.items():
            cache.put(tag, d)
            storec.update(d)
    print(f"[a11/r2] folds done in {time.time() - T0:.0f} s", flush=True)
    set_freq("nominal")
    R = eval2(store2)
    RC = eval_cyc(storec)
    return R, RC


def verdict2(res1, R, RC):
    V = {}
    for scheme, rsets, inps, models in PLAN2:
        for rs in rsets:
            for inp in inps:
                for m in models:
                    r = R[scheme][rs][inp][m]
                    V.setdefault(scheme, {}).setdefault(rs, {}).setdefault(inp, {})[m] = {
                        "pooled_R2": r["pooled_R2"],
                        "interior_pooled_R2": r["interior"]["pooled_R2"],
                        "edge_pooled_R2": r["edge"]["pooled_R2"],
                        "interior_all_positive": r["interior"]["all_levels_positive"],
                        "peak_level_R2": r["per_level"].get("20", {}).get("R2"),
                    }
    out = {
        "table": V,
        "invalid_label_leak": "rate-band folds with the nominal label as input are not a held-out-frequency"
        " test: in every rate band all test recordings carry a label value that also occurs in the training rows"
        " (per_level n_test_label_seen_in_train), so they are excluded from every list below",
    }
    anyp = lambda f: sorted(  # noqa: E731
        f"{s}/{rs}/{i}/{m}"
        for s in V
        for rs in V[s]
        for i in V[s][rs]
        for m in V[s][rs][i]
        if f(V[s][rs][i][m]) and not (s in ("band", "bandq") and i == "nominal")
    )
    out["pooled_positive"] = anyp(lambda d: d["pooled_R2"] > 0)
    out["interior_pooled_positive"] = anyp(lambda d: d["interior_pooled_R2"] is not None and d["interior_pooled_R2"] > 0)
    out["interior_every_level_positive"] = anyp(lambda d: d["interior_all_positive"])
    out["edge_pooled_positive"] = anyp(lambda d: d["edge_pooled_R2"] is not None and d["edge_pooled_R2"] > 0)
    out["peak_20Hz_positive"] = anyp(lambda d: d["peak_level_R2"] is not None and d["peak_level_R2"] > 0)
    cv = {}
    for s in ("cyc_band", "cyc_nom"):
        for rs in C_ROWS:
            for m in C_MODELS:
                r = RC[s][rs][m]
                cv[f"{s}/{rs}/{m}"] = {
                    "pooled_R2": r["pooled_R2"],
                    "pooled_recording_R2": r["pooled_recording_R2"],
                    "band20_R2": r["per_level"]["20"]["R2"],
                    "band20_recording_R2": r["per_level"]["20"]["recording_R2"],
                    "interior_all_positive": r["interior"]["all_levels_positive"],
                }
    out["cycles"] = cv
    return out


def build_macros2(R, RC):
    mac("frqNClean", str(len(CLEAN)), "round2.clean_rows (length)")
    for v in LEVELS:
        mac("frqCleanN" + A2.NUM_WORD[v] + "Hz", str(int((FRQ_NOM[CLEAN] == v).sum())), f"round2.folds.clean.clean.nominal.gp.per_level.{v}.n")
        mac("frqBandN" + A2.NUM_WORD[v] + "Hz", str(int((band_of(FIN["measured"]) == v).sum())), f"round2.folds.band.n75.measured.gp.per_level.{v}.n")
        c = RC["coverage"][str(v)]
        mac("frqCycBandN" + A2.NUM_WORD[v] + "Hz", str(c["n_cycles"]), f"round2.cycles.coverage.{v}.n_cycles")
        mac("frqCycBandOther" + A2.NUM_WORD[v] + "Hz", str(c["n_cycles_other_label"]), f"round2.cycles.coverage.{v}.n_cycles_other_label")
    mac("frqCycN", str(CYC_META["n_with_rate"]), "round2.cycles.meta.n_with_rate")
    for scheme, rsets, inps, models in PLAN2:
        for rs in rsets:
            rsm = {"n75": "", "n72": "Excl", "clean": ""}[rs]
            for inp in inps:
                for m in models:
                    r = R[scheme][rs][inp][m]
                    base = f"frq{SCH_WORD[scheme]}{rsm}{IN_MAC[inp]}{MAC[m]}"
                    key = f"round2.folds.{scheme}.{rs}.{inp}.{m}"
                    mac(base + "Rsq", f2(r["pooled_R2"]), key + ".pooled_R2")
                    for band, bm in (("interior", "Int"), ("edge", "Edge")):
                        if r[band]["pooled_R2"] is not None:
                            mac(base + bm + "Rsq", f2(r[band]["pooled_R2"]), f"{key}.{band}.pooled_R2")
                    for k, v in r["per_level"].items():
                        if v["R2"] is None:
                            continue
                        w = Q_WORD[int(k)] if scheme == "bandq" else "Freq" + A2.NUM_WORD[int(k)]
                        mac(base + w + "Rsq", f2(v["R2"]), f"{key}.per_level.{k}.R2")
    for s, sw in (("cyc_band", "CycBand"), ("cyc_nom", "CycNomfold")):
        for rs in C_ROWS:
            rsm = "" if rs == "n75" else "Excl"
            for m in C_MODELS:
                r = RC[s][rs][m]
                base = f"frq{sw}{rsm}{MAC.get(m, m)}"
                key = f"round2.cycles.{s}.{rs}.{m}"
                mac(base + "Rsq", f2(r["pooled_R2"]), key + ".pooled_R2")
                mac(base + "RecRsq", f2(r["pooled_recording_R2"]), key + ".pooled_recording_R2")
                for k, v in r["per_level"].items():
                    w = "Freq" + A2.NUM_WORD[int(k)]
                    mac(base + w + "Rsq", f2(v["R2"]), f"{key}.per_level.{k}.R2")
                    if v["recording_R2"] is not None:
                        mac(base + w + "RecRsq", f2(v["recording_R2"]), f"{key}.per_level.{k}.recording_R2")


def write_tables2(out, res1, R, RC):
    tab = out / "tables"
    T = {}

    def cell(x):
        return "--" if x is None else f2(x)

    for rs, suf in (("n75", ""), ("n72", "_n72")):
        L = [
            r"\footnotesize\setlength{\tabcolsep}{2pt}%",
            r"\begin{tabular}{@{}lllcccccccc@{}}",
            r"\toprule",
            r"Folds & $f$ input & Model & 1 & 2 & 3 & 4 & 5 & Pooled & Interior & Edge \\",
            r"\midrule",
        ]
        blocks = [("nom1", "nominal"), ("nom1", "measured"), ("band", "measured"), ("band", "nominal"), ("bandq", "measured")]
        if rs == "n75":
            blocks += [("clean", "nominal"), ("clean", "measured")]
        for sch, inp in blocks:
            for j, m in enumerate(R2_REF):
                if sch == "nom1":
                    r = res1[rs][inp][m]
                    pls = [r["per_level"][str(v)]["R2"] for v in LEVELS]
                    row = [r["pooled_R2"], r["interior"]["pooled_R2"], r["edge"]["pooled_R2"]]
                    sname = SCH_TEX["nom"]
                else:
                    rr = "clean" if sch == "clean" else rs
                    r = R[sch][rr][inp][m]
                    pls = [r["per_level"].get(str(v), {}).get("R2") for v in labels_for(sch)]
                    row = [r["pooled_R2"], r["interior"]["pooled_R2"], r["edge"]["pooled_R2"]]
                    sname = SCH_TEX[sch] + (r"$^\ddagger$" if sch == "band" and inp == "nominal" else "")
                L.append(
                    f"{sname if j == 0 else ''} & {IN_TEX[inp] if j == 0 else ''} & {LABEL[m]} & "
                    + " & ".join(cell(x) for x in pls + row)
                    + r" \\"
                )
            L.append(r"\addlinespace[2pt]")
        L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
        name = f"tab_a11_r2_folds{suf}.tex"
        (tab / name).write_text("\n".join(L) + "\n")
        T[name] = (
            f"round 2 A/B ({rs}): R2 per held-out group (columns 1-5 = 5, 10, 15, 20, 25 Hz labels or rate bands;"
            " quantile rows: lowest to highest rate quintile), pooled, interior and edge band-pooled R2;"
            " double dagger = label input under rate-band folds, label values of the test band occur in"
            " training, not a held-out-frequency test"
        )
    # peak table: 20 Hz held out
    L = [
        r"\footnotesize\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabular}{@{}lllccccc@{}}",
        r"\toprule",
        r"Folds & $f$ input & Model & $R^2$ at 20 Hz & MAE (V) & Bias (V) & $f_0$ (Hz) & Pooled $R^2$ \\",
        r"\midrule",
    ]
    for sch, rs, inp in (("nom", "n75", "nominal"), ("nom", "n75", "measured"), ("band", "n75", "measured"), ("clean", "clean", "nominal")):
        for j, m in enumerate(["lawgp"] + PEAK):
            if m == "lawgp":
                if sch == "nom":
                    r = res1[rs][inp][m]
                else:
                    r = R[sch][rs][inp][m]
                f0 = None
            else:
                r = R[sch][rs][inp][m]
                fi = r["fold_info"]["20"]
                f0 = fi.get("f0", np.median(fi.get("f0_k", [np.nan])) if "f0_k" in fi else None)
            p = r["per_level"]["20"]
            L.append(
                f"{SCH_TEX[sch] if j == 0 else ''} & {IN_TEX[inp] if j == 0 else ''} & {LABEL[m]} & "
                f"{cell(p['R2'])} & {C3.fnum(p['MAE'], 3)} & {C3.fnum(p['bias'], 3)} & "
                f"{'--' if f0 is None else C3.fnum(f0, 1)} & {f2(r['pooled_R2'])} \\\\"
            )
        L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    (tab / "tab_a11_r2_peak.tex").write_text("\n".join(L) + "\n")
    T["tab_a11_r2_peak.tex"] = "round 2 D: held-out 20 Hz (label fold or rate band), n75 / clean; f0 = training-fold law peak (median over compositions for the hierarchical law)"
    # cycles table
    L = [
        r"\footnotesize\setlength{\tabcolsep}{2pt}%",
        r"\begin{tabular}{@{}lllccccccc@{}}",
        r"\toprule",
        r"Folds & Rows & Model & 5 & 10 & 15 & 20 & 25 & Pooled & Pooled (rec.) \\",
        r"\midrule",
    ]
    for s, sname in (("cyc_band", "instantaneous-rate bands"), ("cyc_nom", "label folds")):
        for rs in C_ROWS:
            for j, m in enumerate(C_MODELS):
                r = RC[s][rs][m]
                L.append(
                    f"{sname if j == 0 and rs == 'n75' else ''} & {rs if j == 0 else ''} & {LABEL.get(m, m)} & "
                    + " & ".join(cell(r["per_level"].get(str(v), {}).get("R2")) for v in LEVELS)
                    + f" & {f2(r['pooled_R2'])} & {f2(r['pooled_recording_R2'])} \\\\"
                )
            L.append(r"\addlinespace[2pt]")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
    (tab / "tab_a11_r2_cycles.tex").write_text("\n".join(L) + "\n")
    T["tab_a11_r2_cycles.tex"] = "round 2 C: per-cycle R2 per held-out band, pooled per cycle and per recording (mean of test cycles)"
    return T


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--quick", action="store_true", help="n75 only, gp/lawgp/transfer/mf (smoke)"
    )
    ap.add_argument("--no-round2", action="store_true", help="round 1 only")
    args = ap.parse_args()
    assert args.n_jobs <= 4, "coordinator cap: n_jobs <= 4"
    T0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = M10.Cache(out / ("a11_cache_quick" if args.quick else "a11_cache"))
    rowsets = ["n75"] if args.quick else ["n75", "n72"]
    global MODELS
    if args.quick:
        MODELS = ["gp", "lawgp"] + TRANSFER + ORACLE + MF
    store, jobs = {}, []
    for m in MODELS:
        for inp in INPUTS:
            for rs in rowsets:
                tag = f"{m}_{inp}_{rs}"
                if cache.has(tag):
                    store.update(cache.get(tag))
                    continue
                jobs += [delayed(job)(m, inp, rs, fi) for fi in range(len(LEVELS))]
    print(f"[a11] {len(jobs)} fold jobs to run", flush=True)
    if jobs:
        res_jobs = Parallel(n_jobs=args.n_jobs)(jobs)
        bucket = {}
        for key, mu, info in res_jobs:
            bucket.setdefault(f"{key[0]}_{key[1]}_{key[2]}", {})[key] = (mu, info)
        for tag, d in bucket.items():
            cache.put(tag, d)
            store.update(d)
    print(f"[a11] folds done in {time.time() - T0:.0f} s", flush=True)
    set_freq("nominal")
    res, PRED, FIG = evaluate(store, rowsets)
    J = {
        "task": "E6 FREQ (analysis protocol section 5)",
        "target": TARGET,
        "seed": SEED,
        "quick_mode": bool(args.quick),
        "n_conditions": int(A2.N),
        "excluded_rows_n72": DUP_ROWS,
        "folds": "leave-one-nominal-frequency-level-out (5 folds), labels from targets_design freq_Hz",
        "bands": {"interior": INTERIOR, "edge": EDGE},
        "freq_inputs": {
            "nominal": "targets_design freq_Hz (the label)",
            "measured": "a1_recordings tap_rate_used_Hz (timing only); missing -> nominal",
            "measured_agree": f"tap_rate_used_Hz where |spectral_rate_Hz - tap| / tap <= {AGREE_TOL}, else nominal",
        },
        "rates": {
            "agree_tolerance": AGREE_TOL,
            "n_tap_off_nominal_gt10pct": int((RATES.rel_dev_tap_vs_nominal.abs() > 0.10).sum()),
            "n_missing": int(RATES.missing_tap_rate.sum()),
            "n_disagree": int((~RATES.timing_estimates_agree).sum()),
            "disagree_ids": RATES.condition_id[~RATES.timing_estimates_agree].tolist(),
            "tap_off_nominal_gt10pct_ids": RATES.condition_id[
                RATES.rel_dev_tap_vs_nominal.abs() > 0.10
            ].tolist(),
            "weak_periodicity_ids": RATES.condition_id[
                RATES.flag_weak_periodicity
            ].tolist(),
            "table": RATES.to_dict(orient="records"),
        },
        "salama_shape": SAL,
        "models": {
            "gp": "plain GP (a5_common gp)",
            "lawgp": "LawGP (a5_common physgp)",
            "prod": "ProductGP (a5_common m1, seed 0)",
            "lorspec": "ProductGP with Lorentzian-spectrum kernel in f (a10_methods.predict_lorspec)",
            "tr_lor_mean": "Salama Lorentzian shape, amplitude (a0 + a1 c + a2 c^2) F refitted per fold, no GP",
            "tr_lor_gp": "tr_lor_mean + plain GP on the residual",
            "tr_pow_mean": "Salama power-law shape f^b, amplitude refitted per fold, no GP",
            "tr_pow_gp": "tr_pow_mean + plain GP on the residual",
            "oracle_shape_gp": "LEAKY DIAGNOSTIC: twin/law shape (f0, gamma fitted on all rows incl. the held-out level), amplitude per fold + GP",
            "mf_ar1": "twin low fidelity (in-fold law, in-fold kappa, h(c)) x rho + plain GP (AR(1) multi-fidelity)",
            "mf_nar": "GP on (x, twin low fidelity) (nonlinear autoregressive multi-fidelity)",
        },
        "twin_import": "a6_twin.response and a6_twin.h_factor imported (main block not run); the twin's"
        " full-grid law and kappa are not used because they were fitted on all 75 recordings",
        "results": res,
    }
    if not args.quick:
        J["a10_reproduction_max_abs_diff_R2"] = a10_check(res)
    J["verdict"] = verdict(res)
    tabs = write_tables(out, res) if not args.quick else {}
    if not args.quick and not args.no_round2:
        R2res, RC = run_round2(cache, args.n_jobs)
        J["round2"] = {
            "description": "A folds on measured-rate bands (edges 7.5/12.5/17.5/22.5 Hz; quantile"
            " sensitivity), B clean subset (tap rate within 10 percent of the label, copies dropped),"
            " C per-cycle continuum (instantaneous rate from inter-peak intervals, leave-one-rate-band-out,"
            " training drops every recording with a test cycle), D peak-shape prior (law mean with"
            " training-fold f0, gamma; hierarchical law with f0, gamma pooled across compositions,"
            " kappa by inner leave-one-group-out folds of the training rows)",
            "clean_rows": CLEAN.tolist(),
            "band_edges_Hz": BAND_EDGES,
            "folds": R2res,
            "cycles": RC,
            "verdict": verdict2(res, R2res, RC),
            "verdict_text": "Under measured-rate folds (measured rate as input) the held-out-frequency"
            " collapse becomes zero skill (pooled R2 about 0), not recovered skill; the interior 10 and"
            " 15 Hz levels interpolate only under label folds; the 20 Hz peak and the 5 and 25 Hz edges"
            " are predicted by no method (measured rate, rate-band folds, clean subset, per-cycle"
            " continuum, transfer and twin priors, hierarchical peak prior).",
        }
        tabs.update(write_tables2(out, res, R2res, RC))
    if not args.quick:
        build_macros(res, J)
        if "round2" in J:
            build_macros2(J["round2"]["folds"], J["round2"]["cycles"])
        names = [m[0] for m in MACROS]
        assert len(names) == len(set(names))
        lines = [
            "% numbers_a11.tex -- generated by code/a11_freq.py (E6 FREQ); do not edit"
        ]
        lines += [rf"\newcommand{{\{n}}}{{{v}}}% {k}" for n, v, k in MACROS]
        (out / "numbers_a11.tex").write_text("\n".join(lines) + "\n")
        J["macros"] = {n: {"value": v, "key": k} for n, v, k in MACROS}
    J["index"] = {
        "tables": tabs,
        "macros": "numbers_a11.tex (prefix frq; Excl = n72; Nom/Meas/Agree = frequency input)",
        "a11_predictions.csv": "out-of-fold predictions",
        "a11_figdata_levels.csv": "results.<rowset>.<input>.<model>.per_level",
    }
    J["runtime_s"] = time.time() - T0
    suf = "_quick" if args.quick else ""
    (out / f"a11_freq{suf}.json").write_text(json.dumps(M10.jsonable(J), indent=1))
    pd.DataFrame(PRED).to_csv(out / f"a11_predictions{suf}.csv", index=False)
    pd.DataFrame(FIG).to_csv(out / f"a11_figdata_levels{suf}.csv", index=False)
    v = J["verdict"]["n75"]
    print(
        f"[a11] done in {time.time() - T0:.0f} s | interior positive pooled: "
        f"{v['interior']['positive_band_pooled_R2']} | edge positive pooled: {v['edge']['positive_band_pooled_R2']}"
    )


if __name__ == "__main__":
    main()
