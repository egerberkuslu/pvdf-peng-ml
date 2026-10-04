#!/usr/bin/env python3
"""E2 DYNFORCE: dynamic force recovery by RC deconvolution and an open-circuit sensitivity.

One command (run from the repo root):
    python code/a7_dynforce.py

Physics (project notes). The device is a charge source q(t) = d33 F(t) with capacitance C_p,
loaded by the resistive DAQ input R_in = 144 kOhm. Kirchhoff at the terminal gives
    C_p dV/dt + V / R_in = dq/dt,   so   V(jw) = G(jw) Q(jw),  G = jw R_in / (1 + jw R_in C_p),
and V / V_oc = H(jw) = jw R_in C_p / (1 + jw R_in C_p), a first-order high-pass with corner
1 / (2 pi R_in C_p) (0.14 to 5.8 kHz for C_p 0.19 to 7.7 nF, far above the 5 to 25 Hz drive).

Inversion. G vanishes at w = 0, so q is recovered by frequency-domain Tikhonov regularization
    Q_hat = conj(G) V / (|G|^2 + lambda),  lambda = (2 pi f_lam R_in)^2,
which equals the Wiener filter for white acquisition noise and a white charge prior. Because
|G| ~ w R_in below the corner, the regularizer acts as a high-pass on q with corner f_lam; f_lam
is swept over 0.25 to 4 Hz (below the lowest 5 Hz label). The recorded excursions are one-sample
spikes at 1 kHz, so q is effectively the running integral of V divided by R_in plus C_p V; the
continuous-time G is evaluated on the DFT bins (zero padding to 8192 samples), and the area of
each current pulse is known only through the one or two samples that hit it.

Per tap. Taps are the fixed-period windows of a1_segment (period chosen by choose_period, as in
a1_specimen.py). Inside each window q is linearly detrended; the per-tap amplitude is the
peak-to-peak (primary) and the RMS of the detrended q; the recording amplitude is the mean over
taps. Amplitudes are normalized within each specimen (one specimen per composition, same d33) by
the specimen mean, so the recovered relative dynamic force is A_i / mean_specimen(A).

Tests. Spearman of the normalized amplitude against nominal force per specimen and pooled
(bootstrap CIs), within-cell ordering of the three forces at fixed specimen and frequency
(Monte Carlo permutation null), amplitude ratios 2N:1N and 3N:1N against 2 and 3, and the
within-cell force exponent beta of log A on log F (beta = 1 means proportional to the label).

Models. LOO, held-out force and held-out frequency for the plain GP and LawGP (a2_common
fold_predict, the paper's code path) with (a) the recovered amplitude as the force input
(per recording, which is circular in every scheme because the input comes from the recording
whose descriptor is the target, and a leave-self-out specimen-by-force cell mean, which is
transductive under held-out force because the other recordings of the held-out level are all in
the test fold; valid for LOO and held-out frequency only), and (b) an open-circuit estimate
as the target: scalar V_oc = V / |H(j 2 pi f_label)| and waveform V_oc(t) = q_hat(t) / C_p, with
composition-dependent C_p from literature permittivity (three scenarios) and a uniform 1.9 nF.
Every headline is repeated without recordings 52, 54, 73 (n = 72, duplicated-recordings note in the README).

Writes (only under results/):
    a7_dynforce.json, a7_recordings.csv, a7_figdata_*.csv, numbers_a7.tex (macros prefix dyn),
Force ratios and the exponent carry a cell bootstrap CI and a specimen-cluster bootstrap CI
(5 clusters); the per-specimen range is reported beside the pooled value.
Seeds: 42 for bootstrap and permutation draws; the GP is deterministic (no restarts).
Parallel fold fits use n_jobs = 4 with one BLAS thread each.
"""
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # BLAS threads, set before numpy loads

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_common as A  # noqa: E402
from a1_segment import choose_period, segment  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

T0 = time.time()
ROOT = A.ROOT
REV = A.REV
DATA = os.path.join(REV, "data")
RES = os.path.join(REV, "results")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)

SEED = 42
N_BOOT = 2000
N_PERM = 100000
N_JOBS = 4
FS = 1000.0
R_IN = 144e3  # Ohm, NI USB-6009 input (project notes)
CP_SET = (0.19e-9, 1.9e-9, 7.7e-9)  # F, plausible range (project notes)
CP_PRIMARY = 1.9e-9
FLAM_SET = (0.25, 0.5, 1.0, 2.0, 4.0)  # Hz, Tikhonov corner
FLAM_PRIMARY = 1.0
NPAD = 8192
EPS0 = 8.854e-12
AREA = 12.25e-4  # m^2, electrode area (source study Eq. 3)
THICK = 155e-6  # m, mid of the 140 to 170 um range (source study p. 25484)
THICK_RANGE = (140e-6, 170e-6)
F_MEAN_N = (
    2.0  # mean nominal force of the design (N): scale of the recovered force input
)
DUP_IDS = [52, 54, 73]
FORCES = (1, 2, 3)
FREQS = (5, 10, 15, 20, 25)
COMPS = list(A.COMP_ORDER)

# ----------------------------------------------------------------------------- data
L = pd.read_parquet(os.path.join(DATA, "long.parquet"))
S = pd.read_parquet(os.path.join(DATA, "series_index.parquet")).sort_values("series_id")
D = pd.read_parquet(os.path.join(DATA, "targets_design.parquet")).reset_index(drop=True)
W = np.stack(
    [g.sort_values("time")["voc"].to_numpy(float) for _, g in L.groupby("series_id")]
)
assert W.shape == (75, 1000)
assert (S["material"].to_numpy() == D["composition"].to_numpy()).all()
assert (S["force_newton"].to_numpy() == D["force_N"].to_numpy()).all()
assert (S["freq_hz"].to_numpy() == D["freq_Hz"].to_numpy()).all()
assert (D["composition"].to_numpy() == A.COMP).all() and np.allclose(
    D["force_N"], A.FRC
)
XD = W - W.mean(axis=1, keepdims=True)
N = 75
COMP = D["composition"].to_numpy()
FRC = D["force_N"].to_numpy(float)
FRQ = D["freq_Hz"].to_numpy(float)
ALL = np.arange(N)
KEEP = np.array([i for i in range(N) if i not in DUP_IDS])
SUBSETS = {"all": ALL, "n72": KEEP}
TGT = {t: D[t].to_numpy(float) for t in A.TARGETS}
REC_A1 = pd.read_csv(os.path.join(RES, "a1_recordings.csv"))


def fl(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------- deconvolution
def G_of(w, C):
    return 1j * w * R_IN / (1 + 1j * w * R_IN * C)


def H_of(f, C):
    """|V / V_oc| of the first-order RC high-pass at frequency f (Hz)."""
    x = 2 * np.pi * np.asarray(f, float) * R_IN * C
    return x / np.sqrt(1 + x**2)


def deconv(x, C, flam):
    """Tikhonov (white-noise Wiener) estimate of the charge q(t) in coulomb from V(t) in volt."""
    n = len(x)
    X = np.fft.rfft(x, NPAD)
    w = 2 * np.pi * np.fft.rfftfreq(NPAD, 1.0 / FS)
    G = G_of(w, C)
    lam = (2 * np.pi * flam * R_IN) ** 2
    Q = np.conj(G) * X / (np.abs(G) ** 2 + lam)
    q = np.fft.irfft(Q, NPAD)[:n]
    return q - q.mean()


# ----------------------------------------------------------------------------- segmentation
SEG = []
for i in range(N):
    p, pinfo = choose_period(XD[i], FRQ[i])
    r = segment(XD[i], FRQ[i], period=p)
    SEG.append(
        {
            "windows": r["windows"],
            "n_cycles": r["n_cycles"],
            "tap_rate_used_Hz": pinfo["tap_rate_used_Hz"],
            "period_source": pinfo["period_source"],
        }
    )
seg_check = {
    "n_cycles_equal_a1": int(
        sum(SEG[i]["n_cycles"] == int(REC_A1.loc[i, "n_cycles"]) for i in range(N))
    ),
    "tap_rate_max_abs_diff_Hz": fl(
        max(
            abs(SEG[i]["tap_rate_used_Hz"] - REC_A1.loc[i, "tap_rate_used_Hz"])
            for i in range(N)
        )
    ),
}
print(
    f"[a7] segmentation reproduces a1 in {seg_check['n_cycles_equal_a1']}/75 recordings"
)


def tap_amps(q, windows):
    """Mean over taps of the detrended per-tap peak-to-peak and RMS of q."""
    pp, rms = [], []
    for w_ in windows:
        seg_ = q[w_["start"] : w_["end"]]
        t = np.arange(len(seg_))
        if len(seg_) >= 3:
            seg_ = seg_ - np.polyval(np.polyfit(t, seg_, 1), t)
        pp.append(np.ptp(seg_))
        rms.append(np.sqrt(np.mean(seg_**2)))
    pp, rms = np.asarray(pp), np.asarray(rms)
    return (
        float(pp.mean()),
        float(rms.mean()),
        float(pp.std(ddof=1) / pp.mean()) if len(pp) > 1 else np.nan,
    )


AMP = {}  # (C, flam) -> dict(pp=array, rms=array, cv=array)
for C in CP_SET:
    for flam in FLAM_SET:
        out = np.array(
            [tap_amps(deconv(XD[i], C, flam), SEG[i]["windows"]) for i in range(N)]
        )
        AMP[(C, flam)] = {"pp": out[:, 0], "rms": out[:, 1], "cv": out[:, 2]}
# the recorded voltage with the same tap segmentation (reference for "does deconvolution change anything")
_v = np.array([tap_amps(XD[i], SEG[i]["windows"]) for i in range(N)])
AMP_V = {"pp": _v[:, 0], "rms": _v[:, 1]}
PRIM = AMP[(CP_PRIMARY, FLAM_PRIMARY)]
print(f"[a7] deconvolution done ({time.time() - T0:.1f} s)")


# ----------------------------------------------------------------------------- ordering statistics
def normalize(a, idx):
    """a / specimen mean over rows idx (NaN outside idx)."""
    out = np.full(N, np.nan)
    for c in COMPS:
        m = idx[COMP[idx] == c]
        out[m] = a[m] / a[m].mean()
    return out


def cells(idx):
    """(composition, freq) -> {force: row} over rows idx."""
    out = {}
    for i in idx:
        out.setdefault((COMP[i], FRQ[i]), {})[int(FRC[i])] = i
    return out


def within_beta(an, cl):
    """Within-cell slope of log(an) on log(F) over a list of cell dicts."""
    num = den = 0.0
    for c in cl:
        rows = list(c.values())
        if len(rows) < 2:
            continue
        lx = np.log(FRC[rows])
        ly = np.log(an[rows])
        num += np.sum((lx - lx.mean()) * (ly - ly.mean()))
        den += np.sum((lx - lx.mean()) ** 2)
    return num / den if den > 0 else np.nan


def cell_ratios(an, cl):
    r21, r31 = [], []
    for c in cl:
        if 1 in c and 2 in c:
            r21.append(an[c[2]] / an[c[1]])
        if 1 in c and 3 in c:
            r31.append(an[c[3]] / an[c[1]])
    return np.asarray(r21), np.asarray(r31)


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y)[0])


def ci(draws):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return (
        [fl(np.percentile(d, 2.5)), fl(np.percentile(d, 97.5))]
        if len(d)
        else [None, None]
    )


def cell_rho(an, c):
    fs = sorted(c)
    return spearman(np.array(fs, float), an[[c[k] for k in fs]])


# Monte Carlo null for the mean within-cell Spearman over k complete cells (orderings uniform)
_RHO3 = np.array(
    [
        spearman(np.array([1.0, 2, 3]), np.array(p, float))
        for p in [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    ]
)


def perm_p_mean_rho(obs, k, rng):
    null = _RHO3[rng.integers(0, 6, size=(N_PERM, k))].mean(axis=1)
    return float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))


def ordering_stats(a, idx, boot=True, seed=SEED):
    """Ordering and scaling of amplitude a (raw, per recording) against the nominal force."""
    rng = np.random.default_rng(seed)
    an = normalize(a, idx)
    cl_all = cells(idx)
    res = {"per_specimen": {}, "pooled": {}}
    for c in COMPS:
        rows = idx[COMP[idx] == c]
        cl = [v for (cc, _), v in cl_all.items() if cc == c]
        full = [v for v in cl if len(v) == 3]
        r21, r31 = cell_ratios(an, cl)
        d = {
            "n": int(len(rows)),
            "spearman": fl(spearman(FRC[rows], an[rows])),
            "n_cells_complete": len(full),
            "n_cells_concordant": int(
                sum(an[v[1]] < an[v[2]] < an[v[3]] for v in full)
            ),
            "mean_within_cell_rho": fl(np.mean([cell_rho(an, v) for v in full]))
            if full
            else None,
            "ratio_2_1_gmean": fl(gmean(r21)),
            "ratio_3_1_gmean": fl(gmean(r31)),
            "ratio_2_1_cells": [fl(x) for x in r21],
            "ratio_3_1_cells": [fl(x) for x in r31],
            "beta": fl(within_beta(an, cl)),
            "mean_by_force": {
                str(k): fl(np.mean(an[rows][FRC[rows] == k])) for k in FORCES
            },
        }
        if boot:
            bs_rho, bs_b, bs21, bs31 = [], [], [], []
            for _ in range(N_BOOT):
                s = rng.choice(rows, len(rows), replace=True)
                bs_rho.append(spearman(FRC[s], an[s]))
                cs = [cl[j] for j in rng.integers(0, len(cl), len(cl))]
                bs_b.append(within_beta(an, cs))
                x21, x31 = cell_ratios(an, cs)
                bs21.append(gmean(x21))
                bs31.append(gmean(x31))
            d["spearman_ci"], d["beta_ci"] = ci(bs_rho), ci(bs_b)
            d["ratio_2_1_ci"], d["ratio_3_1_ci"] = ci(bs21), ci(bs31)
        res["per_specimen"][c] = d
    cl = list(cl_all.values())
    full = [v for v in cl if len(v) == 3]
    r21, r31 = cell_ratios(an, cl)
    rhos = [cell_rho(an, v) for v in full]
    k_conc = int(sum(an[v[1]] < an[v[2]] < an[v[3]] for v in full))
    P = res["pooled"]
    P.update(
        {
            "n": int(len(idx)),
            "spearman": fl(spearman(FRC[idx], an[idx])),
            "n_cells_complete": len(full),
            "n_cells_concordant": k_conc,
            "concordant_binom_p_vs_one_sixth": fl(
                stats.binomtest(k_conc, len(full), 1 / 6, alternative="greater").pvalue
            ),
            "mean_within_cell_rho": fl(np.mean(rhos)),
            "mean_within_cell_rho_perm_p": perm_p_mean_rho(
                np.mean(rhos), len(full), rng
            ),
            "n_cells_rho_positive": int(np.sum(np.array(rhos) > 0)),
            "ratio_2_1_gmean": fl(gmean(r21)),
            "ratio_3_1_gmean": fl(gmean(r31)),
            "ratio_2_1_median": fl(np.median(r21)),
            "ratio_3_1_median": fl(np.median(r31)),
            "n_cells_ratio_3_1_below_1": int(np.sum(r31 < 1)),
            "n_cells_ratio_3_1_in_2_to_4": int(np.sum((r31 >= 2) & (r31 <= 4))),
            "beta": fl(within_beta(an, cl)),
            "mean_by_force": {
                str(k): fl(np.mean(an[idx][FRC[idx] == k])) for k in FORCES
            },
            "mean_by_freq": {
                str(k): fl(np.mean(an[idx][FRQ[idx] == k])) for k in FREQS
            },
        }
    )
    if boot:
        bs_rho, bs_rho_cl, bs_b, bs_b_cl, bs21, bs31 = [], [], [], [], [], []
        bs21_cl, bs31_cl = [], []
        by_c = {c: idx[COMP[idx] == c] for c in COMPS}
        cl_by_c = {c: [v for (cc, _), v in cl_all.items() if cc == c] for c in COMPS}
        for _ in range(N_BOOT):
            # stratified by specimen: resample recordings inside each specimen
            s = np.concatenate(
                [rng.choice(by_c[c], len(by_c[c]), replace=True) for c in COMPS]
            )
            bs_rho.append(spearman(FRC[s], an[s]))
            cs = [cl[j] for j in rng.integers(0, len(cl), len(cl))]
            bs_b.append(within_beta(an, cs))
            x21, x31 = cell_ratios(an, cs)
            bs21.append(gmean(x21))
            bs31.append(gmean(x31))
            # specimen-cluster bootstrap (5 clusters)
            cc = rng.choice(COMPS, len(COMPS), replace=True)
            s2 = np.concatenate([by_c[c] for c in cc])
            bs_rho_cl.append(spearman(FRC[s2], an[s2]))
            cs_cl = [v for c in cc for v in cl_by_c[c]]
            bs_b_cl.append(within_beta(an, cs_cl))
            x21, x31 = cell_ratios(an, cs_cl)
            bs21_cl.append(gmean(x21))
            bs31_cl.append(gmean(x31))
        P["spearman_ci"] = ci(bs_rho)
        P["spearman_ci_specimen_cluster"] = ci(bs_rho_cl)
        P["beta_ci"] = ci(bs_b)
        P["beta_ci_specimen_cluster"] = ci(bs_b_cl)
        P["ratio_2_1_ci"], P["ratio_3_1_ci"] = ci(bs21), ci(bs31)
        P["ratio_2_1_ci_specimen_cluster"] = ci(bs21_cl)
        P["ratio_3_1_ci_specimen_cluster"] = ci(bs31_cl)
    # spread between specimens behind the pooled value
    PS = res["per_specimen"]
    for k in ("ratio_2_1_gmean", "ratio_3_1_gmean", "beta"):
        v = [PS[c][k] for c in COMPS if PS[c][k] is not None]
        P[k.replace("_gmean", "") + "_specimen_range"] = [fl(min(v)), fl(max(v))]
    return res, an


ORD, ORDN = {}, {}
for sub, idx in SUBSETS.items():
    ORD[sub], ORDN[sub] = {}, {}
    ORD[sub]["q_pp"], ORDN[sub]["q_pp"] = ordering_stats(PRIM["pp"], idx)
    ORD[sub]["q_rms"], ORDN[sub]["q_rms"] = ordering_stats(PRIM["rms"], idx)
    ORD[sub]["V_rms"], ORDN[sub]["V_rms"] = ordering_stats(TGT["rms_Voc"], idx)
    ORD[sub]["V_pp"], ORDN[sub]["V_pp"] = ordering_stats(TGT["Vpp"], idx)
print(f"[a7] ordering statistics done ({time.time() - T0:.1f} s)")

# regularization and C_p sensitivity (point estimates; primary CIs above)
SENS = []
for sub, idx in SUBSETS.items():
    for C in CP_SET:
        for flam in FLAM_SET:
            for kind in ("pp", "rms"):
                a = AMP[(C, flam)][kind]
                o, _ = ordering_stats(a, idx, boot=False)
                p = o["pooled"]
                SENS.append(
                    {
                        "subset": sub,
                        "C_p_nF": C * 1e9,
                        "f_lam_Hz": flam,
                        "amplitude": kind,
                        "spearman_pooled": p["spearman"],
                        "mean_within_cell_rho": p["mean_within_cell_rho"],
                        "n_cells_concordant": p["n_cells_concordant"],
                        "n_cells_complete": p["n_cells_complete"],
                        "ratio_2_1_gmean": p["ratio_2_1_gmean"],
                        "ratio_3_1_gmean": p["ratio_3_1_gmean"],
                        "beta": p["beta"],
                        "spearman_vs_primary_amplitude": fl(
                            spearman(a[idx], PRIM[kind][idx])
                        ),
                        "spearman_vs_Vrms": fl(spearman(a[idx], TGT["rms_Voc"][idx])),
                        "median_tap_cv": fl(np.nanmedian(AMP[(C, flam)]["cv"][idx])),
                    }
                )
SENS = pd.DataFrame(SENS)
print(f"[a7] sensitivity grid done ({time.time() - T0:.1f} s)")

# ----------------------------------------------------------------------------- permittivity scenarios
# relative permittivity ratios to pristine PVDF read from Fig. 7 of the source study at 1 kHz
# (digitized; the text states the 3 wt% ratio 1.57). The absolute values in that figure are
# below 1, which is not physical for a dielectric, so only the ratios are used.
SRC_RATIO = {
    "PVDF": 1.0,
    "PVDF+BaTiO3": 0.66,
    "PVDF+BaTiO3+%1CNT": 0.63,
    "PVDF+BaTiO3+%2CNT": 0.98,
    "PVDF+BaTiO3+%3CNT": 1.57,
}
EPS_SCEN = {
    "mat": {
        "label": "electrospun mat",
        "eps": {c: 2.0 * SRC_RATIO[c] for c in COMPS},
        "basis": "pristine electrospun PVDF mat eps_r = 2.0 (1.1 to 2.2 at 1 kHz for fiber mats, castkovaStructurePropertiesRelationship2020, Table 4) times the source-study composition ratios (kocPiezoelectricNanogeneratorsBased2025, Fig. 7)",
    },
    "dense": {
        "label": "dense film",
        "eps": {c: 14.0 * SRC_RATIO[c] for c in COMPS},
        "basis": "dense alpha-PVDF eps_r about 14 (castkovaStructurePropertiesRelationship2020) times the source-study composition ratios",
    },
    "lit": {
        "label": "composite literature",
        "eps": {
            "PVDF": 14.0,
            "PVDF+BaTiO3": 21.0,
            "PVDF+BaTiO3+%1CNT": 59.0,
            "PVDF+BaTiO3+%2CNT": 300.0,
            "PVDF+BaTiO3+%3CNT": 300.0,
        },
        "basis": "dense composites from the literature: PVDF 14 (castkova2020); +15 vol% BaTiO3 +50% (bouharrasDielectricCharacterizationCoreShell2023); BaTiO3/MWNT/PVDF 11.5/0.35/88.15 vol% eps_r 59 (zhaoBaTiO3MWNTsPolyvinylidene2019) for 1 wt% CNT; MWNT/PVDF near 2 vol% eps_r about 300 (wangCarbonNanotubeComposites2005) for 2 and 3 wt% CNT (about 2.5 and 3.8 vol% at 15 vol% BaTiO3), which lie above the percolation thresholds reported there (dangGiantDielectricPermittivities2007)",
    },
}
CP_UNIFORM = CP_PRIMARY


def cp_of(eps):
    return EPS0 * eps * AREA / THICK


CP_TAB = []
for k, sc in EPS_SCEN.items():
    sc["C_p_F"] = {c: cp_of(sc["eps"][c]) for c in COMPS}
    for c in COMPS:
        C = sc["C_p_F"][c]
        CP_TAB.append(
            {
                "scenario": k,
                "composition": c,
                "eps_r": sc["eps"][c],
                "C_p_nF": C * 1e9,
                "C_p_nF_t140": EPS0 * sc["eps"][c] * AREA / THICK_RANGE[0] * 1e9,
                "C_p_nF_t170": EPS0 * sc["eps"][c] * AREA / THICK_RANGE[1] * 1e9,
                "corner_Hz": 1 / (2 * np.pi * R_IN * C),
                "H_5Hz": float(H_of(5, C)),
                "H_25Hz": float(H_of(25, C)),
            }
        )
CP_TAB = pd.DataFrame(CP_TAB)


def cp_vec(scen):
    if scen == "uniform":
        return np.full(N, CP_UNIFORM)
    return np.array([EPS_SCEN[scen]["C_p_F"][c] for c in COMP])


VOC_SCAL, VOC_WAVE = {}, {}
for scen in ["uniform", "mat", "dense", "lit"]:
    Cv = cp_vec(scen)
    Hn = H_of(FRQ, Cv)
    VOC_SCAL[scen] = {t: TGT[t] / Hn for t in A.TARGETS}
    vw = np.array([deconv(XD[i], Cv[i], FLAM_PRIMARY) / Cv[i] for i in range(N)])
    vw = vw - vw.mean(axis=1, keepdims=True)
    VOC_WAVE[scen] = {
        "rms_Voc": np.sqrt(np.mean(vw**2, axis=1)),
        "Vpp": np.ptp(vw, axis=1),
        "Vmax": np.abs(vw).max(axis=1),
    }

# ----------------------------------------------------------------------------- GP / LawGP reruns
FORCE_COL = A.FEAT.index("force_N")


def force_inputs(a, idx):
    """Recovered relative force on the nominal scale: per recording and leave-self-out cell mean.

    Both inputs are built once over all rows of the subset, outside the folds. The per-recording
    input is circular in every scheme. The cell mean is transductive under held-out force: it
    averages the other recordings of the held-out level (all in the test fold) and normalize()
    averages over test rows; it is valid for LOO and held-out frequency only.
    """
    an = normalize(a, idx)
    rec = F_MEAN_N * an
    cell = np.full(N, np.nan)
    for i in idx:
        m = idx[(COMP[idx] == COMP[i]) & (FRC[idx] == FRC[i]) & (idx != i)]
        cell[i] = F_MEAN_N * an[m].mean()
    return rec, cell


SCEN = (
    {}
)  # name -> dict(label, kind, per-subset {"X": (N,4), "F": (N,), "y": {t: (N,)}})
for sub, idx in SUBSETS.items():
    base = {"X": A.X.copy(), "F": A.FRC.copy(), "y": dict(TGT)}
    SCEN.setdefault(
        "label", {"label": "nominal label (paper)", "kind": "baseline", "sub": {}}
    )["sub"][sub] = base
    for C in CP_SET:
        rec, cell = force_inputs(AMP[(C, FLAM_PRIMARY)]["pp"], idx)
        tag = {0.19e-9: "Lo", 1.9e-9: "", 7.7e-9: "Hi"}[C]
        for nm, fv, lab in (
            ("frec", rec, "recovered force, per recording"),
            ("fcell", cell, "recovered force, leave-self-out cell mean"),
        ):
            if tag and nm == "fcell":
                continue
            Xs = A.X.copy()
            Xs[:, FORCE_COL] = np.where(np.isfinite(fv), fv, A.FRC)
            key = nm + tag
            SCEN.setdefault(
                key,
                {
                    "label": f"{lab}, C_p {C * 1e9:g} nF",
                    "kind": "force_input",
                    "sub": {},
                },
            )["sub"][sub] = {"X": Xs, "F": Xs[:, FORCE_COL].copy(), "y": dict(TGT)}
    for scen in ["uniform", "mat", "dense", "lit"]:
        lab = EPS_SCEN[scen]["label"] if scen != "uniform" else "uniform 1.9 nF"
        SCEN.setdefault(
            f"ocs_{scen}",
            {"label": f"scalar V_oc, {lab}", "kind": "voc_scalar", "sub": {}},
        )["sub"][sub] = {"X": A.X.copy(), "F": A.FRC.copy(), "y": VOC_SCAL[scen]}
        SCEN.setdefault(
            f"ocw_{scen}",
            {"label": f"waveform V_oc, {lab}", "kind": "voc_wave", "sub": {}},
        )["sub"][sub] = {"X": A.X.copy(), "F": A.FRC.copy(), "y": VOC_WAVE[scen]}

MODELS = ["gp", "physgp"]
MODEL_LABEL = {"gp": "GP", "physgp": "LawGP"}
SCHEMES = ["loo", "force", "frequency"]


def folds_for(scheme, idx):
    if scheme == "loo":
        return [(tr, te, str(int(te[0]))) for tr, te in A.loo_folds(idx)]
    return [
        (tr, te, A.level_key(scheme, v)) for tr, te, v in A.group_folds(scheme, idx=idx)
    ]


def run_fold(model, Xm, Fv, y, tr, te):
    sys.path.insert(0, HERE)
    import a2_common as AA

    X0, F0 = AA.X, AA.FRC
    AA.X, AA.FRC = Xm, Fv
    try:
        mu, sd, _ = AA.fold_predict(model, tr, te, y)
    finally:
        AA.X, AA.FRC = X0, F0
    return mu


JOBS, KEYS = [], []
for sname, sc in SCEN.items():
    for sub, idx in SUBSETS.items():
        dat = sc["sub"][sub]
        for t in A.TARGETS:
            for m in MODELS:
                for scheme in SCHEMES:
                    for tr, te, lv in folds_for(scheme, idx):
                        KEYS.append((sname, sub, t, m, scheme, lv, te))
                        JOBS.append(
                            delayed(run_fold)(
                                m, dat["X"], dat["F"], dat["y"][t], tr, te
                            )
                        )
print(f"[a7] running {len(JOBS)} fold fits")
OUT = Parallel(n_jobs=N_JOBS, batch_size=8)(JOBS)
print(f"[a7] fold fits done ({time.time() - T0:.1f} s)")

OOF = {}
for (sname, sub, t, m, scheme, lv, te), mu in zip(KEYS, OUT):
    arr = OOF.setdefault((sname, sub, t, m, scheme), np.full(N, np.nan))
    arr[te] = mu


def lvl_r2(y, mu):
    return fl(r2_score(y, mu))


GPR = {}  # sname -> sub -> t -> model -> scheme -> {"pooled_R2", "levels": {lv: R2}}
for (sname, sub, t, m, scheme), mu in OOF.items():
    idx = SUBSETS[sub]
    y = SCEN[sname]["sub"][sub]["y"][t]
    d = {
        "pooled_R2": fl(r2_score(y[idx], mu[idx])),
        "pooled_MAE": fl(mean_absolute_error(y[idx], mu[idx])),
    }
    if scheme != "loo":
        lv = {}
        for tr, te, k in folds_for(scheme, idx):
            lv[k] = lvl_r2(y[te], mu[te])
        d["levels"] = lv
        d["n_levels_negative"] = int(sum(v < 0 for v in lv.values()))
        d["mean_within_level_R2"] = fl(np.mean(list(lv.values())))
    GPR.setdefault(sname, {}).setdefault(sub, {}).setdefault(t, {}).setdefault(m, {})[
        scheme
    ] = d

# paper values (a2_physgp.json is the source of numbers_a2.tex; LOO GP equals \benchLoo*Gp of numbers_a3.tex)
A2 = json.load(open(os.path.join(RES, "a2_physgp.json")))
PAPER = {"all": {}, "n72": {}}
for t in A.TARGETS:
    PAPER["all"][t], PAPER["n72"][t] = {}, {}
    for m in MODELS:
        PAPER["all"][t][m] = {"loo": {"pooled_R2": A2["pooled"][t]["loo"][m]["R2"]}}
        PAPER["n72"][t][m] = {
            "loo": {
                "pooled_R2": A2["sensitivity_excluding_duplicates"]["loo"][t][m]["R2"]
            }
        }
        for ax in ("force", "frequency"):
            PAPER["all"][t][m][ax] = {
                "pooled_R2": A2["pooled"][t][ax][m]["R2"],
                "levels": {k: v["R2"] for k, v in A2["per_level"][t][ax][m].items()},
            }
            PAPER["n72"][t][m][ax] = {
                "pooled_R2": A2["sensitivity_excluding_duplicates"]["pooled"][t][ax][m][
                    "R2"
                ],
                "levels": {
                    k: v["R2"]
                    for k, v in A2["sensitivity_excluding_duplicates"]["per_level"][t][
                        ax
                    ][m].items()
                },
            }
# the label scenario must reproduce the paper
repro = []
for sub in SUBSETS:
    for t in A.TARGETS:
        for m in MODELS:
            for sch in SCHEMES:
                repro.append(
                    abs(
                        GPR["label"][sub][t][m][sch]["pooled_R2"]
                        - PAPER[sub][t][m][sch]["pooled_R2"]
                    )
                )
                if sch != "loo":
                    for k, v in PAPER[sub][t][m][sch]["levels"].items():
                        repro.append(abs(GPR["label"][sub][t][m][sch]["levels"][k] - v))
REPRO_MAX = float(np.max(repro))
print(f"[a7] label scenario reproduces a2: max |dR2| = {REPRO_MAX:.2e}")


# ----------------------------------------------------------------------------- conclusions
K1_TOL = 0.05  # LOO R2 drop of the plain GP that counts as a change of the interpolation claim


def conclusions(g, ref):
    """Paper conclusions evaluated on one scenario result g[t][m][scheme]; ref = label scenario, same subset."""
    out = {}
    for t in A.TARGETS:
        r = g[t]
        out[t] = {
            "K1_gp_loo_holds": r["gp"]["loo"]["pooled_R2"]
            >= ref[t]["gp"]["loo"]["pooled_R2"] - K1_TOL,
            "K2_gp_beats_lawgp_loo": r["gp"]["loo"]["pooled_R2"]
            > r["physgp"]["loo"]["pooled_R2"],
            "K3_lawgp_beats_gp_held_out_force": r["physgp"]["force"]["pooled_R2"]
            > r["gp"]["force"]["pooled_R2"],
            "K4_lawgp_beats_gp_at_3N": r["physgp"]["force"]["levels"]["3"]
            > r["gp"]["force"]["levels"]["3"],
            "K5_held_out_frequency_negative_both": max(
                r["gp"]["frequency"]["pooled_R2"], r["physgp"]["frequency"]["pooled_R2"]
            )
            < 0,
        }
    return out


K_LABEL = {
    "K1_gp_loo_holds": "plain GP LOO $R^2$ within 0.05 of the nominal-label value",
    "K2_gp_beats_lawgp_loo": "plain GP beats LawGP within the grid (LOO)",
    "K3_lawgp_beats_gp_held_out_force": "LawGP beats plain GP at a held-out force (pooled)",
    "K4_lawgp_beats_gp_at_3N": "LawGP beats plain GP at held-out \\SI{3}{\\newton}",
    "K5_held_out_frequency_negative_both": "held-out frequency negative for both models",
}
CONC = {s: {sub: conclusions(GPR[s][sub], GPR["label"][sub]) for sub in SUBSETS} for s in SCEN}
CHANGES = {}
for s in SCEN:
    CHANGES[s] = {}
    for sub in SUBSETS:
        ch = []
        for t in A.TARGETS:
            for k, v in CONC[s][sub][t].items():
                if v != CONC["label"][sub][t][k]:
                    ch.append(f"{t}:{k}:{CONC['label'][sub][t][k]}->{v}")
        CHANGES[s][sub] = ch

# ----------------------------------------------------------------------------- outputs: csv
rec = pd.DataFrame(
    {
        "condition_id": ALL,
        "composition": COMP,
        "cnt_pct": D["cnt_pct"],
        "force_N": FRC,
        "freq_Hz": FRQ,
        "duplicate_excluded_in_n72": [i in DUP_IDS for i in ALL],
        "n_taps": [SEG[i]["n_cycles"] for i in ALL],
        "tap_rate_used_Hz": [SEG[i]["tap_rate_used_Hz"] for i in ALL],
        "period_source": [SEG[i]["period_source"] for i in ALL],
        "rms_Voc": TGT["rms_Voc"],
        "Vpp": TGT["Vpp"],
        "Vmax": TGT["Vmax"],
    }
)
for C in CP_SET:
    nm = f"{C * 1e9:g}nF".replace(".", "p")
    a = AMP[(C, FLAM_PRIMARY)]
    rec[f"q_pp_pC_{nm}"] = a["pp"] * 1e12
    rec[f"q_rms_pC_{nm}"] = a["rms"] * 1e12
    rec[f"q_pp_tap_cv_{nm}"] = a["cv"]
    rec[f"q_pp_norm_{nm}"] = normalize(a["pp"], ALL)
    rec[f"q_rms_norm_{nm}"] = normalize(a["rms"], ALL)
    rec[f"F_rec_N_{nm}"] = force_inputs(a["pp"], ALL)[0]
rec["F_cell_N_1p9nF"] = force_inputs(PRIM["pp"], ALL)[1]
rec["q_pp_norm_1p9nF_n72"] = normalize(PRIM["pp"], KEEP)
rec["Vrms_norm"] = normalize(TGT["rms_Voc"], ALL)
for scen in ["uniform", "mat", "dense", "lit"]:
    rec[f"C_p_nF_{scen}"] = cp_vec(scen) * 1e9
    for t in A.TARGETS:
        rec[f"Voc_scalar_{scen}_{t}"] = VOC_SCAL[scen][t]
        rec[f"Voc_wave_{scen}_{t}"] = VOC_WAVE[scen][t]
rec.to_csv(os.path.join(RES, "a7_recordings.csv"), index=False)

fig_amp = rec[
    [
        "condition_id",
        "composition",
        "force_N",
        "freq_Hz",
        "duplicate_excluded_in_n72",
        "q_pp_norm_1p9nF",
        "q_rms_norm_1p9nF",
        "Vrms_norm",
    ]
]
fig_amp.to_csv(os.path.join(RES, "a7_figdata_amplitude.csv"), index=False)

rows = []
an = normalize(PRIM["pp"], ALL)
av = normalize(TGT["rms_Voc"], ALL)
for (c, f_), cd in cells(ALL).items():
    rows.append(
        {
            "composition": c,
            "freq_Hz": f_,
            "ratio_2_1_q": an[cd[2]] / an[cd[1]],
            "ratio_3_1_q": an[cd[3]] / an[cd[1]],
            "ratio_2_1_Vrms": av[cd[2]] / av[cd[1]],
            "ratio_3_1_Vrms": av[cd[3]] / av[cd[1]],
            "contains_duplicate": any(cd[k] in DUP_IDS for k in cd),
        }
    )
pd.DataFrame(rows).to_csv(os.path.join(RES, "a7_figdata_ratios.csv"), index=False)

wf = {"t_s": np.arange(1000) / FS}
for i in (58, 40, 0):
    wf[f"V_{i}"] = XD[i]
    for C in CP_SET:
        wf[f"q_pC_{i}_{C * 1e9:g}nF"] = deconv(XD[i], C, FLAM_PRIMARY) * 1e12
    for flam in FLAM_SET:
        wf[f"q_pC_{i}_flam{flam:g}Hz"] = deconv(XD[i], CP_PRIMARY, flam) * 1e12
pd.DataFrame(wf).to_csv(os.path.join(RES, "a7_figdata_waveforms.csv"), index=False)
SENS.to_csv(os.path.join(RES, "a7_figdata_regularization.csv"), index=False)
CP_TAB.to_csv(os.path.join(RES, "a7_figdata_cp.csv"), index=False)

r2rows = []
for s in SCEN:
    for sub in SUBSETS:
        for t in A.TARGETS:
            for m in MODELS:
                for sch in SCHEMES:
                    d = GPR[s][sub][t][m][sch]
                    r2rows.append(
                        {
                            "scenario": s,
                            "subset": sub,
                            "target": t,
                            "model": MODEL_LABEL[m],
                            "scheme": sch,
                            "level": "pooled",
                            "R2": d["pooled_R2"],
                            "paper_R2": PAPER[sub][t][m][sch]["pooled_R2"],
                        }
                    )
                    for k, v in d.get("levels", {}).items():
                        r2rows.append(
                            {
                                "scenario": s,
                                "subset": sub,
                                "target": t,
                                "model": MODEL_LABEL[m],
                                "scheme": sch,
                                "level": k,
                                "R2": v,
                                "paper_R2": PAPER[sub][t][m][sch]["levels"][k],
                            }
                        )
pd.DataFrame(r2rows).to_csv(os.path.join(RES, "a7_figdata_r2.csv"), index=False)

# ----------------------------------------------------------------------------- macros
MAC = []
INDEX = {}
WORD = {
    1: "One",
    2: "Two",
    3: "Three",
    5: "Five",
    10: "Ten",
    15: "Fifteen",
    20: "Twenty",
    25: "TwentyFive",
}
TW = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}
SW = {
    "label": "Label",
    "frec": "Rec",
    "fcell": "Cell",
    "frecLo": "RecLo",
    "frecHi": "RecHi",
    "ocs_uniform": "OcScalUni",
    "ocs_mat": "OcScalMat",
    "ocs_dense": "OcScalDense",
    "ocs_lit": "OcScalLit",
    "ocw_uniform": "OcWaveUni",
    "ocw_mat": "OcWaveMat",
    "ocw_dense": "OcWaveDense",
    "ocw_lit": "OcWaveLit",
}
MW = {"gp": "GP", "physgp": "LawGP"}
SCHW = {"loo": "Loo", "force": "Force", "frequency": "Freq"}


def num(x, nd):
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        if float(s) == 0:
            s = s[1:]
        else:
            return r"\ensuremath{-}" + s[1:]
    return s


def mac(name, value, comment, key=None):
    assert re.fullmatch(r"dyn[A-Za-z]+", name), name
    MAC.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {comment}")
    if key:
        INDEX[name] = key


mac("dynRin", r"\SI{144}{\kilo\ohm}", "DAQ input resistance")
for C, nm in zip(CP_SET, ("Lo", "Mid", "Hi")):
    mac(f"dynCp{nm}", f"{C * 1e9:g}", "C_p (nF) of the deconvolution sweep")
    mac(f"dynCorner{nm}", f"{1 / (2 * np.pi * R_IN * C) / 1e3:.2f}", "RC corner (kHz)")
mac("dynFlamPrimary", f"{FLAM_PRIMARY:g}", "Tikhonov corner f_lam (Hz), primary")
mac("dynFlamMin", f"{min(FLAM_SET):g}", "Tikhonov corner sweep minimum (Hz)")
mac("dynFlamMax", f"{max(FLAM_SET):g}", "Tikhonov corner sweep maximum (Hz)")
mac("dynNBoot", str(N_BOOT), "bootstrap draws")
mac(
    "dynSegAgree",
    str(seg_check["n_cycles_equal_a1"]),
    "recordings whose tap count equals a1 (of 75)",
)

for sub, SUF in (("all", ""), ("n72", "Dup")):
    for amp_key, AW in (
        ("q_pp", "Q"),
        ("q_rms", "QRms"),
        ("V_rms", "Vrec"),
        ("V_pp", "VrecPp"),
    ):
        P = ORD[sub][amp_key]["pooled"]
        base = f"ordering.{sub}.{amp_key}.pooled"
        mac(
            f"dyn{SUF}Spearman{AW}",
            num(P["spearman"], 2),
            f"pooled Spearman, {amp_key}, {sub}",
            base + ".spearman",
        )
        mac(
            f"dyn{SUF}Spearman{AW}Lo",
            num(P["spearman_ci"][0], 2),
            "95% CI low (stratified bootstrap)",
        )
        mac(f"dyn{SUF}Spearman{AW}Hi", num(P["spearman_ci"][1], 2), "95% CI high")
        mac(
            f"dyn{SUF}Spearman{AW}ClLo",
            num(P["spearman_ci_specimen_cluster"][0], 2),
            "95% CI low (specimen cluster)",
        )
        mac(
            f"dyn{SUF}Spearman{AW}ClHi",
            num(P["spearman_ci_specimen_cluster"][1], 2),
            "95% CI high (specimen cluster)",
        )
        mac(
            f"dyn{SUF}CellRho{AW}",
            num(P["mean_within_cell_rho"], 2),
            "mean within-cell Spearman",
            base + ".mean_within_cell_rho",
        )
        mac(
            f"dyn{SUF}CellRho{AW}P",
            num(P["mean_within_cell_rho_perm_p"], 4)
            if P["mean_within_cell_rho_perm_p"] >= 1e-4
            else r"$<10^{-4}$",
            "permutation p",
        )
        mac(
            f"dyn{SUF}CellsConc{AW}",
            str(P["n_cells_concordant"]),
            "cells ordered 1N<2N<3N",
            base + ".n_cells_concordant",
        )
        mac(
            f"dyn{SUF}CellsTotal{AW}",
            str(P["n_cells_complete"]),
            "complete specimen-by-frequency cells",
        )
        mac(
            f"dyn{SUF}CellsRhoPos{AW}",
            str(P["n_cells_rho_positive"]),
            "cells with positive within-cell rho",
        )
        mac(
            f"dyn{SUF}RatioTwo{AW}",
            num(P["ratio_2_1_gmean"], 2),
            "geometric-mean ratio 2N:1N",
            base + ".ratio_2_1_gmean",
        )
        mac(
            f"dyn{SUF}RatioTwo{AW}Lo",
            num(P["ratio_2_1_ci"][0], 2),
            "95% CI low (cell bootstrap)",
        )
        mac(f"dyn{SUF}RatioTwo{AW}Hi", num(P["ratio_2_1_ci"][1], 2), "95% CI high")
        mac(
            f"dyn{SUF}RatioThree{AW}",
            num(P["ratio_3_1_gmean"], 2),
            "geometric-mean ratio 3N:1N",
            base + ".ratio_3_1_gmean",
        )
        mac(f"dyn{SUF}RatioThree{AW}Lo", num(P["ratio_3_1_ci"][0], 2), "95% CI low")
        mac(f"dyn{SUF}RatioThree{AW}Hi", num(P["ratio_3_1_ci"][1], 2), "95% CI high")
        for rk, RW in (("ratio_2_1", "RatioTwo"), ("ratio_3_1", "RatioThree")):
            mac(
                f"dyn{SUF}{RW}{AW}ClLo",
                num(P[rk + "_ci_specimen_cluster"][0], 2),
                "95% CI low (specimen-cluster bootstrap, 5 clusters)",
                base + f".{rk}_ci_specimen_cluster.0",
            )
            mac(
                f"dyn{SUF}{RW}{AW}ClHi",
                num(P[rk + "_ci_specimen_cluster"][1], 2),
                "95% CI high (specimen-cluster bootstrap, 5 clusters)",
                base + f".{rk}_ci_specimen_cluster.1",
            )
        for rk, RW in (("ratio_2_1", "RatioTwo"), ("ratio_3_1", "RatioThree"), ("beta", "Exp")):
            mac(
                f"dyn{SUF}{RW}{AW}SpecMin",
                num(P[rk + "_specimen_range"][0], 2),
                "smallest per-specimen value behind the pooled one",
                base + f".{rk}_specimen_range.0",
            )
            mac(
                f"dyn{SUF}{RW}{AW}SpecMax",
                num(P[rk + "_specimen_range"][1], 2),
                "largest per-specimen value behind the pooled one",
                base + f".{rk}_specimen_range.1",
            )
        mac(
            f"dyn{SUF}RatioThreeBelowOne{AW}",
            str(P["n_cells_ratio_3_1_below_1"]),
            "cells with 3N:1N ratio below 1",
        )
        mac(
            f"dyn{SUF}Exp{AW}",
            num(P["beta"], 2),
            "within-cell force exponent beta",
            base + ".beta",
        )
        mac(
            f"dyn{SUF}Exp{AW}Lo", num(P["beta_ci"][0], 2), "95% CI low (cell bootstrap)"
        )
        mac(f"dyn{SUF}Exp{AW}Hi", num(P["beta_ci"][1], 2), "95% CI high")
        mac(
            f"dyn{SUF}Exp{AW}ClLo",
            num(P["beta_ci_specimen_cluster"][0], 2),
            "95% CI low (specimen cluster)",
        )
        mac(
            f"dyn{SUF}Exp{AW}ClHi",
            num(P["beta_ci_specimen_cluster"][1], 2),
            "95% CI high (specimen cluster)",
        )
        if amp_key == "q_pp":
            for c in COMPS:
                ps = ORD[sub][amp_key]["per_specimen"][c]
                cw = A.COMP_MACRO[c]
                mac(
                    f"dyn{SUF}Spearman{AW}{cw}",
                    num(ps["spearman"], 2),
                    f"Spearman, {c}",
                )
                mac(
                    f"dyn{SUF}Spearman{AW}{cw}Lo",
                    num(ps["spearman_ci"][0], 2),
                    "95% CI low",
                )
                mac(
                    f"dyn{SUF}Spearman{AW}{cw}Hi",
                    num(ps["spearman_ci"][1], 2),
                    "95% CI high",
                )
                mac(f"dyn{SUF}Exp{AW}{cw}", num(ps["beta"], 2), f"force exponent, {c}")
                mac(
                    f"dyn{SUF}CellsConc{AW}{cw}",
                    str(ps["n_cells_concordant"]),
                    f"ordered cells, {c}",
                )
mac(
    "dynSpearmanQVrec",
    num(spearman(PRIM["pp"], TGT["rms_Voc"]), 2),
    "Spearman of recovered q_pp and recorded V_rms over 75 recordings",
)
mac(
    "dynSpearmanQVrecNorm",
    num(spearman(normalize(PRIM["pp"], ALL), normalize(TGT["rms_Voc"], ALL)), 2),
    "same, after within-specimen normalization",
)
for sub, SUF in (("all", ""), ("n72", "Dup")):
    P = ORD[sub]["q_pp"]["pooled"]
    for k in FREQS:
        mac(f"dyn{SUF}QMeanFreq{WORD[k]}", num(P["mean_by_freq"][str(k)], 2),
            f"mean normalized q_pp at the {k} Hz label, {sub}", f"ordering.{sub}.q_pp.pooled.mean_by_freq.{k}")
    for k in FORCES:
        mac(f"dyn{SUF}QMeanForce{WORD[k]}", num(P["mean_by_force"][str(k)], 2),
            f"mean normalized q_pp at the {k} N label, {sub}", f"ordering.{sub}.q_pp.pooled.mean_by_force.{k}")
    mac(f"dyn{SUF}CellsConcQBinomP", f"\\num{{{P['concordant_binom_p_vs_one_sixth']:.1e}}}",
        f"binomial p of the ordered-cell count against chance 1/6, {sub}")
sa = SENS[SENS.subset == "all"]
mac(
    "dynSensSpearmanMin",
    num(sa.spearman_pooled.min(), 2),
    "pooled Spearman, minimum over C_p x f_lam x amplitude type",
)
mac(
    "dynSensSpearmanMax",
    num(sa.spearman_pooled.max(), 2),
    "pooled Spearman, maximum over the sweep",
)
mac("dynSensExpMin", num(sa.beta.min(), 2), "force exponent, minimum over the sweep")
mac("dynSensExpMax", num(sa.beta.max(), 2), "force exponent, maximum over the sweep")
mac(
    "dynSensRatioThreeMin",
    num(sa.ratio_3_1_gmean.min(), 2),
    "3N:1N ratio, minimum over the sweep",
)
mac(
    "dynSensRatioThreeMax",
    num(sa.ratio_3_1_gmean.max(), 2),
    "3N:1N ratio, maximum over the sweep",
)
mac(
    "dynSensAmpCorrMin",
    num(sa.spearman_vs_primary_amplitude.min(), 3),
    "Spearman of swept amplitude with the primary, minimum",
)
sp = sa[sa.amplitude == "pp"]
mac(
    "dynSensConcMin",
    str(int(sp.n_cells_concordant.min())),
    "ordered cells, minimum over the q_pp sweep",
)
mac(
    "dynSensConcMax",
    str(int(sp.n_cells_concordant.max())),
    "ordered cells, maximum over the q_pp sweep",
)
for k in ("mat", "dense", "lit"):
    ct = CP_TAB[CP_TAB.scenario == k]
    KW = {"mat": "Mat", "dense": "Dense", "lit": "Lit"}[k]
    mac(f"dynCp{KW}Min", f"{ct.C_p_nF.min():.2f}", f"C_p (nF) minimum, {k} scenario")
    mac(f"dynCp{KW}Max", f"{ct.C_p_nF.max():.2f}", f"C_p (nF) maximum, {k} scenario")
    mac(f"dynH{KW}Max", f"{ct.H_25Hz.max():.3f}", f"largest |H| at 25 Hz, {k} scenario")
mac(
    "dynReproMax",
    f"\\num{{{REPRO_MAX:.1e}}}",
    "max |R2 difference| of the label scenario against a2_physgp.json",
)
for s in SCEN:
    for sub, SUF in (("all", ""), ("n72", "Dup")):
        for t in A.TARGETS:
            for m in MODELS:
                for sch in SCHEMES:
                    d = GPR[s][sub][t][m][sch]
                    nd = 3 if sch == "loo" else 2
                    mac(
                        f"dyn{SUF}{SW[s]}{SCHW[sch]}{TW[t]}{MW[m]}",
                        num(d["pooled_R2"], nd),
                        f"pooled R2, {s}, {sub}",
                        f"gp.{s}.{sub}.{t}.{m}.{sch}.pooled_R2",
                    )
                    if sch == "force" or (sch == "frequency" and t == "rms_Voc"):
                        for k, v in d["levels"].items():
                            mac(
                                f"dyn{SUF}{SW[s]}{SCHW[sch]}{WORD[int(k)]}{TW[t]}{MW[m]}",
                                num(v, 2),
                                f"within-level R2 at {k}, {s}, {sub}",
                            )
        mac(
            f"dyn{SUF}{SW[s]}NChanged" if s != "label" else f"dyn{SUF}LabelNChanged",
            str(len(CHANGES[s][sub])),
            f"conclusion flags (of 15) that differ from the paper, {s}, {sub}",
        )

_names = [m_.split("}")[0] for m_ in MAC]
assert len(_names) == len(set(_names)), "duplicate macro names"
with open(os.path.join(RES, "numbers_a7.tex"), "w") as fh:
    fh.write(
        "% numbers_a7.tex: generated by code/a7_dynforce.py; keys in a7_dynforce.json -> index\n"
    )
    fh.write("\n".join(MAC) + "\n")


# ----------------------------------------------------------------------------- tables
def T(x, nd=2):
    return num(x, nd).replace(r"\ensuremath{-}", "$-$") if x is not None else "n/a"


def cistr(c, nd=2):
    return f"[{T(c[0], nd)}, {T(c[1], nd)}]"


TABLE_NOTES = {}


def write_table(fname, lines, notes):
    """Table body plus its notes: as comment lines on top and as <name>_note.tex for the caption."""
    TABLE_NOTES[fname] = notes
    head = [f"% note: {n}" for n in notes]
    open(os.path.join(TAB, fname), "w").write("\n".join(head + lines) + "\n")
    open(os.path.join(TAB, fname.replace(".tex", "_note.tex")), "w").write(
        " ".join(notes) + "\n"
    )


# ordering table
lines = [
    r"\begin{tabular}{l r rl rl c rr rl}",
    r"\toprule",
    r" & & \multicolumn{2}{c}{Spearman, recovered $q$} & \multicolumn{2}{c}{Spearman, recorded \RMS} & Ordered & \multicolumn{2}{c}{Ratio to \SI{1}{\newton}} & \multicolumn{2}{c}{Force exponent $\beta$} \\",
    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
    r"Specimen & $n$ & $\rho$ & 95\% CI & $\rho$ & 95\% CI & cells & \SI{2}{\newton} & \SI{3}{\newton} & $\beta$ & 95\% CI \\",
    r"\midrule",
]
for sub, head in (
    ("all", "All recordings"),
    ("n72", r"Without recordings 52, 54, 73 ($n = 72$)"),
):
    lines.append(rf"\multicolumn{{11}}{{l}}{{{head}}} \\")
    Q, V = ORD[sub]["q_pp"], ORD[sub]["V_rms"]
    for c in COMPS:
        q, v = Q["per_specimen"][c], V["per_specimen"][c]
        lines.append(
            f"{A.COMP_TEX[c]} & {q['n']} & {T(q['spearman'])} & {cistr(q['spearman_ci'])} & {T(v['spearman'])} & {cistr(v['spearman_ci'])} & "
            f"{q['n_cells_concordant']}/{q['n_cells_complete']} & {T(q['ratio_2_1_gmean'])} & {T(q['ratio_3_1_gmean'])} & {T(q['beta'])} & {cistr(q['beta_ci'])} \\\\"
        )
    q, v = Q["pooled"], V["pooled"]
    lines.append(
        f"Pooled & {q['n']} & {T(q['spearman'])} & {cistr(q['spearman_ci'])} & {T(v['spearman'])} & {cistr(v['spearman_ci'])} & "
        f"{q['n_cells_concordant']}/{q['n_cells_complete']} & {T(q['ratio_2_1_gmean'])} & {T(q['ratio_3_1_gmean'])} & {T(q['beta'])} & {cistr(q['beta_ci'])} \\\\"
    )
    if sub == "all":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
ORD_NOTES = []
for sub, head in (("all", "all recordings"), ("n72", "$n = 72$")):
    q = ORD[sub]["q_pp"]["pooled"]
    ORD_NOTES.append(
        f"Pooled recovered-$q$ ratios, {head}: \\SI{{2}}{{\\newton}}:\\SI{{1}}{{\\newton}} {T(q['ratio_2_1_gmean'])} "
        f"(cell bootstrap {cistr(q['ratio_2_1_ci'])}, specimen-cluster bootstrap {cistr(q['ratio_2_1_ci_specimen_cluster'])}, "
        f"per-specimen range {T(q['ratio_2_1_specimen_range'][0])} to {T(q['ratio_2_1_specimen_range'][1])}); "
        f"\\SI{{3}}{{\\newton}}:\\SI{{1}}{{\\newton}} {T(q['ratio_3_1_gmean'])} "
        f"(cell {cistr(q['ratio_3_1_ci'])}, specimen cluster {cistr(q['ratio_3_1_ci_specimen_cluster'])}, "
        f"per-specimen range {T(q['ratio_3_1_specimen_range'][0])} to {T(q['ratio_3_1_specimen_range'][1])}); "
        f"$\\beta$ {T(q['beta'])} (cell {cistr(q['beta_ci'])}, specimen cluster {cistr(q['beta_ci_specimen_cluster'])}, "
        f"per-specimen range {T(q['beta_specimen_range'][0])} to {T(q['beta_specimen_range'][1])})."
    )
write_table("tab_a7_ordering.tex", lines, ORD_NOTES)

# sensitivity table (all recordings)
lines = [
    r"\begin{tabular}{rr l rrrrrr}",
    r"\toprule",
    r"$C_p$ (\si{\nano\farad}) & $f_\lambda$ (\si{\hertz}) & Amplitude & Pooled $\rho$ & Cell $\bar\rho$ & Ordered & \SI{2}{\newton}:\SI{1}{\newton} & \SI{3}{\newton}:\SI{1}{\newton} & $\beta$ \\",
    r"\midrule",
]
for _, r in SENS[SENS.subset == "all"].iterrows():
    lines.append(
        f"{r.C_p_nF:g} & {r.f_lam_Hz:g} & {'peak-to-peak' if r.amplitude == 'pp' else 'RMS'} & {T(r.spearman_pooled)} & {T(r.mean_within_cell_rho)} & "
        f"{int(r.n_cells_concordant)}/{int(r.n_cells_complete)} & {T(r.ratio_2_1_gmean)} & {T(r.ratio_3_1_gmean)} & {T(r.beta)} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_a7_sensitivity.tex"), "w").write("\n".join(lines) + "\n")

# permittivity / capacitance table
lines = [
    r"\begin{tabular}{l l r r r r r}",
    r"\toprule",
    r"Scenario & Specimen & $\varepsilon_r$ & $C_p$ (\si{\nano\farad}) & Corner (\si{\hertz}) & $|H|$ at \SI{5}{\hertz} & $|H|$ at \SI{25}{\hertz} \\",
    r"\midrule",
]
for k in ("mat", "dense", "lit"):
    for j, (_, r) in enumerate(CP_TAB[CP_TAB.scenario == k].iterrows()):
        lab = EPS_SCEN[k]["label"].capitalize() if j == 0 else ""
        lines.append(
            f"{lab} & {A.COMP_TEX[r.composition]} & {r.eps_r:.1f} & {r.C_p_nF:.2f} & {r.corner_Hz:.0f} & {r.H_5Hz:.4f} & {r.H_25Hz:.4f} \\\\"
        )
    lines.append(r"\midrule" if k != "lit" else r"\bottomrule")
lines.append(r"\end{tabular}")
open(os.path.join(TAB, "tab_a7_cp.tex"), "w").write("\n".join(lines) + "\n")

SC_ORDER = [
    "label",
    "frec",
    "fcell",
    "frecLo",
    "frecHi",
    "ocs_uniform",
    "ocs_mat",
    "ocs_dense",
    "ocs_lit",
    "ocw_uniform",
    "ocw_mat",
    "ocw_dense",
    "ocw_lit",
]
TRANS = r"$^\dagger$"  # transductive under held-out force (cell mean)
CIRC = r"$^\ddagger$"  # circular in every scheme (per-recording force)
NOTE_TRANS = (
    TRANS + " Transductive under held-out force: the leave-self-out cell mean averages the other "
    "recordings of the held-out force level, all of which are in the test fold, and the within-specimen "
    "normalization averages over test rows; valid for LOO and held-out frequency only."
)
NOTE_CIRC = (
    CIRC + " Circular in every scheme (LOO, held-out force, held-out frequency): the force input is "
    "computed from the same recording whose descriptor is the target."
)
SC_TEX = {
    "label": "Nominal label (paper, recomputed)",
    "frec": r"Recovered force, per recording, \SI{1.9}{\nano\farad}",
    "fcell": r"Recovered force, leave-self-out cell mean",
    "frecLo": r"Recovered force, per recording, \SI{0.19}{\nano\farad}",
    "frecHi": r"Recovered force, per recording, \SI{7.7}{\nano\farad}",
    "ocs_uniform": r"Scalar $V_\mathrm{oc}$, uniform \SI{1.9}{\nano\farad}",
    "ocs_mat": r"Scalar $V_\mathrm{oc}$, electrospun-mat $\varepsilon_r$",
    "ocs_dense": r"Scalar $V_\mathrm{oc}$, dense-film $\varepsilon_r$",
    "ocs_lit": r"Scalar $V_\mathrm{oc}$, composite-literature $\varepsilon_r$",
    "ocw_uniform": r"Waveform $V_\mathrm{oc}$, uniform \SI{1.9}{\nano\farad}",
    "ocw_mat": r"Waveform $V_\mathrm{oc}$, electrospun-mat $\varepsilon_r$",
    "ocw_dense": r"Waveform $V_\mathrm{oc}$, dense-film $\varepsilon_r$",
    "ocw_lit": r"Waveform $V_\mathrm{oc}$, composite-literature $\varepsilon_r$",
}


for _s in ("frec", "frecLo", "frecHi"):
    SC_TEX[_s] += CIRC


def mark(s, scheme):
    """Footnote mark of a cell of scenario s under scheme."""
    return TRANS if (s == "fcell" and scheme == "force") else ""


def pooled_table(targets, fname, sub="all"):
    lines = [
        r"\begin{tabular}{l rr rr rr r}",
        r"\toprule",
        r" & \multicolumn{2}{c}{LOO} & \multicolumn{2}{c}{Held-out force} & \multicolumn{2}{c}{Held-out frequency} & Changed \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Scenario & GP & LawGP & GP & LawGP & GP & LawGP & flags \\",
        r"\midrule",
    ]
    for ti, t in enumerate(targets):
        if len(targets) > 1:
            lines.append(rf"\multicolumn{{8}}{{l}}{{{A.TARGET_TEX[t]}}} \\")
        pp = PAPER[sub][t]
        lines.append(
            "Paper value (a2) & "
            + " & ".join(
                T(pp[m][sch]["pooled_R2"], 3 if sch == "loo" else 2)
                for sch in SCHEMES
                for m in MODELS
            )
            + r" & \\"
        )
        for s in SC_ORDER:
            g = GPR[s][sub][t]
            nch = sum(1 for c in CHANGES[s][sub] if c.startswith(t + ":"))
            lines.append(
                SC_TEX[s]
                + " & "
                + " & ".join(
                    T(g[m][sch]["pooled_R2"], 3 if sch == "loo" else 2) + mark(s, sch)
                    for sch in SCHEMES
                    for m in MODELS
                )
                + f" & {nch}/5 \\\\"
            )
        if ti < len(targets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(fname, lines, [NOTE_TRANS, NOTE_CIRC])


pooled_table(["rms_Voc"], "tab_a7_pooled.tex")
pooled_table(list(A.TARGETS), "tab_a7_pooled_all.tex")
pooled_table(list(A.TARGETS), "tab_a7_pooled_dup.tex", sub="n72")


def level_table(scheme, levels, fname, t="rms_Voc", sub="all"):
    unit = "newton" if scheme == "force" else "hertz"
    cols = " ".join(["rr"] * len(levels))
    lines = [
        rf"\begin{{tabular}}{{l {cols}}}",
        r"\toprule",
        " & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{\SI{{{lv}}}{{\{unit}}}}}" for lv in levels
        )
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(len(levels))
        ),
        "Scenario & " + " & ".join(["GP & LawGP"] * len(levels)) + r" \\",
        r"\midrule",
    ]
    pp = PAPER[sub][t]
    lines.append(
        "Paper value (a2) & "
        + " & ".join(
            T(pp[m][scheme]["levels"][str(lv)]) for lv in levels for m in MODELS
        )
        + r" \\"
    )
    for s in SC_ORDER:
        g = GPR[s][sub][t]
        lines.append(
            SC_TEX[s]
            + " & "
            + " & ".join(
                T(g[m][scheme]["levels"][str(lv)]) + mark(s, scheme)
                for lv in levels
                for m in MODELS
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(
        fname, lines, ([NOTE_TRANS] if scheme == "force" else []) + [NOTE_CIRC]
    )


level_table("force", FORCES, "tab_a7_levels_force.tex")
level_table("frequency", FREQS, "tab_a7_levels_freq.tex")
for t in ("Vpp", "Vmax"):
    level_table("force", FORCES, f"tab_a7_levels_force_{t.lower()}.tex", t=t)

# conclusions table: in how many of the three targets each paper conclusion holds
lines = [
    r"\begin{tabular}{l " + "c" * 6 + "}",
    r"\toprule",
    r"Conclusion & Paper & Rec. force$^\ddagger$ & Cell force & Scalar $V_\mathrm{oc}$ & Waveform $V_\mathrm{oc}$ & $n=72$ paper \\",
    r"\midrule",
]
K_FORCE = ("K3_lawgp_beats_gp_held_out_force", "K4_lawgp_beats_gp_at_3N")
for k, lab in K_LABEL.items():

    def cnt(s, sub="all"):
        return sum(CONC[s][sub][t][k] for t in A.TARGETS)

    def rng_(ss):
        v = [cnt(s) for s in ss]
        return f"{min(v)}" if min(v) == max(v) else f"{min(v)} to {max(v)}"

    lines.append(
        f"{lab} & {cnt('label')}/3 & {cnt('frec')}/3 & {cnt('fcell')}/3{TRANS if k in K_FORCE else ''} & {rng_(['ocs_uniform', 'ocs_mat', 'ocs_dense', 'ocs_lit'])}/3 & "
        f"{rng_(['ocw_uniform', 'ocw_mat', 'ocw_dense', 'ocw_lit'])}/3 & {cnt('label', 'n72')}/3 \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_a7_conclusions.tex", lines, [NOTE_TRANS, NOTE_CIRC])

# ----------------------------------------------------------------------------- json
SOURCES = [
    {
        "key": "kocPiezoelectricNanogeneratorsBased2025",
        "doi": "10.1007/s10853-025-11872-9",
        "use": "composition ratios of eps_r at 1 kHz (Fig. 7, text p. 25490: 3 wt% CNT 1.57 x PVDF, others below PVDF); electrode area, thickness, 15 vol% BaTiO3",
    },
    {
        "key": "castkovaStructurePropertiesRelationship2020",
        "doi": "10.3390/nano10061221",
        "use": "electrospun PVDF fiber mats eps_r 1.1 to 2.2 at 1 kHz (Table 4); dense alpha-PVDF about 14; porosity lowers eps_r",
    },
    {
        "key": "bouharrasDielectricCharacterizationCoreShell2023",
        "doi": "10.3390/polym15030595",
        "use": "PVDF + 15 vol% BaTiO3: eps_r about +50%",
    },
    {
        "key": "zhaoBaTiO3MWNTsPolyvinylidene2019",
        "doi": "10.1021/acsomega.8b02504",
        "use": "BaTiO3/MWNT/PVDF 11.5/0.35/88.15 vol%: eps_r 59, MWNT threshold below 0.4 vol%",
    },
    {
        "key": "wangCarbonNanotubeComposites2005",
        "doi": "10.1063/1.1996842",
        "use": "MWNT/PVDF threshold 1.61 vol%, eps_r about 300 near 2 vol%",
    },
    {
        "key": "dangGiantDielectricPermittivities2007",
        "doi": "10.1002/adma.200600703",
        "use": "giant permittivity of MWNT/PVDF near percolation (qualitative support for the high scenario; cited by the source study as ref. 42)",
    },
]
J = {
    "meta": {
        "task": "E2 DYNFORCE (a7)",
        "run": "20260926-130456",
        "command": "python code/a7_dynforce.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "runtime_s": None,
        "model": "V(jw) = G(jw) Q(jw), G = jw R_in / (1 + jw R_in C_p); V/V_oc = H = jw R_in C_p / (1 + jw R_in C_p)",
        "R_in_Ohm": R_IN,
        "C_p_sweep_F": list(CP_SET),
        "C_p_primary_F": CP_PRIMARY,
        "inversion": "frequency-domain Tikhonov, Q = conj(G) V / (|G|^2 + lambda), lambda = (2 pi f_lam R_in)^2; equals a Wiener filter with white noise and white charge prior; acts as a high-pass on q at f_lam; zero padding to %d samples; q demeaned"
        % NPAD,
        "f_lam_sweep_Hz": list(FLAM_SET),
        "f_lam_primary_Hz": FLAM_PRIMARY,
        "tap_amplitude": "fixed-period windows from a1_segment.choose_period/segment on the recorded voltage; q linearly detrended inside each window; peak-to-peak (primary) and RMS; recording amplitude = mean over taps",
        "normalization": "within specimen (composition): A_i / mean over that specimen's recordings in the subset",
        "force_input": "F_hat = 2 N x normalized amplitude (2 N = mean nominal force of the design); per recording (circular: the input is computed from the waveform whose descriptor is the target) and leave-self-out mean over the other recordings of the same specimen and nominal force",
        "voc_scalar": "V_oc = descriptor / |H(j 2 pi f_label)| with C_p per composition",
        "voc_wave": "V_oc(t) = q_hat(t) / C_p (f_lam primary), descriptors RMS, peak-to-peak, max |.| after demeaning",
        "C_p_geometry": {
            "area_m2": AREA,
            "thickness_m": THICK,
            "thickness_range_m": list(THICK_RANGE),
            "eps0": EPS0,
        },
        "caveats": [
            "the recorded excursions are one-sample spikes at 1 kHz, so the area of each current pulse (hence the charge step) is sampled by one or two points; the recovered q is close to the running integral of V / R_in and inherits the aliasing of the acquisition",
            "the nominal 1, 2, 3 N are static weights; the recovered quantity is a relative dynamic force (d33 F_dyn up to a specimen factor) and the static preload is not recoverable",
            "using the per-recording recovered amplitude as the force input is circular in every scheme (LOO, held-out force, held-out frequency), since input and target come from the same recording; these rows are marked with a double dagger in the tables",
            "the leave-self-out cell-mean force input is transductive under held-out force: it averages the other recordings of the held-out force level, all in the test fold, and the within-specimen normalization averages over test rows; it is valid for LOO and held-out frequency only (dagger in the tables); 'LawGP beats GP at a held-out force' under this input is not a held-out-force result",
            "the pooled force ratios hide a wide spread between specimens (per-specimen range in ordering.<subset>.<amplitude>.pooled.*_specimen_range); the specimen-cluster bootstrap (5 clusters) gives the CI that reflects it",
            "the source study's absolute eps_r values are below 1 (not physical), so only its composition ratios are used; the absolute scale spans an electrospun-mat and two dense-film scenarios",
            "leakage resistance R_p and the DAQ wiring mode are unknown (project notes); R_p >> R_in is assumed",
        ],
        "outputs": [
            "a7_dynforce.json",
            "a7_recordings.csv",
            "a7_figdata_amplitude.csv",
            "a7_figdata_ratios.csv",
            "a7_figdata_waveforms.csv",
            "a7_figdata_regularization.csv",
            "a7_figdata_cp.csv",
            "a7_figdata_r2.csv",
            "numbers_a7.tex",
            "tables/tab_a7_ordering.tex",
            "tables/tab_a7_sensitivity.tex",
            "tables/tab_a7_cp.tex",
            "tables/tab_a7_pooled.tex",
            "tables/tab_a7_pooled_all.tex",
            "tables/tab_a7_pooled_dup.tex",
            "tables/tab_a7_levels_force.tex",
            "tables/tab_a7_levels_freq.tex",
            "tables/tab_a7_levels_force_vpp.tex",
            "tables/tab_a7_levels_force_vmax.tex",
            "tables/tab_a7_conclusions.tex",
            "tables/tab_a7_*_note.tex (table notes: dagger, double dagger, ratio CIs)",
            "a7_manifest.json",
        ],
        "excluded_n72": DUP_IDS,
    },
    "segmentation_check": seg_check,
    "ordering": ORD,
    "sensitivity": SENS.to_dict(orient="records"),
    "amplitude_vs_recorded": {
        "spearman_q_pp_vs_Vrms": fl(spearman(PRIM["pp"], TGT["rms_Voc"])),
        "spearman_q_pp_vs_Vrms_normalized": fl(
            spearman(normalize(PRIM["pp"], ALL), normalize(TGT["rms_Voc"], ALL))
        ),
        "spearman_q_pp_vs_V_tap_pp": fl(spearman(PRIM["pp"], AMP_V["pp"])),
    },
    "permittivity": {
        "source_ratios": SRC_RATIO,
        "scenarios": {
            k: {
                "label": v["label"],
                "eps_r": v["eps"],
                "basis": v["basis"],
                "C_p_nF": {c: v["C_p_F"][c] * 1e9 for c in COMPS},
            }
            for k, v in EPS_SCEN.items()
        },
        "table": CP_TAB.to_dict(orient="records"),
        "sources": SOURCES,
    },
    "scenarios": {
        s: {
            "label": SCEN[s]["label"],
            "kind": SCEN[s]["kind"],
            "valid_schemes": (
                []
                if s.startswith("frec")
                else ["loo", "frequency"] if s == "fcell" else list(SCHEMES)
            ),
            "validity_note": (
                "circular in every scheme"
                if s.startswith("frec")
                else "transductive under held-out force"
                if s == "fcell"
                else "no leakage from the force input"
            ),
        }
        for s in SCEN
    },
    "table_notes": TABLE_NOTES,
    "gp": GPR,
    "paper_values": PAPER,
    "paper_values_source": "results/a2_physgp.json (pooled, per_level, sensitivity_excluding_duplicates), the source of numbers_a2.tex; LOO GP equals \\benchLoo<T>Gp in numbers_a3.tex",
    "label_scenario_reproduction_max_abs_dR2": REPRO_MAX,
    "conclusions": {
        "definitions": K_LABEL,
        "values": CONC,
        "changed_vs_label": CHANGES,
    },
    "index": INDEX,
}
J["meta"]["runtime_s"] = round(time.time() - T0, 1)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return fl(o)
    return o


json.dump(_clean(J), open(os.path.join(RES, "a7_dynforce.json"), "w"), indent=1)

# ----------------------------------------------------------------------------- console summary
P = ORD["all"]["q_pp"]["pooled"]
print(
    f"[a7] recovered q_pp: pooled rho {P['spearman']:.2f} {P['spearman_ci']}, cells ordered {P['n_cells_concordant']}/{P['n_cells_complete']}, "
    f"ratios {P['ratio_2_1_gmean']:.2f} {P['ratio_3_1_gmean']:.2f}, beta {P['beta']:.2f} {P['beta_ci']}"
)
for s in SC_ORDER:
    g = GPR[s]["all"]["rms_Voc"]
    print(
        f"[a7] {s:12s} Vrms LOO {g['gp']['loo']['pooled_R2']:.3f}/{g['physgp']['loo']['pooled_R2']:.3f} "
        f"force {g['gp']['force']['pooled_R2']:.2f}/{g['physgp']['force']['pooled_R2']:.2f} "
        f"freq {g['gp']['frequency']['pooled_R2']:.2f}/{g['physgp']['frequency']['pooled_R2']:.2f} changed {len(CHANGES[s]['all'])}"
    )

# ----------------------------------------------------------------------------- manifest
sys.stdout.flush()
print(f"[a7] done in {time.time() - T0:.1f} s")
