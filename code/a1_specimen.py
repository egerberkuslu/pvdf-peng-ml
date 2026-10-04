#!/usr/bin/env python3
"""A1: specimen structure, acquisition repeatability and cluster-aware statistics.

One command (run from the repo root):
    python code/a1_specimen.py
    python code/a1_specimen.py --exclude-duplicates
        (or A1_EXCLUDE_DUPLICATES=1): drops the scaled-copy recordings in
        EXCLUDE from every primary analysis (duplicated-recordings note in the README option d)

Reads only data/long.parquet, series_index.parquet and
targets_design.parquet. Writes
    results/a1_specimen.json          every number, plus an index
    results/a1_cycles.csv             per-cycle amplitudes
    results/a1_recordings.csv         per-recording cycle statistics
    results/a1_predictions.csv        out-of-fold GP predictions
    results/a1_bootstrap_draws.csv    bootstrap R^2 draws
    results/tables/tab_spec_{repeatability,splithalf,anova,bootstrap,loso}.tex
    results/numbers_a1.tex            \\spec* macros
    figures/fig_spec_{cycles,splithalf}.{pdf,png}
    figures/captions_a1.md

Sections: (1) cycle-level amplitude repeatability, (2) split-half
pseudo-replicates and GP refits, (3) split-half ANOVA whose error term is
acquisition repeatability, (4) specimen-cluster vs record-level bootstrap of the
within-grid LOO R^2, (5) held-out specimen (= held-out composition) per level.
Seed 0 everywhere.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # joblib/loky workers inherit this

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a1_segment import (  # noqa: E402
    EDGE_ENERGY_MIN,
    LOCK_MIN,
    PEAK_HEIGHT_FRAC,
    HALF_ACF_MIN,
    PARITY_MIN,
    PERIOD_MARGIN,
    RATE_DEV_MAX,
    SPEC_BAND,
    choose_period,
    segment,
    spectral_rate,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel as CK  # noqa: E402
from sklearn.gaussian_process.kernels import Matern, WhiteKernel  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from peng_paths import ROOT_STR as ROOT  # repository root (env PENG_ROOT overrides)
REV = ROOT
DATA = os.path.join(REV, "data")
RES = os.path.join(REV, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(REV, "figures")
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)

SEED = 0
N_BOOT = 2000
# recordings that are scaled copies of other recordings (duplicate audit below,
# duplicated-recordings note in the README): 52 = 2.96 x 47, 54 = 1.74 x 49, 73 ~ 7.3 x 5
EXCLUDE = [52, 54, 73]
EXCLUDE_MODE = "--exclude-duplicates" in sys.argv or os.environ.get(
    "A1_EXCLUDE_DUPLICATES", ""
).lower() in ("1", "true", "yes")
DUP_R_MIN = 0.95  # |r| threshold of the duplicate audit
DUP_MAX_LAG = 30  # samples (30 ms)
DUP_EXACT_RESID = 1e-9  # relative residual below which a copy is exact
TARGETS = ["rms_Voc", "Vpp", "Vmax"]  # energy proxy is not reported (analysis protocol 3)
TNAME = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}  # macro stems
TTEX = {"rms_Voc": r"\RMS", "Vpp": r"\Vpp", "Vmax": r"\Vmax"}
CYC = {"rms_Voc": "rms", "Vpp": "vpp", "Vmax": "vmax"}  # cycle-level analogue
COMPS = [
    "PVDF",
    "PVDF+BaTiO3",
    "PVDF+BaTiO3+%1CNT",
    "PVDF+BaTiO3+%2CNT",
    "PVDF+BaTiO3+%3CNT",
]
COMP_STEM = {
    "PVDF": "Pvdf",
    "PVDF+BaTiO3": "Bto",
    "PVDF+BaTiO3+%1CNT": "CntOne",
    "PVDF+BaTiO3+%2CNT": "CntTwo",
    "PVDF+BaTiO3+%3CNT": "CntThree",
}
COMP_TEX = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": r"PVDF/BaTiO\textsubscript{3}",
    "PVDF+BaTiO3+%1CNT": r"+1 wt\% MWCNT",
    "PVDF+BaTiO3+%2CNT": r"+2 wt\% MWCNT",
    "PVDF+BaTiO3+%3CNT": r"+3 wt\% MWCNT",
}
COMP_LABEL = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "PVDF/BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "+1 wt% MWCNT",
    "PVDF+BaTiO3+%2CNT": "+2 wt% MWCNT",
    "PVDF+BaTiO3+%3CNT": "+3 wt% MWCNT",
}
NUMW = {
    1: "One",
    2: "Two",
    3: "Three",
    5: "Five",
    10: "Ten",
    15: "Fifteen",
    20: "Twenty",
    25: "TwentyFive",
}
# house style (shared figure style, values copied)
CBLUE, CTEAL, CAMBER, CRED, CGREY = (
    "#3B6FB6",
    "#2A9D8F",
    "#E08D2F",
    "#C8553D",
    "#6C757D",
)
COMP_COLOR = dict(zip(COMPS, [CGREY, CBLUE, CTEAL, CRED, CAMBER]))
COMP_MARK = dict(zip(COMPS, ["o", "s", "^", "D", "v"]))


def f(x):
    """plain float for JSON"""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def pct(a, q=(2.5, 25, 50, 75, 97.5)):
    a = np.asarray(a, float)
    return {f"p{str(p).replace('.', '_')}": float(np.percentile(a, p)) for p in q}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
L = pd.read_parquet(os.path.join(DATA, "long.parquet"))
S = pd.read_parquet(os.path.join(DATA, "series_index.parquet")).sort_values("series_id")
D = pd.read_parquet(os.path.join(DATA, "targets_design.parquet")).reset_index(drop=True)
W = np.stack(
    [g.sort_values("time")["voc"].to_numpy(float) for _, g in L.groupby("series_id")]
)
assert W.shape == (75, 1000)
N_ALL = 75
# series_id order equals design-table row order
assert (S["material"].to_numpy() == D["composition"].to_numpy()).all()
assert (S["force_newton"].to_numpy() == D["force_N"].to_numpy()).all()
assert (S["freq_hz"].to_numpy() == D["freq_Hz"].to_numpy()).all()
XD = W - W.mean(axis=1, keepdims=True)  # recording-demeaned waveform


def cond_label(i):
    return f"{COMP_LABEL[D.loc[i, 'composition']]}, {int(D.loc[i, 'force_N'])} N, {int(D.loc[i, 'freq_Hz'])} Hz"


# ----------------------------------------------------------------------------
# (0) duplicate audit on all 75 recordings (independent of EXCLUDE_MODE)
# ----------------------------------------------------------------------------
def duplicate_audit(Xd, max_lag=DUP_MAX_LAG):
    """Largest |Pearson r| between every pair of demeaned waveforms over lags
    -max_lag..max_lag samples (overlapping part only)."""
    n, T_ = Xd.shape
    best = np.zeros((n, n))
    blag = np.zeros((n, n), int)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            A_, B_ = Xd[:, lag:], Xd[:, : T_ - lag]
        else:
            A_, B_ = Xd[:, : T_ + lag], Xd[:, -lag:]
        Az = A_ - A_.mean(1, keepdims=True)
        Az /= np.linalg.norm(Az, axis=1, keepdims=True)
        Bz = B_ - B_.mean(1, keepdims=True)
        Bz /= np.linalg.norm(Bz, axis=1, keepdims=True)
        Rm = Az @ Bz.T  # Rm[i, j] = corr(x_i[t + lag], x_j[t])
        m = np.abs(Rm) > np.abs(best)
        best[m] = Rm[m]
        blag[m] = lag
    np.fill_diagonal(best, 0.0)
    return best, blag


DUP_R, DUP_LAG = duplicate_audit(XD)
_iu = np.triu_indices(N_ALL, 1)
_absr = np.abs(DUP_R[_iu])
dup_pairs = []
for k in np.argsort(-_absr):
    if _absr[k] < DUP_R_MIN:
        break
    i0, j0 = int(_iu[0][k]), int(_iu[1][k])
    copy, orig = max(i0, j0), min(i0, j0)  # later condition id = copy
    lag = int(DUP_LAG[copy, orig])  # x_copy[t + lag] ~ x_orig[t]
    if lag >= 0:
        a_, b_ = XD[copy, lag:], XD[orig, : 1000 - lag]
    else:
        a_, b_ = XD[copy, : 1000 + lag], XD[orig, -lag:]
    scale = float(a_ @ b_ / (b_ @ b_))
    resid = float(np.linalg.norm(a_ - scale * b_) / np.linalg.norm(a_))
    dup_pairs.append({
        "copy": copy, "original": orig, "copy_condition": cond_label(copy),
        "original_condition": cond_label(orig), "lag_samples": lag,
        "r": float(DUP_R[copy, orig]), "scale": scale,
        "relative_residual": resid, "exact": bool(resid < DUP_EXACT_RESID),
    })
dup_pairs.sort(key=lambda p_: p_["copy"])
_rest = [v for k, v in enumerate(_absr) if v < DUP_R_MIN]
DUPLICATES = {
    "method": "Pearson r between every pair of recording-demeaned waveforms at lags -%d..%d samples on the overlapping part, best |r| per pair; pairs with |r| >= %.2f are listed with the least-squares scale copy = scale x original at the best lag and the relative residual ||copy - scale x original|| / ||copy||; the later condition id is called the copy; exact if residual < %.0e"
    % (DUP_MAX_LAG, DUP_MAX_LAG, DUP_R_MIN, DUP_EXACT_RESID),
    "n_pairs_tested": int(len(_absr)),
    "pairs": dup_pairs,
    "n_pairs": len(dup_pairs),
    "n_exact": int(sum(p_["exact"] for p_ in dup_pairs)),
    "max_abs_r_other_pairs": float(max(_rest)),
    "exclude_list": EXCLUDE,
    "copies_found": sorted(p_["copy"] for p_ in dup_pairs),
}
assert sorted(DUPLICATES["copies_found"]) == sorted(EXCLUDE), DUPLICATES["copies_found"]

# ----------------------------------------------------------------------------
# primary sample: all 75 recordings, or 72 with --exclude-duplicates
# ----------------------------------------------------------------------------
KEEP_IDS = [i for i in range(N_ALL) if not (EXCLUDE_MODE and i in EXCLUDE)]
W, XD = W[KEEP_IDS], XD[KEEP_IDS]
D = D.iloc[KEEP_IDS].reset_index(drop=True)
CID = np.array(KEEP_IDS)  # original condition id of every analyzed row
N = len(CID)


def descriptors(x):
    """The three targets as defined by the design table: after removing the
    mean of the analyzed segment, V_rms = std, V_pp = max - min,
    |V|_max = max |v|."""
    x = x - x.mean()
    return {
        "rms_Voc": float(np.sqrt(np.mean(x**2))),
        "Vpp": float(np.ptp(x)),
        "Vmax": float(np.abs(x).max()),
    }


full_desc = pd.DataFrame([descriptors(W[i]) for i in range(N)])
desc_check = {t: float(np.abs(full_desc[t] - D[t]).max()) for t in TARGETS}
assert all(v < 1e-9 for v in desc_check.values()), desc_check

FREQ = D["freq_Hz"].to_numpy()
FORCE = D["force_N"].to_numpy()
COMP = D["composition"].to_numpy()
SPEC = np.array([COMPS.index(c) for c in COMP])  # specimen id 0..4
# rows that survive the duplicate exclusion (all True in exclude mode)
DUPKEEP = ~np.isin(CID, EXCLUDE)

J = {
    "meta": {
        "task": "A1",
        "seed": SEED,
        "n_boot": N_BOOT,
        "inputs": [
            "data/long.parquet",
            "data/series_index.parquet",
            "data/targets_design.parquet",
        ],
        "target_definitions": "after subtracting the segment mean: V_rms = RMS, V_pp = max-min, |V|_max = max|v|; reproduce targets_design.parquet to max abs error listed in descriptor_check",
        "descriptor_check_max_abs_error": desc_check,
        "design": "5 specimens (one per composition) x 3 forces x 5 frequencies, one 1 s recording (1000 samples at 1 kHz) per condition; specimen and composition are confounded",
        "exclude_duplicates": bool(EXCLUDE_MODE),
        "n_conditions_analyzed": int(N),
        "condition_ids_analyzed": [int(c) for c in CID],
        "condition_id_note": "condition_id = row index of data/targets_design.parquet (0-74) in every output",
    },
    "duplicates": DUPLICATES,
}

# ----------------------------------------------------------------------------
# (1) cycle-level amplitude repeatability
# ----------------------------------------------------------------------------
rec_rows, cyc_rows, segs = [], [], []
for i in range(N):
    p_used, pinfo = choose_period(XD[i], FREQ[i])
    r = segment(XD[i], FREQ[i], period=p_used)
    segs.append(r)
    r_alt = segment(XD[i], FREQ[i])  # nominal period, sensitivity
    row = {
        "condition_id": int(CID[i]),
        "composition": COMP[i],
        "force_N": int(FORCE[i]),
        "freq_Hz": int(FREQ[i]),
        "expected_cycles_nominal": int(FREQ[i]),
        **pinfo,
        "expected_cycles_used": int(round(1000.0 / p_used)),
        "n_cycles": r["n_cycles"],
        "n_full_cycles": r["n_full_cycles"],
        "n_peaks_crosscheck": r["n_peaks_crosscheck"],
        "lock_concentration": r["lock_concentration"],
        "flag_weak_periodicity": bool(r["lock_concentration"] < LOCK_MIN),
        "flag_too_few_cycles": bool(r["n_cycles"] < 3),
        "flag_rate_deviates": bool(abs(pinfo["rate_rel_dev_from_nominal"]) > RATE_DEV_MAX),
        "n_cycles_nominal_period": r_alt["n_cycles"],
        "spectral_rate_Hz": spectral_rate(XD[i]),
        "duplicate_copy_of": next((p_["original"] for p_ in dup_pairs if p_["copy"] == CID[i]), -1),
    }
    row["flag_rate_deviates_spectral"] = bool(
        abs(row["spectral_rate_Hz"] / FREQ[i] - 1.0) > RATE_DEV_MAX
    )
    for k in ("rms", "vpp", "vmax"):
        a = r[k]
        row[f"{k}_mean"] = float(a.mean())
        row[f"{k}_sd"] = float(a.std(ddof=1))
        row[f"{k}_cv_pct"] = float(100 * a.std(ddof=1) / a.mean())
        row[f"{k}_se_rel_pct"] = float(100 * a.std(ddof=1) / np.sqrt(len(a)) / a.mean())
        row[f"{k}_logsd"] = float(np.log10(a).std(ddof=1))
        b = r_alt[k]
        row[f"{k}_cv_pct_nominal_period"] = float(100 * b.std(ddof=1) / b.mean())
    rec_rows.append(row)
    for c, w in enumerate(r["windows"]):
        cyc_rows.append(
            {
                "condition_id": int(CID[i]),
                "cycle": c,
                "start_sample": w["start"],
                "end_sample": w["end"],
                "full_window": w["full"],
                "edge_energy_frac": w["energy_frac"],
                "rms": float(r["rms"][c]),
                "vpp": float(r["vpp"][c]),
                "vmax": float(r["vmax"][c]),
            }
        )
REC = pd.DataFrame(rec_rows)
CYCD = pd.DataFrame(cyc_rows)
REC["segmentation_failed"] = REC["flag_weak_periodicity"] | REC["flag_too_few_cycles"]
CLEAN = ~REC["segmentation_failed"] & ~REC["flag_rate_deviates"]
CYCD.to_csv(os.path.join(RES, "a1_cycles.csv"), index=False)


def cv_summary(v):
    v = np.asarray(v, float)
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


rep = {
    "segmentation": {
        "method": "fixed-period windows of one tap period, phase-aligned to the minimum of the folded mean-square profile; truncated edge windows kept only if they retain >= %.2f of the folded energy"
        % EDGE_ENERGY_MIN,
        "period_rule": "tap period = nominal 1/f unless the envelope-autocorrelation period folds the signal with a phase-lock concentration higher by >= %.2f; recordings whose chosen tap rate differs from the nominal label by more than %.0f%% are listed as rate_deviates (data finding, not dropped)"
        % (PERIOD_MARGIN, 100 * RATE_DEV_MAX),
        "why_not_peak_detection": "excursions are one-sample spikes of very unequal height; a height-thresholded peak detector (distance 0.8/f s, height >= %.2f max|v|) finds fewer peaks than cycles at high f, dropping small cycles and biasing the CV downward; its count is kept as a cross-check"
        % PEAK_HEIGHT_FRAC,
        "flag_rule": "segmentation_failed if phase-lock concentration at the chosen period < %.2f or fewer than 3 cycles; failed and rate-deviating recordings stay in the primary summary and are excluded in sensitivity summaries"
        % LOCK_MIN,
        "n_period_from_acf": int((REC["period_source"] == "acf").sum()),
        "rate_estimator": "the deviation count n_rate_deviates uses the tap period chosen for segmentation: envelope-autocorrelation lag (smallest local maximum >= %.1f of the largest, 30-260 samples) with the half-period check (accept P/2 if parity >= %.2f and ACF(P/2)/ACF(P) >= %.2f), then kept only if it phase-locks at least %.2f better than the nominal period; n_rate_deviates_spectral is an independent cross-check from the dominant frequency of the rectified signal spectrum in %.0f-%.0f Hz, which can land on a harmonic"
        % (0.8, PARITY_MIN, HALF_ACF_MIN, PERIOD_MARGIN, SPEC_BAND[0], SPEC_BAND[1]),
        "n_half_period_accepted": int(REC["half_period_accepted"].sum()),
        "half_period_accepted_ids": [int(c) for c in REC.loc[REC["half_period_accepted"], "condition_id"]],
        "n_rate_deviates_spectral": int(REC["flag_rate_deviates_spectral"].sum()),
        "n_rate_deviates_both": int((REC["flag_rate_deviates"] & REC["flag_rate_deviates_spectral"]).sum()),
        "n_rate_deviates_excluding_copies": int((REC["flag_rate_deviates"] & ~REC["condition_id"].isin(EXCLUDE)).sum()),
        "n_rate_deviates": int(REC["flag_rate_deviates"].sum()),
        "n_clean": int(CLEAN.sum()),
        "rate_deviating_recordings": REC.loc[
            REC["flag_rate_deviates"],
            ["condition_id", "composition", "force_N", "freq_Hz", "tap_rate_used_Hz",
             "rate_rel_dev_from_nominal", "lock_concentration_nominal",
             "lock_concentration_acf", "half_period_accepted", "spectral_rate_Hz",
             "n_cycles", "duplicate_copy_of"],
        ]
        .assign(explained_by=lambda d_: [
            f"scaled copy of condition {o}, not a tapping effect" if o >= 0 else ""
            for o in d_["duplicate_copy_of"]])
        .to_dict("records"),
        "rate_deviates_by_freq": {
            int(k): int(g.sum()) for k, g in REC.groupby("freq_Hz")["flag_rate_deviates"]
        },
        "n_recordings": int(N),
        "n_cycles_total": int(REC["n_cycles"].sum()),
        "n_cycles_min": int(REC["n_cycles"].min()),
        "n_cycles_max": int(REC["n_cycles"].max()),
        "n_recordings_all_expected_cycles": int(
            (REC["n_cycles"] == REC["expected_cycles_used"]).sum()
        ),
        "n_recordings_one_short": int(
            (REC["n_cycles"] == REC["expected_cycles_used"] - 1).sum()
        ),
        "cycles_per_freq": {
            int(k): {"nominal": int(k), "min": int(g.min()), "max": int(g.max())}
            for k, g in REC.groupby("freq_Hz")["n_cycles"]
        },
        "peaks_crosscheck_per_freq": {
            int(k): {
                "min": int(g.min()),
                "median": float(g.median()),
                "max": int(g.max()),
            }
            for k, g in REC.groupby("freq_Hz")["n_peaks_crosscheck"]
        },
        "n_failed": int(REC["segmentation_failed"].sum()),
        "failed_recordings": REC.loc[
            REC["segmentation_failed"],
            [
                "condition_id",
                "composition",
                "force_N",
                "freq_Hz",
                "n_cycles",
                "lock_concentration",
            ],
        ].to_dict("records"),
        "per_recording": REC[
            [
                "condition_id",
                "composition",
                "force_N",
                "freq_Hz",
                "expected_cycles_nominal",
                "period_source",
                "tap_rate_used_Hz",
                "rate_rel_dev_from_nominal",
                "expected_cycles_used",
                "n_cycles",
                "n_full_cycles",
                "n_peaks_crosscheck",
                "lock_concentration",
                "flag_rate_deviates",
                "segmentation_failed",
            ]
        ].to_dict("records"),
    }
}
for k in ("rms", "vpp", "vmax"):
    blk = {
        "all": cv_summary(REC[f"{k}_cv_pct"]),
        "all_excluding_failed": cv_summary(
            REC.loc[~REC["segmentation_failed"], f"{k}_cv_pct"]
        ),
        "all_nominal_period": cv_summary(REC[f"{k}_cv_pct_nominal_period"]),
        "clean_only": cv_summary(REC.loc[CLEAN, f"{k}_cv_pct"]),
        "excluding_duplicates": cv_summary(REC.loc[DUPKEEP, f"{k}_cv_pct"]),
        "by_freq": {
            int(a): cv_summary(g) for a, g in REC.groupby("freq_Hz")[f"{k}_cv_pct"]
        },
        "by_force": {
            int(a): cv_summary(g) for a, g in REC.groupby("force_N")[f"{k}_cv_pct"]
        },
        "by_composition": {
            a: cv_summary(g) for a, g in REC.groupby("composition")[f"{k}_cv_pct"]
        },
        "relative_se_of_recording_mean_pct": cv_summary(REC[f"{k}_se_rel_pct"]),
    }
    # Kruskal-Wallis across levels (descriptive)
    for fac in ("freq_Hz", "force_N", "composition"):
        groups = [g.to_numpy() for _, g in REC.groupby(fac)[f"{k}_cv_pct"]]
        h, p = stats.kruskal(*groups)
        blk[f"kruskal_{fac}"] = {"H": float(h), "p": float(p)}
    # within-recording vs between-condition spread
    n_i = REC["n_cycles"].to_numpy()
    lin = CYCD.groupby("condition_id")[k]
    s2 = lin.var(ddof=1).to_numpy()
    means = lin.mean().to_numpy()
    within = float(np.sqrt(np.sum((n_i - 1) * s2) / np.sum(n_i - 1)))
    between = float(np.std(means, ddof=1))
    logc = CYCD.assign(lg=np.log10(CYCD[k])).groupby("condition_id")["lg"]
    ls2 = logc.var(ddof=1).to_numpy()
    lmeans = logc.mean().to_numpy()
    lwithin = float(np.sqrt(np.sum((n_i - 1) * ls2) / np.sum(n_i - 1)))
    lbetween = float(np.std(lmeans, ddof=1))
    # one-way random-effects ICC(1) on log10 cycle amplitude, unequal n (k0)
    Ncyc, g = n_i.sum(), len(n_i)
    grand = np.log10(CYCD[k]).mean()
    ssb = float(np.sum(n_i * (lmeans - grand) ** 2))
    ssw = float(np.sum((n_i - 1) * ls2))
    msb, msw = ssb / (g - 1), ssw / (Ncyc - g)
    k0 = (Ncyc - np.sum(n_i**2) / Ncyc) / (g - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    tcol = {"rms": "rms_Voc", "vpp": "Vpp", "vmax": "Vmax"}[k]
    blk["spread"] = {
        "within_recording_pooled_sd_V": within,
        "between_condition_sd_of_recording_mean_cycle_amplitude_V": between,
        "between_over_within_ratio": between / within,
        "between_condition_sd_of_recording_target_V": float(np.std(D[tcol], ddof=1)),
        "within_recording_pooled_sd_log10": lwithin,
        "between_condition_sd_log10": lbetween,
        "between_over_within_ratio_log10": lbetween / lwithin,
        "icc1_log10": float(icc),
        "icc1_k0": float(k0),
        "icc1_msb": msb,
        "icc1_msw": msw,
        "fraction_variance_between_conditions_log10": float(icc),
    }
    rep[k] = blk
J["repeatability"] = rep

# ----------------------------------------------------------------------------
# (2) split-half pseudo-replicates
# ----------------------------------------------------------------------------
HA = pd.DataFrame([descriptors(W[i, :500]) for i in range(N)])
HB = pd.DataFrame([descriptors(W[i, 500:]) for i in range(N)])
for t in TARGETS:
    REC[f"halfA_{t}"] = HA[t].to_numpy()
    REC[f"halfB_{t}"] = HB[t].to_numpy()
    REC[f"full_{t}"] = D[t].to_numpy(float)
REC.to_csv(os.path.join(RES, "a1_recordings.csv"), index=False)


def lin_ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sab = np.mean((a - a.mean()) * (b - b.mean()))
    return float(2 * sab / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = D[FEAT].to_numpy(float)


def gp():  # copied from code/baseline/physgp_analysis.py
    k = CK(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def fold_plain(tr, te, y, Xm):
    xs = StandardScaler().fit(Xm[tr])
    g = gp().fit(xs.transform(Xm[tr]), y[tr])
    mu, sd = g.predict(xs.transform(Xm[te]), return_std=True)
    return te, mu, sd


def loo_folds(n):
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


def run_cv(y, folds, Xm=None):
    Xm = X if Xm is None else Xm
    mu, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
    out = Parallel(n_jobs=-1)(delayed(fold_plain)(tr, te, y, Xm) for tr, te in folds)
    for te, m, s in out:
        mu[te], sd[te] = m, s
    assert not np.isnan(mu).any()
    return mu, sd


LOO = loo_folds(N)
LOSO = [(np.where(SPEC != s)[0], np.where(SPEC == s)[0]) for s in range(5)]

legacy_phys = json.load(
    open(os.path.join(REV, "results", "baseline", "physgp_results.json"))
)
pred_rows = []


def add_preds(target, scheme, y_true, mu, sd, train_on, held=None):
    for i in range(N):
        pred_rows.append(
            {
                "condition_id": int(CID[i]),
                "target": target,
                "model": "gp",
                "split_scheme": scheme,
                "train_on": train_on,
                "held_out_level": held[i] if held is not None else "",
                "y_true": float(y_true[i]),
                "y_pred": float(mu[i]),
                "sd": float(sd[i]),
                "lower": float(mu[i] - 1.96 * sd[i]),
                "upper": float(mu[i] + 1.96 * sd[i]),
            }
        )


split = {}
oof_full = {}
for t in TARGETS:
    a, b, yf = HA[t].to_numpy(), HB[t].to_numpy(), D[t].to_numpy(float)
    rel = (a - b) / ((a + b) / 2)
    la, lb = np.log10(a), np.log10(b)
    rng = np.random.default_rng(SEED)
    bi = rng.integers(0, N, (N_BOOT, N))
    ccc_b = np.array([lin_ccc(a[j], b[j]) for j in bi])
    lccc_b = np.array([lin_ccc(la[j], lb[j]) for j in bi])
    w = stats.wilcoxon(a, b)
    noise_var_full = float(np.mean((a - b) ** 2) / 4)
    ceiling = 1 - noise_var_full / float(np.var(yf))
    # GP refits
    mu_f, sd_f = run_cv(yf, LOO)
    mu_a, sd_a = run_cv(a, LOO)
    mu_b, sd_b = run_cv(b, LOO)
    oof_full[t] = mu_f
    add_preds(t, "loo", yf, mu_f, sd_f, "full")
    add_preds(t, "loo_halfA", a, mu_a, sd_a, "halfA")
    add_preds(t, "loo_halfB", b, mu_b, sd_b, "halfB")
    split[t] = {
        "halfA_mean_V": float(a.mean()),
        "halfB_mean_V": float(b.mean()),
        "ccc_raw": lin_ccc(a, b),
        "ccc_raw_ci95_record_boot": [
            float(np.percentile(ccc_b, 2.5)),
            float(np.percentile(ccc_b, 97.5)),
        ],
        "ccc_log10": lin_ccc(la, lb),
        "ccc_log10_ci95_record_boot": [
            float(np.percentile(lccc_b, 2.5)),
            float(np.percentile(lccc_b, 97.5)),
        ],
        "pearson_log10": float(np.corrcoef(la, lb)[0, 1]),
        "abs_rel_diff_pct": cv_summary(100 * np.abs(rel)),
        "abs_rel_diff_p90_pct": float(np.percentile(100 * np.abs(rel), 90)),
        "signed_rel_diff_mean_pct": float(100 * rel.mean()),
        "wilcoxon_halfA_vs_halfB": {
            "statistic": float(w.statistic),
            "p": float(w.pvalue),
        },
        "acquisition_noise_ceiling_R2": float(ceiling),
        "noise_ceiling_assumption": "full-recording noise variance = mean((A-B)^2)/4, exact for a mean of two halves, approximate for V_pp and |V|_max which are extremes over the two halves",
        "loo_R2_full": float(r2_score(yf, mu_f)),
        "loo_MAE_full": float(mean_absolute_error(yf, mu_f)),
        "loo_R2_legacy_full": float(legacy_phys[t]["gp_LOO_R2"]),
        "loo_R2_A_on_A": float(r2_score(a, mu_a)),
        "loo_R2_A_on_B": float(r2_score(b, mu_a)),
        "loo_R2_B_on_B": float(r2_score(b, mu_b)),
        "loo_R2_B_on_A": float(r2_score(a, mu_b)),
        "loo_R2_A_on_full": float(r2_score(yf, mu_a)),
        "loo_R2_B_on_full": float(r2_score(yf, mu_b)),
        "loo_MAE_A_on_A": float(mean_absolute_error(a, mu_a)),
        "loo_MAE_B_on_B": float(mean_absolute_error(b, mu_b)),
        "ccc_boot_percentiles": pct(ccc_b),
    }
J["split_half"] = {
    "definition": "half A = samples 0-499, half B = samples 500-999 of each 1 s recording; each descriptor recomputed after removing the half's own mean; LOO = within-grid leave-one-condition-out plain GP (legacy gp()), X standardized inside the fold; X_on_Y = trained by LOO on half X values, scored against half Y values",
    "ci_method": "percentile, %d record-level bootstrap resamples, seed %d"
    % (N_BOOT, SEED),
    "per_target": split,
    "halves": {"A": HA.to_dict("list"), "B": HB.to_dict("list")},
}

# ----------------------------------------------------------------------------
# (3) split-half ANOVA (log10 primary, raw secondary)
# ----------------------------------------------------------------------------
TERMS = [
    ("C(composition)", "composition", "Composition (specimen)"),
    ("C(force_N)", "force", "Force"),
    ("C(freq_Hz)", "frequency", "Frequency"),
    ("C(composition):C(force_N)", "composition:force", r"Composition $\times$ force"),
    (
        "C(composition):C(freq_Hz)",
        "composition:frequency",
        r"Composition $\times$ frequency",
    ),
    ("C(force_N):C(freq_Hz)", "force:frequency", r"Force $\times$ frequency"),
    (
        "C(composition):C(force_N):C(freq_Hz)",
        "composition:force:frequency",
        r"Composition $\times$ force $\times$ frequency",
    ),
]
FORM3 = "y ~ C(composition) * C(force_N) * C(freq_Hz)"
FORM2 = (
    "y ~ C(composition) + C(force_N) + C(freq_Hz) + C(composition):C(force_N)"
    " + C(composition):C(freq_Hz) + C(force_N):C(freq_Hz)"
)


TERM_KEYS = {"A": "C(composition)", "B": "C(force_N)", "C": "C(freq_Hz)",
             "AB": "C(composition):C(force_N)", "AC": "C(composition):C(freq_Hz)",
             "BC": "C(force_N):C(freq_Hz)",
             "ABC": "C(composition):C(force_N):C(freq_Hz)"}
# Type II comparisons: (term, terms of the reduced model, terms of the fuller model)
TYPE2 = [
    ("A", ["B", "C", "BC"], ["A", "B", "C", "BC"]),
    ("B", ["A", "C", "AC"], ["A", "B", "C", "AC"]),
    ("C", ["A", "B", "AB"], ["A", "B", "C", "AB"]),
    ("AB", ["A", "B", "C", "AC", "BC"], ["A", "B", "C", "AB", "AC", "BC"]),
    ("AC", ["A", "B", "C", "AB", "BC"], ["A", "B", "C", "AB", "AC", "BC"]),
    ("BC", ["A", "B", "C", "AB", "AC"], ["A", "B", "C", "AB", "AC", "BC"]),
    ("ABC", ["A", "B", "C", "AB", "AC", "BC"], list(TERM_KEYS)),
]


def _fit_terms(dat, terms):
    fit = smf.ols("y ~ " + " + ".join(TERM_KEYS[t] for t in terms), data=dat).fit()
    return float(fit.ssr), int(np.linalg.matrix_rank(fit.model.exog)), fit


def anova_block(dat):
    """Type II SS by explicit nested-model comparison with rank-based df.
    Equals statsmodels anova_lm(typ=2) for the balanced 75-condition design
    and stays correct when cells are empty (duplicate exclusion)."""
    sse, rank_full, fit = _fit_terms(dat, list(TERM_KEYS))
    df_e = len(dat) - rank_full
    mse = sse / df_e
    names = {key: name for key, name, _ in TERMS}
    out = {}
    for term, red, fuller in TYPE2:
        rss_r, rk_r, _ = _fit_terms(dat, red)
        rss_f, rk_f, _ = _fit_terms(dat, fuller)
        ss, df_t = rss_r - rss_f, rk_f - rk_r
        F_ = (ss / df_t) / mse
        out[names[TERM_KEYS[term]]] = {
            "df": int(df_t), "SS": float(ss), "F": float(F_),
            "p": float(stats.f.sf(F_, df_t, df_e)), "partial_eta2": float(ss / (ss + sse)),
        }
    out["acquisition_repeatability"] = {"df": int(df_e), "SS": float(sse), "MS": float(mse)}
    order = sorted([n for _, n, _ in TERMS], key=lambda n: -out[n]["partial_eta2"])
    return out, order, float(fit.rsquared)


anova = {
    "specification": "three-factor factorial ANOVA, OLS on 2N rows (N conditions x 2 half-recordings), composition x force x frequency with all two-way interactions and the three-way interaction as a separate term; Type II SS by nested-model comparison with rank-based df (identical to anova_lm typ=2 for the balanced 75-condition design); the error term (df = 2N - rank) is the between-half variation inside each recording = acquisition repeatability, not pure error; partial eta^2 = SS/(SS+SS_error)",
    "log10": {},
    "raw": {},
    "legacy_spec_full_recording_log10": {},
}
for t in TARGETS:
    for scale in ("log10", "raw"):
        tr = np.log10 if scale == "log10" else (lambda v: v)
        dat = pd.DataFrame(
            {
                "y": np.r_[tr(HA[t].to_numpy()), tr(HB[t].to_numpy())],
                "composition": np.r_[COMP, COMP],
                "force_N": np.r_[FORCE, FORCE],
                "freq_Hz": np.r_[FREQ, FREQ],
                "half": ["A"] * N + ["B"] * N,
            }
        )
        out, order, r2 = anova_block(dat)
        anova[scale][t] = {
            "terms": out,
            "ordering_by_partial_eta2": order,
            "model_R2": r2,
        }
    # legacy specification on the full recording (three-way pooled into residual)
    dat = pd.DataFrame(
        {
            "y": np.log10(D[t].to_numpy(float)),
            "composition": COMP,
            "force_N": FORCE,
            "freq_Hz": FREQ,
        }
    )
    fit = smf.ols(FORM2, data=dat).fit()
    tab = sm.stats.anova_lm(fit, typ=2)
    sse = float(tab.loc["Residual", "sum_sq"])
    anova["legacy_spec_full_recording_log10"][t] = {
        "residual_df": int(fit.df_resid),
        "residual_MS": sse / fit.df_resid,
        "partial_eta2": {
            n: float(tab.loc[k, "sum_sq"]) / (float(tab.loc[k, "sum_sq"]) + sse)
            for k, n, _ in TERMS[:6]
        },
    }
J["anova_split_half"] = anova

# ----------------------------------------------------------------------------
# (4) record-level vs specimen-cluster bootstrap of within-grid LOO R^2
# ----------------------------------------------------------------------------
def boot_draws(n):
    """Record and cluster draws; the same code path serves the primary run and
    the duplicate-excluded sensitivity so that the latter equals the primary
    of a --exclude-duplicates run."""
    rng_ = np.random.default_rng(SEED)
    return rng_.integers(0, n, (N_BOOT, n)), rng_.integers(0, 5, (N_BOOT, 5))


def boot_r2(y, mu, spec, rec_idx, clu_idx):
    mem = [np.where(spec == s_)[0] for s_ in range(5)]
    rb = np.array([r2_score(y[j], mu[j]) for j in rec_idx])
    cb = []
    for c in clu_idx:
        j = np.concatenate([mem[s_] for s_ in c])
        cb.append(r2_score(y[j], mu[j]))
    return rb, np.array(cb)


REC_IDX, CLU_IDX = boot_draws(N)
members = [np.where(SPEC == s)[0] for s in range(5)]
multisets = {tuple(sorted(c)) for c in CLU_IDX.tolist()}
n_distinct_spec = np.array([len(set(c)) for c in CLU_IDX.tolist()])
boot = {
    "method": "percentile bootstrap of R^2 computed on the fixed out-of-fold within-grid LOO predictions of the plain GP (no model refit), as in the legacy record-level bootstrap (regression/reg_stats.py boot_ci); record level resamples the N conditions, cluster level resamples the 5 specimens with replacement and takes all conditions of each drawn specimen; both use the same %d draws for every target, seed %d"
    % (N_BOOT, SEED),
    "n_boot": N_BOOT,
    "seed": SEED,
    "cluster_possible_multisets": 126,
    "cluster_distinct_multisets_drawn": len(multisets),
    "cluster_draws_single_specimen_only": int((n_distinct_spec == 1).sum()),
    "cluster_distinct_specimens_per_draw_counts": {
        int(k): int((n_distinct_spec == k).sum()) for k in range(1, 6)
    },
    "per_target": {},
}
draws = {}
for t in TARGETS:
    y, mu = D[t].to_numpy(float), oof_full[t]
    rb, cb = boot_r2(y, mu, SPEC, REC_IDX, CLU_IDX)
    draws[f"{t}_record"] = rb
    draws[f"{t}_cluster"] = cb
    per_spec = {
        COMPS[s]: float(r2_score(y[members[s]], mu[members[s]])) for s in range(5)
    }
    boot["per_target"][t] = {
        "loo_R2": float(r2_score(y, mu)),
        "record_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
        "cluster_ci95": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
        "record_percentiles": pct(rb),
        "cluster_percentiles": pct(cb),
        "record_sd": float(rb.std(ddof=1)),
        "cluster_sd": float(cb.std(ddof=1)),
        "within_specimen_loo_R2": per_spec,
    }
pd.DataFrame(draws).rename_axis("draw").to_csv(
    os.path.join(RES, "a1_bootstrap_draws.csv")
)
pd.DataFrame(CLU_IDX, columns=[f"spec{i}" for i in range(5)]).rename_axis(
    "draw"
).to_csv(os.path.join(RES, "a1_bootstrap_cluster_draws.csv"))
J["bootstrap"] = boot

# ----------------------------------------------------------------------------
# (5) held-out specimen (= held-out composition), plain GP
# ----------------------------------------------------------------------------
loso = {
    "definition": "train on the conditions of four specimens, predict all conditions of the fifth (15 each for the full design); within-level R^2 and MAE use only the held-out conditions (their own mean in the R^2 denominator); pooled R^2 uses all out-of-fold predictions with the global mean (legacy)",
    "per_target": {},
}
for t in TARGETS:
    y = D[t].to_numpy(float)
    mu, sd = run_cv(y, LOSO)
    add_preds(t, "held_out_specimen", y, mu, sd, "full", held=COMP)
    lv = {}
    for s in range(5):
        m = members[s]
        lv[COMPS[s]] = {
            "n": int(len(m)),
            "within_R2": float(r2_score(y[m], mu[m])),
            "MAE": float(mean_absolute_error(y[m], mu[m])),
            "mean_true_V": float(y[m].mean()),
            "mean_pred_V": float(mu[m].mean()),
        }
    w2 = [lv[c]["within_R2"] for c in COMPS]
    loso["per_target"][t] = {
        "per_specimen": lv,
        "pooled_R2": float(r2_score(y, mu)),
        "pooled_MAE": float(mean_absolute_error(y, mu)),
        "pooled_R2_legacy": float(legacy_phys[t]["gp_leave_one_composition_out_R2"]),
        "within_R2_min": float(min(w2)),
        "within_R2_max": float(max(w2)),
        "within_R2_median": float(np.median(w2)),
        "n_specimens_within_R2_negative": int(sum(v < 0 for v in w2)),
    }
J["held_out_specimen"] = loso

# ----------------------------------------------------------------------------
# (6) duplicate sensitivity: headline results without the scaled copies
# ----------------------------------------------------------------------------
POS = np.where(DUPKEEP)[0]
n_s = len(POS)
sens = {
    "definition": "headline results recomputed without recordings %s (scaled copies, duplicated-recordings note in the README) with the same code paths as the primary analysis; equals the primary results of a --exclude-duplicates run"
    % EXCLUDE,
    "excluded_ids": [int(c) for c in CID[~DUPKEEP]],
    "n": int(n_s),
    "cv_median": {k: float(np.median(REC.loc[DUPKEEP, f"{k}_cv_pct"])) for k in ("rms", "vpp", "vmax")},
    "per_target": {},
}
rec_s, clu_s = boot_draws(n_s)
sens["cluster_distinct_multisets_drawn"] = len({tuple(sorted(c)) for c in clu_s.tolist()})
Xs = X[POS]
for t in TARGETS:
    a, b, yf = HA[t].to_numpy()[POS], HB[t].to_numpy()[POS], D[t].to_numpy(float)[POS]
    mu_s, _ = run_cv(yf, loo_folds(n_s), Xm=Xs)
    rb, cb = boot_r2(yf, mu_s, SPEC[POS], rec_s, clu_s)
    dat = pd.DataFrame({
        "y": np.r_[np.log10(a), np.log10(b)],
        "composition": np.r_[COMP[POS], COMP[POS]],
        "force_N": np.r_[FORCE[POS], FORCE[POS]],
        "freq_Hz": np.r_[FREQ[POS], FREQ[POS]],
    })
    aout, aorder, _ = anova_block(dat)
    sens["per_target"][t] = {
        "ccc_raw": lin_ccc(a, b),
        "acquisition_noise_ceiling_R2": float(1 - np.mean((a - b) ** 2) / 4 / np.var(yf)),
        "loo_R2": float(r2_score(yf, mu_s)),
        "loo_MAE": float(mean_absolute_error(yf, mu_s)),
        "record_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
        "cluster_ci95": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
        "anova_log10_terms": aout,
        "anova_ordering_by_partial_eta2": aorder,
        "anova_ordering_same_as_primary": aorder == J["anova_split_half"]["log10"][t]["ordering_by_partial_eta2"],
    }
J["duplicate_sensitivity"] = sens
pd.DataFrame(pred_rows).to_csv(os.path.join(RES, "a1_predictions.csv"), index=False)

# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------
MACROS = {}  # name -> (string, json key)


def fmt(v, d):
    return f"{v:.{d}f}"


def sci(v, d=2):
    m, e = f"{v:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def mac(name, value, d, key):
    assert not any(ch.isdigit() for ch in name), name
    s = fmt(value, d) if isinstance(value, float) else str(value)
    MACROS[name] = (s, key)
    return s


# --- macros: repeatability ---
R = J["repeatability"]
mac(
    "specNCyclesTotal",
    R["segmentation"]["n_cycles_total"],
    0,
    "repeatability.segmentation.n_cycles_total",
)
mac(
    "specNCyclesMin",
    R["segmentation"]["n_cycles_min"],
    0,
    "repeatability.segmentation.n_cycles_min",
)
mac(
    "specNCyclesMax",
    R["segmentation"]["n_cycles_max"],
    0,
    "repeatability.segmentation.n_cycles_max",
)
mac(
    "specNRecAllCycles",
    R["segmentation"]["n_recordings_all_expected_cycles"],
    0,
    "repeatability.segmentation.n_recordings_all_expected_cycles",
)
mac(
    "specNRecOneShort",
    R["segmentation"]["n_recordings_one_short"],
    0,
    "repeatability.segmentation.n_recordings_one_short",
)
mac(
    "specNSegFailed",
    R["segmentation"]["n_failed"],
    0,
    "repeatability.segmentation.n_failed",
)
mac("specNRateDeviates", R["segmentation"]["n_rate_deviates"], 0, "repeatability.segmentation.n_rate_deviates")
mac("specNPeriodAcf", R["segmentation"]["n_period_from_acf"], 0, "repeatability.segmentation.n_period_from_acf")
mac("specNClean", R["segmentation"]["n_clean"], 0, "repeatability.segmentation.n_clean")
for fq in (5, 10, 15, 20, 25):
    mac(f"specNRateDeviates{NUMW[fq]}Hz", R["segmentation"]["rate_deviates_by_freq"][fq], 0, f"repeatability.segmentation.rate_deviates_by_freq.{fq}")
for k, st in (("rms", "Rms"), ("vpp", "Vpp"), ("vmax", "Vmax")):
    b = R[k]
    mac(f"specCvMedian{st}", b["all"]["median"], 1, f"repeatability.{k}.all.median")
    mac(f"specCvQlo{st}", b["all"]["q25"], 1, f"repeatability.{k}.all.q25")
    mac(f"specCvQhi{st}", b["all"]["q75"], 1, f"repeatability.{k}.all.q75")
    mac(f"specCvMin{st}", b["all"]["min"], 1, f"repeatability.{k}.all.min")
    mac(f"specCvMax{st}", b["all"]["max"], 1, f"repeatability.{k}.all.max")
    mac(
        f"specCvMedianNoFail{st}",
        b["all_excluding_failed"]["median"],
        1,
        f"repeatability.{k}.all_excluding_failed.median",
    )
    mac(
        f"specCvMedianNominal{st}",
        b["all_nominal_period"]["median"],
        1,
        f"repeatability.{k}.all_nominal_period.median",
    )
    mac(
        f"specCvMedianClean{st}",
        b["clean_only"]["median"],
        1,
        f"repeatability.{k}.clean_only.median",
    )
    mac(
        f"specRelSeMedian{st}",
        b["relative_se_of_recording_mean_pct"]["median"],
        1,
        f"repeatability.{k}.relative_se_of_recording_mean_pct.median",
    )
    for fq in (5, 10, 15, 20, 25):
        mac(
            f"specCvMedian{st}{NUMW[fq]}Hz",
            b["by_freq"][fq]["median"],
            1,
            f"repeatability.{k}.by_freq.{fq}.median",
        )
    for fo in (1, 2, 3):
        mac(
            f"specCvMedian{st}{NUMW[fo]}N",
            b["by_force"][fo]["median"],
            1,
            f"repeatability.{k}.by_force.{fo}.median",
        )
    sp = b["spread"]
    mac(
        f"specSdWithin{st}",
        sp["within_recording_pooled_sd_V"],
        3,
        f"repeatability.{k}.spread.within_recording_pooled_sd_V",
    )
    mac(
        f"specSdBetween{st}",
        sp["between_condition_sd_of_recording_mean_cycle_amplitude_V"],
        3,
        f"repeatability.{k}.spread.between_condition_sd_of_recording_mean_cycle_amplitude_V",
    )
    mac(
        f"specSdRatio{st}",
        sp["between_over_within_ratio"],
        1,
        f"repeatability.{k}.spread.between_over_within_ratio",
    )
    mac(
        f"specSdWithinLog{st}",
        sp["within_recording_pooled_sd_log10"],
        3,
        f"repeatability.{k}.spread.within_recording_pooled_sd_log10",
    )
    mac(
        f"specSdBetweenLog{st}",
        sp["between_condition_sd_log10"],
        3,
        f"repeatability.{k}.spread.between_condition_sd_log10",
    )
    mac(
        f"specSdRatioLog{st}",
        sp["between_over_within_ratio_log10"],
        1,
        f"repeatability.{k}.spread.between_over_within_ratio_log10",
    )
    mac(f"specIccLog{st}", sp["icc1_log10"], 2, f"repeatability.{k}.spread.icc1_log10")
    mac(
        f"specKwFreqP{st}",
        fmt_p(b["kruskal_freq_Hz"]["p"]),
        0,
        f"repeatability.{k}.kruskal_freq_Hz.p",
    )

# --- macros: split half ---
for t in TARGETS:
    st, s = TNAME[t], J["split_half"]["per_target"][t]
    mac(f"specCcc{st}", s["ccc_raw"], 3, f"split_half.per_target.{t}.ccc_raw")
    mac(
        f"specCccLo{st}",
        s["ccc_raw_ci95_record_boot"][0],
        3,
        f"split_half.per_target.{t}.ccc_raw_ci95_record_boot[0]",
    )
    mac(
        f"specCccHi{st}",
        s["ccc_raw_ci95_record_boot"][1],
        3,
        f"split_half.per_target.{t}.ccc_raw_ci95_record_boot[1]",
    )
    mac(f"specCccLog{st}", s["ccc_log10"], 3, f"split_half.per_target.{t}.ccc_log10")
    mac(
        f"specRelDiffMedian{st}",
        s["abs_rel_diff_pct"]["median"],
        1,
        f"split_half.per_target.{t}.abs_rel_diff_pct.median",
    )
    mac(
        f"specRelDiffPNinety{st}",
        s["abs_rel_diff_p90_pct"],
        1,
        f"split_half.per_target.{t}.abs_rel_diff_p90_pct",
    )
    mac(
        f"specNoiseCeiling{st}",
        s["acquisition_noise_ceiling_R2"],
        3,
        f"split_half.per_target.{t}.acquisition_noise_ceiling_R2",
    )
    mac(
        f"specSplitHalfLoo{st}A",
        s["loo_R2_A_on_A"],
        3,
        f"split_half.per_target.{t}.loo_R2_A_on_A",
    )
    mac(
        f"specSplitHalfLoo{st}B",
        s["loo_R2_B_on_B"],
        3,
        f"split_half.per_target.{t}.loo_R2_B_on_B",
    )
    mac(
        f"specSplitHalfLoo{st}AonB",
        s["loo_R2_A_on_B"],
        3,
        f"split_half.per_target.{t}.loo_R2_A_on_B",
    )
    mac(
        f"specSplitHalfLoo{st}BonA",
        s["loo_R2_B_on_A"],
        3,
        f"split_half.per_target.{t}.loo_R2_B_on_A",
    )
    mac(
        f"specLooFull{st}",
        s["loo_R2_full"],
        3,
        f"split_half.per_target.{t}.loo_R2_full",
    )
    mac(
        f"specLooLegacy{st}",
        s["loo_R2_legacy_full"],
        3,
        f"split_half.per_target.{t}.loo_R2_legacy_full",
    )

# --- macros: ANOVA ---
A3 = J["anova_split_half"]["log10"]
mac(
    "specAnovaErrorDf",
    A3["rms_Voc"]["terms"]["acquisition_repeatability"]["df"],
    0,
    "anova_split_half.log10.rms_Voc.terms.acquisition_repeatability.df",
)
for t in TARGETS:
    st = TNAME[t]
    tt = A3[t]["terms"]
    tw = tt["composition:force:frequency"]
    mac(
        f"specAnovaThreeWayF{st}",
        tw["F"],
        1,
        f"anova_split_half.log10.{t}.terms.composition:force:frequency.F",
    )
    mac(
        f"specAnovaThreeWayP{st}",
        fmt_p(tw["p"]),
        0,
        f"anova_split_half.log10.{t}.terms.composition:force:frequency.p",
    )
    mac(
        f"specAnovaThreeWayEta{st}",
        tw["partial_eta2"],
        2,
        f"anova_split_half.log10.{t}.terms.composition:force:frequency.partial_eta2",
    )
    for nm, stem in (
        ("composition", "Comp"),
        ("force", "Force"),
        ("frequency", "Freq"),
    ):
        mac(
            f"specAnovaEta{stem}{st}",
            tt[nm]["partial_eta2"],
            3,
            f"anova_split_half.log10.{t}.terms.{nm}.partial_eta2",
        )
        mac(
            f"specAnovaF{stem}{st}",
            tt[nm]["F"],
            1,
            f"anova_split_half.log10.{t}.terms.{nm}.F",
        )

# --- macros: bootstrap ---
B = J["bootstrap"]
mac("specBootN", B["n_boot"], 0, "bootstrap.n_boot")
mac(
    "specBootCluDistinct",
    B["cluster_distinct_multisets_drawn"],
    0,
    "bootstrap.cluster_distinct_multisets_drawn",
)
mac(
    "specBootCluPossible",
    B["cluster_possible_multisets"],
    0,
    "bootstrap.cluster_possible_multisets",
)
mac(
    "specBootCluSingle",
    B["cluster_draws_single_specimen_only"],
    0,
    "bootstrap.cluster_draws_single_specimen_only",
)
for t in TARGETS:
    st, b = TNAME[t], B["per_target"][t]
    mac(f"specBootLoo{st}", b["loo_R2"], 3, f"bootstrap.per_target.{t}.loo_R2")
    mac(
        f"specBootRecLo{st}",
        b["record_ci95"][0],
        3,
        f"bootstrap.per_target.{t}.record_ci95[0]",
    )
    mac(
        f"specBootRecHi{st}",
        b["record_ci95"][1],
        3,
        f"bootstrap.per_target.{t}.record_ci95[1]",
    )
    mac(
        f"specBootCluLo{st}",
        b["cluster_ci95"][0],
        3,
        f"bootstrap.per_target.{t}.cluster_ci95[0]",
    )
    mac(
        f"specBootCluHi{st}",
        b["cluster_ci95"][1],
        3,
        f"bootstrap.per_target.{t}.cluster_ci95[1]",
    )

# --- macros: held-out specimen ---
for t in TARGETS:
    st, lo = TNAME[t], J["held_out_specimen"]["per_target"][t]
    mac(
        f"specLosoPooled{st}",
        lo["pooled_R2"],
        2,
        f"held_out_specimen.per_target.{t}.pooled_R2",
    )
    mac(
        f"specLosoWithinMin{st}",
        lo["within_R2_min"],
        2,
        f"held_out_specimen.per_target.{t}.within_R2_min",
    )
    mac(
        f"specLosoWithinMax{st}",
        lo["within_R2_max"],
        2,
        f"held_out_specimen.per_target.{t}.within_R2_max",
    )
    mac(
        f"specLosoNegative{st}",
        lo["n_specimens_within_R2_negative"],
        0,
        f"held_out_specimen.per_target.{t}.n_specimens_within_R2_negative",
    )
    for c in COMPS:
        mac(
            f"specLosoWithin{st}{COMP_STEM[c]}",
            lo["per_specimen"][c]["within_R2"],
            2,
            f"held_out_specimen.per_target.{t}.per_specimen.{c}.within_R2",
        )
        mac(
            f"specLosoMae{st}{COMP_STEM[c]}",
            lo["per_specimen"][c]["MAE"],
            3,
            f"held_out_specimen.per_target.{t}.per_specimen.{c}.MAE",
        )

# --- macros: duplicate audit and duplicate sensitivity ---
DU = J["duplicates"]
mac("specDupNPairs", DU["n_pairs"], 0, "duplicates.n_pairs")
mac("specDupNExact", DU["n_exact"], 0, "duplicates.n_exact")
mac("specDupMaxOtherR", DU["max_abs_r_other_pairs"], 2, "duplicates.max_abs_r_other_pairs")
mac("specDupNExcl", J["duplicate_sensitivity"]["n"], 0, "duplicate_sensitivity.n")
for q, pr in zip(("A", "B", "C"), DU["pairs"]):
    kk = f"duplicates.pairs[{DU['pairs'].index(pr)}]"
    mac(f"specDupCopy{q}", pr["copy"], 0, kk + ".copy")
    mac(f"specDupOrig{q}", pr["original"], 0, kk + ".original")
    mac(f"specDupScale{q}", pr["scale"], 4 if pr["exact"] else 2, kk + ".scale")
    mac(f"specDupR{q}", pr["r"], 3, kk + ".r")
    mac(f"specDupResid{q}", "$" + sci(pr["relative_residual"], 1) + "$", 0, kk + ".relative_residual")
DS = J["duplicate_sensitivity"]
for k, st in (("rms", "Rms"), ("vpp", "Vpp"), ("vmax", "Vmax")):
    mac(f"specDupCvMedian{st}Excl", DS["cv_median"][k], 1, f"duplicate_sensitivity.cv_median.{k}")
mac("specDupBootCluDistinctExcl", DS["cluster_distinct_multisets_drawn"], 0, "duplicate_sensitivity.cluster_distinct_multisets_drawn")
ORDER_TEX = {"composition": "composition", "force": "force", "frequency": "frequency",
             "composition:force": r"composition$\times$force",
             "composition:frequency": r"composition$\times$frequency",
             "force:frequency": r"force$\times$frequency",
             "composition:force:frequency": "three-way interaction"}
for t in TARGETS:
    st, d_ = TNAME[t], DS["per_target"][t]
    base = f"duplicate_sensitivity.per_target.{t}"
    mac(f"specDupCcc{st}Excl", d_["ccc_raw"], 3, base + ".ccc_raw")
    mac(f"specDupNoiseCeiling{st}Excl", d_["acquisition_noise_ceiling_R2"], 3, base + ".acquisition_noise_ceiling_R2")
    mac(f"specDupLoo{st}Excl", d_["loo_R2"], 3, base + ".loo_R2")
    mac(f"specDupBootRecLo{st}Excl", d_["record_ci95"][0], 3, base + ".record_ci95[0]")
    mac(f"specDupBootRecHi{st}Excl", d_["record_ci95"][1], 3, base + ".record_ci95[1]")
    mac(f"specDupBootCluLo{st}Excl", d_["cluster_ci95"][0], 3, base + ".cluster_ci95[0]")
    mac(f"specDupBootCluHi{st}Excl", d_["cluster_ci95"][1], 3, base + ".cluster_ci95[1]")
    mac(f"specDupAnovaOrder{st}Excl", ", ".join(ORDER_TEX[o] for o in d_["anova_ordering_by_partial_eta2"]), 0, base + ".anova_ordering_by_partial_eta2")
    mac(f"specDupAnovaOrderSame{st}Excl", "yes" if d_["anova_ordering_same_as_primary"] else "no", 0, base + ".anova_ordering_same_as_primary")
    mac(f"specAnovaOrder{st}", ", ".join(ORDER_TEX[o] for o in J["anova_split_half"]["log10"][t]["ordering_by_partial_eta2"]), 0, f"anova_split_half.log10.{t}.ordering_by_partial_eta2")
SG = J["repeatability"]["segmentation"]
mac("specNRateDeviatesSpectral", SG["n_rate_deviates_spectral"], 0, "repeatability.segmentation.n_rate_deviates_spectral")
mac("specNRateDeviatesBoth", SG["n_rate_deviates_both"], 0, "repeatability.segmentation.n_rate_deviates_both")
mac("specNRateDeviatesNoCopy", SG["n_rate_deviates_excluding_copies"], 0, "repeatability.segmentation.n_rate_deviates_excluding_copies")
mac("specNHalfPeriod", SG["n_half_period_accepted"], 0, "repeatability.segmentation.n_half_period_accepted")

# ----------------------------------------------------------------------------
# tables (tabular bodies only)
# ----------------------------------------------------------------------------
TABLE_KEYS = {}


def write_table(name, body, keys):
    with open(os.path.join(TAB, f"tab_spec_{name}.tex"), "w") as fh:
        fh.write(
            f"% generated by code/a1_specimen.py; JSON: results/a1_specimen.json -> {', '.join(keys)}\n"
        )
        fh.write(body)
    TABLE_KEYS[f"results/tables/tab_spec_{name}.tex"] = keys


def cvcell(s):
    return f"{s['median']:.1f} [{s['q25']:.1f}, {s['q75']:.1f}]"


# repeatability
lines = [
    r"\begin{tabular}{lrrlll}",
    r"\toprule",
    r"Group & Rec. & Cycles & CV \RMS{} (\%) & CV \Vpp{} (\%) & CV \Vmax{} (\%) \\",
    r"\midrule",
]


def rep_row(label, mask, sel, cyc_col="n_cycles"):
    sub = REC[mask]
    cyc = (
        f"{int(sub[cyc_col].min())}--{int(sub[cyc_col].max())}"
        if sub[cyc_col].min() != sub[cyc_col].max()
        else f"{int(sub[cyc_col].min())}"
    )
    cells = [cvcell(sel(R[k])) for k in ("rms", "vpp", "vmax")]
    return f"{label} & {len(sub)} & {cyc} & " + " & ".join(cells) + r" \\"


for fq in (5, 10, 15, 20, 25):
    lines.append(
        rep_row(f"{fq} Hz", REC["freq_Hz"] == fq, lambda b, fq=fq: b["by_freq"][fq])
    )
lines.append(r"\midrule")
for fo in (1, 2, 3):
    lines.append(
        rep_row(f"{fo} N", REC["force_N"] == fo, lambda b, fo=fo: b["by_force"][fo])
    )
lines.append(r"\midrule")
lines.append(rep_row("All recordings", REC["freq_Hz"] > 0, lambda b: b["all"]))
lines.append(
    rep_row(
        "Excluding weak periodicity",
        ~REC["segmentation_failed"],
        lambda b: b["all_excluding_failed"],
    )
)
lines.append(
    rep_row("Rate matches label, periodic", CLEAN, lambda b: b["clean_only"])
)
lines.append(
    rep_row("Nominal period for all", REC["freq_Hz"] > 0, lambda b: b["all_nominal_period"], "n_cycles_nominal_period")
)
lines.append(
    rep_row("Without the three copies", DUPKEEP, lambda b: b["excluding_duplicates"])
)
lines.append(r"\midrule")
lines.append(
    r"\multicolumn{3}{l}{Within-recording SD, $\log_{10}$} & "
    + " & ".join(
        f"{R[k]['spread']['within_recording_pooled_sd_log10']:.3f}"
        for k in ("rms", "vpp", "vmax")
    )
    + r" \\"
)
lines.append(
    r"\multicolumn{3}{l}{Between-condition SD, $\log_{10}$} & "
    + " & ".join(
        f"{R[k]['spread']['between_condition_sd_log10']:.3f}"
        for k in ("rms", "vpp", "vmax")
    )
    + r" \\"
)
lines.append(
    r"\multicolumn{3}{l}{Between/within ratio} & "
    + " & ".join(
        f"{R[k]['spread']['between_over_within_ratio_log10']:.1f}"
        for k in ("rms", "vpp", "vmax")
    )
    + r" \\"
)
lines.append(
    r"\multicolumn{3}{l}{ICC(1), $\log_{10}$} & "
    + " & ".join(f"{R[k]['spread']['icc1_log10']:.2f}" for k in ("rms", "vpp", "vmax"))
    + r" \\"
)
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table(
    "repeatability",
    "\n".join(lines),
    [
        "repeatability.{rms,vpp,vmax}.by_freq",
        "repeatability.{rms,vpp,vmax}.by_force",
        "repeatability.{rms,vpp,vmax}.all",
        "repeatability.{rms,vpp,vmax}.all_excluding_failed",
        "repeatability.{rms,vpp,vmax}.spread",
        "repeatability.segmentation.per_recording",
    ],
)

# split half
SH = J["split_half"]["per_target"]
rows = [
    (
        "Lin CCC, half A vs half B",
        lambda s: f"{s['ccc_raw']:.3f} [{s['ccc_raw_ci95_record_boot'][0]:.3f}, {s['ccc_raw_ci95_record_boot'][1]:.3f}]",
    ),
    (
        r"Lin CCC, $\log_{10}$",
        lambda s: f"{s['ccc_log10']:.3f} [{s['ccc_log10_ci95_record_boot'][0]:.3f}, {s['ccc_log10_ci95_record_boot'][1]:.3f}]",
    ),
    (
        r"$|$A$-$B$|$ / mean, median (\%)",
        lambda s: f"{s['abs_rel_diff_pct']['median']:.1f}",
    ),
    (
        r"$|$A$-$B$|$ / mean, 90th pct. (\%)",
        lambda s: f"{s['abs_rel_diff_p90_pct']:.1f}",
    ),
    (
        r"Acquisition-noise ceiling on $R^2$",
        lambda s: f"{s['acquisition_noise_ceiling_R2']:.3f}",
    ),
    None,
    (r"LOO $R^2$, full recording (this run)", lambda s: f"{s['loo_R2_full']:.3f}"),
    (
        r"LOO $R^2$, full recording (submitted)",
        lambda s: f"{s['loo_R2_legacy_full']:.3f}",
    ),
    (r"LOO $R^2$, trained A, scored A", lambda s: f"{s['loo_R2_A_on_A']:.3f}"),
    (r"LOO $R^2$, trained A, scored B", lambda s: f"{s['loo_R2_A_on_B']:.3f}"),
    (r"LOO $R^2$, trained B, scored B", lambda s: f"{s['loo_R2_B_on_B']:.3f}"),
    (r"LOO $R^2$, trained B, scored A", lambda s: f"{s['loo_R2_B_on_A']:.3f}"),
]
lines = [
    r"\begin{tabular}{llll}",
    r"\toprule",
    r"Quantity & \RMS & \Vpp & \Vmax \\",
    r"\midrule",
]
for r_ in rows:
    if r_ is None:
        lines.append(r"\midrule")
        continue
    lab, fn = r_
    lines.append(f"{lab} & " + " & ".join(fn(SH[t]) for t in TARGETS) + r" \\")
DSP = J["duplicate_sensitivity"]["per_target"]
lines.append(r"\midrule")
lines.append(r"\multicolumn{4}{l}{Without the three copies, $n=" + str(J["duplicate_sensitivity"]["n"]) + r"$} \\")
for lab, key in (("Lin CCC, half A vs half B", "ccc_raw"),
                 (r"Acquisition-noise ceiling on $R^2$", "acquisition_noise_ceiling_R2"),
                 (r"LOO $R^2$, full recording", "loo_R2")):
    lines.append(f"{lab} & " + " & ".join(f"{DSP[t][key]:.3f}" for t in TARGETS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table("splithalf", "\n".join(lines), ["split_half.per_target", "duplicate_sensitivity.per_target"])

# ANOVA (log10)
lines = [
    r"\begin{tabular}{lr" + "rrr" * 3 + "}",
    r"\toprule",
    r" & & \multicolumn{3}{c}{$\log_{10}$\RMS} & \multicolumn{3}{c}{$\log_{10}$\Vpp} & \multicolumn{3}{c}{$\log_{10}$\Vmax} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
    r"Term & df & $F$ & $p$ & $\eta^2_p$ & $F$ & $p$ & $\eta^2_p$ & $F$ & $p$ & $\eta^2_p$ \\",
    r"\midrule",
]
for i, (_, nm, lab) in enumerate(TERMS):
    if i == 6:
        lines.append(r"\midrule")
    cells = []
    for t in TARGETS:
        tt = A3[t]["terms"][nm]
        cells += [f"{tt['F']:.1f}", fmt_p(tt["p"]), f"{tt['partial_eta2']:.3f}"]
    lines.append(
        f"{lab} & {A3['rms_Voc']['terms'][nm]['df']} & " + " & ".join(cells) + r" \\"
    )
lines.append(r"\midrule")
lines.append(
    r"Acquisition repeatability (error) & "
    + str(A3["rms_Voc"]["terms"]["acquisition_repeatability"]["df"])
    + " & "
    + " & ".join(
        r"\multicolumn{3}{c}{MS $= " + sci(A3[t]["terms"]["acquisition_repeatability"]["MS"]) + "$}"
        for t in TARGETS
    )
    + r" \\"
)
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table(
    "anova", "\n".join(lines), ["anova_split_half.log10.{rms_Voc,Vpp,Vmax}.terms"]
)

# bootstrap
lines = [
    r"\begin{tabular}{lllll}",
    r"\toprule",
    r"Target & LOO $R^2$ & Record-level 95\% CI & Specimen-cluster 95\% CI & Distinct cluster resamples \\",
    r"\midrule",
]
for t in TARGETS:
    b = B["per_target"][t]
    lines.append(
        f"{TTEX[t]} & {b['loo_R2']:.3f} & [{b['record_ci95'][0]:.3f}, {b['record_ci95'][1]:.3f}] & "
        f"[{b['cluster_ci95'][0]:.3f}, {b['cluster_ci95'][1]:.3f}] & {B['cluster_distinct_multisets_drawn']} of {B['cluster_possible_multisets']}"
        + r" \\"
    )
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{Without the three copies, $n=" + str(J["duplicate_sensitivity"]["n"]) + r"$} \\")
for t in TARGETS:
    b = J["duplicate_sensitivity"]["per_target"][t]
    lines.append(
        f"{TTEX[t]} & {b['loo_R2']:.3f} & [{b['record_ci95'][0]:.3f}, {b['record_ci95'][1]:.3f}] & "
        f"[{b['cluster_ci95'][0]:.3f}, {b['cluster_ci95'][1]:.3f}] & {J['duplicate_sensitivity']['cluster_distinct_multisets_drawn']} of {B['cluster_possible_multisets']}"
        + r" \\"
    )
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table(
    "bootstrap",
    "\n".join(lines),
    [
        "bootstrap.per_target",
        "bootstrap.cluster_distinct_multisets_drawn",
        "bootstrap.cluster_possible_multisets",
        "duplicate_sensitivity.per_target",
        "duplicate_sensitivity.cluster_distinct_multisets_drawn",
    ],
)

# duplicates
lines = [r"\begin{tabular}{rlrlrrl}", r"\toprule",
         r"Copy & Copy condition & Original & Original condition & Scale & $r$ & Rel.\ residual \\",
         r"\midrule"]
for pr in J["duplicates"]["pairs"]:
    cc = pr["copy_condition"].replace("%", r"\%").replace("$_3$", r"\textsubscript{3}")
    oc = pr["original_condition"].replace("%", r"\%").replace("$_3$", r"\textsubscript{3}")
    lines.append(
        f"{pr['copy']} & {cc} & {pr['original']} & {oc} & "
        f"{pr['scale']:.4f} & {pr['r']:.3f} & ${sci(pr['relative_residual'], 1)}$" + r" \\"
    )
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table("duplicates", "\n".join(lines), ["duplicates.pairs"])

# held-out specimen
LS = J["held_out_specimen"]["per_target"]
lines = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r" & \multicolumn{2}{c}{\RMS} & \multicolumn{2}{c}{\Vpp} & \multicolumn{2}{c}{\Vmax} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Held-out specimen & $R^2$ & MAE (V) & $R^2$ & MAE (V) & $R^2$ & MAE (V) \\",
    r"\midrule",
]
for c in COMPS:
    cells = []
    for t in TARGETS:
        v = LS[t]["per_specimen"][c]
        cells += [f"{v['within_R2']:.2f}", f"{v['MAE']:.3f}"]
    lines.append(f"{COMP_TEX[c]} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append(
    r"Pooled (global mean) & "
    + " & ".join(
        f"{LS[t]['pooled_R2']:.2f} & {LS[t]['pooled_MAE']:.3f}" for t in TARGETS
    )
    + r" \\"
)
lines += [r"\bottomrule", r"\end{tabular}", ""]
write_table("loso", "\n".join(lines), ["held_out_specimen.per_target"])

# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#DDDDDD",
        "pdf.fonttype": 42,
        "savefig.bbox": "standard",
        "mathtext.fontset": "dejavusans",
        "savefig.pad_inches": 0.02,
    }
)
TW = 17.4 / 2.54
EXAMPLE = {"composition": "PVDF+BaTiO3+%2CNT", "force_N": 2, "freqs": (5, 20)}


# rate-deviating example: condition 58 (+2 wt% MWCNT, 3 N, 20 Hz label), the same
# recording A4 cites (finding #5); asserted to be rate-deviating
RATE_EXAMPLE_ID = 58
RATE_EXAMPLE = int(np.where(CID == RATE_EXAMPLE_ID)[0][0])  # row position
assert bool(REC.loc[RATE_EXAMPLE, "flag_rate_deviates"])


def ex_id(fq):
    return int(
        np.where(
            (COMP == EXAMPLE["composition"])
            & (FORCE == EXAMPLE["force_N"])
            & (FREQ == fq)
        )[0][0]
    )


fig = plt.figure(figsize=(TW, 5.3), layout="constrained")
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])
gtop = outer[0].subgridspec(1, 3)
gbot = outer[1].subgridspec(1, 4)
axw = [fig.add_subplot(gtop[0, q]) for q in range(3)]
t_ms = np.arange(1000)
wave_ids = [ex_id(fq) for fq in EXAMPLE["freqs"]] + [RATE_EXAMPLE]
for ax, i, letter in zip(axw, wave_ids, "abc"):
    r = segs[i]
    ax.plot(t_ms, XD[i], color=CBLUE, lw=0.6)
    for w in r["windows"]:
        ax.axvline(w["start"], color=CGREY, lw=0.5, ls=":")
        seg = XD[i, w["start"] : w["end"]]
        ax.plot(w["start"] + np.argmax(seg), seg.max(), "v", color=CRED, ms=3)
        ax.plot(w["start"] + np.argmin(seg), seg.min(), "^", color=CAMBER, ms=3)
    ax.axvline(r["windows"][-1]["end"], color=CGREY, lw=0.5, ls=":")
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Time (ms)")
    if letter == "a":
        ax.set_ylabel("Recorded voltage (V)")
    ax.set_title(f"({letter}) {FREQ[i]} Hz label, {r['n_cycles']} cycles")
axr = fig.add_subplot(gbot[0, 0])
for c in COMPS:
    m = REC["composition"] == c
    jit = (COMPS.index(c) - 2) * 0.35
    ax_x = REC.loc[m, "freq_Hz"] + jit
    axr.scatter(ax_x, REC.loc[m, "tap_rate_used_Hz"], s=9, color=COMP_COLOR[c],
                marker=COMP_MARK[c], lw=0, zorder=3, label=COMP_LABEL[c])
dv = REC["flag_rate_deviates"]
axr.scatter(REC.loc[dv, "freq_Hz"] + (REC.loc[dv, "composition"].map(COMPS.index) - 2) * 0.35,
            REC.loc[dv, "tap_rate_used_Hz"], s=24, facecolor="none", edgecolor="black", lw=0.5, zorder=4)
xx = np.array([2.5, 27.5])
axr.plot(xx, xx, color=CGREY, lw=0.7, ls="--")
axr.fill_between(xx, xx * (1 - RATE_DEV_MAX), xx * (1 + RATE_DEV_MAX), color=CGREY, alpha=0.12, lw=0)
axr.set_xlim(2.5, 27.5); axr.set_ylim(0, 28)
axr.set_xticks([5, 10, 15, 20, 25])
axr.set_xlabel("Nominal $f$ (Hz)")
axr.set_ylabel("Tap rate used (Hz)")
axr.set_title("(d)")
axc = [fig.add_subplot(gbot[0, q]) for q in (1, 2, 3)]
for ax, k, lab, letter in zip(
    axc,
    ("rms", "vpp", "vmax"),
    (r"$V_\mathrm{rms}$", r"$V_\mathrm{pp}$", r"$|V|_\mathrm{max}$"),
    "efg",
):
    freqs = [5, 10, 15, 20, 25]
    data = [REC.loc[REC["freq_Hz"] == fq, f"{k}_cv_pct"].to_numpy() for fq in freqs]
    ax.boxplot(data, positions=range(5), widths=0.55, showfliers=False,
               medianprops={"color": "black", "lw": 1}, boxprops={"lw": 0.6},
               whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
    for j, fq in enumerate(freqs):
        sub = REC[REC["freq_Hz"] == fq]
        for c in COMPS:
            ss = sub[sub["composition"] == c]
            jitter = (COMPS.index(c) - 2) * 0.07
            ax.scatter(np.full(len(ss), j + jitter), ss[f"{k}_cv_pct"], s=9,
                       color=COMP_COLOR[c], marker=COMP_MARK[c], lw=0, zorder=3)
            fl = ss[ss["flag_rate_deviates"]]
            ax.scatter(np.full(len(fl), j + jitter), fl[f"{k}_cv_pct"], s=22,
                       facecolor="none", edgecolor="black", lw=0.5, zorder=4)
            wk = ss[ss["segmentation_failed"]]
            ax.scatter(np.full(len(wk), j + jitter), wk[f"{k}_cv_pct"], s=18,
                       marker="x", color="black", lw=0.6, zorder=5)
    ax.set_xticks(range(5), [str(v) for v in freqs])
    ax.set_xlabel("Nominal $f$ (Hz)")
    if k == "rms":
        ax.set_ylabel("Cycle CV (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"({letter}) {lab}")
h, l = axr.get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=5,
           handletextpad=0.2, columnspacing=1.0)
fig.savefig(os.path.join(FIG, "fig_spec_cycles.pdf"), metadata={"CreationDate": None, "ModDate": None})
fig.savefig(os.path.join(FIG, "fig_spec_cycles.png"), dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(TW, 2.9), layout="constrained")
for ax, t, lab, letter in zip(
    axs,
    TARGETS,
    (r"$V_\mathrm{rms}$", r"$V_\mathrm{pp}$", r"$|V|_\mathrm{max}$"),
    "abc",
):
    a, b = HA[t].to_numpy(), HB[t].to_numpy()
    for c in COMPS:
        m = COMP == c
        ax.scatter(
            a[m],
            b[m],
            s=12,
            color=COMP_COLOR[c],
            marker=COMP_MARK[c],
            lw=0,
            label=COMP_LABEL[c] if t == "rms_Voc" else None,
        )
    lo, hi = min(a.min(), b.min()) / 1.3, max(a.max(), b.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color=CGREY, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{lab}, first 0.5 s (V)")
    ax.set_ylabel(f"{lab}, second 0.5 s (V)")
    ax.set_title(f"({letter})")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=5,
    handletextpad=0.2,
    columnspacing=1.0,
)
fig.savefig(os.path.join(FIG, "fig_spec_splithalf.pdf"), metadata={"CreationDate": None, "ModDate": None})
fig.savefig(os.path.join(FIG, "fig_spec_splithalf.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# captions (dense declarative prose; numbers from JSON)
# ----------------------------------------------------------------------------
seg = R["segmentation"]
DSx = J["duplicate_sensitivity"]
DUx = J["duplicates"]
nfq = int((REC["freq_Hz"] == 5).sum())
cap = f"""# Captions generated by A1 (generated by code/a1_specimen.py)

## fig_spec_cycles

Cycle-level amplitude repeatability inside single recordings. Panels (a) and (b) show the demeaned recorded voltage of the PVDF/BaTiO3 specimen with 2 wt% MWCNT at {EXAMPLE['force_N']} N over the full 1 s recording at the 5 Hz and 20 Hz labels, and panel (c) shows the same specimen at 3 N under the 20 Hz label, where segmentation at the autocorrelation period finds only {segs[RATE_EXAMPLE]["n_cycles"]} impact cycles in 1 s. Dotted lines mark the fixed-period cycle windows, phase-aligned to the quietest point of the folded mean-square profile, and the markers show the maximum and minimum inside each window. Panel (d) plots the tap rate used for segmentation against the nominal frequency label for all {N} recordings, with the dashed identity line and a shaded band of plus or minus {100 * RATE_DEV_MAX:.0f}%. Black rings mark the {seg['n_rate_deviates']} recordings whose tap rate falls outside that band. The rate comes from the envelope autocorrelation with a half-period check and replaces the nominal period only where it locks the signal clearly better. Panels (e) to (g) show, for each recording, the coefficient of variation across the cycles of that recording of the per-cycle RMS voltage, peak-to-peak voltage and peak absolute voltage, grouped by nominal frequency and colored by composition. Boxes span the interquartile range over the recordings at each nominal frequency, rings repeat the rate flag and crosses mark the {seg['n_failed']} recording with weak periodicity. All flagged recordings remain in every summary. Each coefficient of variation is computed over the {seg['n_cycles_min']} to {seg['n_cycles_max']} cycles of one recording on one specimen, so it measures acquisition repeatability and not specimen-to-specimen or fabrication variability.

## fig_spec_splithalf

Split-half agreement of the three voltage descriptors. Each point is one of the {N} recordings, with the descriptor computed on the first 0.5 s on the horizontal axis and on the second 0.5 s on the vertical axis, each half demeaned by its own mean. Panels (a), (b) and (c) show the RMS voltage, the peak-to-peak voltage and the peak absolute voltage on logarithmic axes, and the dashed line is the identity. The two halves come from the same recording of the same specimen, so their agreement measures acquisition repeatability and does not replace an independent replicate specimen.

## tab_spec_repeatability

Coefficient of variation of the per-cycle amplitude across the tapping cycles of one recording, summarized as the median and interquartile range over the recordings in each group. Rec. is the number of recordings in the group and Cycles is the range of cycles segmented per recording. Frequency groups use the nominal label. The primary rows segment every recording at its chosen tap period, which is the nominal period unless the envelope autocorrelation period locks the signal clearly better. The sensitivity rows exclude the recording with weak periodicity, keep only the {seg['n_clean']} recordings whose tap rate matches the label within {100 * RATE_DEV_MAX:.0f}% and that lock well, force the nominal period on all {N} recordings, or drop the three recordings that are scaled copies of other recordings. The lower block compares the pooled within-recording standard deviation of the log10 cycle amplitude with the standard deviation of the recording-mean log10 amplitude across the {N} conditions, and ICC(1) is the share of log10 cycle-amplitude variance that lies between conditions. All variation in this table arises inside single recordings of single specimens and is acquisition repeatability.

## tab_spec_splithalf

Agreement between the first and second 0.5 s of each recording and the within-grid leave-one-out R2 of the plain Gaussian process refitted on each half. Brackets give 95% percentile intervals from {N_BOOT} record-level bootstrap resamples. The acquisition-noise ceiling is one minus the mean squared half difference divided by four and by the variance of the full-recording target, which is the largest R2 a perfect model could reach if the split-half difference were the only noise. Trained A, scored B means leave-one-out predictions from models fitted to first-half values and compared with second-half values of the held-out condition. The bottom block repeats the agreement, the ceiling and the full-recording leave-one-out R2 on the {DSx['n']} recordings that remain after removing the three scaled copies.

## tab_spec_anova

Three-factor factorial ANOVA on log10 descriptors with the two half-recordings of each condition entering as pseudo-replicates, giving {2 * N} observations. The model contains the three main effects, the three two-way interactions and the three-way interaction as a separate term, with Type II sums of squares. The error term has {A3['rms_Voc']['terms']['acquisition_repeatability']['df']} degrees of freedom and is the variation between the two halves of one recording, so it is acquisition repeatability and not pure error. Composition is confounded with specimen, so the composition row tests differences among five individual specimens. Partial eta squared is the term sum of squares divided by the sum of itself and the error sum of squares.

## tab_spec_bootstrap

Within-grid leave-one-out R2 of the plain Gaussian process with two bootstrap 95% percentile intervals computed on the same fixed out-of-fold predictions over {N_BOOT} resamples. The record-level interval resamples the {N} conditions as if they were independent. The specimen-cluster interval resamples the five specimens with replacement and keeps all conditions of each drawn specimen. Only {B['cluster_possible_multisets']} distinct specimen multisets exist, and the last column gives how many of them the resamples visited. The cluster interval reuses the within-grid leave-one-out predictions, in which every test condition had the other conditions of its own specimen in the training set, and five clusters make a percentile interval anti-conservative. It therefore describes the spread of within-grid accuracy across specimens and is not an interval for a newly fabricated device, which Table tab_spec_loso addresses by holding out whole specimens. The bottom block repeats the analysis on the {DSx['n']} recordings that remain after removing the three scaled copies.

## tab_spec_loso

Held-out specimen evaluation of the plain Gaussian process, which equals held-out composition because each composition is one specimen. Each row trains on the conditions of the other four specimens and predicts every condition of the named specimen. R2 and MAE are computed within the held-out conditions with their own mean in the R2 denominator. The pooled row uses all {N} out-of-fold predictions and the global mean, as in the first-stage analysis.

## tab_spec_duplicates

Recordings that are scaled copies of other recordings, found by correlating every pair of the {N_ALL} demeaned waveforms at lags up to {DUP_MAX_LAG} ms and listing every pair with an absolute correlation of at least {DUP_R_MIN}. Scale is the least-squares factor that maps the original onto the copy and the relative residual is the norm of the copy minus the scaled original divided by the norm of the copy. Copy is the later condition index. {DUx['n_exact']} of the {DUx['n_pairs']} pairs are exact multiples to numerical precision, and no other pair exceeds an absolute correlation of {DUx['max_abs_r_other_pairs']:.2f}. Independent acquisitions cannot produce exact multiples, so these recordings are reported with sensitivity results computed without them.
"""
with open(os.path.join(FIG, "captions_a1.md"), "w") as fh:
    fh.write(cap)

# ----------------------------------------------------------------------------
# macros file, index, JSON
# ----------------------------------------------------------------------------
with open(os.path.join(RES, "numbers_a1.tex"), "w") as fh:
    fh.write(
        "% numbers_a1.tex: generated by code/a1_specimen.py; source results/a1_specimen.json\n"
    )
    for name, (s, key) in MACROS.items():
        fh.write(f"\\newcommand{{\\{name}}}{{{s}}}% {key}\n")

J["macros"] = {n: {"value": s, "json_key": k} for n, (s, k) in MACROS.items()}
J["index"] = {
    **TABLE_KEYS,
    "results/numbers_a1.tex": "macros (each entry lists its json_key)",
    "figures/fig_spec_cycles.pdf": [
        "repeatability.segmentation.per_recording",
        "repeatability.{rms,vpp,vmax}.by_freq",
        "results/a1_recordings.csv",
        "results/a1_cycles.csv",
    ],
    "figures/fig_spec_splithalf.pdf": ["split_half.halves"],
    "results/a1_predictions.csv": "out-of-fold plain GP predictions: loo (full), loo_halfA, loo_halfB, held_out_specimen",
    "results/a1_bootstrap_draws.csv": "bootstrap.per_target record and cluster R^2 draws",
    "results/a1_bootstrap_cluster_draws.csv": "specimen indices of each cluster draw",
    "example_recordings_fig_spec_cycles": {
        "composition": EXAMPLE["composition"],
        "force_N": EXAMPLE["force_N"],
        "condition_ids": [int(CID[ex_id(fq)]) for fq in EXAMPLE["freqs"]],
        "rate_example_condition_id": RATE_EXAMPLE_ID,
    },
    "results/tables/tab_spec_duplicates.tex": ["duplicates.pairs"],
    "duplicate sensitivity macros specDup*Excl": ["duplicate_sensitivity"],
}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(os.path.join(RES, "a1_specimen.json"), "w") as fh:
    json.dump(clean(J), fh, indent=1)


print(
    "A1 done: cycles {}-{} per recording, {} flagged; median cycle CV Vpp {:.1f}%; "
    "split-half CCC rms {:.3f}; LOO rms full {:.3f} A {:.3f} B {:.3f}; "
    "cluster CI rms [{:.3f}, {:.3f}] over {} distinct resamples; {} macros".format(
        seg["n_cycles_min"],
        seg["n_cycles_max"],
        seg["n_failed"],
        R["vpp"]["all"]["median"],
        SH["rms_Voc"]["ccc_raw"],
        SH["rms_Voc"]["loo_R2_full"],
        SH["rms_Voc"]["loo_R2_A_on_A"],
        SH["rms_Voc"]["loo_R2_B_on_B"],
        B["per_target"]["rms_Voc"]["cluster_ci95"][0],
        B["per_target"]["rms_Voc"]["cluster_ci95"][1],
        B["cluster_distinct_multisets_drawn"],
        len(MACROS),
    )
)
