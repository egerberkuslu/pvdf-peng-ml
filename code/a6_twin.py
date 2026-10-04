#!/usr/bin/env python3
r"""Task E1 (Protocol V4): a simulated generator of the tapping rig with known ground truth.

One command from the repository root
    python code/a6_twin.py
writes, under results/ only,
    a6_twin.json                    every number, with an "index" dict
    a6_replicates.csv               one row per synthetic grid (registry)
    a6_figdata_coverage.csv         figure panel: coverage against specimen CV
    a6_figdata_scope.csv            figure panel: scope recovery (LawGP minus GP per axis)
    tables/tab_a6_*.tex             booktabs tabular bodies
    numbers_a6.tex                  \newcommand macros, prefix twin

1. The twin (one recording of condition (c, F, f) on specimen s)
------------------------------------------------------------------
Tap train.  The shaker is driven at the nominal label f; the realized tap rate is
    r = f (1 + delta),   delta ~ empirical deviations at the same nominal force
drawn from a1_recordings.csv column rate_rel_dev_from_nominal (project notes),
recordings 52, 54, 73 excluded (duplicated-recordings note in the README); unsampled forces pool the two
neighbouring force levels.  Tap times t_k = phi + k / r + e_k, phi ~ U(0, 1/r),
e_k ~ N(0, sigma_j^2), inside the 1 s record; sigma_j is the robust SD of the
inter-tap intervals of the 72 real waveforms divided by sqrt(2) (peaks above 40
percent of the maximum).  Without the jitter every pulse of an integer-period train
is sampled at the same sub-millisecond phase, which the real recordings do not show.

Charge and loading.  Each tap is a Gaussian force pulse
    F_k(t) = Fhat_k exp(-(t - t_k)^2 / (2 sigma_t^2)),   q(t) = d33(c) s sum_k F_k(t)
(q = d33 F, Koc et al. 2025 Eqs. 10 to 12 via project notes).  The DAQ input is a
resistor R_in = 144 kOhm in parallel with the film capacitance C_p (R_p >> R_in,
project notes), so the recorded voltage solves
    C_p dV/dt + V / R_in = dq/dt,
i.e. V is R_in dq/dt low-passed at f_c = 1 / (2 pi R_in C_p); the equation is
integrated exactly on a 2 us grid and sampled at 1 kHz (1000 samples, as recorded).
    C_p = eps0 eps_r(c) A / t,   A = 12.25 cm^2, t = 155 um (140 to 170 um),
    eps_r(pristine) = 10 (mid row of the R1 capacitance table, project notes),
    eps_r / eps_r(pristine) = 1.00, 0.90, 0.90, 0.90, 1.57 for PVDF, +BaTiO3, +1, +2,
    +3 wt% MWCNT (Koc et al. 2025 Fig. 7 as read in project notes: 3 wt%
    is 1.57 times pristine, the other composites below pristine; 0.90 is our
    assumption), linear between 2 and 3 wt% for unsampled compositions.
This gives C_p = 0.63 to 1.1 nF, inside the 0.19 to 7.7 nF range of project notes;
the optimum simulation also runs with C_p(3 wt%) = 7.7 nF (upper end) and with
negligible C_p (pure current mode).  The force-pulse SD sigma_t is calibrated (below);
the share of above-half-maximum excursions lasting one sample, which R1 found to be
all of them in the real recordings, is reported as a fidelity statistic.

Composition, force and frequency (the ground truth).  The per-tap amplitude is
    d33(c) s Fhat_k = M(c, F, f) s (r / f)^(kappa - 1/2) a_k / (C0 sqrt(f)),
    C0 = sqrt((1 + beta^2) int g_ref(t)^2 dt),
with g_ref the recorded pulse of a unit-charge tap at the pristine C_p, so that
    sqrt(E[V_rms^2]) = M(c, F, f) s (r / f)^kappa h(c),  h(c) = ||g_c|| / ||g_ref||.
M is the ground-truth response surface:
    law-true   M = (a0 + a1 c + a2 c^2) F L(f; f0, gamma),
               L(f; f0, gamma) = (gamma/2)^2 / ((f - f0)^2 + (gamma/2)^2),
               (a0, a1, a2, f0, gamma) = the A2 five-parameter fit to the full real
               V_rms grid (a2_common.fit_law5; project notes);
    two-mode   M = (b0 + b1 c + b2 c^2) F [w L(f; f1, g1) + (1 - w) L(f; f2, g2)],
               the best two-mode fit to the same real grid (bounded least squares,
               f1 in [15, 25] Hz, f2 in [3, 14] Hz); the single Lorentzian is then
               misspecified while force and composition factors keep the law's form.
The pristine specimen has c = 0 like PVDF/BaTiO3, as in the paper's law; pristine and
PVDF/BaTiO3 differ in the twin only through C_p and their specimen draws.
kappa is the slope of log(y / law) on log(r / f) over the 72 real recordings (0 would
mean the tap-rate deviation leaves V_rms unchanged, 1/2 that each tap carries a fixed
charge); its estimate is stored in the JSON.

Specimen variability.  s = exp(sigma_s z - sigma_s^2 / 2), z ~ N(0, 1),
sigma_s^2 = ln(1 + CV^2), CV in {0, 10, 20, 30} percent (Protocol V4 E1), one draw
per fabricated specimen, shared by every recording of that specimen.

Acquisition noise.  Per-tap amplitude jitter a_k = exp(sigma_a z_k - sigma_a^2)
(E[a_k^2] = 1); a background AR(1) process with lag-one correlation rho (median over
the quiet samples of the 75 real waveforms) and SD beta times the RMS of the pulse
train; quantization to the level spacing Delta of the recorded waveforms
(data/long.parquet).  (sigma_t, sigma_a) are calibrated by grid search
so that the twin's median cycle-to-cycle CV of V_rms equals the measured 21.1 percent
(a1_recordings.csv rms_cv_pct, project notes) and its SD of log(V_rms first half /
second half) equals the measured value (a1 halfA/halfB), and beta so that the median
robust background SD over V_rms matches; the cycle CVs of V_pp and |V|_max, their
split-half statistics, the split-half concordances and the one-sample share are left
uncalibrated and reported as fidelity checks (the twin under-disperses the peak
descriptors, which is why only V_rms goes through the protocol).

Descriptor.  V_rms = SD of the 1000 samples (a1_specimen.descriptors).  Only V_rms is
run through the protocol (the primary target of the paper).

2. Designs
----------
one      one specimen per composition, 15 recordings each (the real design);
three    three specimens per composition, specimen (i_F + i_f) mod 3 records the cell
         (a Latin assignment: every specimen sees all five frequencies);
fifteen  one specimen per recording (fully exchangeable).
All three have the same 75 conditions and inputs, so only exchangeability differs.
Extra recordings per grid: a repeat recording of every condition on the same specimen,
one on a new specimen of the same composition, and unsampled levels: compositions
0.5, 1.5, 2.5 wt% (with BaTiO3; new specimens), forces 1.5 and 2.5 N, frequencies
7.5, 12.5, 17.5 and 22.5 Hz, each on the existing specimen (specimen 0 of the
composition) and on a new one.

3. The protocol replica (paper code, a2_common / a3_common / a5_common)
-----------------------------------------------------------------------
Plain GP = a2_common.gp(); LawGP = a2_common.fit_law5 on the training rows plus the
plain GP on the residual; ProductGP = a5_common.gp_struct("product", 0).  Held-out
composition, force and frequency levels are refitted from the paper's start (all three
models); pooled R^2 with global denominator and within-level R^2.  LOO for the plain GP
(and LawGP on the real design) with the optimizer warm-started at the full-grid fit.
Jackknife+ (a3_common.jk_bounds) at nominal 95 and 90:
  group-wise   every inner LOO model refitted (hyperparameters re-optimized from the
               outer fold's kernel, LawGP law refitted from the outer law): plain GP
               on all designs, LawGP on the real design;
  within grid  nested LOO for the plain GP with the inner models in closed form
               (outer kernel and scaling kept, rank-one downdate of the kernel inverse,
               per-inner-set y normalization as sklearn);
  unsampled    from the 75 LOO models.
Checks (stored in the JSON): on the real grid the replica reproduces A2's pooled and
LOO R^2 and A3's raw-band counts; warm-started exact refits reproduce cold-start
(A3-style) refits; the closed-form inner models are anti-conservative for the
group-wise construction and are used only within the grid, where an exact nested run
on a subset of grids measures the gap.  ProductGP runs on the real design only, on the
group-wise folds and the full-grid fit.

4. Optimum reproducibility
--------------------------
Simulation only (no model fits), n_opt_rep grids per generator, CV, design and
capacitance scenario: probability that the composition with the highest mean over its
15 recordings, or the composition of the single best recording, is the true optimum
(argmax of M(c) h(c)), and the probability that two independent fabrications agree.

Seeds: replicate r uses data seed r (r = 0 .. n_rep - 1) with common random numbers
across generators, CV levels and designs; calibration uses seeds 1000 to 1003 (search)
and 1000 to 1007 (final statistics); models are deterministic (restarts 0) except
ProductGP (random_state 0, as A5).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_common as A  # noqa: E402
import a3_common as C3  # noqa: E402
import a5_common as B  # noqa: E402

RES = os.path.join(A.REV, "results")
TAB = os.path.join(RES, "tables")
CACHE = os.path.join(RES, "a6_cache")
OUT_JSON = os.path.join(RES, "a6_twin.json")
T0 = time.time()

# ---------------------------------------------------------------- constants
FS = 1000.0  # Hz, DAQ sampling (protocol 1: 1000 samples over 1 s)
NS = 1000
R_IN = 144e3  # Ohm, NI USB-6009 AI input (project notes)
AREA = 12.25e-4  # m^2 electrode area (project notes, Koc 2025 Eq. 3)
THICK = 155e-6  # m, middle of 140 to 170 um (project notes)
EPS0 = 8.8541878128e-12
EPS_R_PRISTINE = 10.0  # R1 capacitance table, middle row (range 3 to 100)
EPS_RATIO_BTO = 0.90  # assumption: "below pristine" (R1 reading of Koc 2025 Fig. 7)
EPS_RATIO_3WT = 1.57  # Koc 2025 Fig. 7 via R1: 3 wt% is 1.57 x pristine
SIGMA_T = 0.3e-3  # s, force-pulse SD (R1: excursions above half max last one sample)
DT_FINE = 2e-6
DUP_IDS = [52, 54, 73]  # duplicated-recordings note in the README
CV_LEVELS = [0.0, 0.10, 0.20, 0.30]
DESIGNS = ["one", "three", "fifteen"]
GENS = ["law", "twomode"]
NOMINAL = {"95": 0.05}  # 90 percent level dropped for runtime (coordinator, 2026-09-26)
Z = C3.Z
PRODUCT_SEED = 0  # a5_common.fold_predict default
UNS_C = [0.5, 1.5, 2.5]
UNS_F = [1.5, 2.5]
UNS_FREQ = [7.5, 12.5, 17.5, 22.5]
FORCES = [1, 2, 3]
FREQS = [5, 10, 15, 20, 25]
CAL_SEEDS = list(range(1000, 1008))

X = A.X  # (75, 4): cnt_pct, is_pristine, force_N, freq_Hz
N = A.N
COMP_IDX = np.array([A.COMP_ORDER.index(c) for c in A.COMP])
F_IDX = np.array([FORCES.index(int(v)) for v in A.FRC])
FR_IDX = np.array([FREQS.index(int(v)) for v in A.FRQ])
Y_REAL = A.df["rms_Voc"].values.astype(float)
AXES = ["composition", "force", "frequency"]
AXIS_GROUPS = {"composition": A.COMP, "force": A.FRC, "frequency": A.FRQ}
CV_WORD = {0: "Zero", 10: "Ten", 20: "Twenty", 30: "Thirty"}
GEN_WORD = {"law": "True", "twomode": "Two"}
DES_WORD = {"one": "One", "three": "Three", "fifteen": "Fifteen"}
AX_WORD = {"composition": "Comp", "force": "Force", "frequency": "Freq"}
MOD_WORD = {"gp": "Gp", "lawgp": "Law", "product": "Prod"}


def lorentz(f, f0, g):
    return (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)


def eps_ratio(c, pristine):
    if pristine:
        return 1.0
    if c <= 2.0:
        return EPS_RATIO_BTO
    return EPS_RATIO_BTO + (min(c, 3.0) - 2.0) * (EPS_RATIO_3WT - EPS_RATIO_BTO)


def cap(c, pristine, eps_p=EPS_R_PRISTINE):
    return EPS0 * eps_p * eps_ratio(c, pristine) * AREA / THICK


# ---------------------------------------------------------------- empirical inputs
def empirical_inputs():
    rec = pd.read_csv(os.path.join(RES, "a1_recordings.csv"))
    long = pd.read_parquet(os.path.join(A.REV, "data", "long.parquet"))
    long = long.sort_values(["series_id", "time"])
    waves = np.stack([long.loc[long.series_id == i, "voc"].values for i in range(N)])
    keep = ~rec.condition_id.isin(DUP_IDS)
    pool = {
        F: rec.loc[keep & (rec.force_N == F), "rate_rel_dev_from_nominal"].values
        for F in FORCES
    }
    # level spacing of the recorded waveforms
    steps = []
    for w in waves:
        u = np.diff(np.unique(w))
        steps.append(u[u > 1e-9].min())
    delta_q = float(np.median(steps))
    bg, rho, one_sample = [], [], []
    for w in waves:
        st = wave_stats(w)
        bg.append(st["bg_ratio"])
        rho.append(st["rho"])
        one_sample.append(st["one_sample_frac"])
    # kappa: log(y / law) on log(r / f), 72 recordings
    lawfit, pv, _ = A.fit_law5(np.arange(N), Y_REAL)
    lr = np.log(Y_REAL / lawfit(np.arange(N)))
    ld = np.log(rec.tap_rate_used_Hz.values / rec.freq_Hz.values)
    k = keep.values
    Xd = np.c_[np.ones(k.sum()), ld[k]]
    coef, *_ = np.linalg.lstsq(Xd, lr[k], rcond=None)
    res = lr[k] - Xd @ coef
    se = float(np.sqrt(res @ res / (k.sum() - 2) * np.linalg.inv(Xd.T @ Xd)[1, 1]))

    def ccc(a, b):
        ma, mb = a.mean(), b.mean()
        return float(2 * np.mean((a - ma) * (b - mb)) / (a.var() + b.var() + (ma - mb) ** 2))

    # tap timing jitter: robust SD of inter-tap intervals / sqrt(2)
    jit = []
    for i in range(N):
        if i in DUP_IDS:
            continue
        x = waves[i] - waves[i].mean()
        per = float(rec.period_used_samples[i])
        a = np.abs(x)
        sel = []
        for c in np.argsort(-a):
            if a[c] < 0.4 * a.max():
                break
            if all(abs(c - s) > per / 2 for s in sel):
                sel.append(c)
        d = np.diff(np.sort(sel))
        d = d[np.abs(d - per) < per / 2]
        if len(d) >= 3:
            jit.append(1.4826 * np.median(np.abs(d - np.median(d))))
    sigma_j = float(np.median(jit)) / np.sqrt(2) / FS
    return {
        "sigma_j_s": sigma_j,
        "sigma_j_n_recordings": len(jit),
        "delta_pool": {int(F): pool[F].tolist() for F in FORCES},
        "delta_q_V": delta_q,
        "bg_ratio_median": float(np.median(bg)),
        "bg_ratio_p10_p90": [float(np.percentile(bg, 10)), float(np.percentile(bg, 90))],
        "rho_median": float(np.median(rho)),
        "one_sample_excursion_frac_median": float(np.median(one_sample)),
        "cycle_cv_median": {
            "rms": float(np.median(rec.rms_cv_pct)) / 100,
            "vpp": float(np.median(rec.vpp_cv_pct)) / 100,
            "vmax": float(np.median(rec.vmax_cv_pct)) / 100,
        },
        "split_half_ccc": {
            t: ccc(rec["halfA_" + t].values, rec["halfB_" + t].values)
            for t in A.TARGETS
        },
        "split_half_logratio_sd": {
            t: float(np.std(np.log(rec["halfA_" + t].values / rec["halfB_" + t].values)))
            for t in A.TARGETS
        },
        "kappa": float(coef[1]),
        "kappa_se": se,
        "kappa_n": int(k.sum()),
        "law_true_params_volt": [float(v) for v in pv],
        "law_log_resid_sd_real": float(np.std(lr[k])),
        "law_log_resid_sd_real_all75": float(np.std(lr)),
    }


def wave_stats(w):
    x = w - w.mean()
    rsd = 1.4826 * np.median(np.abs(x - np.median(x)))
    rms = np.sqrt(np.mean(x**2))
    quiet = np.abs(x) < 3 * rsd
    qq = quiet[:-1] & quiet[1:]
    a, b = x[:-1][qq], x[1:][qq]
    rho = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 else 0.0
    above = np.abs(x) > 0.5 * np.abs(x).max()
    runs, cur = [], 0
    for v in above:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return {
        "bg_ratio": float(rsd / rms) if rms > 0 else np.nan,
        "rho": rho,
        "one_sample_frac": float(np.mean(np.array(runs) == 1)) if runs else np.nan,
    }


def fit_twomode():
    c, F, f = A.CNT, A.FRC, A.FRQ

    def model(p):
        b0, b1, b2, f1, g1, f2, g2, w = p
        return (b0 + b1 * c + b2 * c**2) * F * (w * lorentz(f, f1, g1) + (1 - w) * lorentz(f, f2, g2))

    best = None
    for f2 in (5.0, 8.0, 11.0, 14.0):
        for w in (0.3, 0.6, 0.9):
            r = least_squares(
                lambda p: model(p) - Y_REAL,
                [0.08, 0.04, -0.008, 19.0, 8.0, f2, 10.0, w],
                bounds=([0, -1, -1, 15, 1, 3, 1, 0], [5, 1, 1, 25, 30, 14, 30, 1]),
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
            )
            if best is None or r.cost < best.cost:
                best = r
    law_m, _, _ = A.fit_law5(np.arange(N), Y_REAL)
    sse_law = float(np.sum((law_m(np.arange(N)) - Y_REAL) ** 2))
    return {
        "params": [float(v) for v in best.x],
        "names": ["b0", "b1", "b2", "f1", "g1", "f2", "g2", "w"],
        "sse": float(2 * best.cost),
        "sse_law5": sse_law,
        "n_params": 8,
    }


# ---------------------------------------------------------------- generator
def response(gen, P, c, F, f):
    """Ground-truth response surface M(c, F, f) in volts (V_rms scale)."""
    c, F, f = np.asarray(c, float), np.asarray(F, float), np.asarray(f, float)
    if gen == "law":
        a0, a1, a2, f0, g = P["law"]
        return (a0 + a1 * c + a2 * c**2) * F * lorentz(f, f0, g)
    b0, b1, b2, f1, g1, f2, g2, w = P["twomode"]
    return (b0 + b1 * c + b2 * c**2) * F * (w * lorentz(f, f1, g1) + (1 - w) * lorentz(f, f2, g2))


def pulse_kernel(cp, sigma_t):
    """Recorded voltage of one unit-charge tap (q = unit x Gaussian force pulse)."""
    tau = R_IN * cp
    t = np.arange(-6 * sigma_t, 6 * sigma_t + 12 * tau, DT_FINE)
    q = np.exp(-(t**2) / (2 * sigma_t**2))
    i = np.gradient(q, DT_FINE)
    a = np.exp(-DT_FINE / tau)
    v = lfilter([DT_FINE / cp], [1.0, -a], i)  # C dV/dt + V/R = i, exact exponential step
    return t, v


class Kernels:
    """Pulse kernels per capacitance, cached per process."""

    def __init__(self):
        self.store = {}

    def get(self, cp, sigma_t):
        k = (round(cp * 1e15), round(sigma_t * 1e9))
        if k not in self.store:
            if len(self.store) > 64:
                self.store.clear()
            t, v = pulse_kernel(cp, sigma_t)
            self.store[k] = (t, v, float(np.sqrt(np.sum(v**2) * DT_FINE)))
        return self.store[k]


KER = Kernels()
CP_REF = cap(0.0, True)


def h_factor(c, pristine, sigma_t):
    return KER.get(cap(c, pristine), sigma_t)[2] / KER.get(CP_REF, sigma_t)[2]


def pulse_train(tk, amp, cp, sigma_t):
    t, g, _ = KER.get(cp, sigma_t)
    v = np.zeros(NS)
    if len(tk) == 0:
        return v
    lo = int(np.floor(t[0] * FS)) - 1
    hi = int(np.ceil(t[-1] * FS)) + 1
    offs = np.arange(lo, hi + 1)
    base = np.floor(tk * FS).astype(int)
    idx = base[:, None] + offs[None, :]
    dtm = idx / FS - tk[:, None]
    val = np.interp(dtm, t, g, left=0.0, right=0.0) * amp[:, None]
    ok = (idx >= 0) & (idx < NS)
    np.add.at(v, idx[ok], val[ok])
    return v


def simulate_recording(rng, M, f_label, F, cp, spec, P, want_wave=False):
    """One 1 s recording. Returns (V_rms, extras)."""
    pool = P["delta_pool_by_force"](F)
    delta = pool[rng.integers(len(pool))]
    r = f_label * (1.0 + delta)
    phi = rng.uniform(0.0, 1.0 / r)
    tk = phi + np.arange(int(np.ceil(r)) + 2) / r
    tk = tk + P["sigma_j"] * rng.standard_normal(len(tk))
    tk = np.sort(tk[(tk >= 0.0) & (tk < 1.0)])
    zk = rng.standard_normal(len(tk))
    sa = P["sigma_a"]
    ak = np.exp(sa * zk - sa**2)
    amp = M * spec * (r / f_label) ** (P["kappa"] - 0.5) / (P["C0"] * np.sqrt(f_label)) * ak
    pulse = pulse_train(tk, amp, cp, P["sigma_t"])
    prms = float(np.sqrt(np.mean(pulse**2)))
    rho = P["rho"]
    e = lfilter([np.sqrt(1 - rho**2)], [1.0, -rho], rng.standard_normal(NS + 200))[200:]
    v = pulse + P["beta"] * prms * e
    dq = P["delta_q"]
    if dq > 0:
        v = np.round(v / dq) * dq
    x = v - v.mean()
    vrms = float(np.sqrt(np.mean(x**2)))
    if not want_wave:
        return vrms, None
    return vrms, {"x": x, "tk": tk, "r": r}


def cycle_stats(x, tk, r):
    """Cycle-to-cycle CV of rms, vpp, vmax over full windows centred on the taps."""
    P = FS / r
    out = {"rms": [], "vpp": [], "vmax": []}
    for t in tk:
        a, b = int(round(t * FS - P / 2)), int(round(t * FS + P / 2))
        if a < 0 or b > NS or b - a < 2:
            continue
        s = x[a:b]
        out["rms"].append(np.sqrt(np.mean(s**2)))
        out["vpp"].append(np.ptp(s))
        out["vmax"].append(np.abs(s).max())
    cv = {}
    for k, v in out.items():
        v = np.array(v)
        cv[k] = float(v.std(ddof=1) / v.mean()) if len(v) >= 2 and v.mean() > 0 else np.nan
    return cv


def spec_mult(z, cv):
    if cv <= 0:
        return np.ones_like(z)
    s2 = np.log(1 + cv**2)
    return np.exp(np.sqrt(s2) * z - s2 / 2)


def test_points():
    """Unsampled levels: rows (c, pristine, F, f, axis, comp_index_or_-1)."""
    rows = []
    for c in UNS_C:
        for F in FORCES:
            for f in FREQS:
                rows.append((c, 0.0, F, f, "composition", -1))
    for k, comp in enumerate(A.COMP_ORDER):
        c = [0, 0, 1, 2, 3][k]
        pr = 1.0 if k == 0 else 0.0
        for F in UNS_F:
            for f in FREQS:
                rows.append((c, pr, F, f, "force", k))
        for F in FORCES:
            for f in UNS_FREQ:
                rows.append((c, pr, F, f, "frequency", k))
    df = pd.DataFrame(rows, columns=["c", "pristine", "F", "f", "axis", "comp"])
    return df


TEST = test_points()
XT = TEST[["c", "pristine", "F", "f"]].values.astype(float)


def simulate_grid(gen, cv, design, rep, P, grid_only=False):
    """Synthetic grid + repeats + unsampled-level recordings for one replicate."""
    rng_s = np.random.default_rng([rep, 1])
    zs = rng_s.standard_normal((5, 16))  # 15 grid specimens + 1 new per composition
    z_unc = rng_s.standard_normal(len(UNS_C))  # new specimens of unsampled compositions
    ms = spec_mult(zs, cv)
    mu_unc = spec_mult(z_unc, cv)
    rng = np.random.default_rng([rep, 0])
    if design == "one":
        sidx = np.zeros(N, int)
    elif design == "three":
        sidx = (F_IDX + FR_IDX) % 3
    else:
        sidx = F_IDX * 5 + FR_IDX
    pr = X[:, 1]
    Mg = response(gen, P, A.CNT, A.FRC, A.FRQ)
    eps_p = P.get("eps_p", EPS_R_PRISTINE)
    cps = np.array([cap(A.CNT[i], pr[i] > 0, eps_p) for i in range(N)])
    y = np.empty(N)
    y_rep = np.empty(N)
    y_new = np.empty(N)
    for i in range(N):
        y[i] = simulate_recording(rng, Mg[i], A.FRQ[i], A.FRC[i], cps[i], ms[COMP_IDX[i], sidx[i]], P)[0]
    if grid_only:
        return {"y": y}
    for i in range(N):
        y_rep[i] = simulate_recording(rng, Mg[i], A.FRQ[i], A.FRC[i], cps[i], ms[COMP_IDX[i], sidx[i]], P)[0]
    for i in range(N):
        y_new[i] = simulate_recording(rng, Mg[i], A.FRQ[i], A.FRC[i], cps[i], ms[COMP_IDX[i], 15], P)[0]
    Mt = response(gen, P, XT[:, 0], XT[:, 2], XT[:, 3])
    cpt = np.array([cap(XT[j, 0], XT[j, 1] > 0, eps_p) for j in range(len(XT))])
    yt_exist = np.full(len(XT), np.nan)
    yt_new = np.empty(len(XT))
    for j in range(len(XT)):
        k = TEST.comp.values[j]
        if k >= 0:
            yt_exist[j] = simulate_recording(rng, Mt[j], XT[j, 3], XT[j, 2], cpt[j], ms[k, 0], P)[0]
    for j in range(len(XT)):
        k = TEST.comp.values[j]
        s = ms[k, 15] if k >= 0 else mu_unc[UNS_C.index(XT[j, 0])]
        yt_new[j] = simulate_recording(rng, Mt[j], XT[j, 3], XT[j, 2], cpt[j], s, P)[0]
    return {"y": y, "y_rep": y_rep, "y_new": y_new, "yt_exist": yt_exist, "yt_new": yt_new}


# ---------------------------------------------------------------- calibration
def make_params(emp, twomode, sigma_a, beta, sigma_t=SIGMA_T, eps_p=EPS_R_PRISTINE):
    pool = {int(k): np.array(v) for k, v in emp["delta_pool"].items()}
    P = {
        "law": emp["law_true_params_volt"],
        "twomode": twomode["params"],
        "kappa": emp["kappa"],
        "rho": emp["rho_median"],
        "delta_q": emp["delta_q_V"],
        "sigma_a": sigma_a,
        "beta": beta,
        "sigma_j": emp["sigma_j_s"],
        "sigma_t": sigma_t,
        "eps_p": eps_p,
        "pool": {k: v.tolist() for k, v in pool.items()},
    }
    P["C0"] = float(np.sqrt(1 + beta**2) * KER.get(cap(0.0, True, eps_p), sigma_t)[2])
    attach_pool(P)
    return P


def attach_pool(P):
    pool = {int(k): np.array(v) for k, v in P["pool"].items()}

    def by_force(F):
        if F in pool:
            return pool[F]
        lo, hi = int(np.floor(F)), int(np.ceil(F))
        return np.concatenate([pool[lo], pool[hi]])

    P["delta_pool_by_force"] = by_force
    return P


def calib_stats(P, seeds=CAL_SEEDS, gen="law"):
    """Median cycle CVs, background ratio and one-sample fraction on law-true CV=0 grids."""
    cvs = {"rms": [], "vpp": [], "vmax": []}
    bg, one, halves = [], [], {t: ([], []) for t in A.TARGETS}
    Mg = response(gen, P, A.CNT, A.FRC, A.FRQ)
    for sd in seeds:
        rng = np.random.default_rng([sd, 0])
        for i in range(N):
            cp = cap(A.CNT[i], X[i, 1] > 0)
            _, ex = simulate_recording(rng, Mg[i], A.FRQ[i], A.FRC[i], cp, 1.0, P, want_wave=True)
            cs = cycle_stats(ex["x"], ex["tk"], ex["r"])
            for k in cvs:
                cvs[k].append(cs[k])
            st = wave_stats(ex["x"])
            bg.append(st["bg_ratio"])
            one.append(st["one_sample_frac"])
            for t, (ha, hb) in halves.items():
                a, b = ex["x"][:500], ex["x"][500:]
                d = {
                    "rms_Voc": lambda s: np.std(s),
                    "Vpp": lambda s: np.ptp(s - s.mean()),
                    "Vmax": lambda s: np.abs(s - s.mean()).max(),
                }[t]
                ha.append(d(a))
                hb.append(d(b))

    def ccc(a, b):
        a, b = np.asarray(a), np.asarray(b)
        ma, mb = a.mean(), b.mean()
        return float(2 * np.mean((a - ma) * (b - mb)) / (a.var() + b.var() + (ma - mb) ** 2))

    # CCC per seed (75 recordings each, as the real statistic), then median over seeds
    cc, lsd = {}, {}
    for t, (ha, hb) in halves.items():
        ha, hb = np.array(ha).reshape(len(seeds), N), np.array(hb).reshape(len(seeds), N)
        cc[t] = float(np.median([ccc(ha[s], hb[s]) for s in range(len(seeds))]))
        lsd[t] = float(np.median([np.std(np.log(ha[s] / hb[s])) for s in range(len(seeds))]))
    return {
        "cycle_cv_median": {k: float(np.nanmedian(v)) for k, v in cvs.items()},
        "bg_ratio_median": float(np.nanmedian(bg)),
        "one_sample_frac_median": float(np.nanmedian(one)),
        "split_half_ccc": cc,
        "split_half_logratio_sd": lsd,
    }


def calibrate(emp, twomode, log):
    """Fit (sigma_t, sigma_a) to the median cycle CV of V_rms and the SD of the
    split-half log ratio of V_rms, then beta to the median background ratio. The
    V_pp and |V|_max cycle CVs and split-half statistics, the one-sample excursion
    fraction and the split-half concordances stay uncalibrated (fidelity checks)."""
    tgt = emp["cycle_cv_median"]
    tgt_lsd = emp["split_half_logratio_sd"]["rms_Voc"]
    tgt_bg = emp["bg_ratio_median"]
    seeds = CAL_SEEDS[:4]

    def fit_beta(st_, sa_, be0):
        lo, hi = 0.0, 1.5
        be_ = be0
        for _ in range(12):
            be_ = 0.5 * (lo + hi)
            st = calib_stats(make_params(emp, twomode, sa_, be_, st_), seeds=seeds)
            if st["bg_ratio_median"] > tgt_bg:
                hi = be_
            else:
                lo = be_
        return be_

    def loss(st_, sa_, be_):
        st = calib_stats(make_params(emp, twomode, sa_, be_, st_), seeds=seeds)
        c = st["cycle_cv_median"]
        d = st["split_half_logratio_sd"]["rms_Voc"]
        return (c["rms"] / tgt["rms"] - 1) ** 2 + (d / tgt_lsd - 1) ** 2, st

    be = 0.34
    hist = []
    grid_t = np.array([0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2]) * 1e-3
    grid_a = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    best = None
    for st_ in grid_t:
        for sa_ in grid_a:
            L, st = loss(st_, sa_, be)
            hist.append({"sigma_t": float(st_), "sigma_a": float(sa_), "beta": be, "loss": L})
            if best is None or L < best[0]:
                best = (L, st_, sa_)
    # local refinement (two passes of a shrinking grid), beta refitted in between
    _, st_b, sa_b = best
    for rnd, (ft, fa) in enumerate([(0.25, 0.08), (0.08, 0.025)]):
        be = fit_beta(st_b, sa_b, be)
        cand = [
            (st_b * (1 + ft * i), max(0.0, sa_b + fa * j))
            for i in (-2, -1, 0, 1, 2)
            for j in (-2, -1, 0, 1, 2)
        ]
        best = None
        for st_, sa_ in cand:
            L, st = loss(st_, sa_, be)
            hist.append({"sigma_t": float(st_), "sigma_a": float(sa_), "beta": be, "loss": L})
            if best is None or L < best[0]:
                best = (L, st_, sa_)
        _, st_b, sa_b = best
        log(f"[a6] calibration round {rnd}: sigma_t={st_b * 1e3:.3f} ms sigma_a={sa_b:.3f} beta={be:.3f} loss={best[0]:.4f}")
    be = fit_beta(st_b, sa_b, be)
    final = calib_stats(make_params(emp, twomode, sa_b, be, st_b))
    return float(st_b), float(sa_b), float(be), final, hist


# ---------------------------------------------------------------- protocol replica
def lawq(info, Xq):
    return A.law5((Xq[:, 0], Xq[:, 2], Xq[:, 3]), *info["p_scaled"]) * info["scale"]


def fit_outer(model, tr, y, Xq, init=None, law0=None):
    """Fit on rows tr, predict at Xq. init: kernel to start the GP optimizer from
    (warm start; None = the paper's cold start); law0: fit_law5 info to warm-start the
    law fit (None = the paper's start P0_5)."""
    tr = np.asarray(tr)
    xs = StandardScaler().fit(X[tr])
    info = None
    if model == "gp":
        btr, bq, g = 0.0, 0.0, A.gp()
    elif model == "lawgp":
        if law0 is None:
            m, pv, info = A.fit_law5(tr, y)
        else:
            p, sc, ok = law_refit(tr, y, law0)
            info = {"scale": sc, "ok": ok, "p_scaled": p}
            pv = A.to_volt(p, sc)
            m = lambda idx, p=p, sc=sc: A.law5(A._xt(idx), *p) * sc  # noqa: E731
        btr, bq, g = m(tr), lawq(info, Xq), A.gp()
        info = dict(info, params_volt=pv)
    elif model == "product":
        btr, bq, g = 0.0, 0.0, B.gp_struct("product", PRODUCT_SEED)
    else:
        raise ValueError(model)
    if init is not None:
        g.kernel = init
    g.fit(xs.transform(X[tr]), y[tr] - btr)
    mu, sd = g.predict(xs.transform(Xq), return_std=True)
    return {"mu": bq + mu, "sd": sd, "g": g, "xs": xs, "info": info}


def inner_closed(g, xs, tr, Trows, Xq):
    """Leave-one-out models inside tr with the outer kernel fixed (closed form).

    Trows[i] is the target vector of inner model i on rows tr (entry i ignored).
    Each inner model normalizes its own target (mean, SD over tr minus i) as
    sklearn normalize_y does. Returns (own prediction at x_i, predictions at Xq).
    """
    Zt = xs.transform(X[tr])
    n = len(tr)
    K = g.kernel_(Zt)
    K[np.diag_indices(n)] += g.alpha
    Ainv = np.linalg.inv(K)
    Kq = g.kernel_(Zt, xs.transform(Xq))
    eye = np.eye(n, dtype=bool)
    W = np.where(eye, 0.0, 1.0)
    mu = (Trows * W).sum(1) / (n - 1)
    var = (((Trows - mu[:, None]) ** 2) * W).sum(1) / (n - 1)
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    V = (Trows - mu[:, None]) / sd[:, None]
    V[eye] = 0.0
    U = V @ Ainv
    Alpha = U - (np.diag(U) / np.diag(Ainv))[:, None] * Ainv
    Alpha[eye] = 0.0
    Koff = np.where(eye, 0.0, K)
    own = mu + sd * np.einsum("ij,ij->i", Koff, Alpha)
    pq = mu[:, None] + sd[:, None] * (Alpha @ Kq)
    return own, pq


def law_refit(sub, y, info0):
    s_in = float(np.std(y[sub])) or 1.0
    p0 = np.array(info0["p_scaled"], float).copy()
    p0[:3] *= info0["scale"] / s_in
    lo, hi = np.array(A.BOUNDS_5[0], float), np.array(A.BOUNDS_5[1], float)
    p0 = np.clip(p0, lo, hi)
    p, _, scale, ok = A._fit(A.law5, sub, y, list(p0), A.BOUNDS_5)
    return p, scale, ok


def r2(y, p):
    return float(r2_score(y, p))


def exact_inner(model, tr, te, y, fo):
    """Inner LOO models of the group-wise jackknife+, every one refitted (kernel
    hyperparameters re-optimized, warm-started at the outer fold; LawGP law refitted,
    warm-started). Returns (own predictions at x_i, predictions at te)."""
    own = np.empty(len(tr))
    pq = np.empty((len(tr), len(te)))
    for r_, j in enumerate(tr):
        f_ = fit_outer(model, tr[tr != j], y, np.vstack([X[[j]], X[te]]), init=fo["g"].kernel_, law0=fo["info"])
        own[r_] = f_["mu"][0]
        pq[r_] = f_["mu"][1:]
    return own, pq


def closed_inner(model, tr, te, y, fo):
    """Closed-form inner LOO models (outer kernel and scaling kept; law refitted)."""
    if model == "gp":
        return inner_closed(fo["g"], fo["xs"], tr, np.tile(y[tr], (len(tr), 1)), X[te])
    mtr = np.empty((len(tr), len(tr)))
    mte = np.empty((len(tr), len(te)))
    for r_, i in enumerate(tr):
        p, sc, _ = law_refit(tr[tr != i], y, fo["info"])
        mtr[r_] = A.law5((A.CNT[tr], A.FRC[tr], A.FRQ[tr]), *p) * sc
        mte[r_] = A.law5((A.CNT[te], A.FRC[te], A.FRQ[te]), *p) * sc
    own_r, pq_r = inner_closed(fo["g"], fo["xs"], tr, y[tr][None, :] - mtr, X[te])
    return np.diag(mtr) + own_r, mte + pq_r


def run_protocol(y, test=None, product=False, uns_targets=None, rep_targets=None, exact_models=("gp", "lawgp"), cold_loo=False, loo_models=("gp", "lawgp")):
    """The paper's protocol on one 75-condition V_rms vector y.

    exact_models: models whose group-wise jackknife+ uses exact (warm-started) inner
    refits; the others use the closed form. Both are stored when exact is run.
    cold_loo: LOO outer fits from the paper's cold start (else warm-started at the
    full-grid fit, which reproduces the cold-start LOO on the real grid).
    """
    out = {"loo": {}, "group": {}, "cov": {}, "uns": {}, "law": {}}
    idx = np.arange(N)
    Xq_test = XT if test is not None else np.zeros((0, 4))
    nT = len(Xq_test)
    models = ["gp", "lawgp"] + (["product"] if product else [])
    # ---- full-grid fits (paper start): unsampled-level predictions, law parameters, warm starts
    full = {}
    for model in models:
        fo = fit_outer(model, idx, y, Xq_test if nT else X[:1])
        full[model] = fo
        if model == "lawgp":
            pv = fo["info"]["params_volt"]
            lawfull = A.law5((A.CNT, A.FRC, A.FRQ), *fo["info"]["p_scaled"]) * fo["info"]["scale"]
            out["law"] = {
                "params_volt": [float(v) for v in pv],
                "c_star": float(-pv[1] / (2 * pv[2])) if pv[2] < 0 else float("nan"),
                "log_resid_sd": float(np.std(np.log(np.maximum(y, 1e-9) / np.maximum(lawfull, 1e-9)))),
            }
    # ---- LOO, plain GP and LawGP; within-grid nested jackknife+ (GP, closed form)
    loo = {}
    within = {k: {a: [] for a in NOMINAL} for k in ("rec", "rep", "new")}
    within_raw = {a: [] for a in NOMINAL}
    within_w = {a: [] for a in NOMINAL}
    for model in loo_models:
        mu = np.empty(N)
        sdv = np.empty(N)
        pT = np.empty((N, nT))
        a2neg = 0
        for j in range(N):
            rest = np.delete(idx, j)
            Xq = np.vstack([X[[j]], Xq_test])
            fo = fit_outer(model, rest, y, Xq, init=None if cold_loo else full[model]["g"].kernel_)
            mu[j], sdv[j], pT[j] = fo["mu"][0], fo["sd"][0], fo["mu"][1:]
            if model == "lawgp":
                a2neg += int(fo["info"]["params_volt"][2] < 0)
            if model == "gp":
                own, pq = inner_closed(fo["g"], fo["xs"], rest, np.tile(y[rest], (N - 1, 1)), X[[j]])
                res = np.abs(y[rest] - own)
                for a, al in NOMINAL.items():
                    lo_, hi_ = C3.jk_bounds(pq[:, 0], res, al)
                    within["rec"][a].append(lo_ <= y[j] <= hi_)
                    if rep_targets is not None:
                        within["rep"][a].append(lo_ <= rep_targets["rep"][j] <= hi_)
                        within["new"][a].append(lo_ <= rep_targets["new"][j] <= hi_)
                    within_w[a].append(hi_ - lo_)
                    within_raw[a].append(abs(y[j] - fo["mu"][0]) <= Z[a] * fo["sd"][0])
        loo[model] = {"mu": mu, "sd": sdv, "pT": pT}
        out["loo"][model] = {"r2": r2(y, mu), "mae": float(mean_absolute_error(y, mu))}
        if model == "lawgp":
            out["loo"][model]["a2_negative_folds"] = a2neg
    for k in within:
        for a in NOMINAL:
            if within[k][a]:
                out["cov"][f"within_jk_gp_{k}_{a}"] = [int(np.sum(within[k][a])), N]
    for a in NOMINAL:
        out["cov"][f"within_raw_gp_rec_{a}"] = [int(np.sum(within_raw[a])), N]
        out["cov"][f"within_jk_gp_width_{a}"] = float(np.mean(within_w[a]))
    # ---- group-wise: outer exact fits (paper start); jackknife+ exact and closed form
    for ax in AXES:
        grp = AXIS_GROUPS[ax]
        levels = A.axis_levels(ax)
        out["group"][ax] = {}
        for model in models:
            oof = np.empty(N)
            lvl_r2, lvl_mae, f0s = [], [], []
            variants = [] if model == "product" else (["closed"] + (["exact"] if model in exact_models else []))
            hits = {v: {a: [] for a in NOMINAL} for v in variants}
            lvl_hits = {v: {a: [] for a in NOMINAL} for v in variants}
            hits_raw = {a: [] for a in NOMINAL}
            for lv in levels:
                te = idx[grp == lv]
                tr = idx[grp != lv]
                fo = fit_outer(model, tr, y, X[te])
                oof[te] = fo["mu"]
                lvl_r2.append(r2(y[te], fo["mu"]))
                lvl_mae.append(float(mean_absolute_error(y[te], fo["mu"])))
                if model == "product":
                    continue
                if model == "lawgp":
                    f0s.append(float(fo["info"]["params_volt"][3]))
                for a in NOMINAL:
                    hits_raw[a] += [bool(abs(y[t_] - fo["mu"][k_]) <= Z[a] * fo["sd"][k_]) for k_, t_ in enumerate(te)]
                for v in variants:
                    own, pq = (exact_inner if v == "exact" else closed_inner)(model, tr, te, y, fo)
                    res = np.abs(y[tr] - own)
                    for a, al in NOMINAL.items():
                        lh = []
                        for k_, t_ in enumerate(te):
                            lo_, hi_ = C3.jk_bounds(pq[:, k_], res, al)
                            lh.append(bool(lo_ <= y[t_] <= hi_))
                        hits[v][a] += lh
                        lvl_hits[v][a].append(int(np.sum(lh)))
            rec = {
                "pooled_r2": r2(y, oof),
                "pooled_mae": float(mean_absolute_error(y, oof)),
                "level_r2": lvl_r2,
                "level_mae": lvl_mae,
            }
            if model == "lawgp":
                rec["f0_folds"] = f0s
            out["group"][ax][model] = rec
            for v in variants:
                tag = "jk" if v == "exact" else "jkcf"
                for a in NOMINAL:
                    out["cov"][f"group_{tag}_{model}_{ax}_{a}"] = [int(np.sum(hits[v][a])), N]
                    out["cov"][f"group_{tag}_{model}_{ax}_{a}_levels"] = lvl_hits[v][a]
            if model != "product":
                for a in NOMINAL:
                    out["cov"][f"group_raw_{model}_{ax}_{a}"] = [int(np.sum(hits_raw[a])), N]
    if nT:
        ax_t = TEST.axis.values
        for model in models:
            out["uns"][model] = {}
            for ax in AXES:
                m_ = ax_t == ax
                d = {"r2_new": r2(uns_targets["new"][m_], full[model]["mu"][m_])}
                if ax != "composition":
                    d["r2_exist"] = r2(uns_targets["exist"][m_], full[model]["mu"][m_])
                out["uns"][model][ax] = d
        for model in loo:
            res = np.abs(y - loo[model]["mu"])
            pT = loo[model]["pT"]
            for a, al in NOMINAL.items():
                bnd = np.array([C3.jk_bounds(pT[:, t], res, al) for t in range(nT)])
                for ax in AXES:
                    m_ = ax_t == ax
                    yn = uns_targets["new"][m_]
                    out["cov"][f"uns_jk_{model}_{ax}_new_{a}"] = [
                        int(np.sum((bnd[m_, 0] <= yn) & (yn <= bnd[m_, 1]))),
                        int(m_.sum()),
                    ]
                    if ax != "composition":
                        ye = uns_targets["exist"][m_]
                        out["cov"][f"uns_jk_{model}_{ax}_exist_{a}"] = [
                            int(np.sum((bnd[m_, 0] <= ye) & (ye <= bnd[m_, 1]))),
                            int(m_.sum()),
                        ]
    return out


# ---------------------------------------------------------------- one grid job
def grid_job(gen, cv, design, rep, P_ser, cache_dir):
    fn = os.path.join(cache_dir, f"{gen}_{int(round(cv * 100))}_{design}_{rep}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as fh:
            return pickle.load(fh)
    P = attach_pool(dict(P_ser))
    t0 = time.time()
    sim = simulate_grid(gen, cv, design, rep, P)
    y = sim["y"]
    res = run_protocol(
        y,
        test=True,
        product=(design == "one"),
        uns_targets={"new": sim["yt_new"], "exist": sim["yt_exist"]},
        rep_targets={"rep": sim["y_rep"], "new": sim["y_new"]},
        exact_models=("gp", "lawgp") if design == "one" else ("gp",),
        loo_models=("gp", "lawgp") if design == "one" else ("gp",),
    )
    # optimum reproducibility
    cm = np.array([y[COMP_IDX == k].mean() for k in range(5)])
    res["opt"] = {
        "comp_means": cm.tolist(),
        "best_mean": int(np.argmax(cm)),
        "best_recording_comp": int(COMP_IDX[np.argmax(y)]),
        "best_recording_id": int(np.argmax(y)),
    }
    res["y"] = y.tolist()
    res["meta"] = {"gen": gen, "cv": cv, "design": design, "rep": rep, "seconds": time.time() - t0}
    with open(fn, "wb") as fh:
        pickle.dump(res, fh)
    return res


# ---------------------------------------------------------------- exact-refit check
def exact_group_job(y, model, axis):
    """Group-wise jackknife+ with every inner model refitted (the A3 construction,
    law5 form). Returns hits per nominal level for the given axis."""
    idx = np.arange(N)
    grp = AXIS_GROUPS[axis]
    hits = {a: 0 for a in NOMINAL}
    for v in A.axis_levels(axis):
        te = idx[grp == v]
        tr = idx[grp != v]
        resid = np.empty(len(tr))
        preds = np.empty((len(tr), len(te)))
        for r_, j in enumerate(tr):
            fo = fit_outer(model, tr[tr != j], y, np.vstack([X[[j]], X[te]]))
            resid[r_] = abs(y[j] - fo["mu"][0])
            preds[r_] = fo["mu"][1:]
        for a, al in NOMINAL.items():
            for k_, t_ in enumerate(te):
                lo_, hi_ = C3.jk_bounds(preds[:, k_], resid, al)
                hits[a] += int(lo_ <= y[t_] <= hi_)
    return hits


def exact_within(y):
    """Within-grid nested jackknife+ with every inner model refitted (warm-started at
    its outer LOO model, as the group-wise construction of the main run)."""
    idx = np.arange(N)
    hits = {a: 0 for a in NOMINAL}
    for j in range(N):
        rest = np.delete(idx, j)
        fo = fit_outer("gp", rest, y, X[[j]])
        res = np.empty(N - 1)
        pj = np.empty(N - 1)
        for r_, i in enumerate(rest):
            f_ = fit_outer("gp", rest[rest != i], y, np.vstack([X[[i]], X[[j]]]), init=fo["g"].kernel_)
            res[r_] = abs(y[i] - f_["mu"][0])
            pj[r_] = f_["mu"][1]
        for a, al in NOMINAL.items():
            lo_, hi_ = C3.jk_bounds(pj, res, al)
            hits[a] += int(lo_ <= y[j] <= hi_)
    return hits


def exact_job(tag, y, model, axis, cache_dir):
    fn = os.path.join(cache_dir, f"exact_{tag}_{model}_{axis}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as fh:
            return pickle.load(fh)
    y = np.asarray(y)
    if axis == "within":
        hits = exact_within(y)
    else:
        hits = exact_group_job(y, model, axis)
    out = {"tag": tag, "model": model, "axis": axis, "hits": hits}
    with open(fn, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------- optimum simulation
def opt_job(gen, cv, design, scen, reps, P_ser):
    """Best composition by mean over 15 recordings and by the single best recording."""
    P = attach_pool(dict(P_ser))
    bm = np.zeros(5, int)
    br = np.zeros(5, int)
    for rep in reps:
        y = simulate_grid(gen, cv, design, rep, P, grid_only=True)["y"]
        cm = np.array([y[COMP_IDX == k].mean() for k in range(5)])
        bm[int(np.argmax(cm))] += 1
        br[int(COMP_IDX[np.argmax(y)])] += 1
    return {"gen": gen, "cv": cv, "design": design, "scen": scen, "n": len(reps), "best_mean": bm.tolist(), "best_rec": br.tolist()}


def true_optimum(gen, P):
    """Composition ranking of sqrt(E[V_rms^2]) (separable in F and f)."""
    eps_p = P.get("eps_p", EPS_R_PRISTINE)
    cs = [0.0, 0.0, 1.0, 2.0, 3.0]
    prs = [True, False, False, False, False]
    ref = KER.get(cap(0.0, True, eps_p), P["sigma_t"])[2]
    fac = np.array([
        response(gen, P, c, 1.0, 20.0) * KER.get(cap(c, pr, eps_p), P["sigma_t"])[2] / ref
        for c, pr in zip(cs, prs)
    ])
    cc = np.linspace(0, 3, 3001)
    hc = np.array([KER.get(cap(c, False, eps_p), P["sigma_t"])[2] for c in cc[::100]]) / ref
    hfine = np.interp(cc, cc[::100], hc)
    cont = response(gen, P, cc, 1.0, 20.0) * hfine
    return {
        "factor_by_comp": fac.tolist(),
        "best": int(np.argmax(fac)),
        "gap_best_to_second_pct": float(100 * (np.sort(fac)[-1] / np.sort(fac)[-2] - 1)),
        "c_star_continuous": float(cc[np.argmax(cont)]),
        "h_by_comp": (fac / np.array([response(gen, P, c, 1.0, 20.0) for c in cs])).tolist(),
    }


# ================================================================ aggregation helpers
SIM_VERSION = "v5"  # bump when the generator or the protocol replica changes


def fnum_tex(x, nd):
    """Table number: 3 decimals etc., '$-$' for negatives, '--' for NaN."""
    if x is None or not np.isfinite(x):
        return "--"
    return C3.fnum(x, nd)


def mac_num(x, nd):
    if x is None or not np.isfinite(x):
        return "--"
    v = C3.rhu(x, nd)
    s_ = f"{abs(v):.{nd}f}"
    return (r"\ensuremath{-}" + s_) if v < 0 and float(s_) != 0 else s_


def pct(x):
    return "--" if x is None or not np.isfinite(x) else f"{int(C3.rhu(100 * x, 0))}"


def summ(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
        "p2.5": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p97.5": float(q[4]),
    }


def flat(res):
    m = dict(res["meta"])
    m["cv_pct"] = int(round(m["cv"] * 100))
    for mod, d in res["loo"].items():
        m[f"loo_r2_{mod}"] = d["r2"]
        m[f"loo_mae_{mod}"] = d["mae"]
    for ax, dd in res["group"].items():
        for mod, d in dd.items():
            m[f"pooled_r2_{mod}_{ax}"] = d["pooled_r2"]
            m[f"meanlevel_r2_{mod}_{ax}"] = float(np.mean(d["level_r2"]))
            for lv, v in zip(A.axis_levels(ax), d["level_r2"]):
                m[f"level_r2_{mod}_{ax}_{A.level_key(ax, lv)}"] = v
    for k, v in res["cov"].items():
        if k.endswith("_levels"):
            continue
        if isinstance(v, list):
            m[f"cov_{k}"] = v[0] / v[1]
        else:
            m[k] = v
    for mod, dd in res["uns"].items():
        for ax, d in dd.items():
            for kk, vv in d.items():
                m[f"uns_{kk}_{mod}_{ax}"] = vv
    if res.get("law"):
        pv = res["law"]["params_volt"]
        m["law_f0"], m["law_gamma"], m["law_a2"] = pv[3], pv[4], pv[2]
        m["law_c_star"] = res["law"]["c_star"]
        m["law_log_resid_sd"] = res["law"]["log_resid_sd"]
    m["best_mean"] = res["opt"]["best_mean"]
    m["best_recording_comp"] = res["opt"]["best_recording_comp"]
    return m


def clopper_pct(k, n):
    lo, hi = C3.clopper(k, n)
    return [lo, hi]


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=30)
    ap.add_argument("--n-opt-rep", type=int, default=500)
    ap.add_argument("--n-exact-rep", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    log = lambda s_: print(s_, flush=True)  # noqa: E731

    emp = empirical_inputs()
    two = fit_twomode()
    st_, sa_, be_, fid, cal_hist = calibrate(emp, two, log)
    P = make_params(emp, two, sa_, be_, st_)
    P_ser = {k: v for k, v in P.items() if k != "delta_pool_by_force"}
    phash = hashlib.sha256(json.dumps({"P": P_ser, "v": SIM_VERSION}, sort_keys=True).encode()).hexdigest()[:12]
    cdir = os.path.join(CACHE, phash)
    os.makedirs(cdir, exist_ok=True)
    log(f"[a6] calibrated in {time.time() - T0:.0f} s; cache {cdir}")

    # ---- protocol replica on the real grid
    real = run_protocol(Y_REAL, test=None, product=True)
    log(f"[a6] real-grid replica done at {time.time() - T0:.0f} s")

    # ---- synthetic grids
    reps = list(range(args.n_rep))
    jobs = [(g, cv, d, r) for g in GENS for cv in CV_LEVELS for d in DESIGNS for r in reps]
    # longest jobs (product GP on the real design) first
    jobs.sort(key=lambda j: (j[2] != "one", j[3]))
    res = Parallel(n_jobs=args.jobs, batch_size=1, verbose=0)(
        delayed(grid_job)(g, cv, d, r, P_ser, cdir) for g, cv, d, r in jobs
    )
    log(f"[a6] {len(res)} synthetic grids at {time.time() - T0:.0f} s")

    # ---- exact-refit check of the closed-form inner models (group-wise jackknife+)
    ex_grids = [("real", Y_REAL)]
    for g in GENS:
        for cv in (0.0, 0.20):
            for r in range(args.n_exact_rep):
                rr = next(x for x in res if x["meta"]["gen"] == g and x["meta"]["cv"] == cv and x["meta"]["design"] == "one" and x["meta"]["rep"] == r)
                ex_grids.append((f"{g}_{int(cv * 100)}_one_{r}", np.array(rr["y"])))
    ex_tasks = [(tag, y, "gp", "within") for tag, y in ex_grids]
    ex_tasks += [(tag, y, m, ax) for tag, y in ex_grids for m in ("gp", "lawgp") for ax in AXES]
    ex = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(exact_job)(tag, y, m, ax, cdir) for tag, y, m, ax in ex_tasks
    )
    log(f"[a6] exact-refit check at {time.time() - T0:.0f} s")

    # ---- optimum reproducibility (simulation only, many replicates)
    scen_eps = {"nominal": EPS_R_PRISTINE, "high_cp": 7.7e-9 / cap(3.0, False, 1.0), "no_rc": 1e-4}
    Ps = {}
    for sc, ep in scen_eps.items():
        Pp = make_params(emp, two, sa_, be_, st_, eps_p=ep)
        Ps[sc] = {k: v for k, v in Pp.items() if k != "delta_pool_by_force"}
    chunks = [list(range(i, min(i + 250, args.n_opt_rep))) for i in range(0, args.n_opt_rep, 250)]
    opt_raw = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(opt_job)(g, cv, d, sc, ch, Ps[sc])
        for g in GENS for cv in CV_LEVELS for d in DESIGNS for sc in scen_eps for ch in chunks
    )
    log(f"[a6] optimum simulation at {time.time() - T0:.0f} s")
    truth = {sc: {g: true_optimum(g, attach_pool(dict(Ps[sc]))) for g in GENS} for sc in scen_eps}

    write_outputs(args, emp, two, P_ser, fid, cal_hist, real, res, ex, opt_raw, truth, scen_eps, phash)


# ================================================================ outputs
def write_outputs(args, emp, two, P_ser, fid, cal_hist, real, res, ex, opt_raw, truth, scen_eps, phash):
    os.makedirs(TAB, exist_ok=True)
    R = {"meta": {}, "index": {}}
    df = pd.DataFrame([flat(r) for r in res]).sort_values(["gen", "cv_pct", "design", "rep"])
    df.to_csv(os.path.join(RES, "a6_replicates.csv"), index=False)

    # ---------- real-grid replica check
    a3 = json.load(open(os.path.join(RES, "a3_groupwise_coverage.json")))["groupwise"]["rms_Voc"]
    a3w = json.load(open(os.path.join(RES, "a3_groupwise_coverage.json")))["within_grid_recomputed"]["rms_Voc"]
    a2 = json.load(open(os.path.join(RES, "a2_physgp.json")))
    rep_check = {"pooled_r2": {}, "loo_r2": {}, "coverage_95": {}, "coverage_90": {}}
    rep_check["loo_r2"] = {m: real["loo"][m]["r2"] for m in ("gp", "lawgp")}
    for ax in AXES:
        rep_check["pooled_r2"][ax] = {m: real["group"][ax][m]["pooled_r2"] for m in real["group"][ax]}
    exact_real = {(e["model"], e["axis"]): e["hits"] for e in ex if e["tag"] == "real"}
    for a in NOMINAL:
        d = {}
        d["within_jk_gp"] = {
            "replica_closed_form": real["cov"][f"within_jk_gp_rec_{a}"][0],
            "replica_exact_refit_warm": exact_real[("gp", "within")][a],
            "a3_full_refit": a3w["jackknife_plus"][a]["hits"],
        }
        d["within_raw_gp"] = {
            "replica": real["cov"][f"within_raw_gp_rec_{a}"][0],
            "a3": a3w["raw_gp"][a]["hits"],
        }
        for m, m3 in (("gp", "gp"), ("lawgp", "physgp")):
            for ax in AXES:
                d[f"group_jk_{m}_{ax}"] = {
                    "replica_exact_refit_warm": real["cov"][f"group_jk_{m}_{ax}_{a}"][0],
                    "replica_closed_form": real["cov"][f"group_jkcf_{m}_{ax}_{a}"][0],
                    "replica_exact_refit_cold_law5": exact_real[(m, ax)][a],
                    "a3_full_refit_legacy_law": a3[m3][ax]["jackknife_plus"][a]["hits"],
                }
                d[f"group_raw_{m}_{ax}"] = {
                    "replica": real["cov"][f"group_raw_{m}_{ax}_{a}"][0],
                    "a3": a3[m3][ax]["raw"][a]["hits"],
                }
        rep_check[f"coverage_{a}"] = d
    rep_check["loo_r2_reference_a2"] = {
        "gp": a2["pooled"]["rms_Voc"]["loo"]["gp"]["R2"],
        "lawgp": a2["pooled"]["rms_Voc"]["loo"]["physgp"]["R2"],
    }
    rep_check["loo_warm_start_max_abs_diff_vs_a2"] = float(
        max(abs(rep_check["loo_r2_reference_a2"][m] - rep_check["loo_r2"][m]) for m in ("gp", "lawgp"))
    )
    # pooled reference from the A2 / A5 JSONs
    try:
        refs = {}
        for ax in AXES:
            refs[ax] = {
                "gp": a2["pooled"]["rms_Voc"][ax]["gp"]["R2"],
                "lawgp": a2["pooled"]["rms_Voc"][ax]["physgp"]["R2"],
            }
        rep_check["pooled_r2_reference_a2"] = refs
        rep_check["max_abs_diff_pooled_vs_a2"] = float(
            max(abs(refs[ax][m] - rep_check["pooled_r2"][ax][m]) for ax in AXES for m in ("gp", "lawgp"))
        )
    except Exception as e:  # structure differs: record, do not invent
        rep_check["pooled_r2_reference_a2"] = f"not read: {e!r}"
    R["replica_check"] = rep_check
    R["replica_real_full"] = {k: v for k, v in real.items()}

    # ---------- exact (cold start, the A3 construction) vs warm-start exact vs closed form
    exs = {}
    for e in ex:
        if e["tag"] == "real":
            continue
        g, cvp, _, r = e["tag"].split("_")
        row = df[(df.gen == g) & (df.cv_pct == int(cvp)) & (df.design == "one") & (df.rep == int(r))].iloc[0]
        for a in NOMINAL:
            key = (g, int(cvp), e["model"], e["axis"], a)
            if e["axis"] == "within":
                warm = np.nan
                cf = row[f"cov_within_jk_gp_rec_{a}"]
            else:
                warm = row[f"cov_group_jk_{e['model']}_{e['axis']}_{a}"]
                cf = row[f"cov_group_jkcf_{e['model']}_{e['axis']}_{a}"]
            exs.setdefault(key, []).append((e["hits"][a] / N, warm, cf))
    exact_summary = {}
    for (g, cvp, m, ax, a), v in exs.items():
        v = np.array(v, float)
        exact_summary.setdefault(g, {}).setdefault(str(cvp), {}).setdefault(m, {}).setdefault(ax, {})[a] = {
            "n_grids": int(len(v)),
            "exact_cold_mean": float(v[:, 0].mean()),
            "exact_warm_mean": float(np.nanmean(v[:, 1])) if np.isfinite(v[:, 1]).any() else None,
            "closed_form_mean": float(v[:, 2].mean()),
            "max_abs_diff_warm_minus_cold": float(np.nanmax(np.abs(v[:, 1] - v[:, 0]))) if np.isfinite(v[:, 1]).any() else None,
            "mean_diff_closed_minus_exact": float((v[:, 2] - v[:, 0]).mean()),
        }
    R["exact_refit_check"] = exact_summary

    # ---------- per-setting summaries
    S = {}
    metric_cols = [c for c in df.columns if c.startswith(("loo_", "pooled_", "meanlevel_", "level_", "cov_", "within_jk", "uns_", "law_"))]
    for (g, cvp, d), sub in df.groupby(["gen", "cv_pct", "design"]):
        node = {"n_rep": int(len(sub))}
        node["metrics"] = {c: summ(sub[c].values) for c in metric_cols if sub[c].notna().any()}
        sc = {}
        for ax in AXES:
            k = int(np.sum(sub[f"pooled_r2_lawgp_{ax}"] > sub[f"pooled_r2_gp_{ax}"]))
            sc[ax] = {
                "p_law_beats_gp": k / len(sub),
                "k": k,
                "n": int(len(sub)),
                "ci95": clopper_pct(k, len(sub)),
                "median_delta_r2_law_minus_gp": float(np.median(sub[f"pooled_r2_lawgp_{ax}"] - sub[f"pooled_r2_gp_{ax}"])),
            }
            if f"pooled_r2_product_{ax}" in sub and sub[f"pooled_r2_product_{ax}"].notna().all():
                kp = int(np.sum(sub[f"pooled_r2_product_{ax}"] > sub[f"pooled_r2_gp_{ax}"]))
                sc[ax]["p_product_beats_gp"] = kp / len(sub)
        pat = (
            (sub.pooled_r2_lawgp_force > sub.pooled_r2_gp_force)
            & (sub.pooled_r2_lawgp_composition <= sub.pooled_r2_gp_composition)
            & (sub.pooled_r2_lawgp_frequency <= sub.pooled_r2_gp_frequency)
        )
        k = int(pat.sum())
        sc["real_pattern_force_only"] = {"p": k / len(sub), "k": k, "n": int(len(sub)), "ci95": clopper_pct(k, len(sub))}
        # law wins on all three axes
        allw = (
            (sub.pooled_r2_lawgp_force > sub.pooled_r2_gp_force)
            & (sub.pooled_r2_lawgp_composition > sub.pooled_r2_gp_composition)
            & (sub.pooled_r2_lawgp_frequency > sub.pooled_r2_gp_frequency)
        )
        sc["law_wins_all_axes"] = {"p": float(allw.mean()), "k": int(allw.sum())}
        node["scope"] = sc
        node["optimum_protocol_grids"] = {
            "p_best_mean_true": float(np.mean(sub.best_mean == truth["nominal"][g]["best"])),
            "p_best_recording_true": float(np.mean(sub.best_recording_comp == truth["nominal"][g]["best"])),
        }
        S.setdefault(g, {}).setdefault(str(cvp), {})[d] = node
    R["settings"] = S

    # ---------- optimum reproducibility (big simulation)
    O = {}
    for o in opt_raw:
        key = (o["gen"], int(round(o["cv"] * 100)), o["design"], o["scen"])
        cur = O.setdefault(key, {"n": 0, "best_mean": np.zeros(5, int), "best_rec": np.zeros(5, int)})
        cur["n"] += o["n"]
        cur["best_mean"] += np.array(o["best_mean"])
        cur["best_rec"] += np.array(o["best_rec"])
    OPT = {}
    for (g, cvp, d, sc), v in O.items():
        tb = truth[sc][g]["best"]
        n = v["n"]
        pm = v["best_mean"] / n
        pr_ = v["best_rec"] / n
        OPT.setdefault(sc, {}).setdefault(g, {}).setdefault(str(cvp), {})[d] = {
            "n": int(n),
            "true_best": int(tb),
            "best_mean_counts": v["best_mean"].tolist(),
            "best_recording_counts": v["best_rec"].tolist(),
            "p_best_mean_true": float(pm[tb]),
            "p_best_mean_true_ci95": clopper_pct(int(v["best_mean"][tb]), n),
            "p_best_recording_true": float(pr_[tb]),
            "p_best_recording_true_ci95": clopper_pct(int(v["best_rec"][tb]), n),
            "p_two_fabrications_agree_mean": float(np.sum(pm**2)),
            "p_two_fabrications_agree_recording": float(np.sum(pr_**2)),
        }
    R["optimum"] = OPT
    R["truth"] = truth

    # ---------- the real grid against the twin (real design, D1)
    real_vals = {
        "loo_r2_gp": real["loo"]["gp"]["r2"],
        "loo_r2_lawgp": real["loo"]["lawgp"]["r2"],
        "law_log_resid_sd": real["law"]["log_resid_sd"],
    }
    for ax in AXES:
        for m in ("gp", "lawgp", "product"):
            real_vals[f"pooled_r2_{m}_{ax}"] = real["group"][ax][m]["pooled_r2"]
    for k_ in ("within_jk_gp_rec_95", "group_jk_gp_frequency_95", "group_jk_lawgp_frequency_95", "group_jk_gp_force_95", "group_jk_lawgp_force_95", "group_jk_gp_composition_95", "group_jk_lawgp_composition_95"):
        real_vals["cov_" + k_] = real["cov"][k_][0] / N
    RG = {"real_values": real_vals, "by_setting": {}}
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            sub = df[(df.gen == g) & (df.cv_pct == cvp) & (df.design == "one")]
            RG["by_setting"].setdefault(g, {})[str(cvp)] = {
                k_: {
                    "twin": summ(sub[k_].values),
                    "frac_twin_at_or_below_real": float(np.mean(sub[k_].values <= v)),
                }
                for k_, v in real_vals.items()
                if k_ in sub
            }
    R["real_grid_vs_twin"] = RG

    # ---------- meta, parameters, fidelity
    R["meta"] = {
        "task": "E1 (Protocol V4)",
        "script": "code/a6_twin.py",
        "n_rep": int(args.n_rep),
        "n_opt_rep": int(args.n_opt_rep),
        "n_exact_rep": int(args.n_exact_rep),
        "n_grids": int(len(res)),
        "seeds": {"data": "replicate r uses seed r (0..n_rep-1); specimens SeedSequence([r, 1]), recordings SeedSequence([r, 0])", "calibration": CAL_SEEDS, "product_gp_random_state": PRODUCT_SEED},
        "target": "rms_Voc (V_rms) only",
        "generators": GENS,
        "cv_levels": CV_LEVELS,
        "designs": {"one": "one specimen per composition (real design)", "three": "three specimens per composition, Latin assignment (i_F + i_f) mod 3", "fifteen": "one specimen per recording"},
        "unsampled_levels": {"composition_wt_pct": UNS_C, "force_N": UNS_F, "frequency_Hz": UNS_FREQ},
        "deviation_from_paper": "inner LOO models of nested and group-wise jackknife+ keep the outer kernel hyperparameters and input scaling (closed form); LawGP inner law refitted warm-started; ProductGP only on the one-specimen design and only on group-wise folds and the full-grid fit; see exact_refit_check",
        "runtime_s": float(time.time() - T0),
        "cache": os.path.relpath(os.path.join(CACHE, phash), A.ROOT),
        "sim_version": SIM_VERSION,
    }
    cp_vals = {comp: cap([0, 0, 1, 2, 3][k], k == 0) for k, comp in enumerate(A.COMP_ORDER)}
    R["parameters"] = {
        "R_in_ohm": R_IN,
        "area_m2": AREA,
        "thickness_m": THICK,
        "eps_r_pristine": EPS_R_PRISTINE,
        "eps_ratio": {"PVDF": 1.0, "BaTiO3, 1, 2 wt%": EPS_RATIO_BTO, "3 wt%": EPS_RATIO_3WT},
        "C_p_nF": {k: v * 1e9 for k, v in cp_vals.items()},
        "rc_corner_Hz": {k: 1 / (2 * np.pi * R_IN * v) for k, v in cp_vals.items()},
        "sigma_t_s_calibrated": P_ser["sigma_t"],
        "sigma_a_calibrated": P_ser["sigma_a"],
        "beta_calibrated": P_ser["beta"],
        "sigma_j_s": P_ser["sigma_j"],
        "rho": P_ser["rho"],
        "kappa": P_ser["kappa"],
        "delta_q_V": P_ser["delta_q"],
        "C0": P_ser["C0"],
        "law_true_params_volt": P_ser["law"],
        "twomode": two,
        "delta_pool": emp["delta_pool"],
        "sources": {
            "R_in, area, thickness, C_p range": "project notes; project notes",
            "eps ratios": "Koc et al. 2025 Fig. 7 as read in R1 (3 wt% = 1.57 x pristine; others below pristine, 0.90 assumed)",
            "law-true parameters": "a2_common.fit_law5 on the full real V_rms grid (project notes)",
            "two-mode parameters": "bounded least squares on the full real V_rms grid (this script)",
            "delta": "a1_recordings.csv rate_rel_dev_from_nominal, recordings 52, 54, 73 excluded (project notes, 6.12)",
            "kappa": "slope of log(y/law) on log(tap_rate_used_Hz/freq_Hz), 72 recordings",
            "sigma_j": "robust SD of inter-tap intervals of the real waveforms / sqrt(2) (peaks above 40 percent of max)",
            "rho, bg ratio, delta_q": "data/long.parquet (quiet samples, robust SD / V_rms, level spacing)",
            "sigma_t, sigma_a, beta": "calibrated to a1 median cycle CV of V_rms, split-half log-ratio SD of V_rms, median background ratio",
        },
    }
    R["calibration"] = {
        "targets": {
            "cycle_cv_rms": emp["cycle_cv_median"]["rms"],
            "split_half_logratio_sd_rms": emp["split_half_logratio_sd"]["rms_Voc"],
            "bg_ratio": emp["bg_ratio_median"],
        },
        "twin_final_8_seeds": fid,
        "real": {
            "cycle_cv_median": emp["cycle_cv_median"],
            "split_half_ccc": emp["split_half_ccc"],
            "split_half_logratio_sd": emp["split_half_logratio_sd"],
            "bg_ratio_median": emp["bg_ratio_median"],
            "one_sample_excursion_frac_median": emp["one_sample_excursion_frac_median"],
            "law_log_resid_sd_72": emp["law_log_resid_sd_real"],
            "law_log_resid_sd_75": emp["law_log_resid_sd_real_all75"],
            "kappa_se": emp["kappa_se"],
        },
        "search_history": cal_hist,
    }
    R["meta"]["runtime_s"] = float(time.time() - T0)

    tables_macros_figdata(R, df)
    with open(OUT_JSON, "w") as fh:
        json.dump(C3.to_plain(R), fh, indent=1, allow_nan=True)
    print(f"[a6] wrote {OUT_JSON} in {time.time() - T0:.0f} s")


SCHEMES = [
    ("within_rec", "cov_within_jk_gp_rec_95", "within grid, recorded"),
    ("within_new", "cov_within_jk_gp_new_95", "within grid, new specimen"),
    ("group_comp", "cov_group_jk_gp_composition_95", "held-out composition"),
    ("group_force", "cov_group_jk_gp_force_95", "held-out force"),
    ("group_freq", "cov_group_jk_gp_frequency_95", "held-out frequency"),
    ("uns_comp", "cov_uns_jk_gp_composition_new_95", "unsampled composition"),
    ("uns_force", "cov_uns_jk_gp_force_new_95", "unsampled force, new specimen"),
    ("uns_freq", "cov_uns_jk_gp_frequency_new_95", "unsampled frequency, new specimen"),
]
SCHEME_WORD = {
    "within_rec": "WithinRec", "within_new": "WithinNew", "within_rep": "WithinRep",
    "group_comp": "HeldComp", "group_force": "HeldForce", "group_freq": "HeldFreq",
    "uns_comp": "UnsComp", "uns_force": "UnsForce", "uns_freq": "UnsFreq",
    "uns_force_exist": "UnsForceExist", "uns_freq_exist": "UnsFreqExist",
}


def tables_macros_figdata(R, df):
    S = R["settings"]
    M = []  # (name, value, comment)

    def add(name, val, com):
        assert not any(ch.isdigit() for ch in name), name
        M.append((name, val, com))

    meta = R["meta"]
    add("twinNRep", str(meta["n_rep"]), "synthetic replicates per setting")
    add("twinNGrids", str(meta["n_grids"]), "synthetic grids run through the protocol")
    add("twinNOptRep", str(meta["n_opt_rep"]), "replicates per setting, optimum simulation")
    add("twinNExactRep", str(meta["n_exact_rep"]), "grids per setting in the exact-refit check")
    add("twinRuntimeMin", str(int(round(meta["runtime_s"] / 60))), "script runtime, minutes")
    Pm = R["parameters"]
    add("twinSigmaTMs", mac_num(Pm["sigma_t_s_calibrated"] * 1e3, 2), "calibrated force-pulse SD, ms")
    add("twinSigmaA", mac_num(Pm["sigma_a_calibrated"], 2), "calibrated per-tap log-amplitude jitter")
    add("twinBeta", mac_num(Pm["beta_calibrated"], 2), "calibrated background SD / pulse RMS")
    add("twinSigmaJMs", mac_num(Pm["sigma_j_s"] * 1e3, 1), "tap timing jitter SD, ms (real waveforms)")
    add("twinRho", mac_num(Pm["rho"], 2), "background lag-one correlation (real waveforms)")
    add("twinKappa", mac_num(Pm["kappa"], 2), "exponent of the tap-rate deviation on V_rms (72 real recordings)")
    add("twinKappaSe", mac_num(R["calibration"]["real"]["kappa_se"], 2), "SE of kappa")
    add("twinDeltaQmV", mac_num(Pm["delta_q_V"] * 1e3, 1), "quantization step, mV")
    cpn = list(Pm["C_p_nF"].values())
    add("twinCpMinNf", mac_num(min(cpn), 2), "smallest C_p of the twin, nF")
    add("twinCpMaxNf", mac_num(max(cpn), 2), "largest C_p of the twin, nF")
    rc = list(Pm["rc_corner_Hz"].values())
    add("twinCornerMinHz", str(int(round(min(rc)))), "lowest RC corner of the twin, Hz")
    add("twinCornerMaxHz", str(int(round(max(rc)))), "highest RC corner of the twin, Hz")
    cal = R["calibration"]
    fidv = cal["twin_final_8_seeds"]
    rl = cal["real"]
    add("twinCalCycleCvRmsTwin", mac_num(100 * fidv["cycle_cv_median"]["rms"], 1), "twin median cycle CV V_rms, percent (calibrated)")
    add("twinCalCycleCvRmsReal", mac_num(100 * rl["cycle_cv_median"]["rms"], 1), "real median cycle CV V_rms, percent")
    add("twinCalCycleCvVppTwin", mac_num(100 * fidv["cycle_cv_median"]["vpp"], 1), "twin median cycle CV V_pp (fidelity)")
    add("twinCalCycleCvVppReal", mac_num(100 * rl["cycle_cv_median"]["vpp"], 1), "real median cycle CV V_pp")
    add("twinCalCycleCvVmaxTwin", mac_num(100 * fidv["cycle_cv_median"]["vmax"], 1), "twin median cycle CV |V|max (fidelity)")
    add("twinCalCycleCvVmaxReal", mac_num(100 * rl["cycle_cv_median"]["vmax"], 1), "real median cycle CV |V|max")
    add("twinCalLsdRmsTwin", mac_num(fidv["split_half_logratio_sd"]["rms_Voc"], 3), "twin split-half log-ratio SD V_rms (calibrated)")
    add("twinCalLsdRmsReal", mac_num(rl["split_half_logratio_sd"]["rms_Voc"], 3), "real split-half log-ratio SD V_rms")
    add("twinCalLsdVmaxTwin", mac_num(fidv["split_half_logratio_sd"]["Vmax"], 3), "twin split-half log-ratio SD |V|max (fidelity)")
    add("twinCalLsdVmaxReal", mac_num(rl["split_half_logratio_sd"]["Vmax"], 3), "real split-half log-ratio SD |V|max")
    add("twinCalCccRmsTwin", mac_num(fidv["split_half_ccc"]["rms_Voc"], 3), "twin split-half CCC V_rms (fidelity)")
    add("twinCalCccRmsReal", mac_num(rl["split_half_ccc"]["rms_Voc"], 3), "real split-half CCC V_rms")
    add("twinCalOneSampleTwin", pct(fidv["one_sample_frac_median"]), "twin median share of one-sample excursions, percent")
    add("twinCalOneSampleReal", pct(rl["one_sample_excursion_frac_median"]), "real median share of one-sample excursions, percent")
    two = Pm["twomode"]
    tp = dict(zip(two["names"], two["params"]))
    add("twinTwoFOne", mac_num(tp["f1"], 1), "two-mode generator main mode, Hz")
    add("twinTwoGOne", mac_num(tp["g1"], 1), "two-mode main width, Hz")
    add("twinTwoFTwo", mac_num(tp["f2"], 1), "two-mode second mode, Hz")
    add("twinTwoGTwo", mac_num(tp["g2"], 1), "two-mode second width, Hz")
    add("twinTwoW", pct(tp["w"]), "two-mode weight of the main mode, percent")
    add("twinTwoSseReduction", pct(1 - two["sse"] / two["sse_law5"]), "SSE reduction of the two-mode fit over the law on the real grid, percent")
    for sc_ in ("nominal", "high_cp", "no_rc"):
        sw = {"nominal": "", "high_cp": "HighCp", "no_rc": "NoRc"}[sc_]
        for g in GENS:
            tr_ = R["truth"][sc_][g]
            add(f"twin{GEN_WORD[g]}TrueGap{sw}", mac_num(tr_["gap_best_to_second_pct"], 1), f"{g}: true gap best to second composition, percent ({sc_})")
            if sc_ == "nominal":
                add(f"twin{GEN_WORD[g]}TrueCStar", mac_num(tr_["c_star_continuous"], 2), f"{g}: true continuous optimum, wt% CNT")
                add(f"twin{GEN_WORD[g]}HThree", mac_num(100 * (1 - tr_["h_by_comp"][4] / tr_["h_by_comp"][1]), 1), f"{g}: RC attenuation at 3 wt% relative to BaTiO3, percent")

    # ---------- replica check
    rc_ = R["replica_check"]
    add("twinRepLooGp", mac_num(rc_["loo_r2"]["gp"], 3), "replica LOO R2 plain GP, real grid")
    add("twinRepLooLaw", mac_num(rc_["loo_r2"]["lawgp"], 3), "replica LOO R2 LawGP, real grid")
    if isinstance(rc_.get("max_abs_diff_pooled_vs_a2"), float):
        add("twinRepMaxDiffPooled", f"\\num{{{rc_['max_abs_diff_pooled_vs_a2']:.1e}}}", "max |replica - A2| pooled R2")
    for a in NOMINAL:
        aw = {"95": "NinetyFive", "90": "Ninety"}[a]
        d = rc_[f"coverage_{a}"]
        add(f"twinRepWithinClosed{aw}", str(d["within_jk_gp"]["replica_closed_form"]), f"real within-grid jk+ hits /75, closed form, {a}")
        add(f"twinRepWithinExact{aw}", str(d["within_jk_gp"]["replica_exact_refit_warm"]), f"real within-grid jk+ hits /75, exact warm refit, {a}")
        add(f"twinRepWithinAthree{aw}", str(d["within_jk_gp"]["a3_full_refit"]), f"real within-grid jk+ hits /75, A3 full refit, {a}")
        for m in ("gp", "lawgp"):
            for ax in AXES:
                e = d[f"group_jk_{m}_{ax}"]
                add(f"twinRep{MOD_WORD[m]}{AX_WORD[ax]}Warm{aw}", str(e["replica_exact_refit_warm"]), f"real group-wise jk+ hits /75, exact warm refit (main run), {m} {ax} {a}")
                add(f"twinRep{MOD_WORD[m]}{AX_WORD[ax]}Cold{aw}", str(e["replica_exact_refit_cold_law5"]), f"real group-wise jk+ hits /75, exact cold refit, {m} {ax} {a}")
                add(f"twinRep{MOD_WORD[m]}{AX_WORD[ax]}Closed{aw}", str(e["replica_closed_form"]), f"real group-wise jk+ hits /75, closed form, {m} {ax} {a}")
                add(f"twinRep{MOD_WORD[m]}{AX_WORD[ax]}Athree{aw}", str(e["a3_full_refit_legacy_law"]), f"A3 group-wise jk+ hits /75 {m} {ax} {a}")
    add("twinRepLooWarmMaxDiff", f"\\num{{{rc_['loo_warm_start_max_abs_diff_vs_a2']:.1e}}}", "max |warm-start LOO R2 - A2 cold-start LOO R2|")
    ex_ = R["exact_refit_check"]
    wc = [
        ex_[g][c][m][ax]["95"]["max_abs_diff_warm_minus_cold"]
        for g in ex_ for c in ex_[g] for m in ex_[g][c] for ax in ex_[g][c][m]
        if ex_[g][c][m][ax]["95"]["max_abs_diff_warm_minus_cold"] is not None
    ]
    if wc:
        add("twinWarmColdMaxDiffPts", mac_num(100 * max(wc), 1), "max |warm - cold exact| group-wise coverage over checked grids, points")
    cfd = [
        ex_[g][c][m][ax]["95"]["mean_diff_closed_minus_exact"]
        for g in ex_ for c in ex_[g] for m in ex_[g][c] for ax in ex_[g][c][m]
    ]
    if cfd:
        add("twinClosedDiffMinPts", mac_num(100 * min(cfd), 1), "most negative mean coverage difference closed form minus exact, 95, points")
        add("twinClosedDiffMaxPts", mac_num(100 * max(cfd), 1), "largest mean coverage difference closed form minus exact, 95, points")
    for g in ex_:
        for c in ex_[g]:
            if "within" in ex_[g][c].get("gp", {}):
                e = ex_[g][c]["gp"]["within"]["95"]
                base = f"twin{GEN_WORD[g]}Cv{CV_WORD[int(c)]}ExactWithin"
                add(base, pct(e["exact_cold_mean"]), f"exact nested within-grid jk+ 95 coverage, percent, {g} CV {c}")
                add(base + "Closed", pct(e["closed_form_mean"]), "same grids, closed form")

    # ---------- scope (real design)
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            node = S[g][str(cvp)]["one"]
            base = f"twin{GEN_WORD[g]}Cv{CV_WORD[cvp]}"
            for ax in AXES:
                add(f"{base}Scope{AX_WORD[ax]}", pct(node["scope"][ax]["p_law_beats_gp"]), f"percent of replicates LawGP > GP pooled R2, held-out {ax}")
                for m in ("gp", "lawgp", "product"):
                    k_ = f"pooled_r2_{m}_{ax}"
                    if k_ in node["metrics"]:
                        add(f"{base}Pooled{AX_WORD[ax]}{MOD_WORD[m]}", mac_num(node["metrics"][k_]["median"], 2), f"median pooled R2 {m} held-out {ax}")
            add(f"{base}ScopeReal", pct(node["scope"]["real_pattern_force_only"]["p"]), "percent of replicates with the real pattern (LawGP better on force only)")
            add(f"{base}ScopeAll", pct(node["scope"]["law_wins_all_axes"]["p"]), "percent of replicates with LawGP better on all three axes")
            for m in ("gp", "lawgp"):
                add(f"{base}Loo{MOD_WORD[m]}", mac_num(node["metrics"][f"loo_r2_{m}"]["median"], 3), f"median LOO R2 {m}")
                add(f"{base}Loo{MOD_WORD[m]}Lo", mac_num(node["metrics"][f"loo_r2_{m}"]["p2.5"], 3), f"2.5th percentile LOO R2 {m}")
                add(f"{base}Loo{MOD_WORD[m]}Hi", mac_num(node["metrics"][f"loo_r2_{m}"]["p97.5"], 3), f"97.5th percentile LOO R2 {m}")
            add(f"{base}LawLogSd", mac_num(node["metrics"]["law_log_resid_sd"]["median"], 2), "median SD of log(y / law fit) on the full grid")
            add(f"{base}FreqLawLevelMin", mac_num(min(node["metrics"][f"level_r2_lawgp_frequency_{A.level_key('frequency', v)}"]["median"] for v in FREQS), 2), "smallest median within-level R2 of LawGP over held-out frequencies")
            rg = R["real_grid_vs_twin"]["by_setting"][g][str(cvp)]
            for k_, w_ in (("loo_r2_gp", "LooGp"), ("loo_r2_lawgp", "LooLaw"), ("pooled_r2_gp_frequency", "FreqGp"), ("pooled_r2_lawgp_frequency", "FreqLaw"), ("pooled_r2_lawgp_force", "ForceLaw"), ("pooled_r2_gp_composition", "CompGp"), ("pooled_r2_lawgp_composition", "CompLaw"), ("law_log_resid_sd", "LawLogSd")):
                add(f"{base}Below{w_}", pct(rg[k_]["frac_twin_at_or_below_real"]), f"percent of twin replicates at or below the real value of {k_}")
    rv = R["real_grid_vs_twin"]["real_values"]
    add("twinRealLawLogSd", mac_num(rv["law_log_resid_sd"], 2), "real grid SD of log(y / law fit), 75 recordings")

    # ---------- coverage
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            for d in DESIGNS:
                node = S[g][str(cvp)][d]["metrics"]
                base = f"twin{GEN_WORD[g]}Des{DES_WORD[d]}Cv{CV_WORD[cvp]}"
                for sch, col, _ in SCHEMES:
                    if col in node:
                        add(f"{base}Cov{SCHEME_WORD[sch]}", pct(node[col]["mean"]), f"mean jk+ 95 coverage plain GP, {sch}, percent")
                for col, w_ in (("cov_within_jk_gp_rep_95", "WithinRep"), ("cov_uns_jk_gp_force_exist_95", "UnsForceExist"), ("cov_uns_jk_gp_frequency_exist_95", "UnsFreqExist")):
                    if col in node:
                        add(f"{base}Cov{w_}", pct(node[col]["mean"]), f"mean jk+ 95 coverage plain GP, {col}")
                for ax in AXES:
                    col = f"cov_group_jk_lawgp_{ax}_95"
                    if col in node:  # LawGP group-wise coverage exists only for designs where it was computed
                        add(f"{base}CovLaw{AX_WORD[ax]}", pct(node[col]["mean"]), f"mean group-wise jk+ 95 coverage LawGP, {ax}")
                add(f"{base}Loo", mac_num(node["loo_r2_gp"]["median"], 3), "median LOO R2 plain GP")
                o = R["optimum"]["nominal"][g][str(cvp)][d]
                add(f"{base}OptMean", pct(o["p_best_mean_true"]), "percent: best composition mean is the true optimum")
                add(f"{base}OptRec", pct(o["p_best_recording_true"]), "percent: best single recording on the true optimum composition")
                add(f"{base}OptAgree", pct(o["p_two_fabrications_agree_mean"]), "percent: two independent fabrications agree on the best composition mean")
            for sc_, sw in (("high_cp", "HighCp"), ("no_rc", "NoRc")):
                o = R["optimum"][sc_][g][str(cvp)]["one"]
                add(f"twin{GEN_WORD[g]}Cv{CV_WORD[cvp]}OptMean{sw}", pct(o["p_best_mean_true"]), f"optimum reproducibility, capacitance scenario {sc_}")

    # ---------- tables
    # (1) scope recovery
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Generator & CV (\%) & \multicolumn{3}{c}{LawGP beats plain GP (\% of replicates)} & Force only (\%) & All axes (\%) \\",
         r"\cmidrule(lr){3-5}", r" & & Composition & Force & Frequency & & \\", r"\midrule"]
    gname = {"law": "Law true", "twomode": "Two-mode"}
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            sc = S[g][str(cvp)]["one"]["scope"]
            L.append(f"{gname[g]} & {cvp} & " + " & ".join(pct(sc[ax]["p_law_beats_gp"]) for ax in AXES)
                     + f" & {pct(sc['real_pattern_force_only']['p'])} & {pct(sc['law_wins_all_axes']['p'])} \\\\")
        L.append(r"\midrule")
    rr = R["replica_check"]["pooled_r2"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    L.append("Measured grid & -- & " + " & ".join(yn(rr[ax]["lawgp"] > rr[ax]["gp"]) for ax in AXES) + " & yes & no \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a6_scope.tex"), "w").write("\n".join(L) + "\n")

    # (2) coverage vs CV and design, plain GP jackknife+ 95
    for g, fn in (("law", "tab_a6_coverage.tex"), ("twomode", "tab_a6_coverage_twomode.tex")):
        L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Design & CV (\%) & \multicolumn{2}{c}{Within grid} & \multicolumn{3}{c}{Held-out level} & \multicolumn{3}{c}{Unsampled level, new specimen} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r" & & Recorded & New specimen & Comp. & Force & Freq. & Comp. & Force & Freq. \\", r"\midrule"]
        dname = {"one": "One specimen", "three": "Three specimens", "fifteen": "One per recording"}
        for d in DESIGNS:
            for cv in CV_LEVELS:
                cvp = int(round(cv * 100))
                node = S[g][str(cvp)][d]["metrics"]
                L.append(f"{dname[d]} & {cvp} & " + " & ".join(pct(node[col]["mean"]) for _, col, _ in SCHEMES) + r" \\")
            if d != DESIGNS[-1]:
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, fn), "w").write("\n".join(L) + "\n")

    # (3) optimum reproducibility
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Design & CV (\%) & \multicolumn{3}{c}{Law true} & \multicolumn{3}{c}{Two-mode} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r" & & Mean & Recording & Agree & Mean & Recording & Agree \\", r"\midrule"]
    for d in DESIGNS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            cells = []
            for g in GENS:
                o = R["optimum"]["nominal"][g][str(cvp)][d]
                cells += [pct(o["p_best_mean_true"]), pct(o["p_best_recording_true"]), pct(o["p_two_fabrications_agree_mean"])]
            L.append(f"{dname[d]} & {cvp} & " + " & ".join(cells) + r" \\")
        if d != DESIGNS[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a6_optimum.tex"), "w").write("\n".join(L) + "\n")

    # (4) the real grid against the twin (real design)
    rows = [
        ("loo_r2_gp", "LOO $R^2$, plain GP", 3),
        ("loo_r2_lawgp", "LOO $R^2$, LawGP", 3),
        ("pooled_r2_gp_composition", "Held-out composition, plain GP", 2),
        ("pooled_r2_lawgp_composition", "Held-out composition, LawGP", 2),
        ("pooled_r2_gp_force", "Held-out force, plain GP", 2),
        ("pooled_r2_lawgp_force", "Held-out force, LawGP", 2),
        ("pooled_r2_gp_frequency", "Held-out frequency, plain GP", 2),
        ("pooled_r2_lawgp_frequency", "Held-out frequency, LawGP", 2),
        ("law_log_resid_sd", "SD of $\\log(y/m)$, full-grid law", 2),
    ]
    show = [("law", 0), ("law", 20), ("twomode", 0), ("twomode", 20)]
    L = [r"\begin{tabular}{lr" + "r" * len(show) + "}", r"\toprule",
         r"Quantity & Measured & " + " & ".join(f"{gname[g]}, CV {c}\\%" for g, c in show) + r" \\", r"\midrule"]
    for k_, lab, nd in rows:
        cells = []
        for g, c in show:
            e = R["real_grid_vs_twin"]["by_setting"][g][str(c)][k_]
            cells.append(f"{fnum_tex(e['twin']['median'], nd)} [{fnum_tex(e['twin']['p2.5'], nd)}, {fnum_tex(e['twin']['p97.5'], nd)}]")
        L.append(f"{lab} & {fnum_tex(rv[k_], nd)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a6_realgrid.tex"), "w").write("\n".join(L) + "\n")

    # (5) calibration and fidelity
    L = [r"\begin{tabular}{lrrl}", r"\toprule", r"Quantity & Measured & Twin & Role \\", r"\midrule"]
    L.append(f"Median cycle CV of \\RMS{{}} (\\%) & {mac_num(100 * rl['cycle_cv_median']['rms'], 1)} & {mac_num(100 * fidv['cycle_cv_median']['rms'], 1)} & calibrated \\\\")
    L.append(f"Split-half log-ratio SD of \\RMS{{}} & {mac_num(rl['split_half_logratio_sd']['rms_Voc'], 3)} & {mac_num(fidv['split_half_logratio_sd']['rms_Voc'], 3)} & calibrated \\\\")
    L.append(f"Background robust SD / \\RMS{{}} & {mac_num(rl['bg_ratio_median'], 2)} & {mac_num(fidv['bg_ratio_median'], 2)} & calibrated \\\\")
    L.append(f"Median cycle CV of \\Vpp{{}} (\\%) & {mac_num(100 * rl['cycle_cv_median']['vpp'], 1)} & {mac_num(100 * fidv['cycle_cv_median']['vpp'], 1)} & check \\\\")
    L.append(f"Median cycle CV of \\Vmax{{}} (\\%) & {mac_num(100 * rl['cycle_cv_median']['vmax'], 1)} & {mac_num(100 * fidv['cycle_cv_median']['vmax'], 1)} & check \\\\")
    L.append(f"Split-half log-ratio SD of \\Vmax{{}} & {mac_num(rl['split_half_logratio_sd']['Vmax'], 3)} & {mac_num(fidv['split_half_logratio_sd']['Vmax'], 3)} & check \\\\")
    L.append(f"Split-half CCC of \\RMS{{}} & {mac_num(rl['split_half_ccc']['rms_Voc'], 3)} & {mac_num(fidv['split_half_ccc']['rms_Voc'], 3)} & check \\\\")
    L.append(f"One-sample excursions (\\%) & {pct(rl['one_sample_excursion_frac_median'])} & {pct(fidv['one_sample_frac_median'])} & check \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a6_calibration.tex"), "w").write("\n".join(L) + "\n")

    # (6) exact-refit check: cold exact / warm exact (main run) / closed form, jackknife+ 95
    L = [r"\begin{tabular}{lllrrrr}", r"\toprule",
         r"Grid & CV (\%) & Model & Within grid & Held-out comp. & Held-out force & Held-out freq. \\", r"\midrule"]
    d95 = R["replica_check"]["coverage_95"]
    for m in ("gp", "lawgp"):
        w_ = "--"
        if m == "gp":
            w_ = f"{pct(d95['within_jk_gp']['replica_exact_refit_warm'] / N)} / -- / {pct(d95['within_jk_gp']['replica_closed_form'] / N)}"
        cells = [w_] + [
            f"{pct(d95[f'group_jk_{m}_{ax}']['replica_exact_refit_cold_law5'] / N)} / {pct(d95[f'group_jk_{m}_{ax}']['replica_exact_refit_warm'] / N)} / {pct(d95[f'group_jk_{m}_{ax}']['replica_closed_form'] / N)}"
            for ax in AXES
        ]
        L.append(f"Measured & -- & {'plain GP' if m == 'gp' else 'LawGP'} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for g in GENS:
        for c in ("0", "20"):
            if g not in ex_ or c not in ex_[g]:
                continue
            for m in ("gp", "lawgp"):
                cells = []
                if m == "gp" and "within" in ex_[g][c][m]:
                    e = ex_[g][c][m]["within"]["95"]
                    cells.append(f"{pct(e['exact_cold_mean'])} / -- / {pct(e['closed_form_mean'])}")
                else:
                    cells.append("--")
                for ax in AXES:
                    e = ex_[g][c][m][ax]["95"]
                    cells.append(f"{pct(e['exact_cold_mean'])} / {pct(e['exact_warm_mean'])} / {pct(e['closed_form_mean'])}")
                L.append(f"{gname[g]} & {c} & {'plain GP' if m == 'gp' else 'LawGP'} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_a6_exact.tex"), "w").write("\n".join(L) + "\n")

    # ---------- figure data
    rows = []
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            for d in DESIGNS:
                node = S[g][str(cvp)][d]["metrics"]
                for a in NOMINAL:
                    cols = [
                        ("within_rec", "gp", f"cov_within_jk_gp_rec_{a}"),
                        ("within_rep", "gp", f"cov_within_jk_gp_rep_{a}"),
                        ("within_new", "gp", f"cov_within_jk_gp_new_{a}"),
                    ]
                    for m in ("gp", "lawgp"):
                        for ax, w_ in (("composition", "comp"), ("force", "force"), ("frequency", "freq")):
                            cols.append((f"group_{w_}", m, f"cov_group_jk_{m}_{ax}_{a}"))
                            cols.append((f"uns_{w_}_new", m, f"cov_uns_jk_{m}_{ax}_new_{a}"))
                            if ax != "composition":
                                cols.append((f"uns_{w_}_exist", m, f"cov_uns_jk_{m}_{ax}_exist_{a}"))
                    for sch, m, col in cols:
                        if col not in node:
                            continue
                        e = node[col]
                        rows.append({"generator": g, "design": d, "cv_pct": cvp, "scheme": sch, "model": m, "nominal_pct": int(a),
                                     "coverage_mean": e["mean"], "coverage_se": e["se"], "coverage_p2_5": e["p2.5"], "coverage_p97_5": e["p97.5"], "n_rep": e["n"]})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "a6_figdata_coverage.csv"), index=False)
    rows = []
    for g in GENS:
        for cv in CV_LEVELS:
            cvp = int(round(cv * 100))
            for d in DESIGNS:
                node = S[g][str(cvp)][d]
                for ax in AXES:
                    for m in ("gp", "lawgp", "product"):
                        k_ = f"pooled_r2_{m}_{ax}"
                        if k_ not in node["metrics"]:
                            continue
                        e = node["metrics"][k_]
                        rows.append({"generator": g, "design": d, "cv_pct": cvp, "axis": ax, "model": m,
                                     "pooled_r2_median": e["median"], "pooled_r2_p25": e["p25"], "pooled_r2_p75": e["p75"],
                                     "p_law_beats_gp": node["scope"][ax]["p_law_beats_gp"],
                                     "median_delta_law_minus_gp": node["scope"][ax]["median_delta_r2_law_minus_gp"],
                                     "real_grid_value": rv.get(k_, np.nan), "n_rep": e["n"]})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "a6_figdata_scope.csv"), index=False)

    # ---------- macros file
    lines = ["% numbers_a6.tex, generated by code/a6_twin.py (task E1). Do not edit."]
    seen = set()
    for n_, v_, c_ in M:
        if n_ in seen:
            raise ValueError(f"duplicate macro {n_}")
        seen.add(n_)
        lines.append(f"\\newcommand{{\\{n_}}}{{{v_}}}% {c_}")
    open(os.path.join(RES, "numbers_a6.tex"), "w").write("\n".join(lines) + "\n")
    R["meta"]["n_macros"] = len(M)
    R["index"] = {
        "results/tables/tab_a6_scope.tex": "settings.<gen>.<cv>.one.scope; replica_check.pooled_r2",
        "results/tables/tab_a6_coverage.tex": "settings.law.<cv>.<design>.metrics.cov_* (mean)",
        "results/tables/tab_a6_coverage_twomode.tex": "settings.twomode.<cv>.<design>.metrics.cov_* (mean)",
        "results/tables/tab_a6_optimum.tex": "optimum.nominal.<gen>.<cv>.<design>",
        "results/tables/tab_a6_realgrid.tex": "real_grid_vs_twin",
        "results/tables/tab_a6_calibration.tex": "calibration",
        "results/tables/tab_a6_exact.tex": "exact_refit_check; replica_check.coverage_95",
        "results/a6_figdata_coverage.csv": "settings.*.metrics.cov_*",
        "results/a6_figdata_scope.csv": "settings.*.metrics.pooled_r2_*; settings.*.scope",
        "results/a6_replicates.csv": "one row per synthetic grid (flat metrics)",
        "results/numbers_a6.tex": "macros twin*: meta, parameters, calibration, replica_check, exact_refit_check, settings, optimum, truth, real_grid_vs_twin",
    }


if __name__ == "__main__":
    main()
