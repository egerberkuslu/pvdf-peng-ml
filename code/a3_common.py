"""Shared helpers for task A3 (benchmark, coverage, active-learning and replication tables).

Everything here is copied from the legacy scripts that produced the submitted
manuscript so that recomputed numbers are comparable one to one:
  gp(), fit_law(), law()          <- code/baseline/physgp_analysis.py
  jk_interval()                   <- code/baseline/groupwise_conformal.py
  jk_quantile(), Z                <- code/baseline/calibrated_conformal.py
  clopper()                       <- code/baseline/revision_experiments.py
Run from the repository root.
"""

import json
import os
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy.optimize import curve_fit
from scipy.stats import beta, binom
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
REV = ROOT
LEG = REV / "results" / "baseline"
RES = REV / "results"
TAB = RES / "tables"
FIG = REV / "figures"
CACHE = RES / "a3_cache"
DATA = REV / "data" / "targets_design.parquet"

SEED = 0
TARGETS = ["rms_Voc", "Vpp", "Vmax"]  # energy is excluded from the revision
LEGACY_TKEY = {"rms_Voc": "rms", "Vpp": "vpp", "Vmax": "peak_abs"}  # reg_*.json keys
TEX_T = {"rms_Voc": r"\RMS", "Vpp": r"\Vpp", "Vmax": r"\Vmax"}
CAMEL_T = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}
AXES = ["composition", "force", "frequency"]
LEVELS = {"90": 0.10, "95": 0.05}
Z = {"90": 1.6448536269514722, "95": 1.959963984540054}
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]

df = pd.read_parquet(DATA)
X = df[FEAT].values.astype(float)
CNT = df["cnt_pct"].values.astype(float)
FRC = df["force_N"].values.astype(float)
FRQ = df["freq_Hz"].values.astype(float)
N = len(df)
GROUPS = {
    "composition": df["composition"].values,
    "force": df["force_N"].values,
    "frequency": df["freq_Hz"].values,
}


def yv(tgt):
    return df[tgt].values.astype(float)


def load_legacy(name):
    return json.loads((LEG / name).read_text())


# ---------------------------------------------------------------- models
def gp():
    """Plain GP of the interval and PhysGP analyses (legacy physgp_analysis.gp)."""
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


P0 = [1.0, 0.1, -0.02, 1.0, 19.0, 8.0]
BOUNDS = ([0, -5, -5, 0, 10, 1], [50, 5, 5, 50, 30, 30])


def law(Xt, b0, b1, b2, A, f0, g):
    c, F, f = Xt
    lor = A * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)
    return (b0 + b1 * c + b2 * c**2) * F * lor


def fit_law(tr, y):
    """Legacy PhysGP mean: bounded least squares on the training rows only."""
    scale = float(np.std(y[tr])) or 1.0
    try:
        p, _ = curve_fit(
            law,
            (CNT[tr], FRC[tr], FRQ[tr]),
            y[tr] / scale,
            p0=P0,
            bounds=BOUNDS,
            maxfev=20000,
        )
    except Exception:
        p = np.array(P0)
    return lambda idx: law((CNT[idx], FRC[idx], FRQ[idx]), *p) * scale


def fit_predict(model, tr, query, y):
    """Fit `model` ('gp' or 'physgp') on rows tr, return (mu, sd) at rows query.

    PhysGP refits the law on tr (inside every training set, including every
    inner jackknife+ set) and the GP models y - m(x); sd is the residual-GP sd.
    """
    tr = np.asarray(tr)
    query = np.asarray(query)
    xs = StandardScaler().fit(X[tr])
    if model == "gp":
        g = gp().fit(xs.transform(X[tr]), y[tr])
        mu, sd = g.predict(xs.transform(X[query]), return_std=True)
        return mu, sd
    if model == "physgp":
        m = fit_law(tr, y)
        g = gp().fit(xs.transform(X[tr]), y[tr] - m(tr))
        rmu, rsd = g.predict(xs.transform(X[query]), return_std=True)
        return m(query) + rmu, rsd
    raise ValueError(model)


# ---------------------------------------------------------------- intervals
def jk_bounds(preds, resids, alpha):
    """Jackknife+ (Barber et al. 2021) bounds from ensemble predictions and residuals."""
    m = len(resids)
    kk = int(np.clip(np.ceil((1 - alpha) * (m + 1)), 1, m))
    lo = np.sort(preds - resids)[m - kk]
    hi = np.sort(preds + resids)[kk - 1]
    return float(lo), float(hi)


def clopper(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval (legacy revision_experiments.clopper)."""
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def mcnemar_exact(a_hits, b_hits):
    """Exact two-sided McNemar on paired booleans (legacy groupwise_conformal)."""
    a = np.asarray(a_hits, bool)
    b = np.asarray(b_hits, bool)
    n_b_only = int(np.sum(~a & b))
    n_a_only = int(np.sum(a & ~b))
    m = n_b_only + n_a_only
    p = float(2 * binom.cdf(min(n_b_only, n_a_only), m, 0.5)) if m > 0 else 1.0
    return n_b_only, n_a_only, min(p, 1.0)


def cov_record(hits, widths):
    hits = np.asarray(hits, bool)
    k, n = int(hits.sum()), int(hits.size)
    return {
        "coverage": k / n,
        "hits": k,
        "n": n,
        "ci95": clopper(k, n),
        "ci_method": "Clopper-Pearson exact, 95%",
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
    }


# ---------------------------------------------------------------- formatting
def rhu(x, nd=0):
    """Round half up (the submitted tables were rounded this way)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def fnum(x, nd):
    v = rhu(x, nd)
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def fpct(x):
    return f"{int(rhu(100 * x, 0))}"


def width_nd(tgt):
    return 3 if tgt == "rms_Voc" else 2


def cov_cell(rec):
    """'96\\% (72/75) [89, 99]' with Clopper-Pearson bounds in percent."""
    lo, hi = rec["ci95"]
    return (
        rf"{fpct(rec['coverage'])}\% ({rec['hits']}/{rec['n']}) "
        rf"[{fpct(lo)}, {fpct(hi)}]"
    )


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def to_plain(o):
    """Recursively convert numpy types for json.dump."""
    if isinstance(o, dict):
        return {str(k): to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_plain(o.tolist())
    return o


# ---------------------------------------------------------------- figure style
CM = 1 / 2.54
COL_W = 8.4 * CM
TEXT_W = 17.4 * CM
C_BLUE, C_TEAL, C_AMBER, C_RED, C_GREY = (
    "#3B6FB6",
    "#2A9D8F",
    "#E08D2F",
    "#C8553D",
    "#6C757D",
)
TARGET_LABEL = {
    "rms_Voc": r"$V_{\mathrm{rms}}$",
    "Vpp": r"$V_{\mathrm{pp}}$",
    "Vmax": r"$|V|_{\mathrm{max}}$",
}


def apply_style():
    """House style of protocol section 7 (values mirrored from the A4 board post)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

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
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def save_fig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=150)
