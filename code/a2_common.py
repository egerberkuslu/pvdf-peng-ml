"""Shared helpers for task A2 (PhysGP identifiability, per-level results, selection rule).

Data loading, the plain GP of the submitted manuscript, the legacy six-parameter
voltage law, the identifiable five-parameter reparameterization, and fold-level
predictors. The loader and model code are copied from
code/baseline/physgp_analysis.py and revision_experiments.py so the
legacy scripts never have to run in place.

Reparameterization. The legacy law
    m(x) = (b0 + b1 c + b2 c^2) * F * A (g/2)^2 / ((f - f0)^2 + (g/2)^2)
identifies only the products A*b0, A*b1, A*b2. With a_i = A b_i the law becomes
    m(x) = (a0 + a1 c + a2 c^2) * F * (g/2)^2 / ((f - f0)^2 + (g/2)^2)
with five parameters. Legacy bounds A in [0, 50], b0 in [0, 50], b1, b2 in
[-5, 5] map onto the box a0 in [0, 2500], a1, a2 in [-250, 250] exactly (any
point of that box is reached with A = 50, b = a / 50), so the feasible set of
the least-squares problem is unchanged. The legacy start (b0, b1, b2, A) =
(1, 0.1, -0.02, 1) maps to (a0, a1, a2) = (1, 0.1, -0.02). The fit runs on
y / std(y[train]) as in the legacy code; parameters are reported on the volt
scale (a_i in V N^-1) by multiplying a_i by that scale.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from peng_paths import ROOT_STR as ROOT  # repository root (env PENG_ROOT overrides)
REV = ROOT
DATA = os.path.join(REV, "data", "targets_design.parquet")

df = pd.read_parquet(DATA)
FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
X = df[FEAT].values.astype(float)
CNT = df["cnt_pct"].values.astype(float)
FRC = df["force_N"].values.astype(float)
FRQ = df["freq_Hz"].values.astype(float)
COMP = df["composition"].values
N = len(df)

TARGETS = ["rms_Voc", "Vpp", "Vmax"]  # energy proxy dropped (analysis protocol 3)
TARGET_LABEL = {
    "rms_Voc": r"$V_\mathrm{rms}$",
    "Vpp": r"$V_\mathrm{pp}$",
    "Vmax": r"$|V|_\mathrm{max}$",
}
TARGET_TEX = {"rms_Voc": r"\RMS", "Vpp": r"\Vpp", "Vmax": r"\Vmax"}
TARGET_MACRO = {"rms_Voc": "Rms", "Vpp": "Vpp", "Vmax": "Vmax"}

AXES = {"composition": COMP, "force": FRC, "frequency": FRQ}
COMP_ORDER = [
    "PVDF",
    "PVDF+BaTiO3",
    "PVDF+BaTiO3+%1CNT",
    "PVDF+BaTiO3+%2CNT",
    "PVDF+BaTiO3+%3CNT",
]
COMP_SHORT = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": r"+BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "+1% CNT",
    "PVDF+BaTiO3+%2CNT": "+2% CNT",
    "PVDF+BaTiO3+%3CNT": "+3% CNT",
}
COMP_TEX = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": r"PVDF/BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": r"+1 wt\% CNT",
    "PVDF+BaTiO3+%2CNT": r"+2 wt\% CNT",
    "PVDF+BaTiO3+%3CNT": r"+3 wt\% CNT",
}
COMP_MACRO = {
    "PVDF": "Pristine",
    "PVDF+BaTiO3": "Bto",
    "PVDF+BaTiO3+%1CNT": "CntOne",
    "PVDF+BaTiO3+%2CNT": "CntTwo",
    "PVDF+BaTiO3+%3CNT": "CntThree",
}
NUM_WORD = {
    1: "One",
    2: "Two",
    3: "Three",
    5: "Five",
    10: "Ten",
    15: "Fifteen",
    20: "Twenty",
    25: "TwentyFive",
}


def axis_levels(axis):
    if axis == "composition":
        return list(COMP_ORDER)
    return sorted(np.unique(AXES[axis]).tolist())


def level_key(axis, lvl):
    """JSON key for a level."""
    if axis == "composition":
        return str(lvl)
    return f"{int(lvl)}"


def level_macro(axis, lvl):
    if axis == "composition":
        return "Comp" + COMP_MACRO[lvl]
    if axis == "force":
        return "Force" + NUM_WORD[int(lvl)] + "N"
    return "Freq" + NUM_WORD[int(lvl)] + "Hz"


def level_tex(axis, lvl):
    if axis == "composition":
        return COMP_TEX[lvl]
    if axis == "force":
        return rf"\SI{{{int(lvl)}}}{{\newton}}"
    return rf"\SI{{{int(lvl)}}}{{\hertz}}"


# ---------------------------------------------------------------- folds
def loo_folds(idx=None):
    idx = np.arange(N) if idx is None else np.asarray(idx)
    return [(np.delete(idx, i), idx[[i]]) for i in range(len(idx))]


def group_folds(axis, idx=None):
    """Leave-one-level-out folds along `axis`, restricted to rows `idx`."""
    idx = np.arange(N) if idx is None else np.asarray(idx)
    g = AXES[axis][idx]
    out = []
    for v in axis_levels(axis):
        m = g == v
        if m.any() and (~m).any():
            out.append((idx[~m], idx[m], v))
    return out


# ---------------------------------------------------------------- models
def gp(restarts=0):
    """Plain GP of the submitted manuscript (legacy physgp_analysis.gp).

    restarts > 0 is used only for the optimizer-restart robustness check; the
    restart draws use random_state=0 so the check is deterministic.
    """
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        alpha=1e-10,
        random_state=0 if restarts else None,
    )


# legacy six-parameter law
P0_6 = [1.0, 0.1, -0.02, 1.0, 19.0, 8.0]
BOUNDS_6 = ([0, -5, -5, 0, 10, 1], [50, 5, 5, 50, 30, 30])

# identifiable five-parameter law
P0_5 = [1.0 * 1.0, 0.1 * 1.0, -0.02 * 1.0, 19.0, 8.0]
BOUNDS_5 = ([0.0, -250.0, -250.0, 10.0, 1.0], [2500.0, 250.0, 250.0, 30.0, 30.0])
PNAMES = ["a0", "a1", "a2", "f0", "gamma"]


def lorentz(f, f0, g):
    return (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)


def law6(Xt, b0, b1, b2, A, f0, g):
    c, F, f = Xt
    return (b0 + b1 * c + b2 * c**2) * F * A * lorentz(f, f0, g)


def law5(Xt, a0, a1, a2, f0, g):
    c, F, f = Xt
    return (a0 + a1 * c + a2 * c**2) * F * lorentz(f, f0, g)


def law5_off(Xt, a0, a1, a2, f0, g, d):
    return law5(Xt, a0, a1, a2, f0, g) + d


def _xt(idx):
    return (CNT[idx], FRC[idx], FRQ[idx])


# Optimizer stopping tolerance. The legacy code used the curve_fit defaults
# (ftol = xtol = gtol = 1e-8); at that tolerance the six- and five-parameter
# forms stop about 1e-5 V apart in the law mean although they share one minimum.
# The revision fits with 1e-15 so that the reported law is the converged minimum;
# the legacy tolerance stays available through TOL_LEGACY for the equivalence check.
TOL_TIGHT = 1e-15
TOL_LEGACY = None
FIT_TOL = {"tol": TOL_TIGHT}


def _fit(fn, tr, y, p0, bounds, xt=None, tol="default"):
    """Bounded curve_fit on y[tr]/std(y[tr]) (legacy scaling)."""
    tr = np.asarray(tr)
    scale = float(np.std(y[tr])) or 1.0
    xt = _xt(tr) if xt is None else xt
    ok = True
    try:
        tol = FIT_TOL["tol"] if tol == "default" else tol
        kw = {} if tol is None else {"ftol": tol, "xtol": tol, "gtol": tol}
        p, pcov = curve_fit(fn, xt, y[tr] / scale, p0=p0, bounds=bounds, maxfev=20000, **kw)
    except Exception:
        p, pcov, ok = np.array(p0, dtype=float), None, False
    return p, pcov, scale, ok


def fit_law6(tr, y, tol="default"):
    p, pcov, scale, ok = _fit(law6, tr, y, P0_6, BOUNDS_6, tol=tol)
    return (
        (lambda idx: law6(_xt(idx), *p) * scale),
        p,
        {"scale": scale, "ok": ok, "pcov": pcov},
    )


def to_volt(p5, scale):
    """Scaled-fit parameters -> volt-scale parameters (a_i in V/N; f0, gamma in Hz)."""
    q = np.array(p5, dtype=float).copy()
    q[:3] *= scale
    return q


def fit_law5(tr, y, tol="default"):
    """Fit the five-parameter law on rows tr.

    Returns (predict_fn(idx) -> volts, params on the volt scale [a0,a1,a2,f0,gamma],
    info dict with scale, success flag, scaled params, covariance on the volt scale).
    """
    p, pcov, scale, ok = _fit(law5, tr, y, P0_5, BOUNDS_5, tol=tol)
    J = np.diag([scale, scale, scale, 1.0, 1.0])
    cov_v = None if pcov is None else J @ pcov @ J
    info = {"scale": scale, "ok": ok, "p_scaled": p, "cov_volt": cov_v}
    return (lambda idx: law5(_xt(idx), *p) * scale), to_volt(p, scale), info


def fit_law5_off(tr, y):
    p, pcov, scale, ok = _fit(
        law5_off, tr, y, P0_5 + [0.0], (BOUNDS_5[0] + [-10.0], BOUNDS_5[1] + [10.0])
    )
    return (lambda idx: law5_off(_xt(idx), *p) * scale), p, {"scale": scale, "ok": ok}


def mean_linF(tr, y):
    a, b = np.polyfit(FRC[tr], y[tr], 1)
    return lambda idx: a * FRC[idx] + b


# ---------------------------------------------------------------- fold predictors
MODELS = ("gp", "physgp", "physgp6", "physgp_ltol", "physgp6_ltol", "law", "gp_linF", "physgp_offset")
RESTARTS_CHECK = 4  # suffix '_r4': GP hyperparameters re-optimized from 4 extra random starts


def fold_predict(model, tr, te, y):
    """Out-of-fold prediction for one fold.

    model: 'gp' (plain GP), 'physgp' (five-parameter law mean + GP on residuals),
    'physgp6' (legacy six-parameter law mean, for the equivalence check),
    'law' (law alone, no GP), 'gp_linF' (GP about a linear-in-force mean),
    'physgp_offset' (law with additive offset + GP);
    suffix '_ltol' runs the law fit at the legacy curve_fit default tolerance.
    Returns (mu, sd, extra) with extra = fitted law parameters (volt scale) or None.
    """
    tr, te = np.asarray(tr), np.asarray(te)
    extra = None
    tol = "default"
    restarts = 0
    if model.endswith("_r4"):
        model, restarts = model[: -len("_r4")], RESTARTS_CHECK
    if model.endswith("_ltol"):
        model, tol = model[: -len("_ltol")], TOL_LEGACY
    if model == "gp":
        base_tr, base_te = np.zeros(len(tr)), np.zeros(len(te))
    elif model in ("physgp", "law"):
        m, pv, info = fit_law5(tr, y, tol=tol)
        base_tr, base_te = m(tr), m(te)
        extra = {"params": pv.tolist(), "ok": bool(info["ok"])}
        if model == "law":
            return base_te, np.full(len(te), np.nan), extra
    elif model == "physgp6":
        m, p6, info = fit_law6(tr, y, tol=tol)
        base_tr, base_te = m(tr), m(te)
        extra = {"params6": p6.tolist(), "ok": bool(info["ok"])}
    elif model == "gp_linF":
        m = mean_linF(tr, y)
        base_tr, base_te = m(tr), m(te)
    elif model == "physgp_offset":
        m, p, info = fit_law5_off(tr, y)
        base_tr, base_te = m(tr), m(te)
        extra = {"ok": bool(info["ok"])}
    else:
        raise ValueError(model)
    xs = StandardScaler().fit(X[tr])
    g = gp(restarts).fit(xs.transform(X[tr]), y[tr] - base_tr)
    mu, sd = g.predict(xs.transform(X[te]), return_std=True)
    extra = dict(extra or {})
    extra["gp_log_marginal_likelihood"] = float(g.log_marginal_likelihood_value_)
    return base_te + mu, sd, extra


class FoldCache:
    """In-run cache of fold predictions keyed by (model, target, train rows, test rows).

    Jobs are registered first, deduplicated, run in parallel with joblib, then read back.
    The nested selection rule reuses the outer-fold predictions from the per-level analysis.
    """

    def __init__(self):
        self.store = {}
        self.pending = {}

    @staticmethod
    def key(model, target, tr, te):
        return (
            model,
            target,
            tuple(int(i) for i in np.sort(tr)),
            tuple(int(i) for i in te),
        )

    def want(self, model, target, tr, te):
        k = self.key(model, target, tr, te)
        if k not in self.store:
            self.pending[k] = (model, target, np.asarray(tr), np.asarray(te))
        return k

    def run(self, n_jobs=-1):
        from joblib import Parallel, delayed

        jobs = list(self.pending.items())
        self.pending = {}
        if not jobs:
            return 0

        def one(k, model, target, tr, te):
            y = df[target].values.astype(float)
            return k, fold_predict(model, tr, te, y)

        out = Parallel(n_jobs=n_jobs)(delayed(one)(k, *v) for k, v in jobs)
        for k, res in out:
            self.store[k] = res
        return len(jobs)

    def get(self, model, target, tr, te):
        return self.store[self.key(model, target, tr, te)]
