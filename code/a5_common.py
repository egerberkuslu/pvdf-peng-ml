"""Shared helpers for task A5 (structured kernels against plain GP and PhysGP).

Reference models are copied from code/a2_common.py, which in turn
copies code/baseline/physgp_analysis.py, so the numbers agree with A2
without importing its files:
  plain GP  ConstantKernel x ARD Matern-5/2 over (c, pristine, F, f) + White,
            n_restarts_optimizer 0 (legacy), inputs standardized in the fold.
  PhysGP    five-parameter identifiable law (A2 reparameterization of the
            legacy six-parameter law) fitted inside every training fold on
            y/std(y_train) at tolerance 1e-15, GP on the residual.

New models (all GaussianProcessRegressor, normalize_y=True, inputs standardized
in the fold, n_restarts_optimizer 4, random_state = seed):
  m1  product kernel   C * Matern52(c, pristine) * Matern52(F) * Matern52(f) + White
  m2  additive kernel  C1*Matern52(c, pristine) + C2*Matern52(F) + C3*Matern52(f) + White
  m3  m1 about a linear-in-force prior mean beta*F, beta by least squares on the
      training rows (no intercept), GP on the residual
  m4  m3 on log10(y), predictions back-transformed with 10**mu (posterior median)

scikit-learn 1.7.2 kernels have no active-dimension argument, so SubMatern
subclasses Matern and slices the input columns before evaluating the kernel.
Hyperparameters, bounds and analytic gradients are inherited unchanged.
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

TARGETS = ["rms_Voc", "Vpp", "Vmax"]
AXES = {"composition": COMP, "force": FRC, "frequency": FRQ}
COMP_ORDER = [
    "PVDF",
    "PVDF+BaTiO3",
    "PVDF+BaTiO3+%1CNT",
    "PVDF+BaTiO3+%2CNT",
    "PVDF+BaTiO3+%3CNT",
]

# column groups of X for the structured kernels
DIMS_COMP = (0, 1)
DIMS_F = (2,)
DIMS_FREQ = (3,)

LS_BOUNDS = (1e-2, 1e3)
C_BOUNDS = (1e-3, 1e3)
W_INIT, W_BOUNDS = 1e-3, (1e-6, 1e1)
N_RESTARTS_NEW = 4


def axis_levels(axis):
    if axis == "composition":
        return list(COMP_ORDER)
    return sorted(np.unique(AXES[axis]).tolist())


def level_key(axis, lvl):
    return str(lvl) if axis == "composition" else f"{int(lvl)}"


def loo_folds():
    idx = np.arange(N)
    return [(np.delete(idx, i), idx[[i]]) for i in range(N)]


def group_folds(axis):
    idx = np.arange(N)
    g = AXES[axis]
    return [(idx[g != v], idx[g == v], v) for v in axis_levels(axis)]


# ---------------------------------------------------------------- kernels
class SubMatern(Matern):
    """Matern kernel evaluated on a subset of input columns (dims)."""

    def __init__(
        self, dims=(0,), length_scale=1.0, length_scale_bounds=LS_BOUNDS, nu=2.5
    ):
        super().__init__(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu
        )
        self.dims = dims

    def _sl(self, Z):
        return None if Z is None else np.atleast_2d(np.asarray(Z))[:, list(self.dims)]

    def __call__(self, X, Y=None, eval_gradient=False):
        return super().__call__(self._sl(X), self._sl(Y), eval_gradient=eval_gradient)

    def diag(self, X):
        return np.ones(np.asarray(X).shape[0])


def _sub(dims):
    ls = [1.0] * len(dims) if len(dims) > 1 else 1.0
    return SubMatern(dims=dims, length_scale=ls, length_scale_bounds=LS_BOUNDS, nu=2.5)


def kernel_product():
    return C(1.0, C_BOUNDS) * _sub(DIMS_COMP) * _sub(DIMS_F) * _sub(
        DIMS_FREQ
    ) + WhiteKernel(W_INIT, W_BOUNDS)


def kernel_additive():
    return (
        C(1.0, C_BOUNDS) * _sub(DIMS_COMP)
        + C(1.0, C_BOUNDS) * _sub(DIMS_F)
        + C(1.0, C_BOUNDS) * _sub(DIMS_FREQ)
        + WhiteKernel(W_INIT, W_BOUNDS)
    )


def gp_legacy():
    """Plain GP of the first-stage analysis (legacy physgp_analysis.gp)."""
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    )


def gp_legacy_r4(seed):
    """Plain GP with the restart budget of the new models (sensitivity only)."""
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=k,
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS_NEW,
        alpha=1e-10,
        random_state=seed,
    )


def gp_struct(kind, seed):
    k = kernel_product() if kind == "product" else kernel_additive()
    return GaussianProcessRegressor(
        kernel=k,
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS_NEW,
        alpha=1e-10,
        random_state=seed,
    )


def walk_kernel(k, out=None):
    """Collect (SubMatern | ConstantKernel | WhiteKernel) leaves of a fitted kernel."""
    out = [] if out is None else out
    if hasattr(k, "k1") and hasattr(k, "k2"):
        walk_kernel(k.k1, out)
        walk_kernel(k.k2, out)
    else:
        out.append(k)
    return out


# ---------------------------------------------------------------- PhysGP law (A2 form)
P0_5 = [1.0, 0.1, -0.02, 19.0, 8.0]
BOUNDS_5 = ([0.0, -250.0, -250.0, 10.0, 1.0], [2500.0, 250.0, 250.0, 30.0, 30.0])
TOL = 1e-15


def law5(Xt, a0, a1, a2, f0, g):
    c, F, f = Xt
    return (
        (a0 + a1 * c + a2 * c**2) * F * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)
    )


def _xt(idx):
    return (CNT[idx], FRC[idx], FRQ[idx])


def fit_law5(tr, y):
    tr = np.asarray(tr)
    scale = float(np.std(y[tr])) or 1.0
    try:
        p, _ = curve_fit(
            law5,
            _xt(tr),
            y[tr] / scale,
            p0=P0_5,
            bounds=BOUNDS_5,
            maxfev=20000,
            ftol=TOL,
            xtol=TOL,
            gtol=TOL,
        )
        ok = True
    except Exception:
        p, ok = np.array(P0_5, dtype=float), False
    return (lambda idx: law5(_xt(idx), *p) * scale), ok


# ---------------------------------------------------------------- fold predictors
MODELS_MAIN = ["gp", "physgp", "m1", "m2", "m3", "m4"]
MODELS_SENS = ["gp_r4", "physgp_r4"]


def fold_predict(model, tr, te, y, seed=0):
    """Out-of-fold (mu, sd) on the original voltage scale for one fold.

    For m4, sd is the posterior sd on the log10 scale (intervals are built on
    that scale and back-transformed by the caller).
    """
    tr, te = np.asarray(tr), np.asarray(te)
    xs = StandardScaler().fit(X[tr])
    Xtr, Xte = xs.transform(X[tr]), xs.transform(X[te])
    if model in ("gp", "gp_r4"):
        g = gp_legacy() if model == "gp" else gp_legacy_r4(seed)
        g.fit(Xtr, y[tr])
        return g.predict(Xte, return_std=True)
    if model in ("physgp", "physgp_r4"):
        m, _ = fit_law5(tr, y)
        g = gp_legacy() if model == "physgp" else gp_legacy_r4(seed)
        g.fit(Xtr, y[tr] - m(tr))
        mu, sd = g.predict(Xte, return_std=True)
        return m(te) + mu, sd
    if model in ("m1", "m2"):
        g = gp_struct("product" if model == "m1" else "additive", seed).fit(Xtr, y[tr])
        return g.predict(Xte, return_std=True)
    if model in ("m3", "m4"):
        z = np.log10(y) if model == "m4" else y
        beta = float(np.sum(FRC[tr] * z[tr]) / np.sum(FRC[tr] ** 2))
        g = gp_struct("product", seed).fit(Xtr, z[tr] - beta * FRC[tr])
        mu, sd = g.predict(Xte, return_std=True)
        mu = beta * FRC[te] + mu
        if model == "m4":
            return 10.0**mu, sd
        return mu, sd
    raise ValueError(model)


def fit_full(model, y, seed=0):
    """Fit a structured model on all 75 rows; return per-factor hyperparameters in original units."""
    xs = StandardScaler().fit(X)
    Z = xs.transform(X)
    target = y
    beta = None
    if model in ("m3", "m4"):
        target = np.log10(y) if model == "m4" else y
        beta = float(np.sum(FRC * target) / np.sum(FRC**2))
        target = target - beta * FRC
    kind = "additive" if model == "m2" else "product"
    g = gp_struct(kind, seed).fit(Z, target)
    leaves = walk_kernel(g.kernel_)
    ls = {}
    consts, white = [], None
    for lf in leaves:
        if isinstance(lf, SubMatern):
            vals = np.atleast_1d(lf.length_scale).astype(float)
            for d, v in zip(lf.dims, vals):
                ls[FEAT[d]] = {
                    "standardized": float(v),
                    "original_units": float(v * xs.scale_[d]),
                    "at_bound": bool(
                        np.isclose(v, LS_BOUNDS[0], rtol=1e-3)
                        or np.isclose(v, LS_BOUNDS[1], rtol=1e-3)
                    ),
                }
        elif isinstance(lf, WhiteKernel):
            white = float(lf.noise_level)
        elif isinstance(lf, C):
            consts.append(float(lf.constant_value))
    return {
        "length_scales": ls,
        "constants": consts,
        "white_noise_level_normalized": white,
        "beta_force": beta,
        "log_marginal_likelihood": float(g.log_marginal_likelihood_value_),
        "kernel": str(g.kernel_),
        "input_scale_sd": {FEAT[d]: float(xs.scale_[d]) for d in range(4)},
    }
