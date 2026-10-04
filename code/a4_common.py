"""Shared data loading and model fits for the A4 descriptive figures.

The GP is the protocol plain GP (ARD Matern-5/2 + white noise, inputs
standardized, normalize_y=True, n_restarts_optimizer=0 as in the legacy
surface and uncertainty scripts). The random forest uses the benchmark
settings of regression/reg_common.py (400 trees, random_state=42).
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from peng_paths import ROOT_STR as ROOT  # repository root (env PENG_ROOT overrides)
REV = ROOT
DATA = os.path.join(REV, "data")
LEGACY_RES = os.path.join(REV, "results", "baseline")
FIG_DIR = os.path.join(REV, "figures")
RES_DIR = os.path.join(REV, "results")
TAB_DIR = os.path.join(RES_DIR, "tables")

FEAT = ["cnt_pct", "is_pristine", "force_N", "freq_Hz"]
TARGETS = ["rms_Voc", "Vpp", "Vmax"]  # the energy proxy is never reported


def load_design():
    return pd.read_parquet(os.path.join(DATA, "targets_design.parquet"))


def load_long():
    return pd.read_parquet(os.path.join(DATA, "long.parquet"))


def fit_gp(df, target="rms_Voc"):
    X = df[FEAT].values.astype(float)
    k = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-3, (1e-6, 1e1))
    xs = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=0, alpha=1e-10
    ).fit(xs.transform(X), df[target].values.astype(float))
    return gp, xs


def fit_rf(df, target):
    X = df[FEAT].values.astype(float)
    return RandomForestRegressor(n_estimators=400, random_state=42).fit(
        X, df[target].values.astype(float)
    )


def slice_grid(cnt=2.0, nF=161, nf=201):
    """Force-frequency plane at a fixed CNT loading of the BaTiO3 series."""
    F, Q = np.meshgrid(np.linspace(1, 3, nF), np.linspace(5, 25, nf), indexing="ij")
    G = np.column_stack([np.full(F.size, cnt), np.zeros(F.size), F.ravel(), Q.ravel()])
    return F, Q, G


def dense_grid():
    """Legacy optimum grid of regression/reg_02_surface.py (BaTiO3 series)."""
    c = np.arange(0, 3.01, 0.25)
    F = np.round(np.arange(1, 3.001, 0.1), 3)
    f = np.round(np.arange(5, 25.001, 0.5), 3)
    Cg, Fg, fg = np.meshgrid(c, F, f, indexing="ij")
    return np.column_stack([Cg.ravel(), np.zeros(Cg.size), Fg.ravel(), fg.ravel()])


def lorentz(f, A, f0, g):
    """Lorentzian of the legacy resonance fit, g is the full width."""
    return A * (g / 2) ** 2 / ((f - f0) ** 2 + (g / 2) ** 2)
