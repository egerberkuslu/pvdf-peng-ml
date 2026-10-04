"""Shared style for the result figures (vector PDF, single-column width, seaborn deep palette)."""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ for peng_paths
from peng_paths import RESULTS, FIGURES

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SRC = RESULTS  # derived results written by the analysis scripts
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
W = 3.3  # inches, single-column width of the target journal
DEEP = sns.color_palette("deep")

COMP_KEYS = ["PVDF", "PVDF/BaTiO3", "PVDF/BaTiO3/1wt%MWCNT", "PVDF/BaTiO3/2wt%MWCNT", "PVDF/BaTiO3/3wt%MWCNT"]
COMP_LABEL = ["PVDF", "+BaTiO$_3$", "+1 wt% MWCNT", "+2 wt% MWCNT", "+3 wt% MWCNT"]
COMP_COLOR = [DEEP[7], DEEP[2], DEEP[0], DEEP[1], DEEP[3]]
COMP_MARK = ["o", "s", "^", "D", "v"]

GP_C = DEEP[0]
LAW_C = DEEP[3]


def style():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": ".25", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "savefig.pad_inches": 0.02, "figure.dpi": 150,
        "legend.frameon": False, "mathtext.default": "regular",
    })


def save(fig, name, extra=None):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", bbox_extra_artists=extra)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
