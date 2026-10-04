"""House figure style for the revision (protocol section 7).

Import apply_style() before plotting. All text is 9 pt at final size, and
figures are created at their final LaTeX width so no scaling happens in
\\includegraphics.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CM = 1 / 2.54
COL_W = 8.4 * CM  # single column, inches
TEXT_W = 17.4 * CM  # double column, inches

C_BLUE = "#3B6FB6"
C_TEAL = "#2A9D8F"
C_AMBER = "#E08D2F"
C_RED = "#C8553D"
C_GREY = "#6C757D"

ORDER = [
    "PVDF",
    "PVDF+BaTiO3",
    "PVDF+BaTiO3+%1CNT",
    "PVDF+BaTiO3+%2CNT",
    "PVDF+BaTiO3+%3CNT",
]
LABEL = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "PVDF/BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "+1 wt% CNT",
    "PVDF+BaTiO3+%2CNT": "+2 wt% CNT",
    "PVDF+BaTiO3+%3CNT": "+3 wt% CNT",
}
SHORT = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "1% CNT",
    "PVDF+BaTiO3+%2CNT": "2% CNT",
    "PVDF+BaTiO3+%3CNT": "3% CNT",
}
COLOR = {
    "PVDF": C_GREY,
    "PVDF+BaTiO3": C_BLUE,
    "PVDF+BaTiO3+%1CNT": C_TEAL,
    "PVDF+BaTiO3+%2CNT": C_RED,
    "PVDF+BaTiO3+%3CNT": C_AMBER,
}
MARKER = {
    "PVDF": "o",
    "PVDF+BaTiO3": "s",
    "PVDF+BaTiO3+%1CNT": "^",
    "PVDF+BaTiO3+%2CNT": "D",
    "PVDF+BaTiO3+%3CNT": "v",
}
TARGET_LABEL = {"rms_Voc": r"$V_{\mathrm{rms}}$", "Vpp": r"$V_{\mathrm{pp}}$",
                "Vmax": r"$|V|_{\mathrm{max}}$"}

FS = 9


def apply_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": FS,
            "axes.labelsize": FS,
            "axes.titlesize": FS,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
            "legend.title_fontsize": FS,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.3,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#444444",
            "grid.linewidth": 0.4,
            "grid.color": "#DDDDDD",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "standard",
            "figure.dpi": 150,
        }
    )


def save(fig, path_noext):
    """Save vector PDF plus a 150 dpi PNG preview at the exact figure size."""
    fig.savefig(f"{path_noext}.pdf")
    fig.savefig(f"{path_noext}.png", dpi=150)
    plt.close(fig)
