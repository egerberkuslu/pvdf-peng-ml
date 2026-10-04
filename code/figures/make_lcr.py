"""fig_lcr: (a) Cp(f) and (b) D(f) of the five compositions, 1 kHz to 1 MHz, log-x.
Source: sonuclar/lcr/fig_cp_vs_f.csv, fig_D_vs_f.csv. The dielectric constant is not drawn."""
import numpy as np
import pandas as pd
from _style import *
from matplotlib.ticker import FuncFormatter

style()
cp = pd.read_csv(SRC / "lcr/fig_cp_vs_f.csv")
dd = pd.read_csv(SRC / "lcr/fig_D_vs_f.csv")
khz = FuncFormatter(lambda v, _: f"{v/1e3:g}" if v < 1e6 else "1000")

def panel(name, df, col, ylabel, legend):
    fig, ax = plt.subplots(figsize=(W, 1.85))
    for i, (k, lab, c, mk) in enumerate(zip(COMP_KEYS, COMP_LABEL, COMP_COLOR, COMP_MARK)):
        off = 3 + 4 * i
        s = df[df.composition == k].sort_values("frequency_Hz")
        fr = s.frequency_Hz.to_numpy()
        tg = np.logspace(3.05 + 0.04 * i, 5.9, 8)
        idx = sorted({int(np.argmin(np.abs(np.log10(fr) - np.log10(t)))) for t in tg})
        ax.plot(fr, s[col], color=c, lw=1.0, label=lab, marker=mk, markevery=idx,
                ms=3.6, mec="white", mew=0.5)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 1e6)
    ax.xaxis.set_major_formatter(khz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.8, columnspacing=1.0, labelspacing=0.2)
    return fig, ax

fig, ax = panel("lcr_a", cp, "Cp_pF", "$C_p$ (pF)", True)
ax.set_ylim(60, 320)
save(fig, "lcr_a")
fig, ax = panel("lcr_b", dd, "D", "Loss tangent $D$", False)
ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.04]))
ax.yaxis.set_major_formatter(FixedFormatter(["0.005", "0.01", "0.02", "0.04"]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(0.0042, 0.045)
save(fig, "lcr_b")
