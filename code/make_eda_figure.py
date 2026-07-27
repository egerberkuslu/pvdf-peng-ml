#!/usr/bin/env python3
"""Regenerate the six-panel exploratory-data-analysis figure (manuscript Fig. 2).

Panels: (a) output-target distributions, (b) frequency response at 3 N,
(c) force effect, (d) composition effect on Vpp at 3 N and 20 Hz,
(e) composition-by-frequency RMS map at 3 N, (f) Pearson correlations.
No in-figure suptitle; larger annotation fonts in panel (f) for print.
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_p = (
    "targets_design.parquet"
    if os.path.exists("targets_design.parquet")
    else os.path.join(os.path.dirname(__file__), "..", "data", "targets_design.parquet")
)
df = pd.read_parquet(_p)

ORDER = [
    "PVDF",
    "PVDF+BaTiO3",
    "PVDF+BaTiO3+%1CNT",
    "PVDF+BaTiO3+%2CNT",
    "PVDF+BaTiO3+%3CNT",
]
LABEL = {
    "PVDF": "PVDF",
    "PVDF+BaTiO3": "+BaTiO$_3$",
    "PVDF+BaTiO3+%1CNT": "+1% CNT",
    "PVDF+BaTiO3+%2CNT": "+2% CNT",
    "PVDF+BaTiO3+%3CNT": "+3% CNT",
}
COLOR = {
    "PVDF": "0.82",
    "PVDF+BaTiO3": "0.65",
    "PVDF+BaTiO3+%1CNT": "0.48",
    "PVDF+BaTiO3+%2CNT": "crimson",
    "PVDF+BaTiO3+%3CNT": "0.32",
}

fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6), constrained_layout=True)

# (a) target distributions, log scale
ax = axes[0, 0]
targets = [
    ("rms_Voc", r"$V_{rms}$"),
    ("Vpp", r"$V_{pp}$"),
    ("Vmax", r"$|V|_{max}$"),
    ("energy", "Energy"),
]
rng = np.random.default_rng(0)
for i, (col, lab) in enumerate(targets):
    v = df[col].values
    ax.boxplot(
        v,
        positions=[i],
        widths=0.55,
        showfliers=False,
        boxprops=dict(color="royalblue"),
        medianprops=dict(color="royalblue"),
        whiskerprops=dict(color="royalblue"),
        capprops=dict(color="royalblue"),
    )
    ax.scatter(
        np.full_like(v, i) + rng.uniform(-0.16, 0.16, len(v)),
        v,
        s=10,
        color="cornflowerblue",
        alpha=0.55,
        zorder=3,
        lw=0,
    )
ax.set_yscale("log")
ax.set_xticks(range(4))
ax.set_xticklabels([lab for _, lab in targets], fontsize=9)
ax.set_ylabel("value (log scale)", fontsize=9)
ax.set_title(
    "a  Output-target distributions (N = 75)",
    loc="left",
    fontsize=10,
    fontweight="bold",
)

# (b) frequency response at 3 N
ax = axes[0, 1]
sub = df[df.force_N == 3]
for comp in ORDER:
    g = sub[sub.composition == comp].sort_values("freq_Hz")
    ax.plot(
        g.freq_Hz,
        g.rms_Voc,
        marker="o",
        ms=4,
        color=COLOR[comp],
        lw=2 if comp.endswith("%2CNT") else 1.4,
        label=LABEL[comp],
    )
ax.axvline(20, ls=":", c="0.4", lw=1)
ax.set_xlabel("Frequency (Hz)", fontsize=9)
ax.set_ylabel(r"$V_{rms}$ (V)", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("b  Frequency response at 3 N", loc="left", fontsize=10, fontweight="bold")

# (c) force effect
ax = axes[0, 2]
for comp in ORDER:
    g = df[df.composition == comp].groupby("force_N").rms_Voc.mean()
    ax.plot(
        g.index,
        g.values,
        marker="o",
        ms=4,
        color=COLOR[comp],
        lw=2 if comp.endswith("%2CNT") else 1.4,
    )
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Force (N)", fontsize=9)
ax.set_ylabel(r"mean $V_{rms}$ (V)", fontsize=9)
ax.set_title("c  Force effect", loc="left", fontsize=10, fontweight="bold")

# (d) composition effect on Vpp at 3 N, 20 Hz
ax = axes[1, 0]
cell = df[(df.force_N == 3) & (df.freq_Hz == 20)].set_index("composition").loc[ORDER]
bars = ax.bar(range(5), cell.Vpp.values, color=[COLOR[c] for c in ORDER], width=0.65)
imax = int(np.argmax(cell.Vpp.values))
ax.text(
    imax,
    cell.Vpp.values[imax] + 0.12,
    f"{cell.Vpp.values[imax]:.1f} V",
    ha="center",
    fontsize=9,
    color="crimson",
    fontweight="bold",
)
ax.set_xticks(range(5))
ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel(r"$V_{pp}$ (V)", fontsize=9)
ax.set_title(
    "d  Composition effect at 3 N, 20 Hz", loc="left", fontsize=10, fontweight="bold"
)

# (e) composition-by-frequency RMS map at 3 N
ax = axes[1, 1]
piv = sub.pivot_table(index="composition", columns="freq_Hz", values="rms_Voc").loc[
    ORDER
]
im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels([int(c) for c in piv.columns], fontsize=8.5)
ax.set_yticks(range(5))
ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=8.5)
ax.set_xlabel("Frequency (Hz)", fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label(r"$V_{rms}$ (V)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.set_title(r"e  $V_{rms}$ map at 3 N", loc="left", fontsize=10, fontweight="bold")

# (f) Pearson correlations, larger fonts
ax = axes[1, 2]
cols = [
    ("cnt_pct", "CNT"),
    ("force_N", "Force"),
    ("freq_Hz", "Freq"),
    ("rms_Voc", r"$V_{rms}$"),
    ("Vpp", r"$V_{pp}$"),
    ("Vmax", r"$|V|_{max}$"),
    ("energy", "E"),
]
C = df[[c for c, _ in cols]].corr(method="pearson").values
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(
            j,
            i,
            f"{C[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(C[i, j]) > 0.6 else "black",
        )
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([l for _, l in cols], rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels([l for _, l in cols], fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label("r", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.set_title("f  Pearson correlation", loc="left", fontsize=10, fontweight="bold")

print(
    "force-RMS r =",
    round(C[1, 3], 4),
    "| peak Vpp =",
    round(cell.Vpp.max(), 2),
    "V",
    "| map max =",
    round(piv.values.max(), 3),
    "V",
)

for ext in ("pdf", "png"):
    fig.savefig(f"fig00_eda.{ext}", dpi=200)
print("wrote fig00_eda.pdf/.png")
