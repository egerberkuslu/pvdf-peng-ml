#!/usr/bin/env python3
"""Modern conceptual graphical abstract for the electrospun-PENG ML manuscript.
Four-stage horizontal flow: fabrication -> measurement -> ML surrogate -> design.
Pure matplotlib (no external API). Okabe-Ito palette. Saves PDF + PNG."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Polygon,
    Rectangle,
    Ellipse,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none",
    }
)

# Okabe-Ito
OI = dict(
    orange="#E69F00",
    sky="#56B4E9",
    green="#009E73",
    yellow="#F0E442",
    blue="#0072B2",
    verm="#D55E00",
    purple="#CC79A7",
    grey="#5A5A5A",
    ink="#1A1A1A",
)

fig = plt.figure(figsize=(12.4, 3.5), dpi=300)
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.axis("off")

# ---- card geometry ----
cards = [
    ("Electrospun material", OI["blue"]),
    ("Voltage measurement", OI["orange"]),
    ("ML surrogate", OI["green"]),
    ("Design & reliability", OI["verm"]),
]
n = len(cards)
cw, gap = 0.212, 0.045
x0 = (1 - (n * cw + (n - 1) * gap)) / 2
cy, ch = 0.10, 0.74
xs = [x0 + i * (cw + gap) for i in range(n)]


def card(x, title, color):
    # body
    bg.add_patch(
        FancyBboxPatch(
            (x, cy),
            cw,
            ch,
            mutation_scale=0.02,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#CDD3DA",
            facecolor="white",
            zorder=2,
        )
    )
    # header strip
    bg.add_patch(
        FancyBboxPatch(
            (x, cy + ch - 0.11),
            cw,
            0.11,
            mutation_scale=0.02,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
    )
    bg.text(
        x + cw / 2,
        cy + ch - 0.055,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        zorder=4,
    )


for x, (t, c) in zip(xs, cards):
    card(x, t, c)

# arrows between cards
for i in range(n - 1):
    xa = xs[i] + cw + 0.004
    xb = xs[i + 1] - 0.004
    bg.add_patch(
        FancyArrowPatch(
            (xa, cy + ch / 2 - 0.06),
            (xb, cy + ch / 2 - 0.06),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=OI["grey"],
            zorder=5,
        )
    )


# helper: inset axes inside a card's plot region
def inset(i, rel):  # rel = (rx, ry, rw, rh) within card body plot area
    x = xs[i]
    px, py, pw, ph = x + 0.012, cy + 0.055, cw - 0.024, ch - 0.185
    return fig.add_axes([px + rel[0] * pw, py + rel[1] * ph, rel[2] * pw, rel[3] * ph])


# =====================================================================
# Card 1: electrospinning -> nanofiber mat -> wearable
# =====================================================================
ax = inset(0, (0, 0, 1, 1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
# syringe barrel + needle
ax.add_patch(
    Rectangle(
        (0.06, 0.72),
        0.30,
        0.16,
        facecolor=OI["sky"],
        edgecolor=OI["ink"],
        linewidth=1.0,
    )
)
ax.add_patch(
    Polygon(
        [(0.36, 0.74), (0.36, 0.86), (0.47, 0.80)],
        closed=True,
        facecolor="#B0BEC5",
        edgecolor=OI["ink"],
        linewidth=0.8,
    )
)
ax.text(
    0.20,
    0.945,
    "polymer + BaTiO$_3$ + CNT",
    ha="center",
    va="center",
    fontsize=6.4,
    color=OI["ink"],
)
# HV
ax.text(
    0.55,
    0.90,
    "+HV",
    ha="center",
    va="center",
    fontsize=7.5,
    fontweight="bold",
    color=OI["verm"],
)
# Taylor-cone jet (whipping)
t = np.linspace(0, 1, 200)
jx = 0.47 + 0.11 * t + 0.05 * np.sin(20 * t) * t
jy = 0.80 - 0.42 * t
ax.plot(jx, jy, color=OI["blue"], linewidth=1.3, zorder=3)
# collector drum
ax.add_patch(
    Ellipse(
        (0.62, 0.30),
        0.16,
        0.30,
        facecolor="#ECEFF1",
        edgecolor=OI["ink"],
        linewidth=1.0,
    )
)
ax.add_patch(
    Ellipse(
        (0.62, 0.30), 0.05, 0.30, facecolor="none", edgecolor="#90A4AE", linewidth=0.8
    )
)
# nanofiber mat texture
rng = np.random.default_rng(3)
for _ in range(26):
    xa, ya = rng.uniform(0.05, 0.55), rng.uniform(0.06, 0.26)
    ang = rng.uniform(0, np.pi)
    L = rng.uniform(0.05, 0.11)
    ax.plot(
        [xa, xa + L * np.cos(ang)],
        [ya, ya + L * 0.5 * np.sin(ang)],
        color=OI["blue"],
        alpha=0.55,
        linewidth=0.8,
    )
ax.text(
    0.30,
    0.005,
    "nanofiber mat",
    ha="center",
    va="bottom",
    fontsize=6.4,
    color=OI["ink"],
)
# wearable glyph (shirt)
sx, sy = 0.80, 0.14
shirt = Polygon(
    [
        (sx - 0.10, sy),
        (sx - 0.10, sy + 0.16),
        (sx - 0.14, sy + 0.20),
        (sx - 0.10, sy + 0.26),
        (sx - 0.05, sy + 0.22),
        (sx + 0.05, sy + 0.22),
        (sx + 0.10, sy + 0.26),
        (sx + 0.14, sy + 0.20),
        (sx + 0.10, sy + 0.16),
        (sx + 0.10, sy),
    ],
    closed=True,
    facecolor=OI["green"],
    edgecolor=OI["ink"],
    linewidth=0.8,
    alpha=0.85,
)
ax.add_patch(shirt)
ax.text(
    sx, sy - 0.06, "smart textile", ha="center", va="top", fontsize=6.4, color=OI["ink"]
)

# =====================================================================
# Card 2: Voc waveforms + factorial note
# =====================================================================
ax = inset(1, (0, 0.30, 1, 0.70))
tt = np.linspace(0, 1, 600)


def spikes(amp, f=20):
    base = np.zeros_like(tt)
    for k in np.arange(0.05, 1.0, 1 / f):
        base += amp * np.exp(-(((tt - k) / 0.006) ** 2)) - 0.35 * amp * np.exp(
            -(((tt - k - 0.012) / 0.008) ** 2)
        )
    return base


ax.plot(tt, spikes(1.0), color=OI["orange"], linewidth=1.0, label="2% CNT")
ax.plot(tt, spikes(0.45), color=OI["sky"], linewidth=1.0, label="PVDF")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor("#B0B0B0")
ax.set_ylabel("$V_{oc}$", fontsize=7.5, color=OI["ink"], labelpad=1)
ax.legend(
    fontsize=5.6,
    frameon=False,
    loc="upper right",
    handlelength=1.0,
    borderaxespad=0.1,
    labelspacing=0.2,
)
ax2 = inset(1, (0, 0, 1, 0.24))
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.text(
    0.5,
    0.55,
    r"75 conditions  =  5 comp. $\times$ 3 force $\times$ 5 freq.",
    ha="center",
    va="center",
    fontsize=6.6,
    color=OI["ink"],
)

# =====================================================================
# Card 3: inputs -> model box -> calibration
# =====================================================================
ax = inset(2, (0, 0, 1, 1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
for j, lab in enumerate(["CNT %", "force $F$", "freq. $f$"]):
    yy = 0.82 - j * 0.18
    ax.add_patch(
        FancyBboxPatch(
            (0.02, yy - 0.06),
            0.30,
            0.12,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=0.8,
            edgecolor=OI["green"],
            facecolor="#E6F5EF",
        )
    )
    ax.text(0.17, yy, lab, ha="center", va="center", fontsize=6.6, color=OI["ink"])
    ax.add_patch(
        FancyArrowPatch(
            (0.33, yy),
            (0.44, 0.55),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=OI["grey"],
        )
    )
ax.add_patch(
    FancyBboxPatch(
        (0.44, 0.40),
        0.30,
        0.34,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=OI["green"],
        facecolor="white",
    )
)
ax.text(
    0.59,
    0.57,
    "Gaussian\nprocess +\ntree ensembles",
    ha="center",
    va="center",
    fontsize=6.4,
    color=OI["ink"],
    linespacing=1.15,
)
ax.text(
    0.59,
    0.30,
    "LOO-CV",
    ha="center",
    va="center",
    fontsize=6.4,
    fontweight="bold",
    color=OI["green"],
)
# calibration mini-scatter
cal = fig.add_axes(
    [
        xs[2] + 0.012 + 0.76 * (cw - 0.024),
        cy + 0.075,
        0.24 * (cw - 0.024),
        0.34 * (ch - 0.185),
    ]
)
rng = np.random.default_rng(7)
xv = rng.uniform(0, 1, 22)
yv = xv + rng.normal(0, 0.07, 22)
cal.plot([0, 1], [0, 1], color="#B0B0B0", linewidth=0.8, ls="--")
cal.scatter(xv, yv, s=6, color=OI["green"], edgecolor="white", linewidth=0.2)
cal.set_xticks([])
cal.set_yticks([])
cal.set_xlabel("meas.", fontsize=5.6, labelpad=1)
cal.set_ylabel("pred.", fontsize=5.6, labelpad=1)
for s in cal.spines.values():
    s.set_edgecolor("#B0B0B0")
cal.text(0.05, 0.9, "$R^2$=0.91", fontsize=5.8, color=OI["ink"], va="top")

# =====================================================================
# Card 4: response surface + optimum + reliability
# =====================================================================
ax = inset(3, (0, 0.30, 1, 0.70))
gx = np.linspace(5, 25, 80)
gy = np.linspace(1, 3, 80)
GX, GY = np.meshgrid(gx, gy)
Z = (
    (GY / 3.0) * 1.0 / (1 + ((GX - 19) / 4.0) ** 2)
)  # Lorentzian in freq, rising in force
cs = ax.contourf(GX, GY, Z, levels=12, cmap="viridis")
ax.plot(
    19,
    3.0,
    marker="*",
    markersize=13,
    color="white",
    markeredgecolor=OI["ink"],
    markeredgewidth=0.7,
    zorder=5,
)
ax.set_xlabel("frequency (Hz)", fontsize=6.2, labelpad=1)
ax.set_ylabel("force (N)", fontsize=6.2, labelpad=1)
ax.tick_params(labelsize=5.4, length=2)
ax.set_xticks([5, 15, 25])
ax.set_yticks([1, 2, 3])
axn = inset(3, (0, 0, 1, 0.11))
axn.axis("off")
axn.set_xlim(0, 1)
axn.set_ylim(0, 1)
axn.text(
    0.5,
    0.35,
    r"optimum $\approx$ 2% CNT, 3 N, 20 Hz  $\cdot$  reliability map",
    ha="center",
    va="center",
    fontsize=6.0,
    color=OI["ink"],
)

fig.savefig("figures/fig_graphical_abstract.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(
    "figures/fig_graphical_abstract.png", bbox_inches="tight", pad_inches=0.04, dpi=300
)
print("saved figures/fig_graphical_abstract.{pdf,png}")
