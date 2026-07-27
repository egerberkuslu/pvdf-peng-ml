#!/usr/bin/env python3
"""Figures for the PhysGP and calibrated-interval experiments.

(1) fig12_physgp_axes: leave-one-level-out R^2 per held-out axis, plain GP
    against PhysGP, all four targets — shows the force-axis rescue and the
    frequency-axis failure side by side.
(2) fig13_interval_tradeoff: empirical coverage against mean interval width
    for the raw GP intervals, the jackknife+ and the sigma-scaled jackknife+.
Reads physgp_results.json and calibrated_conformal_results.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(name):
    p = (
        name
        if os.path.exists(name)
        else os.path.join(os.path.dirname(__file__), "..", "results", name)
    )
    return json.load(open(p))


phys = load("physgp_results.json")
cal = load("calibrated_conformal_results.json")

TARGETS = [
    ("rms_Voc", r"$V_{rms}$"),
    ("Vpp", r"$V_{pp}$"),
    ("Vmax", r"$|V|_{max}$"),
    ("energy", "Energy"),
]
AXES = [
    ("composition", "held-out composition"),
    ("force", "held-out force"),
    ("frequency", "held-out frequency"),
]

# ---------- Figure 1: axis-wise generalization ----------
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True, sharey=True)
x = np.arange(len(TARGETS))
w = 0.36
for ax, (akey, atitle) in zip(axs, AXES):
    g = [phys[t][f"gp_leave_one_{akey}_out_R2"] for t, _ in TARGETS]
    p = [phys[t][f"physgp_leave_one_{akey}_out_R2"] for t, _ in TARGETS]
    ax.bar(x - w / 2, g, w, label="GP", color="#4878A8")
    ax.bar(x + w / 2, p, w, label="PhysGP", color="#D1495B")
    ax.axhline(0, c="0.2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in TARGETS], fontsize=8.5)
    ax.set_title(atitle, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim(-3.1, 1.05)
    for xi, val in zip(x, p):
        if val < -3.1:
            ax.text(xi + w / 2, -3.02, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#D1495B", rotation=90)
    for xi, val in zip(x, g):
        if val < -3.1:
            ax.text(xi - w / 2, -3.02, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#4878A8", rotation=90)
axs[0].set_ylabel(r"group-wise CV $R^2$", fontsize=9)
axs[0].legend(fontsize=8.5, frameon=False, loc="lower left")
for ext in ("pdf", "png"):
    fig.savefig(f"fig12_physgp_axes.{ext}", dpi=200)
plt.close(fig)

# ---------- Figure 2: coverage-width trade-off ----------
METHODS = [
    ("raw_gp", "raw GP $\\pm z\\sigma$", "o", "#4878A8"),
    ("jackknife_plus", "jackknife+", "s", "#D1495B"),
    ("scaled_jackknife_plus", "$\\sigma$-scaled jackknife+", "^", "#8A7090"),
]
fig, axs = plt.subplots(1, 4, figsize=(10.6, 2.9), constrained_layout=True)
for ax, (t, lab) in zip(axs, TARGETS):
    for m, mlab, marker, color in METHODS:
        for a, nominal in (("90", 0.90), ("95", 0.95)):
            s = cal[t][m][a]
            ax.scatter(
                s["mean_width"],
                s["coverage"],
                marker=marker,
                s=48,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label=mlab if (t == "rms_Voc" and a == "90") else None,
            )
    for nominal, ls in ((0.90, ":"), (0.95, "--")):
        ax.axhline(nominal, c="0.55", lw=0.8, ls=ls, zorder=1)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("mean width", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0.80, 1.01)
axs[0].set_ylabel("empirical coverage", fontsize=9)
fig.legend(
    loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.10)
)
for ext in ("pdf", "png"):
    fig.savefig(f"fig13_interval_tradeoff.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

print("wrote fig12_physgp_axes and fig13_interval_tradeoff")
