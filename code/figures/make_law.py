"""fig_law: identifiability of the LawGP law parameters for the rms target.
(a) relative deviation from the full-grid estimate: bootstrap 95% interval and interquartile range, 75 LOO refits (dots),
    13 group-fold refits (grey range, clipped at the axis edge).
(b) 500 residual-bootstrap draws of the linear and quadratic composition coefficients (a1, a2).
Source: sonuclar/a2_physgp.json identifiability.rms_Voc (full_grid, bootstrap, loo_folds, group_folds)."""
import json
import numpy as np
from _style import *

style()
d = json.load(open(SRC / "a2_physgp.json"))["identifiability"]["rms_Voc"]
P = ["a0", "a1", "a2", "f0", "gamma"]
LAB = ["$a_0$", "$a_1$", "$a_2$", "$f_0$", "$\\gamma$"]
est = np.array([d["full_grid"]["estimate"][p] for p in P])
den = np.abs(est)
rel = lambda v, i: (np.asarray(v) - est[i]) / den[i] * 100.0
pct = d["bootstrap"]["percentiles_2.5_25_50_75_97.5"]
loo = d["loo_folds"]["params"]
grp = d["group_folds"]["spread"]
XL = 125.0

fig, ax = plt.subplots(figsize=(W, 2.0))
rng = np.random.default_rng(1)
for i, p in enumerate(P):
    lo, hi = rel(grp[p]["min"], i), rel(grp[p]["max"], i)
    ax.plot([max(lo, -XL), min(hi, XL)], [i, i], color=".82", lw=5.0, solid_capstyle="butt", zorder=1)
    if hi > XL:
        ax.scatter([XL], [i], marker=">", s=14, color=".6", zorder=2, clip_on=False)
    if lo < -XL:
        ax.scatter([-XL], [i], marker="<", s=14, color=".6", zorder=2, clip_on=False)
    v = rel(loo[p], i)
    ax.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=3.5, color=DEEP[0], alpha=0.45, edgecolor="none", zorder=3)
    q = rel(pct[p], i)
    ax.plot([q[0], q[4]], [i, i], color=DEEP[3], lw=1.0, zorder=4)
    ax.plot([q[1], q[3]], [i, i], color=DEEP[3], lw=3.2, solid_capstyle="butt", zorder=4)
    ax.plot([q[2]], [i], marker="|", color="white", ms=6, mew=1.0, zorder=5)
ax.axvline(0, color=".3", lw=0.7, zorder=0)
ax.set_yticks(range(5))
ax.set_yticklabels(LAB)
ax.invert_yaxis()
ax.set_xlim(-XL - 5, XL + 5)
ax.set_xlabel("Deviation from the full-grid estimate (% of $|$estimate$|$)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(True, axis="x", lw=0.4, alpha=0.4)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h = [Line2D([], [], color=DEEP[3], lw=3.2, label="Bootstrap IQR and 95% interval"),
     Line2D([], [], marker="o", ls="", color=DEEP[0], ms=2.6, alpha=0.7, label="LOO refit"),
     Patch(color=".82", label="Group-fold range")]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
save(fig, "law_a")

# (b) bootstrap scatter of (a1, a2)
draws = np.array(d["bootstrap"]["resamples_volt_scale"])
r = np.corrcoef(draws[:, 1], draws[:, 2])[0, 1]
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.scatter(draws[:, 1], draws[:, 2], s=6, color=DEEP[0], alpha=0.35, edgecolor="none")
ax.scatter([est[1]], [est[2]], marker="*", s=70, color=DEEP[3], edgecolor="white", lw=0.5, zorder=4)
ax.axhline(0, color=".3", lw=0.7)
ax.set_xlabel("$a_1$ (V N$^{-1}$ wt%$^{-1}$)")
ax.set_ylabel("$a_2$ (V N$^{-1}$ wt%$^{-2}$)")
ax.grid(True, lw=0.4, alpha=0.4)
save(fig, "law_b")
print("corr a1,a2 =", round(r, 3), "json corr =", round(d["bootstrap"]["corr"][1][2], 3), "n draws", len(draws),
      "a2<0 frac", d["bootstrap"]["a2_negative_fraction"])
