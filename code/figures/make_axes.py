"""fig_axes: dumbbell of within-level R^2 of the plain GP and LawGP at every held-out level, rms target.
Source: sonuclar/a10_figdata_levels.csv (rowset n75; the same values as a2_physgp.json per_level and
figure_data.fig_phys_axes_clipped_bars). Values below the clip are drawn at the clip with a triangle and printed."""
import json
import pandas as pd
from _style import *

style()
a = pd.read_csv(SRC / "a10_figdata_levels.csv")
a = a[(a.rowset == "n75") & a.model.isin(["gp", "lawgp"])]
CLIP = json.load(open(SRC / "a2_physgp.json"))["figure_data"]["clip_value"]  # -3.0

order = {
    "composition": (["PVDF", "PVDF+BaTiO3", "PVDF+BaTiO3+%1CNT", "PVDF+BaTiO3+%2CNT", "PVDF+BaTiO3+%3CNT"], ["PVDF", "+BaTiO$_3$", "+1 wt% MWCNT", "+2 wt% MWCNT", "+3 wt% MWCNT"]),
    "force": (["1", "2", "3"], ["1 N", "2 N", "3 N"]),
    "frequency": (["5", "10", "15", "20", "25"], ["5 Hz", "10 Hz", "15 Hz", "20 Hz", "25 Hz"]),
}
rows = []
y = 0.0
bounds = {}
heads = []
for ax_name, (keys, labs) in order.items():
    heads.append((y, ax_name.capitalize()))
    y += 1
    y0 = y
    for k, lab in zip(keys, labs):
        s = a[(a.scheme == ax_name) & (a.level.astype(str) == k)].set_index("model").R2_within
        rows.append((y, lab, s["gp"], s["lawgp"]))
        y += 1
    bounds[ax_name] = (y0, y - 1)
    y += 0.35

fig, ax = plt.subplots(figsize=(W, 3.15))
ax.axvline(0, color=".5", lw=0.7)
ax.axvline(CLIP, color=".6", lw=0.6, ls=(0, (3, 2)))
nclip = 0
for yy, lab, g, l in rows:
    gc, lc = max(g, CLIP), max(l, CLIP)
    both = g < CLIP and l < CLIP
    ax.plot([gc, lc], [yy, yy], color=".7", lw=1.6, zorder=1, solid_capstyle="round")
    for v, vc, c, name in ((g, gc, GP_C, "gp"), (l, lc, LAW_C, "law")):
        if v < CLIP:
            nclip += 1
            off = (-0.2 if name == "gp" else 0.2) if both else 0.0
            ax.scatter([vc], [yy + off], marker="<", s=22, color=c, zorder=3, edgecolor="white", lw=0.4)
            txt = (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}").replace("-", "\u2212")
            ax.text(vc + 0.12, yy + off, txt, color=c, fontsize=6.5, va="center", ha="left",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=4)
        else:
            ax.scatter([vc], [yy], marker="o", s=22, color=c, zorder=3, edgecolor="white", lw=0.4)
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows])
ax.invert_yaxis()
ax.set_xlim(CLIP - 0.15, 1.1)
ax.set_xlabel("Within-level $R^2$ (clipped at $-3$)")
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.grid(True, axis="x", lw=0.4, alpha=0.4)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=GP_C, ms=4.5, label="GP"),
     Line2D([], [], marker="o", ls="", color=LAW_C, ms=4.5, label="LawGP")]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.2)
for yh, nm in heads:
    ax.text(-0.52, yh, nm, ha="left", va="center", fontsize=7.5, fontweight="bold", color=".2",
            transform=ax.get_yaxis_transform())
fig.subplots_adjust(left=0.36)
save(fig, "axes")
print("clipped values", nclip)
