"""fig_coverage: coverage-width trade-off at nominal 95% for the rms target.
Within-grid nested LOO intervals of the plain GP (a3_tables.json coverage_within_grid.rms_Voc) and
group-wise held-out-level intervals of the plain GP and LawGP (a3_groupwise_coverage.json groupwise.rms_Voc;
the file labels LawGP as physgp). Raw, jackknife+ and sigma-scaled jackknife+ intervals."""
import json
from _style import *
from matplotlib.lines import Line2D

style()
t = json.load(open(SRC / "a3_tables.json"))["coverage_within_grid"]["rms_Voc"]
g = json.load(open(SRC / "a3_groupwise_coverage.json"))["groupwise"]["rms_Voc"]
METHODS = [("raw", "Raw", "o"), ("jackknife_plus", "Jackknife+", "s"), ("scaled_jackknife_plus", "Scaled jackknife+", "D")]
WITHIN = {"raw": "raw_gp", "jackknife_plus": "jackknife_plus", "scaled_jackknife_plus": "scaled_jackknife_plus"}
SCHEMES = [("within", "Within grid", ".25"), ("composition", "Composition held out", DEEP[4]),
           ("force", "Force held out", DEEP[2]), ("frequency", "Frequency held out", DEEP[1])]

fig, ax = plt.subplots(figsize=(W, 2.5))
ax.axhline(0.95, color=".4", lw=0.7, ls=(0, (4, 2)), zorder=1)
pts = []
for sk, _, col in SCHEMES:
    for mk, _, mm in METHODS:
        if sk == "within":
            r = t[WITHIN[mk]]["95"]
            ax.scatter([r["mean_width"]], [r["coverage"]], marker=mm, s=34, color=col, edgecolor="white", lw=0.5, zorder=3)
            pts.append((sk, mk, "gp", r["mean_width"], r["coverage"], r["hits"], r["n"]))
        else:
            rg, rl = g["gp"][sk][mk]["95"], g["physgp"][sk][mk]["95"]
            ax.plot([rg["mean_width"], rl["mean_width"]], [rg["coverage"], rl["coverage"]], color=col, lw=0.7, alpha=0.5, zorder=2)
            ax.scatter([rg["mean_width"]], [rg["coverage"]], marker=mm, s=34, color=col, edgecolor="white", lw=0.5, zorder=3)
            ax.scatter([rl["mean_width"]], [rl["coverage"]], marker=mm, s=34, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
            pts.append((sk, mk, "gp", rg["mean_width"], rg["coverage"], rg["hits"], rg["n"]))
            pts.append((sk, mk, "law", rl["mean_width"], rl["coverage"], rl["hits"], rl["n"]))
ax.set_xlabel("Mean interval width (V)")
ax.set_ylabel("Empirical coverage")
ax.set_ylim(0.34, 1.02)
ax.set_xlim(0.07, 0.285)
ax.grid(True, lw=0.4, alpha=0.4)
l1 = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=n) for _, n, c in SCHEMES]
leg1 = ax.legend(handles=l1, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0, labelspacing=0.25)
ax.add_artist(leg1)
l2 = [Line2D([], [], marker=m, ls="", color=".35", ms=4.8, label=n) for _, n, m in METHODS]
l2 = [l2[0], None, l2[1], None, l2[2]]
l2[1] = Line2D([], [], marker="o", ls="", color=".35", ms=4.8, label="GP")
l2[3] = Line2D([], [], marker="o", ls="", mfc="white", mec=".35", mew=1.1, ms=4.8, label="LawGP")
leg2 = ax.legend(handles=l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.2, columnspacing=0.9, labelspacing=0.25)
save(fig, "coverage", [leg1, leg2])
for p in pts:
    print(p)
