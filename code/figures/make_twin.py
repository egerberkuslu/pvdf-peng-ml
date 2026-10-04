"""fig_twin: coverage of the simulated generator (jackknife+, nominal 95%, plain GP, 30 replicate grids per setting).
(a) coverage of a recording on a new specimen against the specimen-to-specimen coefficient of variation, for one,
    three and fifteen specimens per composition; bars are the 2.5 to 97.5 percentile range over replicates.
(b) coverage by target type at a coefficient of variation of 20 percent for the same three designs.
Source: sonuclar/a6_figdata_coverage.csv (generator law solid, twomode dashed in (a)), design definitions in a6_twin.json meta."""
import pandas as pd
from _style import *
from matplotlib.lines import Line2D

style()
c = pd.read_csv(SRC / "a6_figdata_coverage.csv")
c = c[(c.model == "gp") & (c.nominal_pct == 95)]
DES = [("one", "1 specimen", DEEP[3], "o"), ("three", "3 specimens", DEEP[1], "s"), ("fifteen", "15 specimens", DEEP[0], "^")]

# (a)
fig, ax = plt.subplots(figsize=(W, 1.95))
ax.axhline(0.95, color=".4", lw=0.7, ls=(0, (4, 2)), zorder=1)
for j, (dk, dl, col, mk) in enumerate(DES):
    for gen, ls, filled in (("twomode", (0, (2, 1.5)), False), ("law", "-", True)):
        s = c[(c.generator == gen) & (c.design == dk) & (c.scheme == "within_new")].sort_values("cv_pct")
        x = s.cv_pct.to_numpy() + (j - 1) * 0.9
        ax.plot(x, s.coverage_mean, color=col, ls=ls, lw=1.0, zorder=2)
        if filled:
            ax.errorbar(x, s.coverage_mean, yerr=[s.coverage_mean - s.coverage_p2_5, s.coverage_p97_5 - s.coverage_mean],
                        fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, alpha=0.8, zorder=2)
            ax.scatter(x, s.coverage_mean, marker=mk, s=22, color=col, edgecolor="white", lw=0.4, zorder=3)
        else:
            ax.scatter(x, s.coverage_mean, marker=mk, s=22, facecolor="white", edgecolor=col, lw=0.8, zorder=3)
ax.set_xticks([0, 10, 20, 30])
ax.set_xlabel("Specimen-to-specimen spread (CV, %)")
ax.set_ylabel("Coverage of a new specimen")
ax.set_ylim(0.58, 1.02)
ax.grid(True, lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=mk, ls="-", color=col, ms=4.5, label=dl) for _, dl, col, mk in DES]
h += [Line2D([], [], color=".3", lw=1.0, label="Law twin"), Line2D([], [], color=".3", lw=1.0, ls=(0, (2, 1.5)), label="Two-mode twin")]
leg = ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=0.9, labelspacing=0.25)
save(fig, "twin_a", [leg])

# (b)
ROWS = [("within_rec", "Grid recording"), ("within_rep", "Repeat, same specimen"), ("within_new", "Repeat, new specimen"),
        ("group_comp", "Composition held out"), ("group_force", "Force held out"), ("group_freq", "Frequency held out"),
        ("uns_comp_new", "Unsampled composition"), ("uns_force_new", "Unsampled force"), ("uns_freq_new", "Unsampled frequency")]
fig, ax = plt.subplots(figsize=(2.6, 2.7))
ax.axvline(0.95, color=".4", lw=0.7, ls=(0, (4, 2)), zorder=1)
for i, (sk, _) in enumerate(ROWS):
    if i in (3, 6):
        ax.axhline(i - 0.5, color=".85", lw=0.5, zorder=0)
    for j, (dk, dl, col, mk) in enumerate(DES):
        s = c[(c.generator == "law") & (c.design == dk) & (c.scheme == sk) & (c.cv_pct == 20)]
        assert len(s) == 1, (sk, dk, len(s))
        s = s.iloc[0]
        y = i + (j - 1) * 0.22
        ax.plot([s.coverage_p2_5, s.coverage_p97_5], [y, y], color=col, lw=0.8, alpha=0.7, zorder=2)
        ax.scatter([s.coverage_mean], [y], marker=mk, s=18, color=col, edgecolor="white", lw=0.4, zorder=3)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[1] for r in ROWS])
ax.invert_yaxis()
ax.set_xlim(0.45, 1.02)
ax.set_xlabel("Coverage at a specimen spread of 20%")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(True, axis="x", lw=0.4, alpha=0.4)
h = [Line2D([], [], marker=mk, ls="", color=col, ms=4.5, label=dl) for _, dl, col, mk in DES]
leg = ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.7)
save(fig, "twin_b", [leg])
