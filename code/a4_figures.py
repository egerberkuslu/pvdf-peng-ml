#!/usr/bin/env python3
"""A4: descriptive and design-space figures, optima table, macros, captions.

One command, run from the repository root:
    python code/a4_figures.py

Writes
    figures/fig_desc_{eda,cnt,surface,unc,resonance,anova,waveforms}.pdf/.png
    results/tables/tab_desc_optima.tex
    results/numbers_a4.tex         (prefix desc)
    results/a4_desc.json           (every number the macros quote)
    figures/captions_a4.md
The pipeline figure (fig_pipeline.pdf) is built separately with the
figure-maker kit from code/a4_pipeline_scene.py.
Only V_rms, V_pp and |V|_max are reported; the squared-voltage proxy is not.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

import a4_common as cm
import a4_style as st

st.apply_style()
np.random.seed(0)
OUT = {}


def fig_path(name):
    return os.path.join(cm.FIG_DIR, f"fig_desc_{name}")


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=st.FS)


df = cm.load_design()
lg = cm.load_long()
assert len(df) == 75


def waveform(material, force_N, freq_Hz):
    s = lg[
        (lg.material == material)
        & (lg.force == f"{force_N} N")
        & (lg.freq_hz == freq_Hz)
    ].sort_values("time")
    assert len(s) == 1000, (material, force_N, freq_Hz, len(s))
    return s.time.values, s.voc.values


# --------------------------------------------------------------- measured facts
best = {}
for t in cm.TARGETS:
    i = df[t].idxmax()
    best[t] = {
        "value": float(df.loc[i, t]),
        "composition": str(df.loc[i, "composition"]),
        "cnt_pct": float(df.loc[i, "cnt_pct"]),
        "force_N": float(df.loc[i, "force_N"]),
        "freq_Hz": float(df.loc[i, "freq_Hz"]),
    }
OUT["best_measured"] = best

# ================================================================ (1) EDA figure
fig, axes = plt.subplots(2, 3, figsize=(st.TEXT_W, 4.9), layout="constrained")

# (a) target distributions
ax = axes[0, 0]
rng = np.random.default_rng(0)
tcols = [st.C_BLUE, st.C_TEAL, st.C_AMBER]
for i, t in enumerate(cm.TARGETS):
    v = df[t].values
    ax.boxplot(
        v,
        positions=[i],
        widths=0.55,
        showfliers=False,
        boxprops=dict(color=tcols[i], lw=0.9),
        medianprops=dict(color=tcols[i], lw=1.4),
        whiskerprops=dict(color=tcols[i], lw=0.9),
        capprops=dict(color=tcols[i], lw=0.9),
    )
    ax.scatter(
        i + rng.uniform(-0.17, 0.17, len(v)),
        v,
        s=7,
        color=tcols[i],
        alpha=0.55,
        lw=0,
        zorder=3,
    )
ax.set_yscale("log")
ax.set_xticks(range(3))
ax.set_xticklabels([st.TARGET_LABEL[t] for t in cm.TARGETS])
ax.set_ylabel("Voltage (V)")
panel_title(ax, "a", "All 75 recordings")

# (b) frequency response, mean over the three forces
ax = axes[0, 1]
fr = df.groupby(["composition", "freq_Hz"]).rms_Voc.mean().reset_index()
for comp in st.ORDER:
    g = fr[fr.composition == comp]
    ax.plot(
        g.freq_Hz,
        g.rms_Voc,
        marker=st.MARKER[comp],
        color=st.COLOR[comp],
        label=st.SHORT[comp],
        ms=3.5,
    )
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"$V_{\mathrm{rms}}$, mean of 3 forces (V)")
panel_title(ax, "b", "Frequency response")

# (c) force effect, mean over the five frequencies
ax = axes[0, 2]
fo = df.groupby(["composition", "force_N"]).rms_Voc.mean().reset_index()
for comp in st.ORDER:
    g = fo[fo.composition == comp]
    ax.plot(
        g.force_N,
        g.rms_Voc,
        marker=st.MARKER[comp],
        color=st.COLOR[comp],
        label=st.SHORT[comp],
        ms=3.5,
    )
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Nominal force (N)")
ax.set_ylabel(r"$V_{\mathrm{rms}}$, mean of 5 freq. (V)")
panel_title(ax, "c", "Force effect")
ax.legend(loc="upper left", handlelength=1.4, labelspacing=0.25, borderaxespad=0.2)

# (d) composition x frequency map at 3 N, one recording per cell
ax = axes[1, 0]
sub3 = df[df.force_N == 3]
piv = sub3.pivot_table(index="composition", columns="freq_Hz", values="rms_Voc").loc[
    st.ORDER
]
im = ax.imshow(1e3 * piv.values, cmap="viridis", aspect="auto")
vmax_map = piv.values.max()
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        val = piv.values[i, j]
        ax.text(
            j,
            i,
            f"{1e3 * val:.0f}",
            ha="center",
            va="center",
            fontsize=st.FS,
            color="white" if val < 0.6 * vmax_map else "black",
        )
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels([int(c) for c in piv.columns])
ax.set_yticks(range(5))
ax.set_yticklabels([st.SHORT[c] for c in st.ORDER])
ax.set_xlabel("Frequency (Hz)")
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
panel_title(ax, "d", r"$V_{\mathrm{rms}}$ at 3 N (mV)")

# (e) Pearson correlation over the 75 conditions, no energy proxy
ax = axes[1, 1]
ccols = [
    ("cnt_pct", "CNT"),
    ("force_N", "Force"),
    ("freq_Hz", "Freq."),
    ("rms_Voc", r"$V_{\mathrm{rms}}$"),
    ("Vpp", r"$V_{\mathrm{pp}}$"),
    ("Vmax", r"$|V|_{\mathrm{max}}$"),
]
Cm = df[[c for c, _ in ccols]].corr(method="pearson").values
# lower triangle without the unit diagonal: rows 1..5, columns 0..4
Low = np.ma.masked_array(Cm[1:, :-1], mask=np.triu(np.ones((5, 5), bool), k=1))
im = ax.imshow(Low, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")


def rfmt(v):
    if abs(v) < 0.005:
        return "0"
    s_ = f"{v:.2f}"
    return s_.replace("0.", ".", 1)


for i in range(5):
    for j in range(i + 1):
        v = Cm[i + 1, j]
        ax.text(
            j,
            i,
            rfmt(v),
            ha="center",
            va="center",
            fontsize=st.FS,
            color="white" if abs(v) > 0.6 else "black",
        )
ax.set_xticks(range(5))
ax.set_xticklabels([l for _, l in ccols[:-1]], rotation=45, ha="right")
ax.set_yticks(range(5))
ax.set_yticklabels([l for _, l in ccols[1:]])
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
cb.set_label("Pearson r")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.outline.set_linewidth(0.5)
panel_title(ax, "e", "Correlation")
OUT["pearson"] = {
    f"{a}__{b}": float(Cm[i, j])
    for i, (a, _) in enumerate(ccols)
    for j, (b, _) in enumerate(ccols)
    if j > i
}

# (f) waveform example
ax = axes[1, 2]
EX = ("PVDF+BaTiO3+%2CNT", 3, 20)
tt, vv = waveform(*EX)
m = tt < 0.5
ax.plot(tt[m], vv[m], color=st.C_RED, lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Recorded voltage (V)")
ax.set_xlim(0, 0.5)
panel_title(ax, "f", "One raw recording")

st.save(fig, fig_path("eda"))
OUT["eda"] = {
    "map_3N_max_V": float(vmax_map),
    "freq_resp_force_mean_max_V": float(fr.rms_Voc.max()),
}
print("eda done")

# ================================================================ (2) CNT effect
fig, ax = plt.subplots(figsize=(st.COL_W, 2.45), layout="constrained")
ser = df[(df.force_N == 3) & (df.is_pristine == 0)].copy()
freqs = sorted(ser.freq_Hz.unique())
fcol = plt.cm.viridis(np.linspace(0.0, 0.78, len(freqs)))
for k, f in enumerate(freqs):
    g = ser[ser.freq_Hz == f].sort_values("cnt_pct")
    ax.plot(
        g.cnt_pct + (k - 2) * 0.045,
        g.rms_Voc,
        ls="none",
        marker=["o", "s", "^", "v", "P"][k],
        ms=3.4,
        mfc="none",
        mec=fcol[k],
        mew=0.9,
        label=f"{int(f)} Hz",
    )
cm_mean = ser.groupby("cnt_pct").rms_Voc.mean()
ax.plot(
    cm_mean.index,
    cm_mean.values,
    color=st.C_RED,
    marker="D",
    ms=4.5,
    lw=1.6,
    label="Mean of 5 freq.",
    zorder=4,
)
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("CNT loading (wt%)")
ax.set_ylabel(r"$V_{\mathrm{rms}}$ at 3 N (V)")
ax.set_xlim(-0.3, 3.3)
ax.legend(
    loc="upper left",
    ncol=2,
    handlelength=1.0,
    columnspacing=0.8,
    labelspacing=0.2,
    borderaxespad=0.2,
)
ax.set_ylim(0, ser.rms_Voc.max() * 1.55)
st.save(fig, fig_path("cnt"))
OUT["cnt_effect_3N"] = {
    "mean_over_freq_V": {str(k): float(v) for k, v in cm_mean.items()},
    "argmax_cnt_pct": float(cm_mean.idxmax()),
}
print("cnt done")

# ================================================= (3) response surfaces and (4) SD
gp_rms, xs_rms = cm.fit_gp(df, "rms_Voc")
F, Q, G = cm.slice_grid(2.0)
mu, sd = gp_rms.predict(xs_rms.transform(G), return_std=True)
mu = mu.reshape(F.shape)
sd = sd.reshape(F.shape)
rf_vpp = cm.fit_rf(df, "Vpp")
vpp = rf_vpp.predict(G).reshape(F.shape)
meas = df[df.cnt_pct == 2.0][["freq_Hz", "force_N"]].drop_duplicates()


def mark_measured(ax):
    ax.scatter(
        meas.freq_Hz,
        meas.force_N,
        s=14,
        c="white",
        edgecolors="#222222",
        linewidths=0.6,
        zorder=4,
        clip_on=False,
    )


fig, axes = plt.subplots(1, 2, figsize=(st.TEXT_W, 2.55), layout="constrained")
pc = axes[0].contourf(Q, F, mu, levels=12, cmap="viridis")
cb = fig.colorbar(pc, ax=axes[0], pad=0.02)
cb.set_label(r"GP mean $V_{\mathrm{rms}}$ (V)")
pc2 = axes[1].pcolormesh(Q, F, vpp, cmap="viridis", shading="nearest", rasterized=True)
cb2 = fig.colorbar(pc2, ax=axes[1], pad=0.02)
cb2.set_label(r"RF $V_{\mathrm{pp}}$ (V)")
for ax, (letter, ttl) in zip(
    axes,
    [("a", r"GP, $V_{\mathrm{rms}}$"), ("b", r"Random forest, $V_{\mathrm{pp}}$")],
):
    mark_measured(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Nominal force (N)")
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.grid(False)
    ax.set_title(f"({letter}) {ttl}", loc="left", fontsize=st.FS, pad=7)
st.save(fig, fig_path("surface"))

i_mu = np.unravel_index(np.argmax(mu), mu.shape)
i_vpp = np.unravel_index(np.argmax(vpp), vpp.shape)
OUT["surface_2wt"] = {
    "gp_rms_max_V": float(mu.max()),
    "gp_rms_argmax_force_N": float(F[i_mu]),
    "gp_rms_argmax_freq_Hz": float(Q[i_mu]),
    "rf_vpp_max_V": float(vpp.max()),
    "rf_vpp_argmax_force_N": float(F[i_vpp]),
    "rf_vpp_argmax_freq_Hz": float(Q[i_vpp]),
    "rf_settings": "RandomForestRegressor(n_estimators=400, random_state=42)",
    "grid": "force 1-3 N (161 pts) x frequency 5-25 Hz (201 pts), cnt 2 wt%",
}
print("surface done")

# GP length scales in original units
ls_std = np.asarray(gp_rms.kernel_.k1.k2.length_scale, dtype=float)
x_sd = xs_rms.scale_
ls_orig = ls_std * x_sd
OUT["gp_rms_full_fit"] = {
    "kernel": str(gp_rms.kernel_),
    "length_scale_standardized": [float(v) for v in ls_std],
    "length_scale_original_units": {n: float(v) for n, v in zip(cm.FEAT, ls_orig)},
}

fig, axes = plt.subplots(1, 2, figsize=(st.TEXT_W, 2.55), layout="constrained")
pa = axes[0].pcolormesh(Q, F, mu, cmap="viridis", shading="gouraud", rasterized=True)
cb = fig.colorbar(pa, ax=axes[0], pad=0.02)
cb.set_label(r"GP mean $V_{\mathrm{rms}}$ (V)")
pb = axes[1].pcolormesh(Q, F, sd, cmap="magma", shading="gouraud", rasterized=True)
cb = fig.colorbar(pb, ax=axes[1], pad=0.02)
cb.set_label(r"GP posterior SD (V)")
for ax, (letter, ttl) in zip(
    axes, [("a", "Posterior mean"), ("b", "Posterior standard deviation")]
):
    mark_measured(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Nominal force (N)")
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.grid(False)
    ax.set_title(f"({letter}) {ttl}", loc="left", fontsize=st.FS, pad=7)
st.save(fig, fig_path("unc"))

# SD statistics (inside the measured box, which is the whole plotted plane)
sd_at_meas = sd[
    np.isin(np.round(F, 6), [1.0, 2.0, 3.0])
    & np.isin(np.round(Q, 6), [5, 10, 15, 20, 25])
]
OUT["unc_2wt"] = {
    "sd_min_V": float(sd.min()),
    "sd_max_V": float(sd.max()),
    "sd_ratio_max_over_min": float(sd.max() / sd.min()),
    "sd_median_V": float(np.median(sd)),
    "sd_at_measured_mean_V": float(sd_at_meas.mean()),
    "n_measured_nodes_in_grid": int(sd_at_meas.size),
    "sd_min_force_N": float(F[np.unravel_index(sd.argmin(), sd.shape)]),
    "sd_min_freq_Hz": float(Q[np.unravel_index(sd.argmin(), sd.shape)]),
    "sd_max_force_N": float(F[np.unravel_index(sd.argmax(), sd.shape)]),
    "sd_max_freq_Hz": float(Q[np.unravel_index(sd.argmax(), sd.shape)]),
}
print("unc done", OUT["unc_2wt"]["sd_min_V"], OUT["unc_2wt"]["sd_ratio_max_over_min"])

# ================================================================ (5) resonance
LEG = json.load(open(os.path.join(cm.LEGACY_RES, "revision_experiments.json")))
LU = LEG["lorentzian_uncertainty"]
fig, axes = plt.subplots(
    1, 5, figsize=(st.TEXT_W, 2.25), layout="constrained", sharey=True
)
fgrid = np.linspace(4, 26, 300)
refit = {}
ymax = df.rms_Voc.max()
for ax, comp in zip(axes, st.ORDER):
    sub = df[df.composition == comp]
    mfr = sub.groupby("freq_Hz").rms_Voc.mean()
    fx, vy = mfr.index.values.astype(float), mfr.values
    # identical call to legacy revision_experiments.py block D
    p, pc = curve_fit(
        cm.lorentz,
        fx,
        vy,
        p0=[vy.max(), 19.0, 8.0],
        bounds=([0, 10, 1], [10 * vy.max(), 30, 30]),
        maxfev=20000,
    )
    se = np.sqrt(np.diag(pc))
    J = LU["per_material"][comp]
    assert abs(p[1] - J["f0"]) < 1e-4 and abs(p[2] - J["gamma"]) < 1e-4, comp
    assert abs(se[1] - J["f0_se"]) < 1e-4 and abs(se[2] - J["gamma_se"]) < 1e-4, comp
    refit[comp] = {"A": float(p[0]), "f0": float(p[1]), "gamma": float(p[2])}
    for Fn, mk in zip([1, 2, 3], ["o", "s", "^"]):
        s = sub[sub.force_N == Fn].sort_values("freq_Hz")
        ax.plot(
            s.freq_Hz,
            s.rms_Voc,
            ls="none",
            marker=mk,
            ms=2.8,
            mfc="none",
            mec="#9AA0A6",
            mew=0.7,
            label=f"{Fn} N" if comp == st.ORDER[0] else None,
        )
    ax.axvspan(
        J["f0"] - J["f0_se"],
        J["f0"] + J["f0_se"],
        color=st.COLOR[comp],
        alpha=0.15,
        lw=0,
    )
    ax.axvline(J["f0"], color=st.COLOR[comp], lw=0.8, ls="--")
    ax.plot(fgrid, cm.lorentz(fgrid, *p), color=st.COLOR[comp], lw=1.3)
    ax.plot(
        fx,
        vy,
        ls="none",
        marker=st.MARKER[comp],
        ms=4,
        color=st.COLOR[comp],
        mec="white",
        mew=0.4,
        zorder=4,
        label="Mean of 3 forces" if comp == st.ORDER[0] else None,
    )
    ax.text(
        0.04,
        0.97,
        f"$f_0$ {J['f0']:.1f} ± {J['f0_se']:.1f}\n"
        f"$\\gamma$  {J['gamma']:.1f} ± {J['gamma_se']:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=st.FS,
        zorder=6,
        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9),
    )
    ax.set_xticks([5, 15, 25])
    ax.set_xlim(3, 27)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(st.SHORT[comp], loc="center", fontsize=st.FS)
axes[0].set_ylabel(r"$V_{\mathrm{rms}}$ (V)")
axes[0].set_ylim(0, ymax * 1.5)
fig.legend(
    loc="outside upper center",
    ncol=4,
    handlelength=1.2,
    columnspacing=1.2,
    borderaxespad=0.1,
)
st.save(fig, fig_path("resonance"))
OUT["resonance"] = {
    "source": "results/baseline/revision_experiments.json -> lorentzian_uncertainty",
    "fit_data": "per composition, V_rms averaged over the three forces at each of the five frequencies (five points, three parameters, 2 residual df)",
    "per_material": LU["per_material"],
    "global_law_rms": {
        k: LU["global_law_rms"][k] for k in ("f0", "f0_se", "gamma", "gamma_se")
    },
    "refit_check": refit,
}
print("resonance done")

# ================================================================ (6) ANOVA
AN = json.load(open(os.path.join(cm.LEGACY_RES, "anova_results.json")))
EFF = [
    ("composition", "Composition"),
    ("force", "Force"),
    ("frequency", "Frequency"),
    ("composition:force", "Comp. × force"),
    ("composition:frequency", "Comp. × freq."),
    ("force:frequency", "Force × freq."),
]
SERIES = [
    ("log_rms", r"$\log_{10} V_{\mathrm{rms}}$", st.C_BLUE, None),
    ("rms_Voc", r"$V_{\mathrm{rms}}$", st.C_TEAL, None),
    ("Vpp", r"$V_{\mathrm{pp}}$", st.C_AMBER, None),
]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


fig, ax = plt.subplots(figsize=(st.COL_W, 3.5), layout="constrained")
h = 0.26
ypos = np.arange(len(EFF))[::-1]
for k, (key, lab, col, _) in enumerate(SERIES):
    eta = [AN[key]["typ2_partial_eta2"][e] for e, _ in EFF]
    pv = [AN[key]["typ2_p_values"][e] for e, _ in EFF]
    yy = ypos + (1 - k) * h
    ax.barh(yy, eta, height=h * 0.92, color=col, label=lab, lw=0)
    for y_, e_, p_ in zip(yy, eta, pv):
        ax.text(e_ + 0.012, y_, stars(p_), va="center", ha="left", fontsize=st.FS)
ax.set_yticks(ypos)
ax.set_yticklabels([l for _, l in EFF])
ax.set_xlim(0, 1.18)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"Partial $\eta^2$ (type II)")
ax.grid(axis="y", visible=False)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.45, 1.13),
    ncol=3,
    handlelength=1.0,
    columnspacing=0.9,
    borderaxespad=0.0,
)
st.save(fig, fig_path("anova"))
OUT["anova"] = {
    "source": "results/baseline/anova_results.json",
    "residual_df": {k: AN[k]["residual_df"] for k, *_ in SERIES},
    "partial_eta2": {k: AN[k]["typ2_partial_eta2"] for k, *_ in SERIES},
    "p_values": {k: AN[k]["typ2_p_values"] for k, *_ in SERIES},
}
print("anova done")

# ================================================================ (7) waveforms
fig, axes = plt.subplots(
    5, 1, figsize=(st.COL_W, 3.9), layout="constrained", sharex=True, sharey=True
)
wf = {}
for ax, f in zip(axes, [5, 10, 15, 20, 25]):
    tt, vv = waveform("PVDF+BaTiO3+%2CNT", 3, f)
    ax.plot(tt, vv, color=st.C_RED, lw=0.6)
    ax.text(
        0.995,
        0.93,
        f"{f} Hz",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=st.FS,
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
    )
    ax.set_yticks([-4, 0, 4])
    wf[str(f)] = {"min_V": float(vv.min()), "max_V": float(vv.max())}
axes[-1].set_xlabel("Time (s)")
axes[-1].set_xlim(0, 1)
fig.supylabel("Recorded voltage (V)", fontsize=st.FS)
st.save(fig, fig_path("waveforms"))
OUT["waveforms_2wt_3N"] = wf
print("waveforms done")

# ================================================================ optima table
dg = cm.dense_grid()
opt = {}
gp_pred = gp_rms.predict(xs_rms.transform(dg))
plateau_tol = {"rms_Voc": 0.01}  # GP: points within 1% of the GP maximum


def plateau(pred, tol_rel):
    sel = pred >= pred.max() * (1 - tol_rel)
    P = dg[sel]
    return {
        "cnt": [float(P[:, 0].min()), float(P[:, 0].max())],
        "force": [float(P[:, 2].min()), float(P[:, 2].max())],
        "freq": [float(P[:, 3].min()), float(P[:, 3].max())],
        "n_points": int(sel.sum()),
    }


i = int(np.argmax(gp_pred))
opt["rms_Voc"] = {
    "model": "GP",
    "pred_max_V": float(gp_pred.max()),
    "argmax": {
        "cnt": float(dg[i, 0]),
        "force": float(dg[i, 2]),
        "freq": float(dg[i, 3]),
    },
    "plateau_1pct": plateau(gp_pred, 0.01),
}
for t in ["Vpp", "Vmax"]:
    rf = cm.fit_rf(df, t)
    pr = rf.predict(dg)
    i = int(np.argmax(pr))
    opt[t] = {
        "model": "RF",
        "pred_max_V": float(pr.max()),
        "argmax": {
            "cnt": float(dg[i, 0]),
            "force": float(dg[i, 2]),
            "freq": float(dg[i, 3]),
        },
        "plateau_1pct": plateau(pr, 0.01),
    }
GA = json.load(open(os.path.join(cm.LEGACY_RES, "ga_search.json")))
ga_pred = [r["predicted_rms_V"] for r in GA["runs"]]
OUT["optima"] = {
    "grid": "cnt 0-3 wt% step 0.25, force 1-3 N step 0.1, frequency 5-25 Hz step 0.5, BaTiO3 series (regression/reg_02_surface.py grid)",
    "plateau_definition": "all grid points whose prediction is within 1% of the model maximum",
    "targets": opt,
    "submitted_table8": {"rms_Voc": 0.432, "Vpp": 7.36, "Vmax": 4.20},
    "ga": {
        "source": "results/baseline/ga_search.json",
        "n_runs": len(GA["runs"]),
        "cnt_range": GA["cnt_range"],
        "force_range": GA["force_range"],
        "freq_range": GA["freq_range"],
        "pred_rms_range_V": [float(min(ga_pred)), float(max(ga_pred))],
    },
}


def rng_txt(a, b, fmt):
    a_, b_ = fmt.format(a), fmt.format(b)
    return a_ if a_ == b_ else f"{a_}--{b_}"


rows = []
tlabel = {"rms_Voc": r"\RMS", "Vpp": r"\Vpp", "Vmax": r"\Vmax"}
for t in cm.TARGETS:
    o = opt[t]
    pl = o["plateau_1pct"]
    b = best[t]
    edge = " (at \\SI{3}{\\newton} edge)" if o["argmax"]["force"] >= 3.0 - 1e-9 else ""
    rows.append(
        f"{tlabel[t]} & {o['model']} & {rng_txt(*pl['cnt'], '{:.2f}')} & "
        f"{rng_txt(*pl['force'], '{:.1f}')} & {rng_txt(*pl['freq'], '{:.1f}')} & "
        f"{o['pred_max_V']:.{2 if t != 'rms_Voc' else 3}f}{edge} & "
        f"{b['value']:.{2 if t != 'rms_Voc' else 3}f} \\\\"
    )
g = OUT["optima"]["ga"]
rows.append(
    f"\\RMS{{}} GA check & GP & {rng_txt(*g['cnt_range'], '{:.2f}')} & "
    f"{rng_txt(*g['force_range'], '{:.2f}')} & {rng_txt(*g['freq_range'], '{:.1f}')} & "
    f"{rng_txt(*g['pred_rms_range_V'], '{:.3f}')} & {best['rms_Voc']['value']:.3f} \\\\"
)
tab = (
    "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
    "Target & Model & CNT (wt\\%) & Force (\\si{\\newton}) & Freq.\\ (\\si{\\hertz}) & "
    "Predicted max.\\ (\\si{\\volt}) & Best measured (\\si{\\volt}) \\\\\n\\midrule\n"
    + "\n".join(rows[:3])
    + "\n\\midrule\n"
    + rows[3]
    + "\n\\bottomrule\n\\end{tabular}\n"
)
os.makedirs(cm.TAB_DIR, exist_ok=True)
open(os.path.join(cm.TAB_DIR, "tab_desc_optima.tex"), "w").write(tab)
print("table done")


# ================================================================ macros
def f2(x, d):
    return f"{x:.{d}f}"


PM = LU["per_material"]
CKEY = {
    "PVDF": "Pvdf",
    "PVDF+BaTiO3": "Bto",
    "PVDF+BaTiO3+%1CNT": "CntOne",
    "PVDF+BaTiO3+%2CNT": "CntTwo",
    "PVDF+BaTiO3+%3CNT": "CntThree",
}
M = {}
M["descBestRms"] = f2(best["rms_Voc"]["value"], 3)
M["descBestVpp"] = f2(best["Vpp"]["value"], 2)
M["descBestVmax"] = f2(best["Vmax"]["value"], 2)
M["descBestCnt"] = f2(best["rms_Voc"]["cnt_pct"], 0)
M["descBestForce"] = f2(best["rms_Voc"]["force_N"], 0)
M["descBestFreq"] = f2(best["rms_Voc"]["freq_Hz"], 0)
M["descMapMaxRms"] = f2(OUT["eda"]["map_3N_max_V"], 3)
for k, v in cm_mean.items():
    M[f"descCntMeanRms{['Zero', 'One', 'Two', 'Three'][int(k)]}"] = f2(v, 3)
M["descCntArgmax"] = f2(OUT["cnt_effect_3N"]["argmax_cnt_pct"], 0)
s = OUT["surface_2wt"]
M["descSurfRmsMax"] = f2(s["gp_rms_max_V"], 3)
M["descSurfRmsMaxForce"] = f2(s["gp_rms_argmax_force_N"], 2)
M["descSurfRmsMaxFreq"] = f2(s["gp_rms_argmax_freq_Hz"], 1)
M["descSurfVppMax"] = f2(s["rf_vpp_max_V"], 2)
u = OUT["unc_2wt"]
M["descSdMinRms"] = f2(u["sd_min_V"], 4)
M["descSdMaxRms"] = f2(u["sd_max_V"], 4)
M["descSdRatioRms"] = f2(u["sd_ratio_max_over_min"], 1)
M["descSdMedianRms"] = f2(u["sd_median_V"], 4)
lo = OUT["gp_rms_full_fit"]["length_scale_original_units"]
M["descLsCnt"] = f2(lo["cnt_pct"], 1)
M["descLsForce"] = f2(lo["force_N"], 1)
M["descLsFreq"] = f2(lo["freq_Hz"], 1)
for comp, key in CKEY.items():
    M[f"descFzero{key}"] = f2(PM[comp]["f0"], 1)
    M[f"descFzeroSe{key}"] = f2(PM[comp]["f0_se"], 1)
    M[f"descGamma{key}"] = f2(PM[comp]["gamma"], 1)
    M[f"descGammaSe{key}"] = f2(PM[comp]["gamma_se"], 1)
f0s = [PM[c]["f0"] for c in st.ORDER]
gms = [PM[c]["gamma"] for c in st.ORDER]
M["descFzeroMin"] = f2(min(f0s), 1)
M["descFzeroMax"] = f2(max(f0s), 1)
M["descGammaMin"] = f2(min(gms), 1)
M["descGammaMax"] = f2(max(gms), 1)
GL = LU["global_law_rms"]
M["descFzeroGlobal"] = f2(GL["f0"], 1)
M["descFzeroSeGlobal"] = f2(GL["f0_se"], 1)
M["descGammaGlobal"] = f2(GL["gamma"], 1)
M["descGammaSeGlobal"] = f2(GL["gamma_se"], 1)
EK = {
    "composition": "Comp",
    "force": "Force",
    "frequency": "Freq",
    "composition:force": "CompForce",
    "composition:frequency": "CompFreq",
    "force:frequency": "ForceFreq",
}
SK = {"log_rms": "LogRms", "rms_Voc": "Rms", "Vpp": "Vpp"}
for key, sk in SK.items():
    for e, ek in EK.items():
        M[f"descEta{sk}{ek}"] = f2(AN[key]["typ2_partial_eta2"][e], 2)
        p = AN[key]["typ2_p_values"][e]
        M[f"descP{sk}{ek}"] = (
            f2(p, 2) if p >= 0.01 else ("<0.001" if p < 1e-3 else f2(p, 3))
        )
M["descAnovaResidDf"] = str(AN["rms_Voc"]["residual_df"])
for t, tk in [("rms_Voc", "Rms"), ("Vpp", "Vpp"), ("Vmax", "Vmax")]:
    o = opt[t]
    d = 3 if t == "rms_Voc" else 2
    M[f"descOpt{tk}Pred"] = f2(o["pred_max_V"], d)
    M[f"descOpt{tk}CntLo"] = f2(o["plateau_1pct"]["cnt"][0], 2)
    M[f"descOpt{tk}CntHi"] = f2(o["plateau_1pct"]["cnt"][1], 2)
    M[f"descOpt{tk}ForceLo"] = f2(o["plateau_1pct"]["force"][0], 1)
    M[f"descOpt{tk}ForceHi"] = f2(o["plateau_1pct"]["force"][1], 1)
    M[f"descOpt{tk}FreqLo"] = f2(o["plateau_1pct"]["freq"][0], 1)
    M[f"descOpt{tk}FreqHi"] = f2(o["plateau_1pct"]["freq"][1], 1)
M["descGaRuns"] = str(g["n_runs"])
M["descGaCntLo"] = f2(g["cnt_range"][0], 2)
M["descGaCntHi"] = f2(g["cnt_range"][1], 2)
M["descGaForceLo"] = f2(g["force_range"][0], 2)
M["descGaForceHi"] = f2(g["force_range"][1], 2)
M["descGaFreqLo"] = f2(g["freq_range"][0], 1)
M["descGaFreqHi"] = f2(g["freq_range"][1], 1)
M["descGaRmsLo"] = f2(g["pred_rms_range_V"][0], 3)
M["descGaRmsHi"] = f2(g["pred_rms_range_V"][1], 3)
r = OUT["pearson"]
M["descCorrForceRms"] = f2(r["force_N__rms_Voc"], 2)
M["descCorrFreqRms"] = f2(r["freq_Hz__rms_Voc"], 2)
M["descCorrCntRms"] = f2(r["cnt_pct__rms_Voc"], 2)
M["descCorrRmsVpp"] = f2(r["rms_Voc__Vpp"], 2)
M["descCorrVppVmax"] = f2(r["Vpp__Vmax"], 2)
for name in M:
    assert not any(ch.isdigit() for ch in name), name
lines = [
    "% numbers_a4.tex: generated by code/a4_figures.py (task A4). Do not edit."
]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
open(os.path.join(cm.RES_DIR, "numbers_a4.tex"), "w").write("\n".join(lines) + "\n")
OUT["macros"] = M

# ================================================================ spectral check
# Dominant 3-60 Hz FFT component of each mean-removed recording against the
# nominal tapping frequency (descriptive; shared with A1 on the board).
spec_rows = []
fr_axis = np.fft.rfftfreq(1000, 0.001)
band = (fr_axis >= 3) & (fr_axis <= 60)
for (mat, frc, fq), s_ in lg.groupby(["material", "force", "freq_hz"]):
    v_ = s_.sort_values("time").voc.values
    X_ = np.abs(np.fft.rfft(v_ - v_.mean()))
    fpk = float(fr_axis[band][np.argmax(X_[band])])
    spec_rows.append((str(mat), str(frc), int(fq), fpk, abs(fpk - fq) <= 1.0))
OUT["spectral_check"] = {
    "definition": "dominant FFT component in 3-60 Hz of the mean-removed 1 s recording within +-1 Hz of the nominal frequency",
    "n_match": int(sum(r[4] for r in spec_rows)),
    "n_total": len(spec_rows),
    "n_match_by_nominal_Hz": {
        str(f): int(sum(r[4] for r in spec_rows if r[2] == f)) for f in (5, 10, 15, 20, 25)
    },
    "per_recording": [
        {"material": r[0], "force": r[1], "freq_Hz": r[2], "dominant_Hz": r[3]}
        for r in spec_rows
    ],
}

OUT["seed"] = 0
OUT["index"] = {
    "fig_desc_eda": ["best_measured", "eda", "pearson"],
    "fig_desc_cnt": ["cnt_effect_3N"],
    "fig_desc_surface": ["surface_2wt", "gp_rms_full_fit"],
    "fig_desc_unc": ["unc_2wt", "gp_rms_full_fit"],
    "fig_desc_resonance": ["resonance"],
    "fig_desc_anova": ["anova"],
    "fig_desc_waveforms": ["waveforms_2wt_3N"],
    "tab_desc_optima": ["optima", "best_measured"],
    "numbers_a4.tex": ["macros"],
    "board finding on periodicity": ["spectral_check"],
    "macro prefixes": {
        "descBest*": "best_measured",
        "descMapMaxRms": "eda.map_3N_max_V",
        "descCntMeanRms*, descCntArgmax": "cnt_effect_3N",
        "descSurf*": "surface_2wt",
        "descSd*": "unc_2wt",
        "descLs*": "gp_rms_full_fit.length_scale_original_units",
        "descFzero*, descGamma*": "resonance.per_material, resonance.global_law_rms",
        "descEta*, descP*, descAnovaResidDf": "anova",
        "descOpt*": "optima.targets",
        "descGa*": "optima.ga",
        "descCorr*": "pearson",
    },
}
json.dump(OUT, open(os.path.join(cm.RES_DIR, "a4_desc.json"), "w"), indent=1)
print(
    f"A4 done: 7 figures, optima table, {len(M)} macros; "
    f"SD min {M['descSdMinRms']} V ratio {M['descSdRatioRms']}; "
    f"GP optimum {M['descOptRmsPred']} V"
)
