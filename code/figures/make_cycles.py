"""fig_cycles: (a) one-second voltage recording of condition 58 with the segmented tap cycles shaded,
(b) per-cycle V_rms divided by the recording mean, strip per frequency label.
Source: sonuclar/a7_figdata_waveforms.csv (V_58), a1_cycles.csv, a1_recordings.csv."""
import numpy as np
import pandas as pd
from _style import *

style()
wf = pd.read_csv(SRC / "a7_figdata_waveforms.csv")
cyc = pd.read_csv(SRC / "a1_cycles.csv")
rec = pd.read_csv(SRC / "a1_recordings.csv")
CID = 58

# (a) waveform with cycle bands
fig, ax = plt.subplots(figsize=(W, 1.75))
c58 = cyc[cyc.condition_id == CID].reset_index(drop=True)
t = wf.t_s.to_numpy()
for i, r in c58.iterrows():
    t0, t1 = r.start_sample / 1000.0, r.end_sample / 1000.0
    col = DEEP[0] if i % 2 == 0 else DEEP[9]
    ax.axvspan(t0, t1, color=col, alpha=0.10 if r.full_window else 0.0, lw=0)
    if not r.full_window:
        ax.axvspan(t0, t1, facecolor="none", edgecolor=".6", hatch="//", lw=0)
ax.plot(t, wf.V_58, color=".15", lw=0.6)
ax.axhline(0, color=".6", lw=0.4)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
save(fig, "cycles_a")

# (b) per-cycle rms relative to the recording mean
d = cyc.merge(rec[["condition_id", "freq_Hz", "rms_mean"]], on="condition_id")
d["ratio"] = d.rms / d.rms_mean
freqs = sorted(d.freq_Hz.unique())
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(W, 1.85))
for j, f in enumerate(freqs):
    s = d[d.freq_Hz == f]
    col = DEEP[j if j < 3 else j + 0]
    cols = [DEEP[0], DEEP[1], DEEP[2], DEEP[3], DEEP[4]][j]
    x = j + rng.uniform(-0.28, 0.28, len(s))
    full = s.full_window.to_numpy()
    ax.scatter(x[full], s.ratio[full], s=7, color=cols, alpha=0.55, edgecolor="none", zorder=2)
    ax.scatter(x[~full], s.ratio[~full], s=9, facecolor="none", edgecolor=cols, lw=0.6, zorder=2)
    q1, q2, q3 = s.ratio.quantile([0.25, 0.5, 0.75])
    ax.plot([j - 0.36, j + 0.36], [q2, q2], color=".1", lw=1.4, zorder=3)
    ax.plot([j, j], [q1, q3], color=".1", lw=0.8, zorder=3)
ax.axhline(1.0, color=".5", lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([f"{int(f)}" for f in freqs])
ax.set_xlabel("Nominal tap frequency (Hz)")
ax.set_ylabel("Cycle $V_{rms}$ / recording mean")
ax.set_ylim(0, None)
ax.grid(True, axis="y", lw=0.4, alpha=0.5)
save(fig, "cycles_b")
print("cycles", len(d), "partial", int((~d.full_window).sum()), "recordings", d.condition_id.nunique())
