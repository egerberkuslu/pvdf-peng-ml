#!/usr/bin/env python3
"""A1 helper: fixed-period, phase-aligned segmentation of one 1 s recording
into tapping cycles, plus per-cycle amplitudes.

Why fixed-period windows and not peak detection: the recorded voltage excursions
are one-sample spikes (1 kHz sampling, see the project notes) whose height varies
strongly from tap to tap. A peak detector with a height threshold skips the
small taps (at 20 to 25 Hz it finds far fewer peaks than cycles), which drops
exactly the cycles that make the amplitude vary and biases the cycle CV
downward. Fixed windows of one tap period keep every cycle regardless of its
amplitude. The window phase is placed at the quietest point of the folded
(phase-averaged) mean-square profile so that a boundary does not cut a tap
burst in two. Peak detection with the minimum distance 0.8/f s is still run as
a cross-check and its count is saved.

Which period: the nominal 1/f of the signal generator is the default. A
second candidate comes from the envelope autocorrelation (smallest lag within
4 to 33 Hz whose peak reaches 0.8 of the largest peak, which avoids picking a
multiple of the period), then a half-period check: the half lag P/2 replaces P
when it stays within the search range, consecutive windows of length P/2
carry similar energy (median energy of the weaker parity at least PARITY_MIN of
the stronger one) and the envelope autocorrelation near P/2 reaches at least
HALF_ACF_MIN of its value at P. Folding concentration alone cannot make this
call because a spike train folded at P/2 concentrates as well as at P, and
parity alone accepts recordings whose ringing fills every half window. The candidate whose
folding concentrates more energy
in half a period wins, and the nominal period is kept unless the
autocorrelation period beats it by at least PERIOD_MARGIN. This is needed
because in a subset of recordings the impact rate visibly differs from the
nominal label (finding #5 of the descriptive analysis, confirmed here).
"""
import numpy as np
from scipy.signal import find_peaks

FS = 1000.0  # Hz, 1000 samples over 1 s
EDGE_ENERGY_MIN = 0.95  # a truncated edge window counts as a cycle only if it
# retains at least this fraction of the folded mean-square profile
LOCK_MIN = 0.60  # phase-locking concentration below this flags a recording
PEAK_HEIGHT_FRAC = 0.20  # cross-check peak detector: height >= 0.2 max|x|
PERIOD_MARGIN = 0.05  # autocorrelation period replaces nominal only if its
# phase-locking concentration is higher by at least this much
RATE_DEV_MAX = 0.10  # chosen period off nominal by more than this is listed
ACF_LAG_MIN, ACF_LAG_MAX, ACF_FRAC = 30, 260, 0.8  # samples (33 to 4 Hz)
PARITY_MIN = 0.6  # half-period accepted if alternate windows carry >= this
# fraction of each other's median energy
HALF_ACF_MIN = 0.5  # ACF(P/2) / ACF(P) needed to accept the half period
SPEC_BAND = (3.0, 30.0)  # Hz, band searched by the spectral cross-check


def folded_profile(x, period):
    """Mean-square profile of x folded at `period` samples, nb bins, circularly
    smoothed with a boxcar of about a tenth of a period."""
    nb = int(round(period))
    t = np.arange(len(x))
    b = np.floor(((t % period) / period) * nb).astype(int) % nb
    cnt = np.bincount(b, minlength=nb)
    prof = np.bincount(b, weights=x**2, minlength=nb) / np.maximum(cnt, 1)
    k = max(3, int(round(nb / 10)))
    ext = np.r_[prof, prof, prof]
    sm = np.convolve(ext, np.ones(k) / k, mode="same")[nb : 2 * nb]
    return sm, nb


def lock_concentration(prof):
    """Largest fraction of folded energy inside any circular half period.
    0.5 means no phase locking at the folding period, 1.0 means all energy
    sits in one half period."""
    nb = len(prof)
    h = max(1, nb // 2)
    ext = np.r_[prof, prof]
    sums = np.array([ext[i : i + h].sum() for i in range(nb)])
    return float(sums.max() / prof.sum()) if prof.sum() > 0 else float("nan")


def segment(x, f, period=None):
    """Segment a demeaned recording x (1000 samples) tapped at f Hz.

    Returns a dict with window bounds, per-cycle amplitudes and diagnostics.
    period (samples) defaults to the nominal FS / f.
    """
    P = FS / f if period is None else float(period)
    prof, nb = folded_profile(x, P)
    phi = (np.argmin(prof) + 0.5) / nb * P  # boundary phase, samples
    conc = lock_concentration(prof)
    n = len(x)
    kmin = int(np.floor(-phi / P)) - 1
    kmax = int(np.ceil((n - phi) / P)) + 1
    windows = []
    for k in range(kmin, kmax):
        b0 = phi + k * P
        b1 = b0 + P
        s_full, e_full = int(np.ceil(b0)), int(np.ceil(b1))
        s, e = max(s_full, 0), min(e_full, n)
        if e - s <= 0:
            continue
        full = s == s_full and e == e_full
        if full:
            frac = 1.0
        else:
            j = np.arange(s_full, e_full)
            bins = np.floor(((j % P) / P) * nb).astype(int) % nb
            w = prof[bins]
            keep = (j >= 0) & (j < n)
            frac = float(w[keep].sum() / w.sum()) if w.sum() > 0 else 0.0
        windows.append(
            {
                "start": s,
                "end": e,
                "full": bool(full),
                "energy_frac": frac,
                "accepted": bool(full or frac >= EDGE_ENERGY_MIN),
            }
        )
    acc = [w for w in windows if w["accepted"]]
    vpp = np.array([np.ptp(x[w["start"] : w["end"]]) for w in acc])
    vmax = np.array([np.abs(x[w["start"] : w["end"]]).max() for w in acc])
    rms = np.array([np.sqrt(np.mean(x[w["start"] : w["end"]] ** 2)) for w in acc])
    pk, _ = find_peaks(
        np.abs(x),
        distance=max(1, int(np.floor(0.8 * P))),
        height=PEAK_HEIGHT_FRAC * np.abs(x).max(),
    )
    return {
        "period_samples": P,
        "boundary_phase_samples": float(phi),
        "lock_concentration": conc,
        "windows": acc,
        "n_windows_considered": len(windows),
        "n_cycles": len(acc),
        "n_full_cycles": int(sum(w["full"] for w in acc)),
        "n_peaks_crosscheck": int(len(pk)),
        "vpp": vpp,
        "vmax": vmax,
        "rms": rms,
    }


def envelope_acf(x):
    e = np.convolve(np.abs(x), np.ones(5) / 5, mode="same")
    e = e - e.mean()
    n = len(e)
    ac = np.correlate(e, e, mode="full")[n - 1 :]
    return ac / ac[0] * n / (n - np.arange(n))


def acf_period(x):
    """Tap period (samples) from the autocorrelation of the smoothed |x|
    envelope, overlap-corrected; the smallest local-maximum lag in
    [ACF_LAG_MIN, ACF_LAG_MAX] reaching ACF_FRAC of the largest one."""
    ac = envelope_acf(x)
    a = ac[ACF_LAG_MIN : ACF_LAG_MAX + 1]
    pk = [j for j in range(1, len(a) - 1) if a[j] >= a[j - 1] and a[j] >= a[j + 1]]
    if not pk:
        return float("nan"), float("nan")
    best = max(a[j] for j in pk)
    j = min(j for j in pk if a[j] >= ACF_FRAC * best)
    return float(ACF_LAG_MIN + j), float(a[j])


def parity_ratio(x, H):
    """Median energy of the weaker parity of consecutive windows of length H
    over that of the stronger parity (near 1 when every window holds a tap,
    near 0 when only every other window does)."""
    prof, nb = folded_profile(x, H)
    phi = (np.argmin(prof) + 0.5) / nb * H
    n = len(x)
    k = int(np.ceil(-phi / H))
    E = []
    while True:
        s0 = int(np.ceil(phi + k * H))
        e0 = int(np.ceil(phi + (k + 1) * H))
        if e0 > n:
            break
        if s0 >= 0:
            E.append(float(np.sum(x[s0:e0] ** 2)))
        k += 1
    E = np.asarray(E)
    if len(E) < 4:
        return float("nan")
    a, b = np.median(E[0::2]), np.median(E[1::2])
    return float(min(a, b) / max(a, b)) if max(a, b) > 0 else float("nan")


def acf_period_checked(x):
    """acf_period followed by the half-period parity check."""
    p_raw, acf_val = acf_period(x)
    par = float("nan")
    acf_ratio = float("nan")
    half = False
    if np.isfinite(p_raw) and p_raw / 2 >= ACF_LAG_MIN:
        par = parity_ratio(x, p_raw / 2)
        ac = envelope_acf(x)
        h = int(round(p_raw / 2))
        acf_ratio = float(ac[h - 3 : h + 4].max() / ac[int(p_raw)])
        half = bool(
            np.isfinite(par) and par >= PARITY_MIN and acf_ratio >= HALF_ACF_MIN
        )
    p = p_raw / 2 if half else p_raw
    return p, {
        "period_acf_raw_samples": p_raw,
        "half_period_parity": par,
        "half_period_acf_ratio": acf_ratio,
        "half_period_accepted": half,
        "acf_peak": acf_val,
    }


def spectral_rate(x):
    """Dominant frequency (Hz) of the rectified, demeaned, Hann-windowed
    signal within SPEC_BAND (zero-padded to 2^15 points). Cross-check only:
    a spike train has harmonics, so this can land on a multiple of the rate."""
    e = np.abs(x)
    e = e - e.mean()
    nfft = 2**15
    P = np.abs(np.fft.rfft(e * np.hanning(len(e)), nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, 1.0 / FS)
    m = (fr >= SPEC_BAND[0]) & (fr <= SPEC_BAND[1])
    return float(fr[m][np.argmax(P[m])])


def choose_period(x, f):
    """Return (period_used, info) following the rule in the module docstring."""
    p_nom = FS / f
    p_acf, ainfo = acf_period_checked(x)
    c_nom = lock_concentration(folded_profile(x, p_nom)[0])
    c_acf = (
        lock_concentration(folded_profile(x, p_acf)[0])
        if np.isfinite(p_acf)
        else float("nan")
    )
    use_acf = np.isfinite(c_acf) and c_acf >= c_nom + PERIOD_MARGIN
    p = p_acf if use_acf else p_nom
    return p, {
        "period_nominal_samples": p_nom,
        "period_acf_samples": p_acf,
        **ainfo,
        "lock_concentration_nominal": c_nom,
        "lock_concentration_acf": c_acf,
        "period_source": "acf" if use_acf else "nominal",
        "period_used_samples": p,
        "tap_rate_used_Hz": FS / p,
        "rate_rel_dev_from_nominal": (p_nom / p) - 1.0,
    }


def best_period(x, f, rel=0.10, step=0.1):
    """Epoch-folding period search (sensitivity only): the period within
    +/- rel of nominal that maximizes the phase-locking concentration."""
    P0 = FS / f
    grid = np.arange(P0 * (1 - rel), P0 * (1 + rel) + 1e-9, step)
    conc = [lock_concentration(folded_profile(x, p)[0]) for p in grid]
    i = int(np.argmax(conc))
    return float(grid[i]), float(conc[i])
