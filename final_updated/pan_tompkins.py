"""pan_tompkins.py -- causal (streaming-compatible) Pan-Tompkins QRS detector, after Pan & Tompkins, IEEE Trans. Biomed. Eng. 32(3), 1985.
Every stage uses only past samples: 5-15 Hz band-pass (forward-only Butterworth), 5-point derivative, squaring, 150 ms moving-window
integration, then adaptive signal/noise peak thresholds (THR1 = NPKI + 0.25 (SPKI - NPKI)), a 200 ms refractory period, a T-wave slope check
inside 360 ms, and search-back with THR2 = 0.5 THR1 when no beat was found within 1.66 x the running RR average. A candidate integration
peak at index i is decided at time i+1 (a local maximum needs one later sample). Simplifications relative to the paper: thresholds are kept on
the integrated signal only (not also on the band-passed one), and the first 2 s of signal initialise the thresholds. The R position is the
largest |sample| of the causal 0.5-40 Hz signal in the 150 ms before the integration peak, so it is also found with past samples only.
Parameters (defaults reproduce the first version): thr_k = threshold fraction between noise and signal level; shift = samples subtracted from every detected R position (a fixed calibration offset). The stages are vectorised for speed (they are causal, so the result equals sample-by-sample processing); only the decision logic loops."""
import numpy as np
from scipy.signal import butter, sosfilt


def pan_tompkins(sig, fs=360, thr_k=0.25, shift=0):
    sig = np.asarray(sig, dtype=np.float64)
    bp = sosfilt(butter(2, [5.0, 15.0], btype="band", fs=fs, output="sos"), sig)
    d = np.zeros_like(bp); d[4:] = (2 * bp[4:] + bp[3:-1] - bp[1:-3] - 2 * bp[:-4]) * fs / 8.0
    w = int(round(0.150 * fs)); c = np.cumsum(np.concatenate([[0.0], d * d]))
    n = np.arange(1, len(d) + 1); integ = (c[n] - c[np.maximum(n - w, 0)]) / np.minimum(n, w)
    cand = np.flatnonzero((integ[1:-1] > integ[:-2]) & (integ[1:-1] >= integ[2:])) + 1
    init = integ[:int(2 * fs)]
    spki, npki = 0.25 * init.max(), 0.5 * init.mean()
    rr1, rr_ok = [], []; avg = 0.8 * fs
    last, last_slope = -10 * fs, 0.0
    acc, buf = [], []                                         # accepted integration peaks; rejected candidates since the last accepted beat
    refr, tw = int(0.200 * fs), int(0.360 * fs)

    def slope(i):
        return float(np.abs(d[max(0, i - w):i + 1]).max())

    def accept(i, v, sb):
        nonlocal spki, last, last_slope, avg
        spki = (0.25 if sb else 0.125) * v + (0.75 if sb else 0.875) * spki
        if acc:
            rr = i - last; rr1.append(rr); rr1[:] = rr1[-8:]
            if rr_ok and 0.92 * np.mean(rr_ok) < rr < 1.16 * np.mean(rr_ok) or not rr_ok:
                rr_ok.append(rr); rr_ok[:] = rr_ok[-8:]
            avg = float(np.mean(rr_ok)) if rr_ok else float(np.mean(rr1))
        acc.append(i); last = i; last_slope = slope(i); buf.clear()

    for i in cand:
        v = integ[i]; thr1 = npki + thr_k * (spki - npki)
        if acc and i - last > 1.66 * avg:                      # search-back among rejected candidates
            ok = [(j, u) for j, u in buf if j - last > refr and u > 0.5 * thr1]
            if ok:
                j, u = max(ok, key=lambda t: t[1]); accept(j, u, True); thr1 = npki + thr_k * (spki - npki)
                buf[:] = [(a, b) for a, b in buf if a > j]
        if i - last < refr:
            continue
        if v > thr1:
            if i - last < tw and slope(i) < 0.5 * last_slope:   # T wave
                npki = 0.125 * v + 0.875 * npki; buf.append((i, v))
            else:
                accept(i, v, False)
        else:
            npki = 0.125 * v + 0.875 * npki; buf.append((i, v))
    acc = np.array(sorted(acc), dtype=np.int64)
    if len(acc) == 0:
        return acc
    f = sosfilt(butter(4, [0.5, 40.0], btype="band", fs=fs, output="sos"), sig)
    r = np.array([max(0, i - w) + int(np.argmax(np.abs(f[max(0, i - w):i + 1]))) for i in acc], dtype=np.int64)
    return np.unique(np.maximum(r - int(shift), 0))