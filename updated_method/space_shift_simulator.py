"""
space_shift_simulator.py

Physiologically-calibrated simulator for aerospace/microgravity-induced
ECG signal shift, built to create a "target domain" for domain-adaptation
experiments (DANN) on top of the MIT-BIH source domain.

DESIGN PRINCIPLE
-----------------
This does NOT add generic Gaussian noise and call it "space." Each
transformation is calibrated to a specific, cited effect reported in
microgravity / head-down-bed-rest (HDBR) cardiovascular literature:

1. HRV spectral redistribution (VLF/LF band power attenuation)
   Otsuka et al. (npj Microgravity, 2015) reported ULF-band HRV power
   decreased 22.2-52.4% and 13.2-53.9% (two sub-bands) during spaceflight
   vs. Earth in 7 astronauts over 24h recordings.
   [https://www.nature.com/articles/npjmgrav201518]

   NOTE ON HONESTY / LIMITATION: the ULF band strictly requires ~24h
   recordings to resolve reliably. MIT-BIH records are ~30 min, so we
   cannot faithfully reproduce the ULF finding. Instead we apply
   attenuation to the VLF and LF bands (0.003-0.04 Hz and 0.04-0.15 Hz),
   which sit within what a ~30 min record can resolve, and which the same
   body of literature also reports as disrupted under autonomic
   dysregulation in microgravity/HDBR. State this explicitly in your
   paper's methods/limitations section - do not claim ULF reproduction.

2. Increased low-frequency T-wave oscillation (Periodic Repolarization
   Dynamics, PRD)
   Baumert et al. (Front. Physiol., 2019) found long-duration microgravity
   exposure increases low-frequency oscillation of the T-wave vector, an
   arrhythmic risk marker.
   [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928004/]
   We add a slow sinusoidal perturbation confined to the T-wave segment.

3. QRS-T angle / axis drift
   The same HDBR study reports subjects developing QRS-T angles >100 deg
   (vs. a healthy population median well below that), linked to elevated
   arrhythmia/sudden-death risk. We approximate this with a small
   per-beat rotation of the two-lead signal (a simplified vectorcardiographic
   proxy - we are NOT computing a true QRS-T angle from 2 leads, and your
   paper should describe it as a lead-vector rotation proxy, not a
   clinically-equivalent QRS-T angle measurement).

USAGE
-----
This module is meant to be run AFTER the baseline pipeline
(ecg_arrhythmia_cnn.py) has downloaded MIT-BIH into ./ECG_Project/mitdb.
It independently re-reads each record's annotations to get the continuous
R-peak sequence (needed for spectral shifting, which requires continuity -
information the baseline pipeline discards after building the beat-level
dataset).

    from space_shift_simulator import build_domain_dataset
    source, target = build_domain_dataset(records, severity="moderate")

`source` and `target` are dicts with keys: X, RR, y, patient_ids, domain
(domain=0 for source/Earth-like, domain=1 for target/space-shifted).
Labels y and patient_ids are IDENTICAL between source and target for a
given beat index - only the signal has been perturbed. This paired design
is what lets you measure "does the model's decision change when only the
physiology-consistent shift is applied."
"""

from pathlib import Path
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline, interp1d

# ==========================================================
# Constants - MUST match ecg_arrhythmia_cnn.py's config
# ==========================================================

import os

FS = 360
LOWCUT, HIGHCUT, FILTER_ORDER = 0.5, 40.0, 4
LEFT_WINDOW = 180
RIGHT_WINDOW = 180

NORMAL_BEATS = {"N", "L", "R", "e", "j"}  # matches AAMI EC57 / base.SYMBOL_TO_AAMI's
# N-class exactly (see ecg_multiclass_cnn.py). Previously this was {"N"} only,
# which mislabeled left/right bundle branch block beats (L, R - conduction
# variants, not true ectopic arrhythmias) as "Abnormal." That inflated the
# apparent abnormal rate from ~9.7% (the project's established DS1 rate) to
# ~23% in this module's own labeling, and meant this script was training on
# a different, messier task definition than every other script in the
# project. If you already ran this before this fix, the cached
# domain_dataset_*.npz files contain the OLD mislabeled data - delete them
# (or they'll be silently reused) and reprocess.
VALID_BEATS = {'N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F',
               '/', 'f', 'Q', 'e', 'j'}

# Same auto-detection logic as ecg_arrhythmia_cnn.py, so this module finds
# the SAME already-downloaded MIT-BIH data - it never re-downloads anything.
if os.path.exists("/content/drive/MyDrive"):
    PROJECT_DIR = Path("/content/drive/MyDrive/ECG_Project")
else:
    PROJECT_DIR = Path("./ECG_Project")

MITDB_DIR = PROJECT_DIR / "mitdb"

# HRV bands (Hz) - standard definitions, VLF/LF chosen (not ULF) per the
# resolvability limitation documented above.
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)

RR_MIN, RR_MAX = 0.3, 2.0  # physiologically plausible RR bounds (s), for clipping

# Severity presets. These are a *sensitivity analysis axis* for your paper -
# report results across all three, don't just pick one. "moderate" is
# calibrated near the lower end of literature-reported magnitudes; treat
# "severe" as a stress-test upper bound, not a claimed real value.
SEVERITY_PRESETS = {
    "mild":     dict(vlf_atten=0.15, lf_atten=0.08, prd_depth=0.06, axis_rot_deg=3.0),
    "moderate": dict(vlf_atten=0.30, lf_atten=0.15, prd_depth=0.12, axis_rot_deg=6.0),
    "severe":   dict(vlf_atten=0.45, lf_atten=0.25, prd_depth=0.20, axis_rot_deg=10.0),
}


# ==========================================================
# Shared signal utilities (mirrors baseline pipeline)
# ==========================================================

def bandpass_filter(signal):
    nyquist = 0.5 * FS
    low, high = LOWCUT / nyquist, HIGHCUT / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)


def zscore_beat(beat):
    mean = beat.mean(axis=1, keepdims=True)
    std = beat.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (beat - mean) / std


# ==========================================================
# 1. HRV spectral-band shift (needs the CONTINUOUS RR sequence)
# ==========================================================

def shift_rr_sequence(r_peaks, vlf_atten, lf_atten, fs_resample=4.0, seed=None):
    """
    Attenuate VLF/LF band power in a record's RR tachogram, calibrated to
    reported spaceflight HRV changes, and return a shifted RR-interval
    array aligned to the same beat indices as the input.

    r_peaks : 1D array of R-peak sample indices (from wfdb annotation).
    vlf_atten / lf_atten : fraction of power to remove from each band
        (0.30 = remove 30% of power in that band).

    Returns: shifted_rr (len = len(r_peaks) - 1), aligned with
             np.diff(r_peaks) i.e. shifted_rr[i] corresponds to the
             interval between r_peaks[i] and r_peaks[i+1].
    """
    rng = np.random.default_rng(seed)

    rr = np.diff(r_peaks) / FS                     # seconds
    beat_times = np.cumsum(rr)                      # time of each RR sample
    beat_times = beat_times - beat_times[0]          # start at 0

    if len(rr) < 20:
        # too short a record/segment to do meaningful spectral shaping
        return rr.copy()

    # Resample RR tachogram onto a uniform time grid (standard HRV practice)
    t_uniform = np.arange(0, beat_times[-1], 1.0 / fs_resample)
    if len(t_uniform) < 20:
        return rr.copy()

    cs = CubicSpline(beat_times, rr)
    rr_uniform = cs(t_uniform)

    # FFT, attenuate target bands, inverse FFT
    n = len(rr_uniform)
    mean_val = rr_uniform.mean()
    spec = np.fft.rfft(rr_uniform - mean_val)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_resample)

    for band, atten in [(VLF_BAND, vlf_atten), (LF_BAND, lf_atten)]:
        mask = (freqs >= band[0]) & (freqs < band[1])
        # power ~ amplitude^2, so scale amplitude by sqrt(1 - atten)
        spec[mask] *= np.sqrt(max(0.0, 1.0 - atten))

    # small random phase jitter in the attenuated bands to avoid a perfectly
    # deterministic transform (keeps the shift from being trivially invertible
    # / reduces the risk of the domain classifier keying on an artifact)
    band_mask_all = ((freqs >= VLF_BAND[0]) & (freqs < LF_BAND[1]))
    phase_jitter = rng.uniform(-0.05, 0.05, size=spec.shape) * band_mask_all
    spec = spec * np.exp(1j * phase_jitter)

    shifted_uniform = np.fft.irfft(spec, n) + mean_val

    # map back onto the original (non-uniform) beat times
    f_interp = interp1d(t_uniform, shifted_uniform, bounds_error=False,
                         fill_value=(shifted_uniform[0], shifted_uniform[-1]))
    shifted_rr = f_interp(beat_times)

    shifted_rr = np.clip(shifted_rr, RR_MIN, RR_MAX)
    return shifted_rr


# ==========================================================
# 2. Per-beat waveform shift (PRD modulation + axis rotation)
# ==========================================================

def apply_prd_modulation(beat, depth, freq_hz=0.08,
                          t_start_frac=0.55, t_end_frac=0.92,
                          rng=None):
    """
    Add a slow sinusoidal perturbation confined to the (approximate)
    T-wave segment of a beat, with a small inter-lead phase offset to
    proxy the "T-wave vector" oscillation reported as PRD in HDBR studies.

    beat : (2, L) array, already per-beat z-scored.
    depth : modulation amplitude, in z-score units (e.g. 0.12 = 12% of
            one standard deviation).
    """
    if rng is None:
        rng = np.random.default_rng()

    length = beat.shape[1]
    start = int(t_start_frac * length)
    end = int(t_end_frac * length)
    t = np.arange(end - start) / FS

    phase = rng.uniform(0, 2 * np.pi)
    mod_lead1 = depth * np.sin(2 * np.pi * freq_hz * t + phase)
    mod_lead2 = depth * np.sin(2 * np.pi * freq_hz * t + phase + np.pi / 6)

    shifted = beat.copy()
    shifted[0, start:end] += mod_lead1
    shifted[1, start:end] += mod_lead2
    return shifted


def apply_axis_rotation(beat, angle_deg, region, rng=None):
    """
    Rotate ONLY the T-wave segment of the two-lead signal relative to the
    (unrotated) rest of the beat, as a simplified proxy for QRS-T axis
    drift reported under HDBR.

    IMPORTANT: rotating the *entire* beat (QRS + T together) by the same
    angle is a rigid rotation and provably does NOT change the angle
    between the QRS vector and T vector - that angle is rotation-invariant.
    To actually simulate QRS-T angle widening you must rotate the T-wave
    relative to the QRS complex, which is what this function does (region
    should be the same T-wave window used by apply_prd_modulation).

    This remains a simplified proxy, not a clinically computed QRS-T angle
    (that requires 3D vectorcardiographic reconstruction, not 2 leads) -
    describe it as such in your paper.
    """
    if rng is None:
        rng = np.random.default_rng()
    start, end = region
    angle = rng.uniform(0.5, 1.0) * angle_deg * rng.choice([-1.0, 1.0])
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    shifted = beat.copy()
    shifted[:, start:end] = rot @ beat[:, start:end]
    return shifted


def shift_beat_waveform(beat, prd_depth, axis_rot_deg,
                         t_start_frac=0.55, t_end_frac=0.92, rng=None):
    length = beat.shape[1]
    region = (int(t_start_frac * length), int(t_end_frac * length))

    beat = apply_prd_modulation(beat, depth=prd_depth,
                                 t_start_frac=t_start_frac, t_end_frac=t_end_frac,
                                 rng=rng)
    beat = apply_axis_rotation(beat, angle_deg=axis_rot_deg, region=region, rng=rng)
    return beat


# ==========================================================
# 3. Full record processing: paired source/target beats
# ==========================================================

def process_record_paired(record_name, severity="moderate", seed=None):
    """
    Reproduces the baseline pipeline's beat extraction for one record, and
    additionally produces a shifted ("target domain") version of every
    beat + its RR features. Labels and patient IDs are identical between
    source and target - only the signal is perturbed.
    """
    params = SEVERITY_PRESETS[severity]
    rng = np.random.default_rng(seed)

    record = wfdb.rdrecord(str(MITDB_DIR / record_name))
    annotation = wfdb.rdann(str(MITDB_DIR / record_name), "atr")

    signals = record.p_signal
    lead1 = bandpass_filter(signals[:, 0])
    lead2 = bandpass_filter(signals[:, 1])
    filtered = np.column_stack((lead1, lead2))

    r_peaks = annotation.sample
    symbols = annotation.symbol

    # continuous RR shift computed once for the whole record
    shifted_rr_full = shift_rr_sequence(
        r_peaks, vlf_atten=params["vlf_atten"], lf_atten=params["lf_atten"],
        seed=seed
    )
    # shifted_rr_full[i] = shifted interval between r_peaks[i], r_peaks[i+1]

    beats_src, rr_src = [], []
    beats_tgt, rr_tgt = [], []
    labels, patient_ids = [], []

    for i in range(1, len(r_peaks) - 2):
        if symbols[i] not in VALID_BEATS:
            continue

        peak = r_peaks[i]
        start = peak - LEFT_WINDOW
        end = peak + RIGHT_WINDOW + 1
        if start < 0 or end > len(filtered):
            continue

        beat = filtered[start:end].T
        beat = zscore_beat(beat)

        # --- source (Earth-like) RR features, matches baseline pipeline ---
        previous_rr = (r_peaks[i] - r_peaks[i - 1]) / FS
        current_rr = (r_peaks[i + 1] - r_peaks[i]) / FS
        next_rr = (r_peaks[i + 2] - r_peaks[i + 1]) / FS
        rr_ratio = current_rr / previous_rr if previous_rr != 0 else 1.0

        # --- target (space-shifted) RR features, from shifted_rr_full ---
        prev_rr_s = shifted_rr_full[i - 1]
        curr_rr_s = shifted_rr_full[i]
        next_rr_s = shifted_rr_full[i + 1]
        rr_ratio_s = curr_rr_s / prev_rr_s if prev_rr_s != 0 else 1.0

        beat_shifted = shift_beat_waveform(
            beat, prd_depth=params["prd_depth"],
            axis_rot_deg=params["axis_rot_deg"], rng=rng
        )

        label = 0 if symbols[i] in NORMAL_BEATS else 1

        beats_src.append(beat)
        rr_src.append([previous_rr, current_rr, next_rr, rr_ratio])

        beats_tgt.append(beat_shifted)
        rr_tgt.append([prev_rr_s, curr_rr_s, next_rr_s, rr_ratio_s])

        labels.append(label)
        patient_ids.append(record_name)

    return (
        np.array(beats_src), np.array(rr_src),
        np.array(beats_tgt), np.array(rr_tgt),
        np.array(labels), np.array(patient_ids)
    )


def build_domain_dataset(records, severity="moderate", seed=42, cache_dir=None):
    """
    Build paired source (Earth) / target (space-shifted) datasets across
    all given records.

    Returns two dicts: source, target, each with keys
    X, RR, y, patient_ids, domain (0=source, 1=target).
    """
    cache_dir = Path(cache_dir) if cache_dir else (PROJECT_DIR / "datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"domain_dataset_{severity}.npz"

    if cache_path.exists():
        print(f"Loading cached domain dataset ({severity})...")
        data = np.load(cache_path, allow_pickle=True)
        source = dict(X=data["X_src"], RR=data["RR_src"], y=data["y"],
                      patient_ids=data["patient_ids"],
                      domain=np.zeros(len(data["y"]), dtype=np.int64))
        target = dict(X=data["X_tgt"], RR=data["RR_tgt"], y=data["y"],
                      patient_ids=data["patient_ids"],
                      domain=np.ones(len(data["y"]), dtype=np.int64))
        return source, target

    all_src_beats, all_src_rr = [], []
    all_tgt_beats, all_tgt_rr = [], []
    all_labels, all_patients = [], []

    for idx, record in enumerate(records):
        print(f"[{idx+1}/{len(records)}] Processing (paired) record {record}...")
        bs, rs, bt, rt, lbl, pid = process_record_paired(
            record, severity=severity, seed=seed + idx
        )
        all_src_beats.append(bs); all_src_rr.append(rs)
        all_tgt_beats.append(bt); all_tgt_rr.append(rt)
        all_labels.append(lbl); all_patients.append(pid)

    X_src = np.concatenate(all_src_beats, axis=0)
    RR_src = np.concatenate(all_src_rr, axis=0)
    X_tgt = np.concatenate(all_tgt_beats, axis=0)
    RR_tgt = np.concatenate(all_tgt_rr, axis=0)
    y = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patients, axis=0)

    np.savez_compressed(
        cache_path, X_src=X_src, RR_src=RR_src, X_tgt=X_tgt, RR_tgt=RR_tgt,
        y=y, patient_ids=patient_ids
    )
    print(f"Paired domain dataset cached at {cache_path}")

    source = dict(X=X_src, RR=RR_src, y=y, patient_ids=patient_ids,
                  domain=np.zeros(len(y), dtype=np.int64))
    target = dict(X=X_tgt, RR=RR_tgt, y=y, patient_ids=patient_ids,
                  domain=np.ones(len(y), dtype=np.int64))
    return source, target


# ==========================================================
# 4. Validation: confirm the shift actually matches literature magnitudes
# ==========================================================

def validate_band_power_shift(record_name, severity="moderate", seed=42):
    """
    Sanity-check figure/printout for your paper's methods section: verifies
    the VLF/LF power reduction achieved by shift_rr_sequence is in the
    ballpark of what was requested (and, by extension, of the cited
    literature magnitudes) for a real record's RR tachogram.
    """
    params = SEVERITY_PRESETS[severity]
    annotation = wfdb.rdann(str(MITDB_DIR / record_name), "atr")
    r_peaks = annotation.sample

    rr = np.diff(r_peaks) / FS
    shifted_rr = shift_rr_sequence(
        r_peaks, vlf_atten=params["vlf_atten"], lf_atten=params["lf_atten"],
        seed=seed
    )

    def band_power(rr_series, band, fs_resample=4.0):
        times = np.cumsum(rr_series) - rr_series[0]
        t_uniform = np.arange(0, times[-1], 1.0 / fs_resample)
        cs = CubicSpline(times, rr_series)
        sig = cs(t_uniform) - cs(t_uniform).mean()
        spec = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs_resample)
        mask = (freqs >= band[0]) & (freqs < band[1])
        return np.sum(np.abs(spec[mask]) ** 2)

    vlf_before = band_power(rr, VLF_BAND)
    vlf_after = band_power(shifted_rr, VLF_BAND)
    lf_before = band_power(rr, LF_BAND)
    lf_after = band_power(shifted_rr, LF_BAND)

    vlf_pct = 100 * (1 - vlf_after / vlf_before) if vlf_before > 0 else float("nan")
    lf_pct = 100 * (1 - lf_after / lf_before) if lf_before > 0 else float("nan")

    print(f"Record {record_name} | severity={severity}")
    print(f"  VLF power reduction: {vlf_pct:.1f}% (requested {params['vlf_atten']*100:.0f}%)")
    print(f"  LF  power reduction: {lf_pct:.1f}% (requested {params['lf_atten']*100:.0f}%)")
    print("  Reference range from literature (ULF band, 24h recordings): 22-52% / 13-54%")
    print("  (VLF/LF used here due to 30-min record length - see module docstring)")

    return dict(vlf_pct=vlf_pct, lf_pct=lf_pct)


if __name__ == "__main__":
    # Quick smoke test on a single record - requires ECG_Project/mitdb to
    # already exist (run ecg_arrhythmia_cnn.py first, or at least its
    # ensure_dataset() step, to download MIT-BIH).
    test_record = "100"
    if (MITDB_DIR / f"{test_record}.dat").exists():
        validate_band_power_shift(test_record, severity="moderate")

        bs, rs, bt, rt, lbl, pid = process_record_paired(test_record, severity="moderate")
        print(f"\nRecord {test_record}: {len(lbl)} beats extracted")
        print(f"Source beats shape: {bs.shape} | Target beats shape: {bt.shape}")
        print(f"Label distribution: {Counter(lbl)}")
        print(f"Mean |source - target| waveform diff: {np.mean(np.abs(bs - bt)):.4f}")
        print(f"Mean |source - target| RR feature diff: {np.mean(np.abs(rs - rt), axis=0)}")
    else:
        print(f"MIT-BIH record {test_record} not found at {MITDB_DIR}. "
              f"Run the baseline pipeline's ensure_dataset() first.")