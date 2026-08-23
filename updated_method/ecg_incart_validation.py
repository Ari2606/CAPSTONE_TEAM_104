"""
ecg_incart_validation.py

Real (not synthetic) cross-database validation: evaluate the MIT-BIH-trained
model, zero-shot and TENT-adapted, on the St. Petersburg INCART 12-lead
Arrhythmia Database - genuinely different recording equipment, population,
and protocol from MIT-BIH. This directly answers the project's most-repeated
honest limitation ("the space-shift is synthetic, not real") by testing
whether TENT also helps under REAL, unseen distribution shift.

Reuses the already-built and validated TENT machinery from
ecg_tent_adaptation.py (freeze_for_tent, run_tent_adaptation,
evaluate_current_model) - only the data loading below is new.

DESIGN CHOICES TO STATE PLAINLY IN THE PAPER, NOT GLOSS OVER:
-----------------------------------------------------------------
1. LEAD SELECTION: INCART provides 12 leads; the model expects 2 (matching
   MIT-BIH's typical config). We select Lead II and V1 as the closest
   available approximation (a limb-like lead + a precordial lead). This is
   a design choice, not a neutral default - a different pair could give
   different results, and that should be disclosed.
2. RESAMPLING: INCART is natively 257 Hz; the model expects 360 Hz (MIT-BIH's
   rate). We resample INCART's signal to 360 Hz BEFORE beat extraction, so
   window semantics (samples relative to R-peak) match what the model was
   trained on.
3. KNOWN COMPOSITION DIFFERENCE: none of the INCART patients had pacemakers
   (consistent with this project's own Q-dropping decision), and INCART
   skews more V-beat-heavy than MIT-BIH. This validates N/V generalization
   well; say less definitively about S given INCART's composition - state
   this as a limitation, don't oversell coverage.
4. RR SCALING: RR features are scaled using the scaler fit on MIT-BIH's
   DS1 training data (the SAME scaler the model was trained with) - not a
   freshly-fit INCART-specific scaler, since the model's weights expect
   that specific input distribution. This is the correct choice for a
   genuine zero-shot generalization test.

First run needs internet access to PhysioNet to download INCART (~75
records, 12-lead - larger than MIT-BIH, budget extra time). Cached after.
"""

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler

import ecg_multiclass_cnn as base
import ecg_dann_space_shift as dann
import ecg_tent_adaptation as tent

device = dann.device
print("Device:", device)

INCART_DIR = base.PROJECT_DIR / "incartdb"
INCART_CACHE = base.PROJECT_DIR / "datasets" / "incart_processed.npz"
INCART_FS = 257            # native INCART sampling rate
TARGET_FS = base.FS        # 360 Hz - match the MIT-BIH-trained model
LEAD_NAMES_WANTED = ["II", "V1"]  # design choice - see module docstring


# ==========================================================
# Data acquisition
# ==========================================================

def ensure_incart_downloaded():
    INCART_DIR.mkdir(parents=True, exist_ok=True)
    record_list = wfdb.get_record_list("incartdb")
    print(f"INCART database has {len(record_list)} records available.")

    already = all((INCART_DIR / f"{r}.dat").exists() for r in record_list)
    if already:
        print("INCART dataset already present.")
        return record_list

    print("Downloading INCART dataset (larger than MIT-BIH - budget extra time)...")
    for r in record_list:
        if not (INCART_DIR / f"{r}.dat").exists():
            wfdb.dl_database("incartdb", dl_dir=str(INCART_DIR), records=[r])
    print("Download complete.")
    return record_list


def select_lead_indices(record):
    """Find indices of the wanted lead names in this record's channel list.
    Returns None for a lead if not found (caller should skip the record)."""
    sig_names = [n.strip().upper() for n in record.sig_name]
    indices = []
    for wanted in LEAD_NAMES_WANTED:
        wanted_u = wanted.upper()
        matches = [i for i, n in enumerate(sig_names) if n == wanted_u]
        if not matches:
            matches = [i for i, n in enumerate(sig_names) if wanted_u in n]
        indices.append(matches[0] if matches else None)
    return indices


def process_incart_record(record_name):
    """Mirrors ecg_multiclass_cnn.py's process_record logic, adapted for
    INCART's 12-lead/257Hz format: select 2 leads, resample to 360Hz, then
    reuse the EXACT SAME filtering/beat-extraction/labeling code as MIT-BIH
    (base.bandpass_filter, base.zscore_beat, base.SYMBOL_TO_AAMI) so
    preprocessing is genuinely identical across databases, not just similar."""
    record = wfdb.rdrecord(str(INCART_DIR / record_name))
    annotation = wfdb.rdann(str(INCART_DIR / record_name), "atr")

    lead_idx = select_lead_indices(record)
    if None in lead_idx:
        print(f"  Skipping {record_name}: couldn't find leads {LEAD_NAMES_WANTED} "
              f"(available: {record.sig_name})")
        return None

    signals = record.p_signal[:, lead_idx]  # (n_samples, 2)

    n_samples_new = int(round(signals.shape[0] * TARGET_FS / INCART_FS))
    signals_resampled = np.zeros((n_samples_new, 2))
    for ch in range(2):
        signals_resampled[:, ch] = resample_poly(signals[:, ch], TARGET_FS, INCART_FS)

    lead1 = base.bandpass_filter(signals_resampled[:, 0])
    lead2 = base.bandpass_filter(signals_resampled[:, 1])
    filtered = np.column_stack((lead1, lead2))

    r_peaks = np.round(annotation.sample * TARGET_FS / INCART_FS).astype(int)
    symbols = annotation.symbol

    beats, rr_features, labels = [], [], []
    for i in range(1, len(r_peaks) - 2):
        if symbols[i] not in base.VALID_BEATS:
            continue
        peak = r_peaks[i]
        start = peak - base.LEFT_WINDOW
        end = peak + base.RIGHT_WINDOW + 1
        if start < 0 or end > len(filtered):
            continue

        beat = filtered[start:end].T
        beat = base.zscore_beat(beat)

        previous_rr = (r_peaks[i] - r_peaks[i - 1]) / TARGET_FS
        current_rr = (r_peaks[i + 1] - r_peaks[i]) / TARGET_FS
        next_rr = (r_peaks[i + 2] - r_peaks[i + 1]) / TARGET_FS
        rr_ratio = current_rr / previous_rr if previous_rr != 0 else 1.0

        aami_class = base.SYMBOL_TO_AAMI[symbols[i]]
        label = 0 if aami_class == "N" else 1  # binary, matching the DANN/TENT task

        beats.append(beat)
        rr_features.append([previous_rr, current_rr, next_rr, rr_ratio])
        labels.append(label)

    if len(labels) == 0:
        return None
    return np.array(beats), np.array(rr_features), np.array(labels)


def build_incart_dataset():
    if INCART_CACHE.exists():
        print("Loading cached INCART dataset...")
        data = np.load(INCART_CACHE)
        return data["X"], data["RR"], data["y"]

    record_list = ensure_incart_downloaded()
    all_beats, all_rr, all_labels = [], [], []
    skipped = []

    for i, r in enumerate(record_list):
        print(f"[{i+1}/{len(record_list)}] Processing INCART record {r}...")
        result = process_incart_record(r)
        if result is None:
            skipped.append(r)
            continue
        b, rr, lbl = result
        all_beats.append(b); all_rr.append(rr); all_labels.append(lbl)

    if skipped:
        print(f"\nSkipped {len(skipped)} records (lead mismatch or no valid beats): {skipped}")

    X = np.concatenate(all_beats, axis=0)
    RR = np.concatenate(all_rr, axis=0)
    y = np.concatenate(all_labels, axis=0)

    INCART_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INCART_CACHE, X=X, RR=RR, y=y)
    print(f"INCART dataset cached: {len(y)} beats, {100*y.mean():.1f}% abnormal")

    return X, RR, y


def get_mitbih_rr_scaler():
    """Refits the SAME scaler the model was trained with (fit on MIT-BIH
    DS1 training RR features, seed-deterministic) - needed since INCART's
    RR features must be scaled the way the model expects, not freshly
    normalized to INCART's own distribution."""
    print("\nRefitting the MIT-BIH training-time RR scaler (for correct zero-shot scaling)...")
    base.ensure_dataset()
    X, RR, y, patient_ids = base.build_dataset()
    (X_train, RR_train, y_train, X_val, RR_val, y_val,
     X_test, RR_test, y_test) = base.patient_wise_split(X, RR, y, patient_ids)
    scaler = StandardScaler()
    scaler.fit(RR_train)
    return scaler


# ==========================================================
# Main
# ==========================================================

def main():
    X_incart, RR_incart_raw, y_incart = build_incart_dataset()
    print(f"\nINCART: {len(y_incart)} beats | Abnormal rate: {100*y_incart.mean():.1f}% "
          f"(MIT-BIH DS1 reference: ~9.7%)")

    scaler = get_mitbih_rr_scaler()
    RR_incart = scaler.transform(RR_incart_raw)

    print("\nLoading pretrained source-only model (trained on MIT-BIH only, never seen INCART)...")
    model = dann.DANNModel().to(device)
    model.load_state_dict(torch.load(dann.MODEL_DIR / "dann_source_only.pth", map_location=device))

    baseline = tent.evaluate_current_model(model, X_incart, RR_incart, y_incart)
    print(f"\n--- Zero-shot on INCART (no adaptation) ---")
    print(f"Accuracy: {baseline['accuracy']:.4f} | F1: {baseline['f1']:.4f} | "
          f"Predicted abnormal rate: {baseline['abnormal_rate']:.4f}")

    print("\n" + "=" * 60)
    print("APPLYING TENT ON REAL CROSS-DATABASE SHIFT (INCART)")
    print("=" * 60)
    tent.freeze_for_tent(model)
    tent.verify_freezing(model)
    adapted_model, history = tent.run_tent_adaptation(model, X_incart, RR_incart, y_incart)
    final = tent.evaluate_current_model(adapted_model, X_incart, RR_incart, y_incart)

    print("\n" + "=" * 60)
    print("RESULTS: Real Cross-Database Validation (INCART)")
    print("=" * 60)
    print(f"{'Condition':<28}{'Accuracy':<12}{'F1':<12}{'Abnormal Rate'}")
    print(f"{'Zero-shot (no adaptation)':<28}{baseline['accuracy']:<12.4f}"
          f"{baseline['f1']:<12.4f}{baseline['abnormal_rate']:.4f}")
    print(f"{'TENT-adapted':<28}{final['accuracy']:<12.4f}"
          f"{final['f1']:<12.4f}{final['abnormal_rate']:.4f}")

    true_rate = y_incart.mean()
    print(f"\nTrue INCART abnormal rate: {true_rate:.4f}")
    print("If TENT's predicted rate moved closer to this AND F1 improved/held, that's a "
          "genuine finding: the same adaptation mechanism validated on synthetic space-shift "
          "also helps under real, unseen cross-database distribution shift.")

    torch.save(adapted_model.state_dict(), dann.MODEL_DIR / "tent_adapted_incart.pth")
    print(f"\nAdapted model saved to {dann.MODEL_DIR / 'tent_adapted_incart.pth'}")


if __name__ == "__main__":
    main()
