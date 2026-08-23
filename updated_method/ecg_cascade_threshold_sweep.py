"""
ecg_cascade_threshold_sweep.py

Reuses the ALREADY-TRAINED Stage 1 / Stage 2 checkpoints (no retraining) to
sweep Stage 1's decision threshold across several operating points, showing
the S-recall vs V/N-precision trade-off explicitly instead of hiding it
behind one chosen threshold.

Why this matters for the paper: a single global threshold can't be
simultaneously lenient enough to catch subtle S beats and strict enough to
keep V/N precision clean (Stage 2's oracle performance is stable throughout
- the bottleneck is entirely Stage 1's one-threshold design). Reporting
this as an explicit table is more rigorous than picking one number and
hoping nobody asks about the trade-off - it also doubles as a legitimate
"operating point" story for the edge-deployment angle (different deployment
contexts would reasonably choose different points on this curve).

Run AFTER ecg_cascade_cnn.py has completed at least once (needs the saved
.pth files in ECG_Project/models/).
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

import ecg_multiclass_cnn as base
import ecg_cascade_cnn as cascade

device = cascade.device

# Thresholds spanning the range actually seen across your runs: 0.17 (lenient,
# best S recall), ~0.4-0.5 (rough middle), 0.90 (strict, best V/N precision)
SWEEP_THRESHOLDS = [0.17, 0.30, 0.50, 0.70, 0.90]


def rebuild_test_data():
    """
    Deterministically reproduces the exact same train/val/test split and RR
    scaling used during training (same SEED, same DS1/DS2 partition, same Q
    filter) - the scaler itself wasn't persisted, but since everything
    upstream is seeded and deterministic, refitting it on the same train
    split reproduces identical scaling.
    """
    base.ensure_dataset()
    X, RR, y, patient_ids = base.build_dataset()
    X, RR, y, patient_ids = cascade.filter_out_class(X, RR, y, patient_ids, "Q")

    (X_train, RR_train, y_train, X_val, RR_val, y_val,
     X_test, RR_test, y_test) = base.patient_wise_split(X, RR, y, patient_ids)

    rr_scaler = StandardScaler()
    rr_scaler.fit(RR_train)  # fit only, matches training-time scaling
    RR_val = rr_scaler.transform(RR_val)
    RR_test = rr_scaler.transform(RR_test)

    y_val_bin, y_val_sub = cascade.build_cascade_labels(y_val)
    y_test_bin, y_test_sub = cascade.build_cascade_labels(y_test)

    return (X_val, RR_val, y_val, y_val_bin, y_val_sub,
            X_test, RR_test, y_test, y_test_bin, y_test_sub)


def load_models():
    stage1_model = cascade.ECGClassifierBinary().to(device)
    stage1_model.load_state_dict(torch.load(
        cascade.MODEL_DIR / "cascade_stage1_binary.pth", map_location=device
    ))
    stage1_model.eval()

    stage2_model = cascade.ECGClassifierMultiClass(num_classes=len(cascade.SUBTYPE_CLASSES)).to(device)
    stage2_model.load_state_dict(torch.load(
        cascade.MODEL_DIR / "cascade_stage2_subtype.pth", map_location=device
    ))
    stage2_model.eval()

    return stage1_model, stage2_model


def run_cascade_at_threshold(stage1_model, stage2_model, X_test, RR_test, y_test, threshold):
    """End-to-end cascade evaluation at a single Stage 1 threshold."""
    final_preds = np.zeros(len(y_test), dtype=np.int64)

    with torch.no_grad():
        ecg_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        rr_t = torch.tensor(RR_test, dtype=torch.float32).to(device)

        s1_out = stage1_model(ecg_t, rr_t)
        s1_probs = torch.softmax(s1_out, dim=1).cpu().numpy()
        s1_pred = (s1_probs[:, 1] >= threshold).astype(int)

        abnormal_idx = np.where(s1_pred == 1)[0]
        if len(abnormal_idx) > 0:
            ecg_ab = ecg_t[abnormal_idx]
            rr_ab = rr_t[abnormal_idx]
            s2_out = stage2_model(ecg_ab, rr_ab)
            s2_pred = torch.argmax(s2_out, dim=1).cpu().numpy()
            subtype_to_aami_idx = {i: base.CLASS_TO_IDX[cascade.SUBTYPE_CLASSES[i]]
                                    for i in range(len(cascade.SUBTYPE_CLASSES))}
            for local_i, global_i in enumerate(abnormal_idx):
                final_preds[global_i] = subtype_to_aami_idx[s2_pred[local_i]]

    macro_f1 = f1_score(y_test, final_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, final_preds, average="weighted", zero_division=0)
    acc = accuracy_score(y_test, final_preds)
    per_class_f1 = f1_score(y_test, final_preds, average=None, labels=list(range(4)), zero_division=0)

    fp_rate_on_normal = (s1_pred[y_test == 0] == 1).mean()  # false-positive rate specifically

    return {
        "threshold": threshold, "accuracy": acc, "macro_f1": macro_f1,
        "weighted_f1": weighted_f1, "per_class_f1": per_class_f1,
        "abnormal_flag_rate": s1_pred.mean(), "fp_rate_on_normal": fp_rate_on_normal
    }


def main():
    print("Rebuilding test data (deterministic, no reprocessing of raw signals needed "
          "beyond the cached dataset)...")
    (X_val, RR_val, y_val, y_val_bin, y_val_sub,
     X_test, RR_test, y_test, y_test_bin, y_test_sub) = rebuild_test_data()

    print("Loading trained Stage 1 / Stage 2 checkpoints...")
    stage1_model, stage2_model = load_models()

    print(f"\nSweeping {len(SWEEP_THRESHOLDS)} thresholds on the TEST set "
          f"(no retraining - using existing checkpoints)...\n")

    results = []
    for t in SWEEP_THRESHOLDS:
        r = run_cascade_at_threshold(stage1_model, stage2_model, X_test, RR_test, y_test, t)
        results.append(r)

    print("=" * 100)
    print("THRESHOLD SENSITIVITY TABLE (end-to-end cascade, test set)")
    print("=" * 100)
    header = f"{'Threshold':<11}{'Accuracy':<10}{'MacroF1':<10}{'WeightedF1':<12}" \
             f"{'N-F1':<8}{'S-F1':<8}{'V-F1':<8}{'F-F1':<8}{'AbnormalFlag%':<15}{'FP-on-N%'}"
    print(header)
    for r in results:
        pcf = r["per_class_f1"]
        print(f"{r['threshold']:<11.2f}{r['accuracy']:<10.4f}{r['macro_f1']:<10.4f}"
              f"{r['weighted_f1']:<12.4f}{pcf[0]:<8.3f}{pcf[1]:<8.3f}{pcf[2]:<8.3f}{pcf[3]:<8.3f}"
              f"{r['abnormal_flag_rate']*100:<15.1f}{r['fp_rate_on_normal']*100:.1f}")

    print("\nInterpretation for the paper:")
    print("- Low threshold (e.g. 0.17): best S/F recall, worst N/V precision - use where missing")
    print("  a rare arrhythmia subtype is costlier than false alarms (e.g. mission-critical screening).")
    print("- High threshold (e.g. 0.90): best N/V precision, worst S recall - use where false-alarm")
    print("  rate must stay low (e.g. resource-constrained onboard triage with human review downstream).")
    print("- Report this full table in the paper rather than a single cherry-picked operating point -")
    print("  it demonstrates the trade-off is characterized, not hidden.")

    # Save as a figure too
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in results]
    plt.figure(figsize=(9, 6))
    for i, cls in enumerate(cascade.AAMI_CLASSES_NO_Q):
        plt.plot(thresholds, [r["per_class_f1"][i] for r in results], marker="o", label=f"{cls} F1")
    plt.plot(thresholds, [r["macro_f1"] for r in results], marker="s", linestyle="--",
              color="black", label="Macro F1")
    plt.xlabel("Stage 1 Decision Threshold")
    plt.ylabel("F1 Score")
    plt.title("End-to-End Cascade: Threshold Sensitivity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = cascade.FIGURE_DIR / "threshold_sensitivity.png"
    plt.savefig(out_path)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
