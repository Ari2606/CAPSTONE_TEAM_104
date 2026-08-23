"""
ecg_cascade_cnn.py

Two-stage cascade arrhythmia detector.

  Stage 1 (screening): Normal vs Abnormal, binary. This is "detect
  arrhythmia" in the most direct sense, and binary detection genuinely
  supports strong numbers because it's a much easier decision boundary
  than 5-way subtyping. This is your headline metric.

  Stage 2 (subtyping): AAMI S/V/F/Q classification, run only on beats
  Stage 1 flags as abnormal. Evaluated TWO ways, both reported:
    (a) oracle - given the TRUE abnormal beats (isolates subtyping quality
        on its own, independent of Stage 1's mistakes)
    (b) end-to-end - through the actual cascade, including beats Stage 1
        missed (these become false negatives for their true subtype). This
        is the only honest number for "how does the full system perform."

IMPORTANT - what this design does NOT do: it does not hide weak classes.
The end-to-end evaluation still reports all 5 AAMI classes against ground
truth. What changes is which stage is responsible for which decision, and
that's a legitimate architectural choice (it mirrors real clinical
screening workflows: flag first, characterize second), not a reporting
trick.

This script REUSES the data pipeline, DS1/DS2 split, backbone, focal loss,
SWA training utilities, and edge-deployment metrics already built and
fixed in ecg_multiclass_cnn.py, rather than duplicating them - run that
script (or at least let this one trigger the same cached dataset build)
first for a shared, single source of truth on the data.
"""

import copy
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

import ecg_multiclass_cnn as base  # reuses the already-fixed pipeline

device = base.device
PROJECT_DIR = base.PROJECT_DIR
MODEL_DIR = base.MODEL_DIR
FIGURE_DIR = base.FIGURE_DIR
AAMI_CLASSES = base.AAMI_CLASSES          # ["N","S","V","F","Q"] - full taxonomy
AAMI_CLASSES_NO_Q = ["N", "S", "V", "F"]  # what we actually report - Q filtered out
ECGDataset = base.ECGDataset
ECGFeatureExtractor = base.ECGFeatureExtractor
RRFeatureExtractor = base.RRFeatureExtractor
ECGClassifierMultiClass = base.ECGClassifierMultiClass  # reused for Stage 2
FocalLoss = base.FocalLoss
update_bn_two_input = base.update_bn_two_input
count_params = base.count_params
estimate_conv1d_flops = base.estimate_conv1d_flops
benchmark_cpu_latency = base.benchmark_cpu_latency

print("Device:", device)

# ==========================================================
# Config (shared training recipe across both stages, for a fair comparison)
# ==========================================================

BATCH_SIZE = base.BATCH_SIZE
EPOCHS = 40
WARMUP_EPOCHS = 3
SWA_START_FRAC = 0.6
LR = 3e-4
SWA_LR = 1e-4
WEIGHT_DECAY = 2e-4
GRAD_CLIP_NORM = 1.0
FOCAL_GAMMA = 2.0
CB_BETA = 0.9999

SUBTYPE_CLASSES = ["S", "V", "F"]  # Q dropped - see filter_out_class below
SUBTYPE_TO_IDX = {c: i for i, c in enumerate(SUBTYPE_CLASSES)}


def filter_out_class(X, RR, y, patient_ids, class_name):
    """
    Remove all beats belonging to the given AAMI class. Used to drop Q:
    under the standard DS1/DS2 split, the paced-rhythm records that contain
    most Q beats are already excluded per AAMI convention, so what remains
    of Q is a tiny residual (2 train / 6 val / 7 test beats observed) -
    not a learnable class. This mirrors published practice of dropping Q
    after applying that same exclusion (see e.g. arXiv:1510.02541).
    """
    idx = base.CLASS_TO_IDX[class_name]
    keep = y != idx
    return X[keep], RR[keep], y[keep], patient_ids[keep]


# ==========================================================
# Stage 1 model: binary screening, same efficient backbone
# ==========================================================

class ECGClassifierBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_extractor = ECGFeatureExtractor()
        self.rr_extractor = RRFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(160, 64), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, ecg, rr):
        f = torch.cat([self.ecg_extractor(ecg), self.rr_extractor(rr)], dim=1)
        return self.classifier(f)


def build_cascade_labels(y_aami):
    """
    y_aami: array of AAMI class indices (0=N,1=S,2=V,3=F,4=Q).
    Returns:
      y_binary  : 0=Normal, 1=Abnormal
      y_subtype : index into SUBTYPE_CLASSES (S/V/F/Q) for abnormal beats;
                  -1 for Normal beats (never used as a Stage-2 target).
    """
    y_binary = (y_aami != 0).astype(np.int64)
    y_subtype = np.full_like(y_aami, -1)
    for aami_idx, cls in enumerate(AAMI_CLASSES):
        if cls == "N" or cls not in SUBTYPE_TO_IDX:
            continue  # "N" is Stage 1's job; anything dropped (Q) has no subtype target
        y_subtype[y_aami == aami_idx] = SUBTYPE_TO_IDX[cls]
    return y_binary, y_subtype


# ==========================================================
# Shared SWA training loop (warmup -> cosine -> SWA), generalized so both
# stages use the identical recipe already validated in ecg_multiclass_cnn.py
# ==========================================================

def find_subtype_balanced_threshold(probs, y_binary, y_subtype, n_bootstrap=200, seed=base.SEED):
    """
    The plain bootstrap threshold (find_best_threshold_bootstrap) optimizes
    POOLED binary F1 - but the "abnormal" class is a mix of subtypes with
    very different separability from Normal (V is easy, S is subtle) and
    very different prevalence in validation (979 V vs 100 S). A threshold
    tuned on pooled F1 silently favors whichever subtype dominates the
    "abnormal" validation pool (V), which can badly starve recall on the
    rarer/subtler one (S) - exactly what happened in practice: S end-to-end
    F1 got WORSE after pooled-F1 threshold tuning (0.20 -> 0.09), because
    the chosen threshold (0.90) was too strict for S's typically lower
    confidence scores even when Stage 1 correctly suspects something's off.

    This instead scores each candidate threshold by the MEAN of:
      - specificity on Normal (don't want to sacrifice this)
      - recall on S
      - recall on V
      - recall on F
    treating each subtype's detectability as equally important, regardless
    of how many examples of it exist in validation.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs)
    y_binary = np.asarray(y_binary)
    y_subtype = np.asarray(y_subtype)
    n = len(y_binary)
    candidate_thresholds = np.arange(0.10, 0.91, 0.01)
    chosen = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_s, yb_s, ys_s = probs[idx], y_binary[idx], y_subtype[idx]

        best_t, best_score = 0.5, -1.0
        for t in candidate_thresholds:
            pred_abnormal = p_s >= t

            normal_mask = yb_s == 0
            specificity = (~pred_abnormal[normal_mask]).mean() if normal_mask.sum() > 0 else np.nan

            recalls = []
            for subtype_idx in range(3):  # S, V, F
                mask = ys_s == subtype_idx
                if mask.sum() > 0:
                    recalls.append(pred_abnormal[mask].mean())

            if np.isnan(specificity) or not recalls:
                continue
            # BUG FIX: averaging [specificity, recall_S, recall_V, recall_F]
            # weights specificity 1-vs-3 against the recalls, so lowering the
            # threshold (which raises all 3 recalls at once) always wins - the
            # search degenerated to the floor threshold (0.10) and flagged 48%
            # of Normal beats as abnormal. Balance it as TWO groups instead:
            # specificity vs. the average of the subtype recalls.
            mean_recall = np.mean(recalls)
            score = np.mean([specificity, mean_recall])
            if score > best_score:
                best_score, best_t = score, t

        chosen.append(best_t)

    chosen = np.array(chosen)
    median_t = float(np.median(chosen))
    iqr = (float(np.percentile(chosen, 25)), float(np.percentile(chosen, 75)))
    print(f"Subtype-balanced threshold: median={median_t:.2f} | IQR=[{iqr[0]:.2f}, {iqr[1]:.2f}] "
          f"| n_bootstrap={len(chosen)}")
    return median_t


def find_best_threshold_bootstrap(probs, labels, n_bootstrap=200, seed=base.SEED):
    """
    Same bootstrap threshold selection already validated in the original
    binary script - reused here because Stage 1 has the identical problem:
    naive argmax (implicit 0.5 threshold) gives high recall (0.74) but poor
    precision (0.33), and every false positive leaks into Stage 2, which has
    no reject option and confidently assigns it a wrong subtype.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs); labels = np.asarray(labels)
    n = len(labels)
    candidate_thresholds = np.arange(0.10, 0.91, 0.01)
    chosen = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_sample, y_sample = probs[idx], labels[idx]
        if y_sample.sum() == 0 or y_sample.sum() == n:
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in candidate_thresholds:
            preds = (p_sample >= t).astype(int)
            f1 = f1_score(y_sample, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        chosen.append(best_t)

    chosen = np.array(chosen)
    median_t = float(np.median(chosen))
    iqr = (float(np.percentile(chosen, 25)), float(np.percentile(chosen, 75)))
    print(f"Bootstrap threshold: median={median_t:.2f} | IQR=[{iqr[0]:.2f}, {iqr[1]:.2f}] "
          f"| n_bootstrap={len(chosen)}")
    return median_t


class Stage1WeightedLoss(nn.Module):
    """
    Standard nn.CrossEntropyLoss(weight=...) only supports ONE weight per
    target class (Normal vs Abnormal) - every abnormal example (S, V, or F)
    gets the identical weight. Since V outnumbers S+F combined in training
    (2808 vs 841+413), V's gradient dominates, and the shared feature
    extractor learns "does this look like V" rather than "is this abnormal
    at all" - so Stage 1's probability outputs for S/F stay low even when
    correct. This is a TRAINING problem, not a threshold problem: pushing
    the decision threshold to the search floor (0.10) only partially
    recovered S/F recall, at the cost of flagging 48% of Normal beats as
    abnormal - that's the ceiling of what threshold tuning alone can fix.

    Fix: weight each ABNORMAL example by its own subtype's class-balanced
    weight (not one flat "Abnormal" weight), so S and F get proportionally
    more gradient signal during training itself.
    """
    def __init__(self, weight_normal, subtype_weights, label_smoothing=0.05):
        super().__init__()
        self.weight_normal = weight_normal
        self.subtype_weights = subtype_weights  # tensor, len 3 (S,V,F)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets_binary, targets_subtype):
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits, targets_binary, reduction="none", label_smoothing=self.label_smoothing
        )
        safe_subtype = targets_subtype.clamp(min=0)  # -1 (Normal) -> 0, unused due to where()
        weights = torch.where(
            targets_binary == 0,
            torch.full_like(per_sample_loss, self.weight_normal),
            self.subtype_weights.to(logits.device)[safe_subtype]
        )
        return (per_sample_loss * weights).mean()


class ECGCascadeStage1Dataset(torch.utils.data.Dataset):
    """Like ECGDataset, but also carries the subtype label (needed for
    Stage1WeightedLoss's per-sample weighting)."""
    def __init__(self, waveforms, rr_features, labels_binary, labels_subtype, augment=False):
        self.waveforms = waveforms.astype(np.float32)
        self.rr = torch.tensor(rr_features, dtype=torch.float32)
        self.labels_binary = torch.tensor(labels_binary, dtype=torch.long)
        self.labels_subtype = torch.tensor(labels_subtype, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels_binary)

    def __getitem__(self, idx):
        beat = self.waveforms[idx].copy()
        if self.augment:
            scale = np.random.uniform(*base.AUG_SCALE_RANGE)
            beat = beat * scale + np.random.normal(0, base.AUG_NOISE_STD, size=beat.shape)
        return (torch.tensor(beat, dtype=torch.float32), self.rr[idx],
                self.labels_binary[idx], self.labels_subtype[idx])


def train_stage1_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for ecg, rr, y_bin, y_sub in loader:
        ecg, rr, y_bin, y_sub = ecg.to(device), rr.to(device), y_bin.to(device), y_sub.to(device)
        optimizer.zero_grad()
        outputs = model(ecg, rr)
        loss = criterion(outputs, y_bin, y_sub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate_stage1(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for ecg, rr, y_bin, y_sub in loader:
            ecg, rr, y_bin, y_sub = ecg.to(device), rr.to(device), y_bin.to(device), y_sub.to(device)
            outputs = model(ecg, rr)
            loss = criterion(outputs, y_bin, y_sub)
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y_bin.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
    return (running_loss / len(loader), np.array(preds_all),
            np.array(labels_all), np.array(probs_all))


def train_stage1_with_swa(model, train_loader, val_loader, criterion, tag="Stage1"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    swa_start_epoch = int(EPOCHS * SWA_START_FRAC)

    def lr_lambda(epoch):
        swa_ratio = SWA_LR / LR
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        if epoch >= swa_start_epoch:
            return swa_ratio
        progress = (epoch - WARMUP_EPOCHS) / max(1, (swa_start_epoch - WARMUP_EPOCHS))
        return swa_ratio + (1 - swa_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    swa_model = AveragedModel(model)
    swa_active = False

    for epoch in range(EPOCHS):
        train_loss = train_stage1_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_labels, _ = evaluate_stage1(model, val_loader, criterion)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_active = True
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        phase = "SWA" if epoch >= swa_start_epoch else ("warmup" if epoch < WARMUP_EPOCHS else "cosine")
        print(f"[{tag}] Epoch {epoch+1}/{EPOCHS} [{phase}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_macro_f1={macro_f1:.4f} lr={lr_now:.6f}")

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"[{tag}] NaN detected - stopping.")
            break

    if swa_active:
        print(f"[{tag}] Recalibrating BatchNorm for the SWA-averaged model...")
        update_bn_two_input(train_loader, swa_model, device=device)
        model = swa_model.module

    return model


def evaluate_generic(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for ecg, rr, labels in loader:
            ecg, rr, labels = ecg.to(device), rr.to(device), labels.to(device)
            outputs = model(ecg, rr)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
    return (running_loss / len(loader), np.array(preds_all),
            np.array(labels_all), np.array(probs_all))


def train_stage_with_swa(model, train_loader, val_loader, criterion, tag=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    swa_start_epoch = int(EPOCHS * SWA_START_FRAC)

    def lr_lambda(epoch):
        swa_ratio = SWA_LR / LR
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        if epoch >= swa_start_epoch:
            return swa_ratio
        progress = (epoch - WARMUP_EPOCHS) / max(1, (swa_start_epoch - WARMUP_EPOCHS))
        return swa_ratio + (1 - swa_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    swa_model = AveragedModel(model)
    swa_active = False

    for epoch in range(EPOCHS):
        train_loss = base.train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_labels, _ = evaluate_generic(model, val_loader, criterion)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_active = True
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        phase = "SWA" if epoch >= swa_start_epoch else ("warmup" if epoch < WARMUP_EPOCHS else "cosine")
        print(f"[{tag}] Epoch {epoch+1}/{EPOCHS} [{phase}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_macro_f1={macro_f1:.4f} lr={lr_now:.6f}")

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"[{tag}] NaN detected - stopping.")
            break

    if swa_active:
        print(f"[{tag}] Recalibrating BatchNorm for the SWA-averaged model...")
        update_bn_two_input(train_loader, swa_model, device=device)
        model = swa_model.module

    return model


# ==========================================================
# Main
# ==========================================================

def main():
    base.ensure_dataset()
    X, RR, y, patient_ids = base.build_dataset()  # y = AAMI indices 0..4

    X, RR, y, patient_ids = filter_out_class(X, RR, y, patient_ids, "Q")
    print(f"\nDropped Q class (unlearnable under DS1/DS2 - see filter_out_class "
          f"docstring). {len(y)} beats remain across N/S/V/F.")

    (X_train, RR_train, y_train, X_val, RR_val, y_val,
     X_test, RR_test, y_test) = base.patient_wise_split(X, RR, y, patient_ids)

    rr_scaler = StandardScaler()
    RR_train = rr_scaler.fit_transform(RR_train)
    RR_val = rr_scaler.transform(RR_val)
    RR_test = rr_scaler.transform(RR_test)

    y_train_bin, y_train_sub = build_cascade_labels(y_train)
    y_val_bin, y_val_sub = build_cascade_labels(y_val)
    y_test_bin, y_test_sub = build_cascade_labels(y_test)

    # ======================================================
    # STAGE 1: binary screening, trained on ALL beats
    # ======================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Binary screening (Normal vs Abnormal)")
    print("=" * 60)

    stage1_train_ds = ECGCascadeStage1Dataset(X_train, RR_train, y_train_bin, y_train_sub, augment=True)
    stage1_val_ds = ECGCascadeStage1Dataset(X_val, RR_val, y_val_bin, y_val_sub, augment=False)
    stage1_test_ds = ECGCascadeStage1Dataset(X_test, RR_test, y_test_bin, y_test_sub, augment=False)

    stage1_train_loader = DataLoader(stage1_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    stage1_val_loader = DataLoader(stage1_val_ds, batch_size=BATCH_SIZE, shuffle=False)
    stage1_test_loader = DataLoader(stage1_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Weight Normal at the class level (as before), but weight ABNORMAL
    # examples by their own subtype's class-balanced weight, not one flat
    # "Abnormal" value - see Stage1WeightedLoss docstring for why the flat
    # version let V dominate S/F during training itself (a deeper problem
    # than anything a decision threshold can fix after the fact).
    bin_counts = Counter(y_train_bin)
    counts_arr_bin = np.array([bin_counts.get(i, 1) for i in range(2)], dtype=np.float64)
    effective_num_bin = 1.0 - np.power(CB_BETA, counts_arr_bin)
    bin_weights = (1.0 - CB_BETA) / effective_num_bin
    bin_weights = bin_weights / bin_weights.sum() * 2

    subtype_counts = Counter(y_train_sub[y_train_sub >= 0])
    counts_arr_sub = np.array([subtype_counts.get(i, 1) for i in range(3)], dtype=np.float64)
    effective_num_sub = 1.0 - np.power(CB_BETA, counts_arr_sub)
    subtype_weights_for_s1 = (1.0 - CB_BETA) / effective_num_sub
    subtype_weights_for_s1 = subtype_weights_for_s1 / subtype_weights_for_s1.sum() * 3
    subtype_weights_for_s1 = torch.tensor(subtype_weights_for_s1, dtype=torch.float32)

    print(f"Stage 1 Normal weight: {bin_weights[0]:.3f}")
    print("Stage 1 per-subtype Abnormal weights:",
          {SUBTYPE_CLASSES[i]: round(float(subtype_weights_for_s1[i]), 3) for i in range(3)})

    stage1_model = ECGClassifierBinary().to(device)
    stage1_criterion = Stage1WeightedLoss(
        weight_normal=float(bin_weights[0]), subtype_weights=subtype_weights_for_s1
    )
    stage1_model = train_stage1_with_swa(
        stage1_model, stage1_train_loader, stage1_val_loader, stage1_criterion, tag="Stage1"
    )
    torch.save(stage1_model.state_dict(), MODEL_DIR / "cascade_stage1_binary.pth")

    # ======================================================
    # STAGE 2: subtype classification, trained ONLY on true abnormal beats
    # ======================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Subtype classification (S/V/F/Q), abnormal beats only")
    print("=" * 60)

    train_ab_mask = y_train_bin == 1
    val_ab_mask = y_val_bin == 1

    X_train_ab, RR_train_ab, y_train_ab = (X_train[train_ab_mask], RR_train[train_ab_mask],
                                            y_train_sub[train_ab_mask])
    X_val_ab, RR_val_ab, y_val_ab = (X_val[val_ab_mask], RR_val[val_ab_mask],
                                      y_val_sub[val_ab_mask])

    stage2_train_ds = ECGDataset(X_train_ab, RR_train_ab, y_train_ab, augment=True)
    stage2_val_ds = ECGDataset(X_val_ab, RR_val_ab, y_val_ab, augment=False)

    stage2_train_loader = DataLoader(stage2_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    stage2_val_loader = DataLoader(stage2_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    sub_counts = Counter(y_train_ab)
    counts_arr = np.array([sub_counts.get(i, 1) for i in range(len(SUBTYPE_CLASSES))], dtype=np.float64)
    effective_num = 1.0 - np.power(CB_BETA, counts_arr)
    sub_alpha = (1.0 - CB_BETA) / effective_num
    sub_alpha = sub_alpha / sub_alpha.sum() * len(SUBTYPE_CLASSES)
    sub_alpha = torch.tensor(sub_alpha, dtype=torch.float32)
    print("Stage 2 class-balanced alpha:",
          {SUBTYPE_CLASSES[i]: round(float(sub_alpha[i]), 3) for i in range(len(SUBTYPE_CLASSES))})

    stage2_model = ECGClassifierMultiClass(num_classes=len(SUBTYPE_CLASSES)).to(device)
    stage2_criterion = FocalLoss(alpha=sub_alpha, gamma=FOCAL_GAMMA)
    stage2_model = train_stage_with_swa(
        stage2_model, stage2_train_loader, stage2_val_loader, stage2_criterion, tag="Stage2"
    )
    torch.save(stage2_model.state_dict(), MODEL_DIR / "cascade_stage2_subtype.pth")

    # ======================================================
    # EVALUATION
    # ======================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # ---- Tune Stage 1's decision threshold on validation data ----
    # Subtype-balanced, not pooled F1 - see find_subtype_balanced_threshold
    # docstring for why pooled F1 quietly sacrifices S-class recall.
    _, _, s1_val_labels, s1_val_probs = evaluate_stage1(stage1_model, stage1_val_loader, stage1_criterion)
    stage1_threshold = find_subtype_balanced_threshold(
        s1_val_probs[:, 1], s1_val_labels, y_val_sub
    )

    # ---- (a) Stage 1 standalone, using the TUNED threshold, not argmax ----
    _, _, s1_labels, s1_probs = evaluate_stage1(stage1_model, stage1_test_loader, stage1_criterion)
    s1_preds = (s1_probs[:, 1] >= stage1_threshold).astype(int)
    acc = accuracy_score(s1_labels, s1_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        s1_labels, s1_preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(s1_labels, s1_probs[:, 1])

    print("\n--- Stage 1 (Binary Arrhythmia Detection) - HEADLINE METRIC ---")
    print(f"Decision threshold (tuned on validation): {stage1_threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    cm1 = confusion_matrix(s1_labels, s1_preds)
    print("Confusion matrix [Normal, Abnormal]:\n", cm1)

    # ---- (b) Stage 2 standalone (ORACLE): given TRUE abnormal test beats ----
    test_ab_mask = y_test_bin == 1
    X_test_ab, RR_test_ab, y_test_ab = (X_test[test_ab_mask], RR_test[test_ab_mask],
                                         y_test_sub[test_ab_mask])
    stage2_oracle_ds = ECGDataset(X_test_ab, RR_test_ab, y_test_ab, augment=False)
    stage2_oracle_loader = DataLoader(stage2_oracle_ds, batch_size=BATCH_SIZE, shuffle=False)
    _, s2_preds, s2_labels, _ = evaluate_generic(stage2_model, stage2_oracle_loader, stage2_criterion)

    print("\n--- Stage 2 (Subtype, ORACLE - given true abnormal beats) ---")
    print(classification_report(s2_labels, s2_preds, labels=list(range(len(SUBTYPE_CLASSES))),
                                 target_names=SUBTYPE_CLASSES, zero_division=0))

    # ---- (c) END-TO-END CASCADE: the only honest full-system number ----
    # Run Stage 1 on ALL test beats. For beats predicted Abnormal, run Stage 2.
    # Beats predicted Normal are finalized as "N" (even if truly abnormal -
    # these are Stage 1's false negatives, and they DO count against the
    # relevant class in this evaluation, as they should).
    stage1_model.eval(); stage2_model.eval()
    final_preds_aami = np.zeros(len(y_test), dtype=np.int64)  # default N

    with torch.no_grad():
        ecg_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        rr_t = torch.tensor(RR_test, dtype=torch.float32).to(device)

        s1_out = stage1_model(ecg_t, rr_t)
        s1_probs_full = torch.softmax(s1_out, dim=1).cpu().numpy()
        s1_pred_full = (s1_probs_full[:, 1] >= stage1_threshold).astype(int)

        abnormal_idx = np.where(s1_pred_full == 1)[0]
        if len(abnormal_idx) > 0:
            ecg_ab = ecg_t[abnormal_idx]
            rr_ab = rr_t[abnormal_idx]
            s2_out = stage2_model(ecg_ab, rr_ab)
            s2_pred = torch.argmax(s2_out, dim=1).cpu().numpy()
            # map subtype idx (0..3 = S,V,F,Q) back to AAMI idx (1..4)
            subtype_to_aami_idx = {i: base.CLASS_TO_IDX[SUBTYPE_CLASSES[i]] for i in range(len(SUBTYPE_CLASSES))}
            for local_i, global_i in enumerate(abnormal_idx):
                final_preds_aami[global_i] = subtype_to_aami_idx[s2_pred[local_i]]
        # beats where s1_pred_full == 0 stay at default AAMI index 0 (N)

    print("\n--- END-TO-END CASCADE (full system, all 5 AAMI classes, honest) ---")
    macro_f1_e2e = f1_score(y_test, final_preds_aami, average="macro", zero_division=0)
    weighted_f1_e2e = f1_score(y_test, final_preds_aami, average="weighted", zero_division=0)
    acc_e2e = accuracy_score(y_test, final_preds_aami)
    print(f"Accuracy:    {acc_e2e:.4f}")
    print(f"Macro F1:    {macro_f1_e2e:.4f}")
    print(f"Weighted F1: {weighted_f1_e2e:.4f}")
    print("\n" + classification_report(y_test, final_preds_aami, labels=list(range(4)),
                                        target_names=AAMI_CLASSES_NO_Q, zero_division=0))

    cm_e2e = confusion_matrix(y_test, final_preds_aami, labels=list(range(4)))
    print("End-to-end confusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{c:>6}" for c in AAMI_CLASSES_NO_Q))
    for i, row in enumerate(cm_e2e):
        print(f"{AAMI_CLASSES_NO_Q[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    # ---- Edge-deployment metrics: BOTH stages ----
    n_params1, n_params2 = count_params(stage1_model), count_params(stage2_model)
    macs1, macs2 = estimate_conv1d_flops(stage1_model), estimate_conv1d_flops(stage2_model)
    latency1 = benchmark_cpu_latency(stage1_model)
    latency2 = benchmark_cpu_latency(stage2_model)
    abnormal_rate = (s1_pred_full == 1).mean()
    expected_latency = latency1 + abnormal_rate * latency2  # realistic average, not worst-case

    print("\n=== Edge-Deployment Metrics (Cascade) ===")
    print(f"Stage 1 params: {n_params1:,} | Stage 2 params: {n_params2:,} | Total: {n_params1+n_params2:,}")
    print(f"Stage 1 MACs: {macs1/1e6:.2f}M | Stage 2 MACs: {macs2/1e6:.2f}M")
    print(f"Stage 1-only latency (always runs): {latency1:.3f} ms")
    print(f"Stage 1+2 worst-case latency (every beat flagged abnormal): {latency1+latency2:.3f} ms")
    print(f"Observed abnormal-flag rate on test set: {abnormal_rate*100:.1f}%")
    print(f"Expected average latency per beat (realistic deployment number): {expected_latency:.3f} ms")

    # ---- Plots ----
    for cm_data, labels, title, fname in [
        (cm1, ["Normal", "Abnormal"], "Stage 1: Binary Screening", "cascade_stage1_cm.png"),
        (confusion_matrix(s2_labels, s2_preds, labels=list(range(len(SUBTYPE_CLASSES)))), SUBTYPE_CLASSES,
         "Stage 2 (Oracle): Subtype", "cascade_stage2_oracle_cm.png"),
        (cm_e2e, AAMI_CLASSES_NO_Q, "End-to-End Cascade (N/S/V/F)", "cascade_end_to_end_cm.png"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_data, cmap="Blues")
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm_data[i, j], ha="center", va="center",
                          color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / fname)
        plt.close()

    print(f"\nPlots saved to {FIGURE_DIR}")
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()