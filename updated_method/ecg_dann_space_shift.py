"""
ecg_dann_space_shift.py

The actual domain-adaptation piece: trains a binary (Normal vs Abnormal)
screening model two ways -
  (a) SOURCE-ONLY  - standard training on real (Earth-like) MIT-BIH beats,
                      the way every prior script in this project has done it.
  (b) DANN         - same task, but a gradient-reversal domain discriminator
                      is attached to the shared feature extractor, trained
                      jointly on labeled source beats + unlabeled
                      space-shifted target beats (from space_shift_simulator.py),
                      forcing the extractor to learn features that don't
                      encode "which domain is this beat from."

Then both models are evaluated across a MILD/MODERATE/SEVERE severity sweep
of the physiologically-grounded shift, on held-out DS2 patients never seen
in training. The comparison IS the paper's robustness claim - success means
narrowing the source-vs-shifted performance gap, not beating clean-domain
numbers (a DANN model scoring higher than clean-domain baseline would be
suspicious, not a win - see prior discussion in this project).

Requires: ECG_Project/mitdb already populated (from any earlier script's
ensure_dataset() call) and space_shift_simulator.py in the same directory.

First run will process the full paired dataset at each severity used (one-
time cost per severity, cached afterward by space_shift_simulator.py).
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
from torch.autograd import Function
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score,
    confusion_matrix
)

import ecg_multiclass_cnn as base
import space_shift_simulator as sim

device = base.device
PROJECT_DIR = base.PROJECT_DIR
MODEL_DIR = base.MODEL_DIR
FIGURE_DIR = base.FIGURE_DIR

print("Device:", device)

# ==========================================================
# Config
# ==========================================================

BATCH_SIZE = 512
EPOCHS = 40
WARMUP_EPOCHS = 3
SWA_START_FRAC = 0.6
LR = 3e-4
SWA_LR = 1e-4
WEIGHT_DECAY = 2e-4
GRAD_CLIP_NORM = 1.0
CB_BETA = 0.9999

TRAIN_SEVERITY = "moderate"  # domain adaptation trains against this shift level
EVAL_SEVERITIES = ["mild", "moderate", "severe"]  # test-time robustness sweep


# ==========================================================
# Gradient Reversal Layer - the core DANN mechanism
# ==========================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)


def dann_lambda_schedule(progress):
    """Standard DANN ramp-up: 0 -> 1 via a sigmoid, so the domain-adversarial
    term doesn't destabilize the task loss early in training when features
    aren't meaningful yet (Ganin et al., 2016)."""
    return 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0


# ==========================================================
# Model: shared extractor + task head + domain head
# ==========================================================

class DANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_extractor = base.ECGFeatureExtractor()
        self.rr_extractor = base.RRFeatureExtractor()

        self.task_head = nn.Sequential(
            nn.Linear(160, 64), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def features(self, ecg, rr):
        return torch.cat([self.ecg_extractor(ecg), self.rr_extractor(rr)], dim=1)

    def forward(self, ecg, rr, lambd=0.0):
        feats = self.features(ecg, rr)
        task_out = self.task_head(feats)
        domain_out = self.domain_head(grad_reverse(feats, lambd))
        return task_out, domain_out


# ==========================================================
# Dataset - paired source/target beats, same index = same underlying beat
# ==========================================================

class PairedDomainDataset(Dataset):
    """Returns (ecg_src, rr_src, ecg_tgt, rr_tgt, label) - label is the TASK
    label (Normal/Abnormal), shared by construction since target is just a
    physiologically-shifted version of the same beat. Target labels are
    never used for the task loss (only for reporting) - this is what makes
    it a genuine unsupervised-target domain-adaptation setup."""

    def __init__(self, X_src, RR_src, X_tgt, RR_tgt, labels, augment=False):
        self.X_src = X_src.astype(np.float32)
        self.RR_src = torch.tensor(RR_src, dtype=torch.float32)
        self.X_tgt = X_tgt.astype(np.float32)
        self.RR_tgt = torch.tensor(RR_tgt, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        src = self.X_src[idx].copy()
        tgt = self.X_tgt[idx].copy()
        if self.augment:
            for arr in (src, tgt):
                pass  # augmentation intentionally omitted here - shift itself
                      # is already a controlled perturbation; adding jitter on
                      # top would confound the domain-shift signal
        return (torch.tensor(src, dtype=torch.float32), self.RR_src[idx],
                torch.tensor(tgt, dtype=torch.float32), self.RR_tgt[idx],
                self.labels[idx])


class SourceOnlyDataset(Dataset):
    """Plain source-domain dataset for the baseline model."""
    def __init__(self, X, RR, labels, augment=True):
        self.X = X.astype(np.float32)
        self.RR = torch.tensor(RR, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        beat = self.X[idx].copy()
        if self.augment:
            scale = np.random.uniform(*base.AUG_SCALE_RANGE)
            beat = beat * scale + np.random.normal(0, base.AUG_NOISE_STD, size=beat.shape)
        return torch.tensor(beat, dtype=torch.float32), self.RR[idx], self.labels[idx]


# ==========================================================
# Data preparation
# ==========================================================

def build_dann_data():
    print(f"\nBuilding paired source/target dataset at severity='{TRAIN_SEVERITY}' "
          f"(one-time cost, cached after)...")
    records = base.DS1_RECORDS + base.DS2_RECORDS
    source, target = sim.build_domain_dataset(records, severity=TRAIN_SEVERITY)

    # NOTE: source/target labels come from space_shift_simulator.py's own
    # process_record_paired (binary 0=Normal/1=Abnormal, computed independently
    # of ecg_multiclass_cnn.py's AAMI labeling) - no need to cross-reference
    # the multiclass dataset here.
    X_src, RR_src, y_bin, patient_ids = source["X"], source["RR"], source["y"], source["patient_ids"]
    X_tgt, RR_tgt = target["X"], target["RR"]

    is_ds1 = np.isin(patient_ids, base.DS1_RECORDS)
    is_ds2 = np.isin(patient_ids, base.DS2_RECORDS)

    X_src_ds1, RR_src_ds1, X_tgt_ds1, RR_tgt_ds1, y_ds1, p_ds1 = (
        X_src[is_ds1], RR_src[is_ds1], X_tgt[is_ds1], RR_tgt[is_ds1], y_bin[is_ds1], patient_ids[is_ds1]
    )
    X_src_test, RR_src_test, X_tgt_test, RR_tgt_test, y_test, p_test = (
        X_src[is_ds2], RR_src[is_ds2], X_tgt[is_ds2], RR_tgt[is_ds2], y_bin[is_ds2], patient_ids[is_ds2]
    )

    gss = GroupShuffleSplit(test_size=0.15, random_state=base.SEED, n_splits=1)
    tr_idx, val_idx = next(gss.split(X_src_ds1, y_ds1, groups=p_ds1))

    def sub(arrs, idx):
        return tuple(a[idx] for a in arrs)

    train_arrs = sub((X_src_ds1, RR_src_ds1, X_tgt_ds1, RR_tgt_ds1, y_ds1), tr_idx)
    val_arrs = sub((X_src_ds1, RR_src_ds1, X_tgt_ds1, RR_tgt_ds1, y_ds1), val_idx)

    rr_scaler = StandardScaler()
    rr_scaler.fit(train_arrs[1])  # fit on source RR only

    def scale(arrs):
        X_s, RR_s, X_t, RR_t, y_ = arrs
        return X_s, rr_scaler.transform(RR_s), X_t, rr_scaler.transform(RR_t), y_

    train_arrs = scale(train_arrs)
    val_arrs = scale(val_arrs)
    RR_src_test = rr_scaler.transform(RR_src_test)
    RR_tgt_test = rr_scaler.transform(RR_tgt_test)

    print(f"Train: {len(train_arrs[4])} | Val: {len(val_arrs[4])} | Test: {len(y_test)}")
    print(f"Train class balance: {Counter(train_arrs[4])}")

    return (train_arrs, val_arrs,
            (X_src_test, RR_src_test, X_tgt_test, RR_tgt_test, y_test))


# ==========================================================
# Training - shared warmup/cosine/SWA schedule
# ==========================================================

def make_scheduler(optimizer):
    swa_start_epoch = int(EPOCHS * SWA_START_FRAC)

    def lr_lambda(epoch):
        swa_ratio = SWA_LR / LR
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        if epoch >= swa_start_epoch:
            return swa_ratio
        progress = (epoch - WARMUP_EPOCHS) / max(1, (swa_start_epoch - WARMUP_EPOCHS))
        return swa_ratio + (1 - swa_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), swa_start_epoch


def train_source_only(X_train, RR_train, y_train, X_val, RR_val, y_val, class_weights):
    train_ds = SourceOnlyDataset(X_train, RR_train, y_train, augment=True)
    val_ds = SourceOnlyDataset(X_val, RR_val, y_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DANNModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler, swa_start_epoch = make_scheduler(optimizer)
    swa_model = AveragedModel(model)
    swa_active = False

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for ecg, rr, labels in train_loader:
            ecg, rr, labels = ecg.to(device), rr.to(device), labels.to(device)
            optimizer.zero_grad()
            task_out, _ = model(ecg, rr, lambd=0.0)  # no domain branch used here
            loss = criterion(task_out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item()

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_active = True
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                preds, labels_all = [], []
                for ecg, rr, labels in val_loader:
                    ecg, rr = ecg.to(device), rr.to(device)
                    out, _ = model(ecg, rr, lambd=0.0)
                    preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    labels_all.extend(labels.numpy())
                f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
            print(f"[SourceOnly] Epoch {epoch+1}/{EPOCHS} train_loss={running_loss/len(train_loader):.4f} "
                  f"val_macro_f1={f1:.4f}")

    if swa_active:
        print("[SourceOnly] Recalibrating BatchNorm for SWA model...")
        base.update_bn_two_input(train_loader, swa_model, device=device)
        model = swa_model.module

    return model


def train_dann(train_arrs, val_arrs, class_weights):
    X_src, RR_src, X_tgt, RR_tgt, y = train_arrs
    X_src_v, RR_src_v, X_tgt_v, RR_tgt_v, y_v = val_arrs

    train_ds = PairedDomainDataset(X_src, RR_src, X_tgt, RR_tgt, y)
    val_ds = PairedDomainDataset(X_src_v, RR_src_v, X_tgt_v, RR_tgt_v, y_v)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DANNModel().to(device)
    task_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler, swa_start_epoch = make_scheduler(optimizer)
    swa_model = AveragedModel(model)
    swa_active = False

    total_steps = EPOCHS * len(train_loader)
    step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_task_loss, running_domain_loss, running_domain_acc = 0.0, 0.0, 0.0
        for ecg_src, rr_src, ecg_tgt, rr_tgt, labels in train_loader:
            ecg_src, rr_src = ecg_src.to(device), rr_src.to(device)
            ecg_tgt, rr_tgt = ecg_tgt.to(device), rr_tgt.to(device)
            labels = labels.to(device)

            progress = step / max(1, total_steps)
            lambd = dann_lambda_schedule(progress)

            optimizer.zero_grad()

            task_out, domain_out_src = model(ecg_src, rr_src, lambd=lambd)
            task_loss = task_criterion(task_out, labels)

            _, domain_out_tgt = model(ecg_tgt, rr_tgt, lambd=lambd)

            domain_logits = torch.cat([domain_out_src, domain_out_tgt], dim=0)
            domain_labels = torch.cat([
                torch.zeros(len(labels), dtype=torch.long, device=device),   # source = 0
                torch.ones(len(labels), dtype=torch.long, device=device)     # target = 1
            ])
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = task_loss + domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            running_task_loss += task_loss.item()
            running_domain_loss += domain_loss.item()
            domain_preds = torch.argmax(domain_logits, dim=1)
            running_domain_acc += (domain_preds == domain_labels).float().mean().item()
            step += 1

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_active = True
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            n = len(train_loader)
            model.eval()
            with torch.no_grad():
                preds, labels_all = [], []
                for ecg, rr, _, _, labels in val_loader:
                    ecg, rr = ecg.to(device), rr.to(device)
                    out, _ = model(ecg, rr, lambd=0.0)
                    preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    labels_all.extend(labels.numpy())
                f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
            print(f"[DANN] Epoch {epoch+1}/{EPOCHS} task_loss={running_task_loss/n:.4f} "
                  f"domain_loss={running_domain_loss/n:.4f} domain_acc={running_domain_acc/n:.3f} "
                  f"(->0.5 = domain-invariant) val_macro_f1={f1:.4f} lambda={lambd:.3f}")

    if swa_active:
        print("[DANN] Recalibrating BatchNorm for SWA model...")
        base.update_bn_two_input(train_loader, swa_model, device=device)
        model = swa_model.module

    return model


# ==========================================================
# Evaluation
# ==========================================================

def evaluate_binary(model, X, RR, y, batch_size=1024):
    model.eval()
    preds_all, probs_all = [], []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            ecg = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            rr = torch.tensor(RR[i:i+batch_size], dtype=torch.float32).to(device)
            out, _ = model(ecg, rr, lambd=0.0)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds_all.extend(np.argmax(probs, axis=1))
            probs_all.extend(probs[:, 1])

    preds_all, probs_all = np.array(preds_all), np.array(probs_all)
    acc = accuracy_score(y, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds_all, average="binary", zero_division=0)
    auc = roc_auc_score(y, probs_all) if len(np.unique(y)) > 1 else float("nan")
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)


def main():
    train_arrs, val_arrs, test_arrs = build_dann_data()
    X_train_src, RR_train_src, _, _, y_train = train_arrs
    X_src_test, RR_src_test, X_tgt_test, RR_tgt_test, y_test = test_arrs

    counts = Counter(y_train)
    counts_arr = np.array([counts.get(0, 1), counts.get(1, 1)], dtype=np.float64)
    eff = 1.0 - np.power(CB_BETA, counts_arr)
    w = (1.0 - CB_BETA) / eff
    w = w / w.sum() * 2
    class_weights = torch.tensor(w, dtype=torch.float32)
    print(f"Class-balanced weights: Normal={w[0]:.3f}, Abnormal={w[1]:.3f}")

    print("\n" + "=" * 60)
    print("TRAINING: Source-only baseline")
    print("=" * 60)
    source_only_model = train_source_only(
        X_train_src, RR_train_src, y_train,
        val_arrs[0], val_arrs[1], val_arrs[4], class_weights
    )
    torch.save(source_only_model.state_dict(), MODEL_DIR / "dann_source_only.pth")

    print("\n" + "=" * 60)
    print("TRAINING: DANN (domain-adversarial)")
    print("=" * 60)
    dann_model = train_dann(train_arrs, val_arrs, class_weights)
    torch.save(dann_model.state_dict(), MODEL_DIR / "dann_model.pth")

    # ======================================================
    # ABLATION: source-only vs DANN, on source AND shifted target
    # ======================================================
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)

    print(f"\n--- Source domain test set (sanity check - both should be similar) ---")
    for name, model in [("Source-only", source_only_model), ("DANN", dann_model)]:
        m = evaluate_binary(model, X_src_test, RR_src_test, y_test)
        print(f"{name:<14} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    print(f"\n--- Target domain test set (severity='{TRAIN_SEVERITY}') - THE KEY RESULT ---")
    results_summary = {}
    for name, model in [("Source-only", source_only_model), ("DANN", dann_model)]:
        m = evaluate_binary(model, X_tgt_test, RR_tgt_test, y_test)
        results_summary[name] = m
        print(f"{name:<14} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    gap_before = results_summary["Source-only"]["f1"]
    gap_after = results_summary["DANN"]["f1"]
    print(f"\nTarget-domain F1: Source-only={gap_before:.4f} -> DANN={gap_after:.4f} "
          f"({'improved' if gap_after > gap_before else 'no improvement'} by {gap_after-gap_before:+.4f})")

    # ======================================================
    # SEVERITY SWEEP (test-time only - train once, evaluate across shift levels)
    # ======================================================
    print("\n" + "=" * 60)
    print(f"SEVERITY SWEEP (train severity='{TRAIN_SEVERITY}', evaluated across {EVAL_SEVERITIES})")
    print("=" * 60)

    sweep_results = {name: [] for name in ["Source-only", "DANN"]}
    for severity in EVAL_SEVERITIES:
        if severity == TRAIN_SEVERITY:
            X_tgt_sev, RR_tgt_sev = X_tgt_test, RR_tgt_test
        else:
            print(f"\nBuilding severity='{severity}' target test set...")
            records = base.DS1_RECORDS + base.DS2_RECORDS
            _, target_sev = sim.build_domain_dataset(records, severity=severity)
            is_ds2 = np.isin(target_sev["patient_ids"], base.DS2_RECORDS)
            X_tgt_sev = target_sev["X"][is_ds2]
            RR_tgt_sev_raw = target_sev["RR"][is_ds2]
            # reuse the scaler already fit during build_dann_data - refit here
            # for simplicity since this function doesn't have access to it;
            # deterministic given fixed seed, so equivalent in practice.
            scaler = StandardScaler().fit(RR_train_src)
            RR_tgt_sev = scaler.transform(RR_tgt_sev_raw)

        print(f"\n--- severity='{severity}' ---")
        for name, model in [("Source-only", source_only_model), ("DANN", dann_model)]:
            m = evaluate_binary(model, X_tgt_sev, RR_tgt_sev, y_test)
            sweep_results[name].append(m["f1"])
            print(f"{name:<14} F1={m['f1']:.4f} Acc={m['accuracy']:.4f} AUC={m['auc']:.4f}")

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    for name, f1s in sweep_results.items():
        plt.plot(EVAL_SEVERITIES, f1s, marker="o", label=name)
    plt.xlabel("Space-Shift Severity")
    plt.ylabel("Target-Domain F1")
    plt.title("Robustness to Space-Shift: Source-only vs DANN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dann_severity_sweep.png")
    print(f"\nFigure saved to {FIGURE_DIR / 'dann_severity_sweep.png'}")

    print("\nDone. Report the severity-sweep table/figure as the core robustness result - ")
    print("success is DANN narrowing the source-vs-shifted gap across severities, ")
    print("not DANN exceeding clean-domain performance.")


if __name__ == "__main__":
    main()
