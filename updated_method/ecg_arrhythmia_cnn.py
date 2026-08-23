"""
ECG Arrhythmia Classification - Residual CNN + RR Fusion
Fixed & improved version, adapted to run locally (macOS, incl. Apple Silicon MPS).

Fixes vs. your original notebook:
  1. Section "Test Evaluation" was evaluating on val_loader instead of test_loader.
     -> Fixed: now correctly uses test_loader.
  2. BEST_THRESHOLD was hardcoded to 0.90, ignoring the threshold you tuned on
     validation data.
     -> Fixed: uses the searched best_threshold.
  3. No per-beat amplitude normalization (patient/lead gain varies a lot in MIT-BIH).
     -> Added: z-score each beat individually after filtering.
  4. No training-time augmentation.
     -> Added: light Gaussian jitter + random amplitude scaling, train split only.
  5. Plain residual blocks.
     -> Added: squeeze-and-excitation (SE) gating, cheap and usually helps ECG CNNs.
  6. Colab / Google Drive specific code.
     -> Replaced with local folders relative to this script + MPS device support.

Run:
    python3 -m venv venv
    source venv/bin/activate
    pip install wfdb neurokit2 torch scikit-learn matplotlib numpy pandas
    python ecg_arrhythmia_cnn.py
"""

import copy
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures instead of trying to pop windows
import matplotlib.pyplot as plt

import wfdb

from scipy.signal import butter, filtfilt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ==========================================================
# 0. Reproducibility & device
# ==========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU
else:
    device = torch.device("cpu")

print("Device:", device)

# ==========================================================
# 1. Project folders
#
# COLAB USERS: mount Drive FIRST in a separate cell, then this path
# persists across runtime resets (no re-downloading MIT-BIH each time):
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# LOCAL (macOS etc.) USERS: leave this as the relative "./ECG_Project"
# path - it'll just live next to this script.
# ==========================================================

import os

if os.path.exists("/content/drive/MyDrive"):
    PROJECT_DIR = Path("/content/drive/MyDrive/ECG_Project")   # Colab + Drive mounted
else:
    PROJECT_DIR = Path("./ECG_Project")                         # local run

MITDB_DIR = PROJECT_DIR / "mitdb"
DATASET_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "figures"
RESULT_DIR = PROJECT_DIR / "results"

for folder in [PROJECT_DIR, MITDB_DIR, DATASET_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. Config
# ==========================================================

RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

FS = 360
LOWCUT, HIGHCUT, FILTER_ORDER = 0.5, 40.0, 4

LEFT_WINDOW = 180
RIGHT_WINDOW = 180
BEAT_LENGTH = 361

NORMAL_BEATS = {"N"}
VALID_BEATS = {'N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F',
               '/', 'f', 'Q', 'e', 'j'}

BATCH_SIZE = 512
EPOCHS = 40
PATIENCE = 10
LR = 3e-4              # was 1e-3 -> too high, caused training instability
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05  # small amount, helps calibration/generalization
GRAD_CLIP_NORM = 1.0    # new: stabilizes updates

AUG_NOISE_STD = 0.02     # relative to unit-normalized beat
AUG_SCALE_RANGE = (0.9, 1.1)


# ==========================================================
# 3. Download dataset if needed
# ==========================================================

def ensure_dataset():
    already = all((MITDB_DIR / f"{r}.dat").exists() for r in RECORDS)
    if already:
        print("MIT-BIH dataset already present.")
        return
    print("Downloading MIT-BIH dataset (this needs internet access)...")
    for r in RECORDS:
        if not (MITDB_DIR / f"{r}.dat").exists():
            wfdb.dl_database("mitdb", dl_dir=str(MITDB_DIR), records=[r])
    print("Download complete.")


# ==========================================================
# 4. Signal processing
# ==========================================================

def bandpass_filter(signal):
    nyquist = 0.5 * FS
    low, high = LOWCUT / nyquist, HIGHCUT / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)


def zscore_beat(beat):
    """Normalize a single beat (channels, length) to zero mean / unit std per channel.
    Fixes amplitude-gain differences across patients/leads."""
    mean = beat.mean(axis=1, keepdims=True)
    std = beat.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (beat - mean) / std


def process_record(record_name):
    record = wfdb.rdrecord(str(MITDB_DIR / record_name))
    annotation = wfdb.rdann(str(MITDB_DIR / record_name), "atr")

    signals = record.p_signal
    lead1 = bandpass_filter(signals[:, 0])
    lead2 = bandpass_filter(signals[:, 1])
    filtered = np.column_stack((lead1, lead2))

    r_peaks = annotation.sample
    symbols = annotation.symbol

    beats, rr_features, labels, patient_ids = [], [], [], []

    for i in range(1, len(r_peaks) - 2):
        if symbols[i] not in VALID_BEATS:
            continue

        peak = r_peaks[i]
        start = peak - LEFT_WINDOW
        end = peak + RIGHT_WINDOW + 1
        if start < 0 or end > len(filtered):
            continue

        beat = filtered[start:end].T          # (2, 361)
        beat = zscore_beat(beat)               # <-- fix: per-beat amplitude normalization

        previous_rr = (r_peaks[i] - r_peaks[i - 1]) / FS
        current_rr = (r_peaks[i + 1] - r_peaks[i]) / FS
        next_rr = (r_peaks[i + 2] - r_peaks[i + 1]) / FS
        rr_ratio = current_rr / previous_rr if previous_rr != 0 else 1.0

        label = 0 if symbols[i] in NORMAL_BEATS else 1

        beats.append(beat)
        rr_features.append([previous_rr, current_rr, next_rr, rr_ratio])
        labels.append(label)
        patient_ids.append(record_name)

    return (np.array(beats), np.array(rr_features),
            np.array(labels), np.array(patient_ids))


def build_dataset():
    cache_path = DATASET_DIR / "processed_dataset.npz"
    if cache_path.exists():
        print("Loading cached processed dataset...")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["RR"], data["y"], data["patient_ids"]

    all_beats, all_rr, all_labels, all_patients = [], [], [], []
    for record in RECORDS:
        print(f"Processing record {record}...")
        b, rr, lbl, pid = process_record(record)
        all_beats.append(b)
        all_rr.append(rr)
        all_labels.append(lbl)
        all_patients.append(pid)

    X = np.concatenate(all_beats, axis=0)
    RR = np.concatenate(all_rr, axis=0)
    y = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patients, axis=0)

    np.savez_compressed(cache_path, X=X, RR=RR, y=y, patient_ids=patient_ids)
    print("Dataset cached at", cache_path)

    counts = Counter(y)
    print(f"Total beats: {len(y)} | Normal: {counts[0]} | Arrhythmia: {counts[1]} "
          f"({100 * counts[1] / len(y):.2f}%)")

    return X, RR, y, patient_ids


# ==========================================================
# 5. Patient-wise split
# ==========================================================

def patient_wise_split(X, RR, y, patient_ids):
    gss = GroupShuffleSplit(test_size=0.20, random_state=SEED, n_splits=1)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    X_train, RR_train, y_train, p_train = X[train_idx], RR[train_idx], y[train_idx], patient_ids[train_idx]
    X_test, RR_test, y_test, p_test = X[test_idx], RR[test_idx], y[test_idx], patient_ids[test_idx]

    gss2 = GroupShuffleSplit(test_size=0.15, random_state=SEED, n_splits=1)
    tr_idx2, val_idx = next(gss2.split(X_train, y_train, groups=p_train))

    X_val, RR_val, y_val, p_val = X_train[val_idx], RR_train[val_idx], y_train[val_idx], p_train[val_idx]
    X_train, RR_train, y_train, p_train = X_train[tr_idx2], RR_train[tr_idx2], y_train[tr_idx2], p_train[tr_idx2]

    # sanity check: no patient leakage
    assert not (set(p_train) & set(p_val))
    assert not (set(p_train) & set(p_test))
    assert not (set(p_val) & set(p_test))

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Train classes:", Counter(y_train))
    print("Val classes:  ", Counter(y_val))
    print("Test classes: ", Counter(y_test))

    return (X_train, RR_train, y_train, X_val, RR_val, y_val,
            X_test, RR_test, y_test)


# ==========================================================
# 6. Dataset / augmentation
# ==========================================================

class ECGDataset(Dataset):
    def __init__(self, waveforms, rr_features, labels, augment=False):
        self.waveforms = waveforms.astype(np.float32)
        self.rr = torch.tensor(rr_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        beat = self.waveforms[idx].copy()

        if self.augment:
            # random amplitude scaling
            scale = np.random.uniform(*AUG_SCALE_RANGE)
            beat = beat * scale
            # gaussian jitter
            beat = beat + np.random.normal(0, AUG_NOISE_STD, size=beat.shape)

        beat_t = torch.tensor(beat, dtype=torch.float32)
        return beat_t, self.rr[idx], self.labels[idx]


# ==========================================================
# 7. Model: SE-Residual CNN + RR fusion
# ==========================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: cheap channel-attention, usually a free accuracy bump."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class ECGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(2, 32),
            nn.MaxPool1d(2),
            ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.squeeze(-1)


class RRFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.network(x)


class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_extractor = ECGFeatureExtractor()
        self.rr_extractor = RRFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, ecg, rr):
        ecg_features = self.ecg_extractor(ecg)
        rr_features = self.rr_extractor(rr)
        features = torch.cat([ecg_features, rr_features], dim=1)
        return self.classifier(features)


# ==========================================================
# 8. Train / evaluate helpers
# ==========================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for ecg, rr, labels in loader:
        ecg, rr, labels = ecg.to(device), rr.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(ecg, rr)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    running_loss = 0.0
    probs_all, preds_all, labels_all = [], [], []

    with torch.no_grad():
        for ecg, rr, labels in loader:
            ecg, rr, labels = ecg.to(device), rr.to(device), labels.to(device)
            outputs = model(ecg, rr)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            probs_all.extend(probs.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    loss = running_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec = recall_score(labels_all, preds_all, zero_division=0)
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    auc_score = roc_auc_score(labels_all, probs_all)

    return {
        "loss": loss, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc_score,
        "probs": np.array(probs_all), "preds": np.array(preds_all),
        "labels": np.array(labels_all)
    }


def find_best_threshold(probs, labels):
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ==========================================================
# 9. Main
# ==========================================================

def main():
    ensure_dataset()
    X, RR, y, patient_ids = build_dataset()

    (X_train, RR_train, y_train, X_val, RR_val, y_val,
     X_test, RR_test, y_test) = patient_wise_split(X, RR, y, patient_ids)

    # RR normalization: fit on train only
    rr_scaler = StandardScaler()
    RR_train = rr_scaler.fit_transform(RR_train)
    RR_val = rr_scaler.transform(RR_val)
    RR_test = rr_scaler.transform(RR_test)

    train_dataset = ECGDataset(X_train, RR_train, y_train, augment=True)
    val_dataset = ECGDataset(X_val, RR_val, y_val, augment=False)
    test_dataset = ECGDataset(X_test, RR_test, y_test, augment=False)

    # NOTE: switched from WeightedRandomSampler to class-weighted loss.
    # Oversampling the minority class every batch (combined with a deep SE-ResNet
    # and LR=1e-3) was causing the training instability seen in your run (val loss
    # swinging between 0.23 and 0.86 while train loss stayed flat). Class-weighted
    # loss corrects for imbalance more gently, without repeatedly duplicating the
    # same minority-class beats within an epoch.
    class_counts = Counter(y_train)
    total = len(y_train)
    class_weights = torch.tensor(
        [total / (2 * class_counts[0]), total / (2 * class_counts[1])],
        dtype=torch.float32
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    history = {k: [] for k in
               ["train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "auc"]}

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion, threshold=0.5)
        scheduler.step(val_metrics["auc"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            history[k].append(val_metrics[k])

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} "
              f"| Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} "
              f"| AUC: {val_metrics['auc']:.4f} | LR: {lr_now:.6f}")

        # Select checkpoint by AUC: it's threshold-independent, so it's a more
        # stable signal than F1@0.5 for picking the best epoch, especially
        # early in training before the decision boundary has settled.
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("Best model updated.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / "best_ecg_classifier.pth")
    print(f"\nBest Validation AUC: {best_auc:.4f}")

    # ---- Threshold tuning on validation set ----
    val_metrics = evaluate(model, val_loader, criterion, threshold=0.5)
    best_threshold, best_val_f1 = find_best_threshold(val_metrics["probs"], val_metrics["labels"])
    print(f"Best threshold (from validation): {best_threshold:.2f} | Val F1 at that threshold: {best_val_f1:.4f}")

    # ---- FIX: evaluate on the TEST set, with the tuned threshold ----
    test_metrics = evaluate(model, test_loader, criterion, threshold=best_threshold)

    print("\n=== Final Test Set Results ===")
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"ROC AUC:   {test_metrics['auc']:.4f}")

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(test_metrics["labels"], test_metrics["preds"],
                                 target_names=["Normal", "Arrhythmia"]))

    # ---- Save plots ----
    plt.figure(figsize=(8, 5))
    plt.plot(history["f1"], label="Val F1")
    plt.plot(history["auc"], label="Val AUC")
    plt.plot(history["accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Validation Metrics over Training")
    plt.grid(True)
    plt.savefig(FIGURE_DIR / "training_curves.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Arrhythmia"])
    plt.yticks([0, 1], ["Normal", "Arrhythmia"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(FIGURE_DIR / "confusion_matrix.png")
    plt.close()

    print(f"\nPlots saved to {FIGURE_DIR}")
    print(f"Model saved to {MODEL_DIR / 'best_ecg_classifier.pth'}")


if __name__ == "__main__":
    main()
