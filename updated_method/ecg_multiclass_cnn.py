"""
ecg_multiclass_cnn.py

Multi-class AAMI (N/S/V/F/Q) arrhythmia classifier - extends the binary
baseline (ecg_arrhythmia_cnn.py) to standard AAMI EC57 beat categories,
which is what the inter-patient literature actually benchmarks against.

WHY THIS MATTERS FOR THE PAPER
--------------------------------
Binary N-vs-not-N pools together morphologically distinct beat types
(ventricular ectopic beats are easy to separate; supraventricular ectopic
beats are subtle) into one bucket, which muddies both the learning signal
and the metric. AAMI EC57 is the standard protocol reviewers will expect,
and per-class reporting lets you make an honest, specific "high 90s" claim
(V-class F1) instead of an overclaimed blanket one.

AAMI EC57 BEAT MAPPING (from MIT-BIH annotation symbols)
-----------------------------------------------------------
N (Normal)                : N, L, R, e, j
S (Supraventricular ectopic): A, a, J, S
V (Ventricular ectopic)    : V, E
F (Fusion)                 : F
Q (Unknown/paced)          : /, f, Q

ARCHITECTURE NOTE
------------------
Kept deliberately efficient (same backbone size as the stabilized binary
model - 3 residual blocks, 32/64/128 channels), NOT scaled up. Given the
edge-deployment requirement, complexity should come from what the paper
demonstrates (domain adaptation, physiologically-grounded shift), not from
an oversized backbone. Model size / FLOPs / CPU latency are reported at
the end of training so you have real numbers for the edge-deployment
section of the paper, not just a claim.
"""

import os
import copy
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wfdb
from scipy.signal import butter, filtfilt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# 0. Reproducibility & device
# ==========================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

# ==========================================================
# 1. Project folders (same Drive/local auto-detect as the binary script,
#    so this reuses the already-downloaded MIT-BIH data - no re-download)
# ==========================================================

if os.path.exists("/content/drive/MyDrive"):
    PROJECT_DIR = Path("/content/drive/MyDrive/ECG_Project")
else:
    PROJECT_DIR = Path("./ECG_Project")

MITDB_DIR = PROJECT_DIR / "mitdb"
DATASET_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "figures"

for folder in [PROJECT_DIR, MITDB_DIR, DATASET_DIR, MODEL_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. Config
# ==========================================================

# ---- Standard inter-patient split (de Chazal et al., 2004) ----
# This is THE standard partition used across the inter-patient ECG
# classification literature - using it means your numbers are directly
# comparable to published results, and it avoids the degenerate class
# distribution you hit with a random patient-wise split (Val Q=5 out of
# 8032 total - Q/S beats are concentrated in a handful of specific
# patients, so a random group split can accidentally starve one split of
# an entire class).
#
# Records 102, 104, 107, 217 are excluded (paced rhythms / atypical
# quality) per the same protocol - this is standard, not a choice we made.
DS1_RECORDS = ['101', '106', '108', '109', '112', '114', '115', '116',
               '118', '119', '122', '124', '201', '203', '205', '207',
               '208', '209', '215', '220', '223', '230']
DS2_RECORDS = ['100', '103', '105', '111', '113', '117', '121', '123',
               '200', '202', '210', '212', '213', '214', '219', '221',
               '222', '228', '231', '232', '233', '234']
RECORDS = DS1_RECORDS + DS2_RECORDS

FS = 360
LOWCUT, HIGHCUT, FILTER_ORDER = 0.5, 40.0, 4
LEFT_WINDOW = 180
RIGHT_WINDOW = 180

# ---- AAMI EC57 class mapping ----
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}

SYMBOL_TO_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
VALID_BEATS = set(SYMBOL_TO_AAMI.keys())

BATCH_SIZE = 512
EPOCHS = 40
WARMUP_EPOCHS = 3
SWA_START_FRAC = 0.6
LR = 3e-4
SWA_LR = 1e-4
WEIGHT_DECAY = 2e-4  # was 1e-4 - train loss hit ~0.0005 while val loss kept climbing to 0.15-0.17
GRAD_CLIP_NORM = 1.0
FOCAL_GAMMA = 2.0
CB_BETA = 0.9999  # class-balanced reweighting (Cui et al., CVPR 2019)

AUG_NOISE_STD = 0.02
AUG_SCALE_RANGE = (0.9, 1.1)


# ==========================================================
# 3. Signal utilities (same as binary pipeline)
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


def ensure_dataset():
    already = all((MITDB_DIR / f"{r}.dat").exists() for r in RECORDS)
    if already:
        print("MIT-BIH dataset already present.")
        return
    print("Downloading MIT-BIH dataset...")
    for r in RECORDS:
        if not (MITDB_DIR / f"{r}.dat").exists():
            wfdb.dl_database("mitdb", dl_dir=str(MITDB_DIR), records=[r])
    print("Download complete.")


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

        beat = filtered[start:end].T
        beat = zscore_beat(beat)

        previous_rr = (r_peaks[i] - r_peaks[i - 1]) / FS
        current_rr = (r_peaks[i + 1] - r_peaks[i]) / FS
        next_rr = (r_peaks[i + 2] - r_peaks[i + 1]) / FS
        rr_ratio = current_rr / previous_rr if previous_rr != 0 else 1.0

        label = CLASS_TO_IDX[SYMBOL_TO_AAMI[symbols[i]]]

        beats.append(beat)
        rr_features.append([previous_rr, current_rr, next_rr, rr_ratio])
        labels.append(label)
        patient_ids.append(record_name)

    return (np.array(beats), np.array(rr_features),
            np.array(labels), np.array(patient_ids))


def build_dataset():
    cache_path = DATASET_DIR / "processed_dataset_multiclass_dschazal.npz"
    if cache_path.exists():
        print("Loading cached multi-class dataset...")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["RR"], data["y"], data["patient_ids"]

    all_beats, all_rr, all_labels, all_patients = [], [], [], []
    for record in RECORDS:
        print(f"Processing record {record}...")
        b, rr, lbl, pid = process_record(record)
        all_beats.append(b); all_rr.append(rr)
        all_labels.append(lbl); all_patients.append(pid)

    X = np.concatenate(all_beats, axis=0)
    RR = np.concatenate(all_rr, axis=0)
    y = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patients, axis=0)

    np.savez_compressed(cache_path, X=X, RR=RR, y=y, patient_ids=patient_ids)
    print(f"Dataset cached at {cache_path}")

    counts = Counter(y)
    print("\nClass distribution (AAMI):")
    for i, c in enumerate(AAMI_CLASSES):
        print(f"  {c}: {counts.get(i, 0)} ({100*counts.get(i,0)/len(y):.2f}%)")

    return X, RR, y, patient_ids


# ==========================================================
# 4. Patient-wise split (same logic as binary pipeline)
# ==========================================================

def patient_wise_split(X, RR, y, patient_ids):
    """
    DS1 (train+val) / DS2 (test) per de Chazal et al. - a fixed partition,
    not a random draw. We carve a validation set out of DS1 patients only
    (still patient-wise disjoint), and DS2 is never touched until final
    test evaluation. This matches standard inter-patient evaluation
    practice in the literature.
    """
    is_ds1 = np.isin(patient_ids, DS1_RECORDS)
    is_ds2 = np.isin(patient_ids, DS2_RECORDS)
    assert is_ds1.sum() + is_ds2.sum() == len(patient_ids), \
        "Some beats belong to records outside DS1/DS2 - check RECORDS list."

    X_ds1, RR_ds1, y_ds1, p_ds1 = X[is_ds1], RR[is_ds1], y[is_ds1], patient_ids[is_ds1]
    X_test, RR_test, y_test, p_test = X[is_ds2], RR[is_ds2], y[is_ds2], patient_ids[is_ds2]

    # patient-wise validation carve-out, DS1 patients only
    gss = GroupShuffleSplit(test_size=0.15, random_state=SEED, n_splits=1)
    tr_idx, val_idx = next(gss.split(X_ds1, y_ds1, groups=p_ds1))

    X_train, RR_train, y_train, p_train = X_ds1[tr_idx], RR_ds1[tr_idx], y_ds1[tr_idx], p_ds1[tr_idx]
    X_val, RR_val, y_val, p_val = X_ds1[val_idx], RR_ds1[val_idx], y_ds1[val_idx], p_ds1[val_idx]

    assert not (set(p_train) & set(p_val))
    assert not (set(p_train) & set(p_test))
    assert not (set(p_val) & set(p_test))

    print("\nTrain:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = Counter(labels)
        dist = ", ".join(f"{AAMI_CLASSES[i]}={counts.get(i,0)}" for i in range(5))
        print(f"  {name}: {dist}")

    return (X_train, RR_train, y_train, X_val, RR_val, y_val,
            X_test, RR_test, y_test)


# ==========================================================
# 5. Dataset with augmentation
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
            scale = np.random.uniform(*AUG_SCALE_RANGE)
            beat = beat * scale + np.random.normal(0, AUG_NOISE_STD, size=beat.shape)
        return torch.tensor(beat, dtype=torch.float32), self.rr[idx], self.labels[idx]


# ==========================================================
# 6. Efficient SE-Residual CNN (same size as the binary model - NOT scaled
#    up, to keep the edge-deployment story real)
# ==========================================================

class SEBlock(nn.Module):
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
        return self.features(x).squeeze(-1)


class RRFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.network(x)


class ECGClassifierMultiClass(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ecg_extractor = ECGFeatureExtractor()
        self.rr_extractor = RRFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(160, 64), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, rr):
        ecg_features = self.ecg_extractor(ecg)
        rr_features = self.rr_extractor(rr)
        features = torch.cat([ecg_features, rr_features], dim=1)
        return self.classifier(features)


# ==========================================================
# 7. Focal loss (multi-class) - targets hard/minority examples directly,
#    a better fit than plain class-weighted CE for the very rare F/Q classes
# ==========================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha  # tensor of per-class weights, or None
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1 - p_t) ** self.gamma * logp_t
        return loss.mean()


# ==========================================================
# 8. Train / eval helpers
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


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for ecg, rr, labels in loader:
            ecg, rr, labels = ecg.to(device), rr.to(device), labels.to(device)
            outputs = model(ecg, rr)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    loss = running_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_all, preds_all, labels=list(range(5)), zero_division=0
    )
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

    return dict(
        loss=loss, accuracy=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
        per_class_precision=precision, per_class_recall=recall,
        per_class_f1=f1, per_class_support=support,
        preds=np.array(preds_all), labels=np.array(labels_all)
    )


def update_bn_two_input(loader, model, device):
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None

    was_training = model.training
    model.train()
    with torch.no_grad():
        for batch in loader:
            ecg, rr = batch[0].to(device), batch[1].to(device)  # works for 3- or 4-tuples
            model(ecg, rr)

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


# ==========================================================
# 9. Edge-deployment metrics: param count, approx FLOPs, CPU latency
# ==========================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_conv1d_flops(model, input_length=361):
    """Rough MACs estimate for Conv1d/Linear layers (dominant cost)."""
    total_macs = 0
    length = input_length
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            out_length = length  # 'same' padding used throughout
            macs = (module.in_channels * module.out_channels *
                    module.kernel_size[0] * out_length / module.groups)
            total_macs += macs
        elif isinstance(module, nn.MaxPool1d):
            length = length // module.kernel_size
        elif isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features
    return total_macs  # multiply-accumulates; ~2x for FLOPs if needed


def benchmark_cpu_latency(model, n_runs=100):
    """Single-beat CPU inference latency - the realistic edge-deployment number."""
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    dummy_ecg = torch.randn(1, 2, 361)
    dummy_rr = torch.randn(1, 4)

    with torch.no_grad():
        for _ in range(10):  # warmup
            cpu_model(dummy_ecg, dummy_rr)

        start = time.perf_counter()
        for _ in range(n_runs):
            cpu_model(dummy_ecg, dummy_rr)
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000  # ms per beat


# ==========================================================
# 10. Main
# ==========================================================

def main():
    ensure_dataset()
    X, RR, y, patient_ids = build_dataset()

    (X_train, RR_train, y_train, X_val, RR_val, y_val,
     X_test, RR_test, y_test) = patient_wise_split(X, RR, y, patient_ids)

    rr_scaler = StandardScaler()
    RR_train = rr_scaler.fit_transform(RR_train)
    RR_val = rr_scaler.transform(RR_val)
    RR_test = rr_scaler.transform(RR_test)

    train_dataset = ECGDataset(X_train, RR_train, y_train, augment=True)
    val_dataset = ECGDataset(X_val, RR_val, y_val, augment=False)
    test_dataset = ECGDataset(X_test, RR_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Class-Balanced focal-loss alpha (Cui et al., CVPR 2019) ----
    # NOT raw inverse frequency. Raw 1/count produced a 143x weight ratio
    # between F (rarest, 413 train examples) and N (58k) in an earlier run,
    # which combined with focal loss's own (1-p)^gamma rarity weighting,
    # caused the model to over-predict F so badly that ~22% of true N beats
    # in the test set were misclassified as F. Effective-number reweighting
    # is deliberately gentler on very rare classes for exactly this reason.
    class_counts = Counter(y_train)
    counts_arr = np.array([class_counts.get(i, 1) for i in range(5)], dtype=np.float64)
    effective_num = 1.0 - np.power(CB_BETA, counts_arr)
    alpha = (1.0 - CB_BETA) / effective_num
    alpha = alpha / alpha.sum() * 5  # normalize so mean weight ~1
    alpha = torch.tensor(alpha, dtype=torch.float32)
    print("\nClass-balanced focal loss alpha:",
          {AAMI_CLASSES[i]: round(float(alpha[i]), 3) for i in range(5)})
    print(f"(max/min ratio = {float(alpha.max()/alpha.min()):.1f}x - "
          f"compare to raw inverse-frequency, which would be far more extreme)")

    model = ECGClassifierMultiClass(num_classes=5).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
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

    history = {k: [] for k in ["train_loss", "val_loss", "macro_f1", "weighted_f1", "accuracy"]}

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_active = True

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["macro_f1"].append(val_metrics["macro_f1"])
        history["weighted_f1"].append(val_metrics["weighted_f1"])
        history["accuracy"].append(val_metrics["accuracy"])

        lr_now = optimizer.param_groups[0]["lr"]
        phase = "SWA" if epoch >= swa_start_epoch else ("warmup" if epoch < WARMUP_EPOCHS else "cosine")
        print(f"\nEpoch {epoch+1}/{EPOCHS} [{phase}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Acc: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} "
              f"| Weighted F1: {val_metrics['weighted_f1']:.4f} | LR: {lr_now:.6f}")
        per_class_str = " | ".join(
            f"{AAMI_CLASSES[i]}: F1={val_metrics['per_class_f1'][i]:.3f}" for i in range(5)
        )
        print(f"Per-class F1 -> {per_class_str}")

        if np.isnan(train_loss) or np.isnan(val_metrics["loss"]):
            print("\nNaN loss detected - stopping training.")
            break

    if swa_active:
        print("\nRecalibrating BatchNorm statistics for the SWA-averaged model...")
        update_bn_two_input(train_loader, swa_model, device=device)
        model = swa_model.module
        print("Using SWA-averaged model as the final model.")

    torch.save(model.state_dict(), MODEL_DIR / "best_ecg_classifier_multiclass.pth")

    # ---- Final test evaluation ----
    test_metrics = evaluate(model, test_loader, criterion)

    print("\n=== Final Test Set Results (Multi-class AAMI) ===")
    print(f"Loss:        {test_metrics['loss']:.4f}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print("\nPer-class breakdown:")
    print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<10}{'Support'}")
    for i in range(5):
        print(f"{AAMI_CLASSES[i]:<8}"
              f"{test_metrics['per_class_precision'][i]:<12.4f}"
              f"{test_metrics['per_class_recall'][i]:<12.4f}"
              f"{test_metrics['per_class_f1'][i]:<10.4f}"
              f"{test_metrics['per_class_support'][i]}")

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"], labels=list(range(5)))
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{c:>6}" for c in AAMI_CLASSES))
    for i, row in enumerate(cm):
        print(f"{AAMI_CLASSES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    print("\n" + classification_report(
        test_metrics["labels"], test_metrics["preds"],
        labels=list(range(5)), target_names=AAMI_CLASSES, zero_division=0
    ))

    # ---- Edge-deployment metrics ----
    n_params = count_params(model)
    macs = estimate_conv1d_flops(model)
    cpu_latency_ms = benchmark_cpu_latency(model)

    print("\n=== Edge-Deployment Metrics ===")
    print(f"Trainable parameters: {n_params:,}")
    print(f"Approx. MACs per inference (Conv1d/Linear only): {macs:,.0f} (~{macs/1e6:.2f} M)")
    print(f"CPU latency per beat: {cpu_latency_ms:.3f} ms (single-sample, no batching)")
    print("(Report these alongside accuracy in the paper's deployment section - "
          "this is what makes the edge claim verifiable rather than asserted.)")

    # ---- Plots ----
    plt.figure(figsize=(8, 5))
    plt.plot(history["macro_f1"], label="Val Macro F1")
    plt.plot(history["weighted_f1"], label="Val Weighted F1")
    plt.plot(history["accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
    plt.title("Multi-class Validation Metrics")
    plt.savefig(FIGURE_DIR / "multiclass_training_curves.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Confusion Matrix (AAMI classes)")
    plt.colorbar()
    plt.xticks(range(5), AAMI_CLASSES); plt.yticks(range(5), AAMI_CLASSES)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig(FIGURE_DIR / "multiclass_confusion_matrix.png")
    plt.close()

    print(f"\nPlots saved to {FIGURE_DIR}")
    print(f"Model saved to {MODEL_DIR / 'best_ecg_classifier_multiclass.pth'}")


if __name__ == "__main__":
    main()