# ECG Arrhythmia Classification – Preprocessing & Augmentation Pipeline

This repository contains the **complete preprocessing pipeline** implemented as part of a research project inspired by the paper:

> **Kanani, P., & Padole, M. (2020). _ECG Heartbeat Arrhythmia Classification Using Time-Series Augmented Signals and Deep Learning Approach_. Procedia Computer Science, 171, 524–531.**

The focus of this repository is **data preprocessing and augmentation**, which plays a critical role in improving deep learning performance on ECG arrhythmia classification tasks.

---

## 📌 Project Overview

Electrocardiogram (ECG) signals are non-stationary, patient-dependent, and highly imbalanced across arrhythmia classes. Instead of relying on heavy signal filtering or handcrafted features, this project implements a **time-domain, morphology-preserving preprocessing pipeline** that:

- Segments ECG signals into individual heartbeats
- Normalizes beats to a standard range
- Applies **lossless time-series augmentation**
- Maps annotations to clinically meaningful AAMI classes
- Produces CNN-ready input data

This preprocessing pipeline is designed to improve:
- Generalization
- Minority-class recognition
- Training stability of deep learning models

---

## 🧠 Dataset

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Sampling Frequency**: 360 Hz
- **Lead Used**: MLII (single-lead ECG)
- **Annotations**: Expert-labeled R-peaks and beat symbols (`.atr` files)

> ⚠️ The dataset is **not included** in this repository. Please download it from PhysioNet and place it in the `dataset/` directory.

---

## 🔁 Preprocessing Pipeline

The following pipeline is fully implemented and verified:

```text
Raw ECG
 → R-peak detection (MIT-BIH annotations)
 → R-to-R beat segmentation
 → Min–max normalization
 → Time & amplitude augmentation
 → Fix length (pad / truncate)
 → AAMI class mapping (5 classes)
 → Label encoding
 → CNN reshape (samples, 300, 1)
```

### 1️⃣ R-Peak Based Beat Segmentation
- Uses expert annotations from MIT-BIH (`.atr` files)
- Each beat is segmented from one R-peak to the next (R–R interval)
- Preserves complete P–QRS–T morphology

### 2️⃣ Min–Max Normalization
- Each beat is scaled to the range **[0, 1]**
- Reduces inter-patient amplitude variation
- Compatible with amplitude-based augmentation

### 3️⃣ Time-Series Augmentation (Core Contribution)

Each ECG beat is augmented using four **lossless transformations**:

| Transformation | Description | Physiological Meaning |
|--------------|------------|----------------------|
| Squeeze | Time compression | Tachycardia |
| Stretch | Time expansion | Bradycardia |
| Amplify | Amplitude scaling | Signal strength variation |
| Shrink | Time + amplitude reduction | Mild tachycardia / low voltage |

All augmented beats retain their original class labels.

### 4️⃣ Fixed-Length Standardization
- All beats are padded or truncated to **300 samples**
- Ensures uniform input size for CNNs

### 5️⃣ AAMI Class Mapping

Original MIT-BIH beat symbols are mapped to **5 standard AAMI classes**:

| AAMI Class | Description |
|-----------|-------------|
| N | Normal beats (including bundle branch blocks) |
| S | Supraventricular ectopic beats |
| V | Ventricular ectopic beats |
| F | Fusion beats |
| Q | Unknown / paced / noise |

This reduces label noise and enables fair comparison with existing literature.

### 6️⃣ CNN-Ready Reshaping

Final data shape:
```text
(samples, 300, 1)
```
- `300` → time steps per beat
- `1` → single ECG channel

---

## 📊 Effectiveness Verification

The effectiveness of preprocessing was evaluated at multiple levels:

### ✔ Signal-Level
- Visual inspection confirms preservation of P–QRS–T morphology
- Time-warped signals verified using DTW and frequency-domain similarity

### ✔ Data-Level
- Dataset size increased **5×** after augmentation
- Signal variance increased slightly (controlled diversity)

### ✔ Model-Level (Recommended)
- (after implementation only)

---

## 📁 Repository Structure

```text
Capstone/
│
├── dataset/                 # MIT-BIH files (.dat, .hea, .atr)
│
├── Paper2/
│   └── preprocessing_paper2.ipynb   # Main preprocessing notebook
│
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
2. Download MIT-BIH Arrhythmia Dataset from PhysioNet
3. Place records inside the `dataset/` folder
4. Open `Paper2/preprocessing_paper2.ipynb`
5. Run cells sequentially

---

## 📦 Dependencies

```bash
pip install numpy scipy matplotlib wfdb scikit-learn tensorflow
```

---

## 📚 Reference

If you use this preprocessing pipeline, please cite:

```bibtex
@article{kanani2020ecg,
  title={ECG heartbeat arrhythmia classification using time-series augmented signals and deep learning approach},
  author={Kanani, Pratik and Padole, Mamta},
  journal={Procedia Computer Science},
  volume={171},
  pages={524--531},
  year={2020}
}
```

---

## ✅ Status

✔ Preprocessing complete
✔ Augmentation verified
✔ CNN-ready dataset generated

---

## 🚀 Next Steps

- Train CNN / ResNet models
- Perform ablation studies (with vs without augmentation)
- Evaluate using F1-score and confusion matrices

---

**Author**: Sukeerthi Vijay  
**Project Type**: Academic / Research (ECG + Deep Learning)

