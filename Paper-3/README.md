# ECG Preprocessing Pipeline — MIT-BIH Arrhythmia Dataset

## Overview

This module implements the preprocessing pipeline used for preparing ECG signals from the MIT-BIH Arrhythmia Database for arrhythmia classification. The preprocessing steps clean the raw ECG signal, extract individual heartbeat segments, and normalize them to create consistent inputs for deep learning models.

The output of this pipeline is a set of normalized heartbeat segments ready for model training.

---

## Dataset Used

The preprocessing is performed on the **MIT-BIH Arrhythmia Database**, which contains ECG recordings sampled at **360 Hz** with expert annotations for heartbeat locations.

Each record includes:

* Raw ECG signal
* Annotation files marking R-peak locations
* Header information

---

## Preprocessing Steps

### 1. ECG Signal Loading

ECG signals are loaded from dataset records, and the MLII lead is selected for analysis.

Purpose:

* Obtain raw heart signal for preprocessing.

---

### 2. Wavelet Threshold Denoising

Wavelet decomposition using the **db4 wavelet** is applied to remove high-frequency noise. Soft thresholding is used so that noisy components are reduced while preserving heartbeat morphology.

Purpose:

* Remove muscle noise and electrical interference.
* Preserve QRS complexes and waveform structure.

---

### 3. Baseline Drift Removal

A high-pass Butterworth filter with zero-phase filtering is used to remove slow baseline drift caused by breathing and motion artifacts.

Purpose:

* Stabilize ECG baseline.
* Improve beat detection and segmentation accuracy.

---

### 4. R-Peak Extraction

R-peak locations are obtained from dataset annotations.

Purpose:

* Identify heartbeat positions for segmentation.

---

### 5. Heartbeat Segmentation

For each R-peak, a window of **270 samples (0.75 seconds)** centered around the peak is extracted.

Purpose:

* Capture one complete heartbeat cycle.
* Create fixed-length inputs for machine learning models.

Output format:

```
(number_of_beats, 270 samples)
```

---

### 6. Z-score Normalization

Each heartbeat segment is normalized using Z-score normalization to center signals around zero and scale them based on variance.

Purpose:

* Standardize signal scale.
* Improve neural network training stability.

---

## Output

The preprocessing pipeline produces:

* Clean heartbeat segments
* Consistent sample length
* Normalized ECG data

This data can be directly used for CNN, LSTM, or hybrid deep learning models.

---

## Requirements

Required Python libraries:

```
numpy
matplotlib
pywt
scipy
wfdb
```

Install using:

```bash
pip install numpy matplotlib pywt scipy wfdb
```

---

## Pipeline Summary

```
Raw ECG
   ↓
Wavelet Threshold Denoising
   ↓
Baseline Drift Removal
   ↓
R-Peak Detection
   ↓
Heartbeat Segmentation
   ↓
Z-score Normalization
   ↓
Preprocessed Dataset
```

---

