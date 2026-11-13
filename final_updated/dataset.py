"""
dataset.py  --  PhysioGuard-Space, Stage 1 (Earth Baseline)

MIT-BIH Arrhythmia Database loader with the strict inter-patient
de Chazal et al. (2004) DS1 / DS2 protocol (0 % patient leakage).

Design notes
------------
* Beats are cut around annotated R-peaks: `pre` samples before and `post`
  samples after -> fixed window length L = pre + post = 256 samples (~0.71 s @ 360 Hz).
  L = 256 keeps the first-layer activation small for the <512 KB SRAM budget.
* Labels follow the AAMI EC57 5-class mapping: N, S, V, F, Q.
* Optional 2-lead output (MLII + companion lead) so that the Stage-2 simulator
  can apply a *true* 2-D cardiac-vector rotation. Default is 1 lead (MLII).
* DS1 is further split BY RECORD into train / val (no beat-level split, so the
  validation set is also patient-disjoint from the training set).
* Beat order inside a record is preserved (`sample_idx`) so that a
  chronological TTA stream (e.g. 10,000 beats) can be built with
  `chronological_stream_indices`.

Tensor-shape legend:  B = batch, C = n_leads (+3 RR-interval channels when
cfg.use_rr_features, i.e. C = cfg.in_channels), L = 256 samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #
FS_MITBIH: int = 360  # Hz

# de Chazal inter-patient split (paced records 102, 104, 107, 217 excluded).
DS1_RECORDS: List[int] = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
                          124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS: List[int] = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
                          212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Record-level validation hold-out drawn from DS1 (has S, V and F morphologies).
# Tunable, but must remain a subset of DS1 and never touch DS2.
DS1_VAL_RECORDS: List[int] = [115, 203, 209, 230]
DS1_TRAIN_RECORDS: List[int] = [r for r in DS1_RECORDS if r not in DS1_VAL_RECORDS]

# AAMI EC57 classes.
CLASS_NAMES: List[str] = ["N", "S", "V", "F", "Q"]
NUM_CLASSES: int = len(CLASS_NAMES)
AAMI_MAP: Dict[str, int] = {
    # N: normal & bundle-branch / escape beats
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S: supraventricular ectopic
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V: ventricular ectopic
    "V": 2, "E": 2,
    # F: fusion
    "F": 3,
    # Q: unknown / paced
    "/": 4, "f": 4, "Q": 4,
}


def assert_no_leakage(train_records: Sequence[int],
                      val_records: Sequence[int],
                      test_records: Sequence[int]) -> None:
    """Hard guarantee of 0 % inter-patient leakage between the three splits."""
    tr, va, te = set(train_records), set(val_records), set(test_records)
    assert not (tr & te), f"LEAKAGE train/test: {sorted(tr & te)}"
    assert not (va & te), f"LEAKAGE val/test: {sorted(va & te)}"
    assert not (tr & va), f"LEAKAGE train/val: {sorted(tr & va)}"
    assert set(DS1_RECORDS).isdisjoint(DS2_RECORDS), "DS1/DS2 definitions overlap"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ECGDataConfig:
    data_dir: str = "./data/mitdb"        # folder holding *.dat/*.hea/*.atr
    cache_dir: str = "./data/cache"
    n_leads: int = 1                      # 1 -> MLII ; 2 -> MLII + companion lead
    pre: int = 96                         # samples before R-peak
    post: int = 160                       # samples after  R-peak  (L = 256)
    norm: str = "record"                  # "record" | "beat" | "none"
    bandpass: bool = True                 # 0.5-40 Hz zero-phase Butterworth
    band: Tuple[float, float] = (0.5, 40.0)
    batch_size: int = 128
    num_workers: int = 0
    balanced_sampling: bool = True        # sqrt-inverse-frequency sampler on train
    seed: int = 0
    # RR-interval (beat timing) features -- de Chazal et al. 2004 (the paper that defines
    # the DS1/DS2 protocol used throughout this project) relies specifically on RR timing
    # to separate S (supraventricular ectopic, often morphologically ~normal but premature)
    # from N. Without this, the model only ever sees beat *shape*, never *when* a beat
    # occurred relative to its neighbors -- a real missing information source, not just a
    # capacity shortfall. Implemented as 3 extra constant-valued channels (pre_rr, post_rr,
    # local_avg_rr-over-9-beats, each a scalar repeated across L samples) concatenated onto
    # the waveform channels, so the existing stem Conv1d (which already mixes across input
    # channels) learns how to use them; no change to the Dataset/DataLoader/model forward
    # signatures is required beyond `in_channels`.
    use_rr_features: bool = True
    n_rr_features: int = 3
    # Patient-relative beat context (causal): add 2 channels built from the SAME patient's previous
    # n_template beats -- their sample-wise median ("this patient's normal beat") and (beat - median).
    # Judging a beat against the patient's own recent beats targets the inter-patient shift directly.
    # 0 = off. Uses only past beats, so it can run on-device (a 16 x 256 buffer + one median per beat).
    n_template: int = 0
    # Extra TRAINING-ONLY recordings from another PhysioNet database (e.g. the MIT-BIH Supraventricular Arrhythmia
    # Database, ids 800-894, recorded at 128 Hz and resampled to 360 Hz). They are appended to the DS1 training set
    # by build_datasets and are never used for validation or test. Not part of the checkpoint data contract.
    extra_dir: str = ""
    extra_records: Tuple[int, ...] = ()
    # Second extra database: INCART (ids 901..975 <-> files I01..I75, 257 Hz, 12 leads; lead II is used).
    extra2_dir: str = ""
    extra2_records: Tuple[int, ...] = ()
    # Third extra database: Icentia11k (ids 1_000_000 + patient*100 + segment <-> pXXXXX/pXXXXX_sYY, 250 Hz, 1 lead,
    # beat labels N/S/V/Q). Q beats are dropped (unclassifiable), normals are subsampled to extra3_max_normal per segment
    # AFTER the RR/template features are computed on the full beat sequence.
    extra3_dir: str = ""
    extra3_records: Tuple[int, ...] = ()
    extra3_max_normal: int = 400

    @property
    def window_len(self) -> int:
        return self.pre + self.post

    @property
    def in_channels(self) -> int:
        """Waveform-only conv input channels (RR features are fused late -- see
        base_cnn.CNNConfig.aux_feat_dim -- so they do NOT add to this)."""
        return self.n_leads + (2 if self.n_template > 0 else 0)

    @property
    def aux_feat_dim(self) -> int:
        """Size of the late-fused auxiliary feature vector (0 if RR features are off)."""
        return self.n_rr_features if self.use_rr_features else 0


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """sig: [T, C] -> [T, C]. Zero-phase (offline) filtering."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def _normalize_record(sig: np.ndarray) -> np.ndarray:
    """Per-record, per-channel z-score. sig: [T, C] -> [T, C]."""
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True) + 1e-8
    return (sig - mu) / sd


def download_mitdb(data_dir: str) -> None:
    """Fetch MIT-BIH from PhysioNet (requires `wfdb` and internet, run once)."""
    import wfdb
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=data_dir,
                     records=[str(r) for r in DS1_RECORDS + DS2_RECORDS])


def _rr_interval_features(samples: np.ndarray, fs: float, local_window: int = 9,
                          n_features: int = 3) -> np.ndarray:
    """
    n_features=3 (default, unchanged): (pre_rr, post_rr, local_avg_rr).
    n_features=5 adds two scale-free RATIO features that directly encode prematurity, the
    defining sign of an S beat:  log(pre_rr / mean of the previous <=9 intervals) and
    log(post_rr / pre_rr)  (a premature beat: first < 0, second > 0, i.e. a compensatory
    pause). Absolute z-scored RR (above) makes the network recover this ratio itself, which
    is hard for an 18K-parameter model and breaks across heart rates (e.g. record 232).

    (Original docstring follows.)
    """
    assert n_features in (3, 5, 7), n_features
    """
    samples: int64 [N] R-peak sample indices, one per KEPT beat, in chronological order
    (ascending within a record -- true for the `ss` list built in `load_record_beats`,
    since wfdb annotations are already time-ordered and we only ever skip/keep in place).

    Returns rr [N, 3] = (pre_rr, post_rr, local_avg_rr), each in seconds, z-normalized
    with a fixed physiological center/scale (0.8 s ~ 75 bpm, 0.3 s spread) and clipped to
    [-3, 3] so a single very long pause doesn't blow up the input scale. Edge beats (first/
    last in a record, where one neighbor doesn't exist) fall back to their one valid
    neighbor, or to the record's median RR if the record has only one beat.

    Caveat: computed from the KEPT beat series (after dropping non-beat annotations and
    beats whose window falls off the record edge), not the fully raw annotation stream.
    The AAMI_MAP beat/non-beat split already matches what de Chazal et al. treat as a beat;
    dropped edge-window beats are a handful per record and have a negligible effect on
    neighboring RR values.
    """
    n = len(samples)
    pre_rr = np.full(n, np.nan, dtype=np.float64)
    post_rr = np.full(n, np.nan, dtype=np.float64)
    local_rr = np.full(n, np.nan, dtype=np.float64)
    s = samples.astype(np.float64)
    for i in range(n):
        if i > 0:
            pre_rr[i] = (s[i] - s[i - 1]) / fs
        if i < n - 1:
            post_rr[i] = (s[i + 1] - s[i]) / fs
        lo = max(0, i - local_window)
        if i > lo:
            local_rr[i] = np.mean(np.diff(s[lo:i + 1])) / fs

    valid = np.concatenate([pre_rr[~np.isnan(pre_rr)], post_rr[~np.isnan(post_rr)]])
    fallback = float(np.median(valid)) if valid.size else 0.8
    pre_rr = np.where(np.isnan(pre_rr), np.where(~np.isnan(post_rr), post_rr, fallback), pre_rr)
    post_rr = np.where(np.isnan(post_rr), np.where(~np.isnan(pre_rr), pre_rr, fallback), post_rr)
    local_rr = np.where(np.isnan(local_rr), pre_rr, local_rr)

    def _z(a: np.ndarray) -> np.ndarray:
        return np.clip((a - 0.8) / 0.3, -3.0, 3.0).astype(np.float32)

    cols = [_z(pre_rr), _z(post_rr), _z(local_rr)]
    if n_features >= 5:
        # mean of the intervals BEFORE the current one (excludes pre_rr itself), so a premature
        # beat is not diluted by its own short interval; fallback = pre_rr (ratio 0)
        prev_avg = pre_rr.copy()
        for i in range(2, n):
            lo = max(0, i - 1 - local_window)
            prev_avg[i] = np.mean(np.diff(s[lo:i])) / fs
        eps = 1e-3
        r_pre = np.log((pre_rr + eps) / (prev_avg + eps))
        r_post = np.log((post_rr + eps) / (pre_rr + eps))
        cols += [np.clip(r_pre / 0.25, -3.0, 3.0).astype(np.float32),
                 np.clip(r_post / 0.25, -3.0, 3.0).astype(np.float32)]
    if n_features == 7:
        # causal LONG-window prematurity ratios (Huang 2014 uses the whole-record mean; Farag 2023 the last
        # 80 / 400 beats): log(pre_rr / mean of the previous <=80 intervals) and of the previous <=400.
        # Uses only past beats, so it can run on-device. Fallback (too little history) = ratio 0.
        d = np.diff(s) / fs                    # d[j] = interval ending at beat j+1
        cs = np.concatenate([[0.0], np.cumsum(d)])
        eps = 1e-3
        for W in (80, 400):
            r = np.zeros(n, dtype=np.float64)
            for i in range(2, n):              # previous intervals = d[max(0,i-1-W) : i-1]
                lo = max(0, i - 1 - W)
                r[i] = np.log((pre_rr[i] + eps) / ((cs[i - 1] - cs[lo]) / (i - 1 - lo) + eps))
            cols.append(np.clip(r / 0.25, -3.0, 3.0).astype(np.float32))
    return np.stack(cols, axis=1)   # [N, 3], [N, 5] or [N, 7]


def add_template_channels(x: np.ndarray, k: int) -> np.ndarray:
    """x [N, C, L] (chronological) -> [N, C+2, L]: appends, for lead 0, the sample-wise MEDIAN of the previous
    <=k beats (the patient's own recent beat; the first beat uses itself) and (beat - median).
    Strictly causal: row i depends only on rows <= i-1 (plus row i itself for the difference)."""
    n, c, L = x.shape
    tmpl = np.empty((n, L), dtype=np.float32)
    for i in range(n):
        tmpl[i] = np.median(x[max(0, i - k):i, 0, :], axis=0) if i > 0 else x[0, 0, :]
    return np.concatenate([x, tmpl[:, None, :], (x[:, 0, :] - tmpl)[:, None, :]], axis=1).astype(np.float32)


def load_record_beats(record: int, cfg: ECGDataConfig
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one MIT-BIH record and cut it into labelled beat windows.

    Returns
    -------
    x       : float32 [N, C, L]   beat windows (C = cfg.n_leads; the waveform ONLY --
              RR features are returned separately and fused late, after pooling, not
              run through the conv stem; see base_cnn.CNNConfig.aux_feat_dim)
    y       : int64   [N]         AAMI class index
    samples : int64   [N]         R-peak sample index (defines chronological order)
    rr      : float32 [N, 3]      (pre_rr, post_rr, local_avg_rr), z-normalized; shape
              [N, 0] if cfg.use_rr_features is False. See `_rr_interval_features`.
    """
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    # "rrL" ("RR, Late-fused") -- deliberately distinct from the older, short-lived
    # "_rr{0|1}{n}" tag used by a channel-injection design (RR baked into x as extra
    # constant-valued conv-input channels) that this project tried and found regressed
    # DS1-val metrics; that tag is kept distinguishable on purpose so a leftover cache
    # from that design is never silently reused as if it matched this (different) schema.
    tag = (f"{record}_c{cfg.n_leads}_p{cfg.pre}_q{cfg.post}_{cfg.norm}"
           f"_bp{int(cfg.bandpass)}_{cfg.band[0]}-{cfg.band[1]}"
           f"_rrL{cfg.n_rr_features if cfg.use_rr_features else 0}"
           + (f"_tm{cfg.n_template}" if cfg.n_template > 0 else ""))
    cache_file = cache / f"{tag}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        return d["x"], d["y"], d["samples"], d["rr"]

    import wfdb  # lazy import so the module is importable without wfdb
    is_extra = record in set(cfg.extra_records)
    is_inc = record in set(cfg.extra2_records)
    is_ic = record in set(cfg.extra3_records)
    if is_ic:
        pat, seg_no = (record - 1_000_000) // 100, (record - 1_000_000) % 100
        path = str(Path(cfg.extra3_dir) / f"p{pat // 1000:02d}" / f"p{pat:05d}" / f"p{pat:05d}_s{seg_no:02d}")
    elif is_inc:
        path = str(Path(cfg.extra2_dir) / f"I{record - 900:02d}")
    else:
        path = str(Path(cfg.extra_dir if is_extra else cfg.data_dir) / str(record))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    names = [n.strip() for n in rec.sig_name]
    if is_ic:                                              # Icentia11k: single lead
        if cfg.n_leads != 1:
            raise ValueError("Icentia11k is single-lead: use n_leads=1")
        lead_idx = [0]
    elif is_inc:                                           # INCART: lead II (+ lead V1 as companion)
        li = names.index("II")
        lead_idx = [li] if cfg.n_leads == 1 else [li, names.index("V1")]
    elif is_extra:                                         # SVDB: ECG1 (+ ECG2); no MLII label
        lead_idx = [0] if cfg.n_leads == 1 else [0, 1]
    else:
        if "MLII" not in names:
            raise ValueError(f"Record {record}: MLII lead not found in {names}")
        lead_idx = [names.index("MLII")]                   # handles record 114 (swapped)
        if cfg.n_leads == 2:
            lead_idx.append(next(i for i in range(len(names)) if i != lead_idx[0]))
        elif cfg.n_leads != 1:
            raise ValueError("n_leads must be 1 or 2")

    sig = rec.p_signal[:, lead_idx].astype(np.float64)     # [T, C]
    fs_rec = float(rec.fs)
    asamples = np.asarray(ann.sample)
    if abs(fs_rec - FS_MITBIH) > 1e-6:                     # resample to 360 Hz, rescale annotation positions
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(FS_MITBIH), int(round(fs_rec)))
        sig = resample_poly(sig, int(FS_MITBIH) // g, int(round(fs_rec)) // g, axis=0)
        asamples = np.round(asamples * FS_MITBIH / fs_rec).astype(np.int64)
        fs_rec = float(FS_MITBIH)
    T = sig.shape[0]
    if cfg.bandpass:
        sig = _bandpass(sig, fs_rec, *cfg.band)            # [T, C]
    if cfg.norm == "record":
        sig = _normalize_record(sig)                       # [T, C]

    xs, ys, ss = [], [], []
    for s, sym in zip(asamples, ann.symbol):
        cls = AAMI_MAP.get(sym)
        if cls is None:                                    # non-beat annotation
            continue
        a, b = int(s) - cfg.pre, int(s) + cfg.post
        if a < 0 or b > T:
            continue
        seg = sig[a:b].T                                   # [C, L]
        if cfg.norm == "beat":
            seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-8)
        xs.append(seg.astype(np.float32))
        ys.append(cls)
        ss.append(int(s))

    x = np.stack(xs).astype(np.float32)                    # [N, C, L]
    if cfg.n_template > 0 and len(x) > 0:
        x = add_template_channels(x, cfg.n_template)       # [N, C+2, L], causal patient-relative context
    y = np.asarray(ys, dtype=np.int64)                     # [N]
    samples = np.asarray(ss, dtype=np.int64)               # [N]

    if cfg.use_rr_features and len(samples) > 0:
        rr = _rr_interval_features(samples, fs_rec, n_features=cfg.n_rr_features)   # [N, 3|5]
    else:
        rr = np.zeros((len(samples), 0), dtype=np.float32)

    if is_ic and len(y) > 0:                               # drop Q, subsample normals (features already computed)
        rng = np.random.default_rng(int(record))
        keep = np.flatnonzero((y != 4) & (y != 0))
        nrm = np.flatnonzero(y == 0)
        if len(nrm) > cfg.extra3_max_normal:
            nrm = rng.choice(nrm, cfg.extra3_max_normal, replace=False)
        sel = np.sort(np.concatenate([keep, nrm]))
        x, y, samples, rr = x[sel], y[sel], samples[sel], rr[sel]
    np.savez_compressed(cache_file, x=x, y=y, samples=samples, rr=rr)
    return x, y, samples, rr


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MITBIHDataset(Dataset):
    """
    In-memory beat dataset.

    __getitem__ -> (x, y) with x: float32 [C, L], y: int64 scalar -- or, when
    cfg.use_rr_features is True, (x, rr, y) with rr: float32 [cfg.n_rr_features].
    Check `self.cfg.use_rr_features` (or `self.cfg.aux_feat_dim > 0`) to know which
    arity a given dataset yields; every caller that may see use_rr_features=True
    must unpack accordingly (train_stage1.py / eval_cross_domain.py both do).

    Extra attributes (used by TTA streams / metrics):
        self.x           [N, C, L]        waveform only, RR never enters the conv input
        self.rr          [N, n_rr_features]   empty (N,0) if use_rr_features is False
        self.y           [N]
        self.record_ids  [N]   patient/record id of every beat
        self.sample_idx  [N]   R-peak index inside its record (chronology)
    `transform` (optional) maps a single beat [C, L] -> [C, L]; this is the hook
    the Stage-2 space-shift simulator plugs into. RR features are NOT passed through
    `transform` -- the simulator corrupts the waveform, not beat timing.
    """

    def __init__(self, records: Sequence[int], cfg: ECGDataConfig,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.cfg = cfg
        self.records = list(records)
        self.transform = transform

        xs, ys, rids, sidx, rrs = [], [], [], [], []
        for r in self.records:
            x, y, s, rr = load_record_beats(r, cfg)
            xs.append(x); ys.append(y); sidx.append(s); rrs.append(rr)
            rids.append(np.full(len(y), r, dtype=np.int64))

        self.x = torch.from_numpy(np.concatenate(xs))                 # [N, C, L]
        self.y = torch.from_numpy(np.concatenate(ys))                 # [N]
        self.record_ids = torch.from_numpy(np.concatenate(rids))      # [N]
        self.sample_idx = torch.from_numpy(np.concatenate(sidx))      # [N]
        self.rr = torch.from_numpy(np.concatenate(rrs))               # [N, n_rr_features or 0]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        x = self.x[i]                                                 # [C, L]
        if self.transform is not None:
            x = self.transform(x)                                     # [C, L]
        if self.cfg.use_rr_features:
            return x, self.rr[i], self.y[i]
        return x, self.y[i]

    # -- utilities ---------------------------------------------------------- #
    def class_counts(self) -> torch.Tensor:
        """[NUM_CLASSES] beat counts."""
        return torch.bincount(self.y, minlength=NUM_CLASSES)

    def class_weights(self, power: float = 0.5) -> torch.Tensor:
        """[NUM_CLASSES] normalized inverse-frequency^power weights for the loss."""
        cnt = self.class_counts().float().clamp_min(1.0)
        w = (cnt.sum() / cnt) ** power
        return w / w.mean()

    def sampler(self, power: float = 0.5, seed: int = 0) -> WeightedRandomSampler:
        """Per-sample weights ~ (1/freq)^power for class-balanced mini-batches."""
        per_class = 1.0 / self.class_counts().float().clamp_min(1.0) ** power   # [K]
        w = per_class[self.y]                                                    # [N]
        g = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(w, num_samples=len(self), replacement=True, generator=g)


# --------------------------------------------------------------------------- #
# Loaders & TTA stream
# --------------------------------------------------------------------------- #
def build_datasets(cfg: ECGDataConfig,
                   train_transform=None, test_transform=None,
                   val_records: Optional[Sequence[int]] = None
                   ) -> Dict[str, MITBIHDataset]:
    """DS1-train / DS1-val (record-disjoint) and DS2-test (patient-disjoint).

    `val_records` (optional) overrides the default DS1_VAL_RECORDS hold-out with another
    subset of DS1 -- used for leave-record-group-out cross-validation (cv_folds.py). DS1
    records NOT in `val_records` become the training set; DS2 is never touched by this.
    """
    if val_records is None:
        tr, va = DS1_TRAIN_RECORDS, DS1_VAL_RECORDS
    else:
        va = [int(r) for r in val_records]
        assert set(va) <= set(DS1_RECORDS), f"val_records must be a subset of DS1: {sorted(set(va) - set(DS1_RECORDS))}"
        tr = [r for r in DS1_RECORDS if r not in set(va)]
    assert_no_leakage(tr, va, DS2_RECORDS)
    extra = [int(r) for r in cfg.extra_records] + [int(r) for r in cfg.extra2_records] + [int(r) for r in cfg.extra3_records]
    assert not (set(extra) & (set(DS1_RECORDS) | set(DS2_RECORDS))), "extra record ids clash with MIT-BIH ids"
    tr = list(tr) + extra                                   # extra data is TRAIN ONLY
    return {
        "train": MITBIHDataset(tr, cfg, train_transform),
        "val":   MITBIHDataset(va, cfg, None),
        "test":  MITBIHDataset(DS2_RECORDS, cfg, test_transform),
    }


def build_dataloaders(cfg: ECGDataConfig) -> Dict[str, DataLoader]:
    ds = build_datasets(cfg)
    train_kw = (dict(sampler=ds["train"].sampler(seed=cfg.seed))
                if cfg.balanced_sampling else dict(shuffle=True))
    return {
        "train": DataLoader(ds["train"], batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, drop_last=True, **train_kw),
        "val":   DataLoader(ds["val"], batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers),
        "test":  DataLoader(ds["test"], batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers),
    }


def chronological_stream_indices(ds: MITBIHDataset, n_beats: Optional[int] = 10_000,
                                 records: Optional[Sequence[int]] = None) -> List[int]:
    """
    Indices that replay the dataset in true acquisition order
    (record by record, R-peak ascending). Used for continuous TTA / drift curves.
    """
    rec = ds.record_ids.numpy()
    smp = ds.sample_idx.numpy()
    order = np.lexsort((smp, rec))                       # primary: record, secondary: time
    if records is not None:
        keep = np.isin(rec[order], np.asarray(list(records)))
        order = order[keep]
    if n_beats is not None:
        order = order[:n_beats]
    return order.tolist()


def build_tta_stream(ds: MITBIHDataset, batch_size: int = 1,
                     n_beats: Optional[int] = 10_000,
                     records: Optional[Sequence[int]] = None) -> DataLoader:
    """Deterministic, order-preserving stream (batch_size=1 mimics beat-by-beat edge TTA)."""
    idx = chronological_stream_indices(ds, n_beats, records)
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------- #
# Self-check:  python dataset.py --data_dir ./data/mitdb [--download]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache",
                   help="preprocessed-beat cache (.npz per record). Point this at a "
                        "persistent location (e.g. a Google Drive path in Colab) so the "
                        "~8min download+preprocessing only ever has to happen once.")
    p.add_argument("--download", action="store_true")
    p.add_argument("--n_leads", type=int, default=1)
    p.add_argument("--no_rr_features", action="store_true",
                   help="disable the 3 RR-interval timing channels (pre_rr, post_rr, "
                        "local_avg_rr); on by default")
    a = p.parse_args()

    if a.download:
        download_mitdb(a.data_dir)
    cfg = ECGDataConfig(data_dir=a.data_dir, cache_dir=a.cache_dir, n_leads=a.n_leads,
                        use_rr_features=not a.no_rr_features)
    print(f"use_rr_features={cfg.use_rr_features}  conv in_channels={cfg.in_channels}  "
          f"late-fused aux_feat_dim={cfg.aux_feat_dim} (RR is fused after pooling, "
          f"not added to in_channels)")
    sets = build_datasets(cfg)
    for k, d in sets.items():
        print(f"{k:5s} beats={len(d):6d}  x={tuple(d.x.shape)}  rr={tuple(d.rr.shape)}  "
              f"class_counts={dict(zip(CLASS_NAMES, d.class_counts().tolist()))}")
    print("Leakage check passed: DS1-train / DS1-val / DS2-test are record-disjoint.")