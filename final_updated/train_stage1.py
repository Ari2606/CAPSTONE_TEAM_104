"""
train_stage1.py  --  PhysioGuard-Space, Stage 1 (Earth Baseline)

Trains `base_cnn.PhysioGuardCNN` on DS1 (de Chazal inter-patient train split) and
model-selects on the DS1 validation split (record-disjoint from DS1-train, see
`dataset.DS1_VAL_RECORDS`) using macro-F1, because the AAMI classes are heavily
imbalanced (N >> S,V >> F,Q) and accuracy alone would hide S/F/Q collapse.

Only `backbone` + `adaptable_head` receive gradients. `frozen_anchor_head` is
inert throughout training (base_cnn already keeps it in eval mode and out of the
optimizer's parameter list); at the very end of training the BEST checkpoint's
weights are reloaded and `model.freeze_anchor()` is called once, which:
  (a) copies the best adaptable_head into frozen_anchor_head and freezes it, and
  (b) snapshots the current dw_bn affine parameters as the TTA "source" state
      used later by `elastic_reset()` (Stage 5).
Doing this AFTER model selection (not at every epoch) means theta_f is the single
best-validated source model, matching the paper's definition of the anchor.

Output: `checkpoint_stage1.pt`, matching the contract documented in
`eval_cross_domain.py` (model_state / cnn_cfg / data_cfg / epoch / val_macro_f1),
so `eval_cross_domain.py --checkpoint checkpoint_stage1.pt` runs unmodified.

Run `python train_stage1.py` with no arguments to execute a fast synthetic
self-test (no PhysioNet download needed) that exercises the exact same `train()`
function real runs use. Run with `--data_dir ...` for the real MIT-BIH run.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment import AugmentConfig, RareClassAugmentedDataset
from base_cnn import CNNConfig, PhysioGuardCNN
from dataset import CLASS_NAMES, NUM_CLASSES, ECGDataConfig, build_datasets
from eval_cross_domain import classification_report, confusion_matrix

_DATA_CONTRACT_FIELDS = ("n_leads", "pre", "post", "norm", "bandpass", "band",
                         "use_rr_features", "n_rr_features", "n_template")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    data_dir: str = "./data/mitdb"
    cache_dir: str = "./data/cache"
    n_leads: int = 1
    use_rr_features: bool = True   # 3 extra RR-interval timing channels, see dataset.py
    checkpoint_path: str = "checkpoint_stage1.pt"

    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4                          # lowered from 1e-3 -- with class weights up to
                                              # 4x (Q) and focal loss's own (1-p)^gamma term,
                                              # 1e-3 was producing large, noisy per-epoch swings
                                              # in val macro-F1 (observed: 0.25-0.45 epoch to
                                              # epoch on real MIT-BIH runs)
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    lr_schedule: bool = True                  # warmup + CosineAnnealingLR over `epochs`
    warmup_epochs: int = 3                    # linear LR warmup 0 -> lr, then cosine decay;
                                              # avoids large early updates while the model is
                                              # still random-ish and class weights are extreme

    use_ema: bool = True                      # exponential moving average of weights --
                                              # directly targets epoch-to-epoch validation
                                              # noise by evaluating/selecting on a SMOOTHED
                                              # copy of the weights instead of the raw,
                                              # noisy per-epoch ones
    ema_decay: float = 0.98

    loss_type: str = "cb_focal"                # "weighted_ce" | "focal" | "cb_focal" | "cb_ce"
                                              # cb_* = class-balanced weighting (Cui et al. 2019,
                                              # effective-number-of-samples) in place of the
                                              # sqrt-inverse-frequency weighting weighted_ce/focal
                                              # use -- see class_balanced_weights() for why this
                                              # targets the Q/F collapse more directly.
    class_weight_power: float = 0.5           # (1/freq)^power, matches dataset.py sampler
                                              # (only used by "weighted_ce"/"focal")
    cb_beta: float = 0.9999                   # only used by "cb_focal"/"cb_ce"
    label_smoothing: float = 0.0              # only used by weighted_ce
    focal_gamma: float = 2.0                  # only used by focal/cb_focal
    balanced_sampling: bool = False           # if True, sampler ALSO balances -> avoid
                                              # double-correcting on top of class-weighted loss

    patience: int = 15                        # early stop on stagnant val macro-F1 -- raised
                                              # from 10: with the warmup+cosine LR schedule, a
                                              # 10-epoch patience was firing before the LR had
                                              # decayed through most of its range (e.g. the real
                                              # run that stopped at epoch 23/40 still had
                                              # lr=1.3e-4, nowhere near the ~0 it reaches by
                                              # epoch 40), cutting the schedule off mid-decay
    min_epochs: int = 5
    min_stop_frac: float = 0.5                # early stopping can't fire before
                                              # max(min_epochs, min_stop_frac * epochs) --
                                              # keeps patience from ending a run before the
                                              # cosine LR schedule has substantially decayed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    log_every: int = 1

    # -- rare-class fixes (data augmentation + decoupled re-training) ------- #
    use_rare_class_augment: bool = False      # wrap ds_train in RareClassAugmentedDataset
    decouple: bool = False                    # Kang et al. 2020 cRT: re-train head on a
                                               # fully class-balanced sampler after normal training
    decouple_epochs: int = 5
    decouple_lr: float = 3e-4

    # -- secondary, honestly-labeled high-accuracy operating point ---------- #
    save_high_accuracy_checkpoint: bool = True   # also save the best-accuracy epoch's
    samples_per_epoch: int = 0        # 0 = len(train set) draws per epoch; >0 = cap the number of (class-balanced) draws per epoch
    num_workers: int = 0              # DataLoader workers for the per-beat augmentation (0 = main process)
    fixed_epochs: bool = False        # no early stop / best-epoch selection: train all epochs, keep the LAST (EMA) weights.
                                      # Removes validation-based selection (4 records, noisy) from the protocol entirely.
    zero_q_weight: bool = False       # class Q has ~4 training beats yet gets the largest loss weight; zero it out
                                                  # weights to a second file, alongside
                                                  # (never instead of) the shipped
                                                  # macro-F1-selected checkpoint


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def class_balanced_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """
    Class-Balanced weighting (Cui et al., CVPR 2019, "Class-Balanced Loss Based
    on Effective Number of Samples"): weight_i = (1-beta) / (1 - beta^n_i),
    renormalized to mean 1.

    Why this instead of sqrt-inverse-frequency (what the rest of this project
    has used so far): plain inverse-frequency weighting treats every extra
    sample in a class as equally informative. In practice, samples within a
    class overlap/correlate (near-duplicate beat morphologies), so the
    MARGINAL value of the n-th sample shrinks as n grows -- the "effective
    number" of samples is 1 + beta + beta^2 + ... + beta^(n-1) = (1-beta^n)/(1-beta),
    not n itself. Weighting by the inverse of that effective number (instead of
    inverse of raw n) gives rare classes relatively more weight than
    sqrt-inverse-frequency does without the instability of plain 1/n
    weighting. beta=0.9999 is the value Cui et al. report working well across
    their long-tailed benchmarks; here it specifically targets the Q/F
    collapse that sqrt-inverse-frequency (class_weight_power=0.5) left
    unresolved across every run so far.
    """
    counts = counts.float().clamp_min(1.0)
    eff_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    w = (1.0 - beta) / eff_num
    return w / w.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss: L = -alpha_t * (1 - p_t)^gamma * log(p_t).
    alpha: [K] per-class weight (same role as CrossEntropy's `weight`, renormalized
    to mean 1 so `gamma` and `alpha` don't fight over the overall loss scale).
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha / alpha.mean())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, K], target: [B]
        logp = F.log_softmax(logits, dim=-1)                      # [B, K]
        logp_t = logp.gather(1, target[:, None]).squeeze(1)       # [B]
        p_t = logp_t.exp()                                        # [B]
        alpha_t = self.alpha.to(logits.device)[target]            # [B]
        loss = -alpha_t * (1.0 - p_t).clamp_min(0.0).pow(self.gamma) * logp_t
        return loss.mean()


class EMA:
    """
    Exponential moving average of model weights, updated after every
    optimizer step. Purpose: the raw per-epoch weights are noisy (a single
    epoch's worth of SGD/Adam updates on batches that rarely contain a
    Q/F beat swings validation macro-F1 by +-0.05-0.10 epoch to epoch, as
    observed across every real MIT-BIH run this project has done). The EMA
    shadow weights average over many such noisy steps, so evaluating AND
    selecting the best checkpoint on the EMA copy -- not the raw model --
    is a standard, well-established fix (used throughout modern vision/
    speech training, e.g. timm, torchvision reference recipes) for exactly
    this symptom. decay=0.98 means the shadow's effective averaging window
    is ~1/(1-0.98)=50 optimizer steps, well inside one epoch (~319 steps
    at batch_size=128 on the real 40810-beat DS1-train set), so the EMA is
    already well "warmed up" by the end of epoch 1.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        # state_dict() can contain non-tensor entries (PhysioGuardCNN's
        # get_extra_state()/set_extra_state() persist `_source_tta`, a plain
        # dict of tensors, under the "_extra_state" key) -- those are copied
        # verbatim rather than EMA-averaged, since they aren't optimized
        # weights at all.
        self.shadow = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            new_val = msd[k]
            if not isinstance(v, torch.Tensor):
                self.shadow[k] = copy.deepcopy(new_val)
            elif v.dtype.is_floating_point:
                v.mul_(self.decay).add_(new_val.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(new_val)             # non-float buffers (e.g. num_batches_tracked)

    def state_dict(self) -> Dict:
        return {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
               for k, v in self.shadow.items()}


def macro_precision(rep: Dict) -> float:
    """Macro-averaged precision across present classes -- the number directly
    comparable to a paper reporting 'average precision' (e.g. MBE 2024's
    68.23%), as opposed to macro-F1 (a different, generally lower number
    once recall is weak)."""
    present = [c for c in CLASS_NAMES if rep["per_class"][c]["support"] > 0]
    if not present:
        return 0.0
    return float(sum(rep["per_class"][c]["precision"] for c in present) / len(present))


def macro_recall(rep: Dict) -> float:
    """Macro-averaged recall (= 'average sensitivity') across present classes --
    the number directly comparable to a paper reporting 'average sensitivity'
    (Romdhane et al. PeerJ CS 2020: 68.55% under the same DS1/DS2 inter-patient
    protocol; MBE 2024: 89.91% under their inter-patient split). NOT the same
    number as binary_screening_metrics()'s sensitivity, which collapses to a
    2-class (Normal vs Arrhythmic) task -- this is the true 5-class macro
    recall, the one actually comparable to those papers' reported figures."""
    present = [c for c in CLASS_NAMES if rep["per_class"][c]["support"] > 0]
    if not present:
        return 0.0
    return float(sum(rep["per_class"][c]["recall"] for c in present) / len(present))


def build_criterion(cfg: TrainConfig, class_weights: torch.Tensor,
                    class_counts: Optional[torch.Tensor] = None) -> nn.Module:
    if cfg.zero_q_weight:
        class_weights = class_weights.clone(); class_weights[CLASS_NAMES.index("Q")] = 0.0
    if cfg.loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    if cfg.loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    if cfg.loss_type in ("cb_focal", "cb_ce"):
        if class_counts is None:
            raise ValueError("cb_focal/cb_ce require class_counts (raw per-class beat counts)")
        cb_weights = class_balanced_weights(class_counts, beta=cfg.cb_beta).to(class_weights.device)
        if cfg.zero_q_weight:
            cb_weights[CLASS_NAMES.index("Q")] = 0.0
        if cfg.loss_type == "cb_ce":
            return nn.CrossEntropyLoss(weight=cb_weights, label_smoothing=cfg.label_smoothing)
        return FocalLoss(alpha=cb_weights, gamma=cfg.focal_gamma)
    raise ValueError(f"unknown loss_type '{cfg.loss_type}' "
                    f"(expected 'weighted_ce', 'focal', 'cb_focal', or 'cb_ce')")


# --------------------------------------------------------------------------- #
# Train / eval epoch
# --------------------------------------------------------------------------- #
def _unpack_batch(batch, device: str):
    """
    MITBIHDataset yields (x, y) normally, or (x, rr, y) when cfg.use_rr_features is
    True (see dataset.py). Returns (x, aux_or_None, y), all moved to `device`.
    """
    if len(batch) == 3:
        x, aux, y = batch
        return x.to(device), aux.to(device), y.to(device)
    x, y = batch
    return x.to(device), None, y.to(device)


def run_train_epoch(model: PhysioGuardCNN, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, cfg: TrainConfig,
                    ema: Optional["EMA"] = None) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x, aux, y = _unpack_batch(batch, cfg.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, aux=aux).logits                           # [B, K]
        loss = criterion(logits, y)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return total_loss / max(n, 1)


def lr_at_epoch(epoch: int, cfg: TrainConfig, total_epochs: int) -> float:
    """Linear warmup 0 -> cfg.lr over cfg.warmup_epochs, then cosine decay
    cfg.lr -> 0 over the remaining epochs. Avoids large early updates while
    the model is still near-random and class weights are extreme (Q weight
    reaches ~4x), which was producing big destabilizing first-epoch steps
    under the old flat-LR + plain-cosine schedule."""
    if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
        return cfg.lr * epoch / cfg.warmup_epochs
    span = max(1, total_epochs - cfg.warmup_epochs)
    t = (epoch - cfg.warmup_epochs) / span
    return 0.5 * cfg.lr * (1.0 + np.cos(np.pi * min(t, 1.0)))


def binary_screening_metrics(rep: Dict) -> Dict:
    """
    Collapse the 5-class confusion matrix into a Normal-vs-Arrhythmic binary
    screening task: N (class 0) is "normal", S/V/F/Q (classes 1-4) are all
    "arrhythmic". This is a DIFFERENT, clinically real task (the "is
    something wrong, yes/no" question an alert system actually asks), not a
    disguised version of the 5-class one -- report it ALONGSIDE the full
    5-class breakdown, never in place of it, and always label it explicitly
    as the binary screening number so a reader can't mistake it for 5-class
    accuracy.

    positive class = "arrhythmic" (S/V/F/Q). Returns accuracy, sensitivity
    (recall on arrhythmic -- the clinically critical number: missed
    arrhythmias), specificity (recall on normal), precision, f1.
    """
    cm = np.array(rep["confusion_matrix"])                 # [5,5], rows=true, cols=pred
    # binary cm: [ [TN, FP], [FN, TP] ] with index 0=normal, 1=arrhythmic
    tn = cm[0, 0]
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    tp = cm[1:, 1:].sum()
    total = tn + fp + fn + tp
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0   # recall on arrhythmic
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0   # recall on normal
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) \
        if (precision + sensitivity) > 0 else 0.0
    return {"accuracy": accuracy, "sensitivity_arrhythmic": sensitivity,
            "specificity_normal": specificity, "precision_arrhythmic": precision, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def macro_f1_excluding(rep: Dict, exclude: Sequence[str]) -> float:
    """
    Supplementary macro-F1 over CLASS_NAMES minus `exclude`, computed from an
    already-built classification_report()'s per_class F1 values. Reported
    ALONGSIDE (never instead of) the full 5-class macro-F1 -- matches the
    precedent in Romdhane et al. (PeerJ CS 2020), who separately reported
    results excluding Q as "practically non-existent" in DS1. Here we use it
    to make the honest, structural F-class scarcity visible as a labelled
    number rather than a silently low F1 buried in the 5-class average.
    """
    f1s = [rep["per_class"][c]["f1"] for c in CLASS_NAMES if c not in exclude]
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


@torch.no_grad()
def run_eval_epoch(model: PhysioGuardCNN, loader: DataLoader, criterion: nn.Module,
                   cfg: TrainConfig) -> Dict:
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for batch in loader:
        x, aux, y = _unpack_batch(batch, cfg.device)
        logits = model(x, aux=aux).logits                           # [B, K]
        loss = criterion(logits, y)
        total_loss += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(-1).cpu().numpy())
    import numpy as np
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    rep = classification_report(confusion_matrix(y_true, y_pred))
    rep["loss"] = total_loss / max(n, 1)
    return rep


# --------------------------------------------------------------------------- #
# Full training routine (shared by CLI and self-test)
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, cnn_cfg: CNNConfig, datasets: Dict[str, Dataset],
         data_contract: Dict) -> Dict:
    """
    datasets: {"train": ..., "val": ...} objects with a MITBIHDataset-compatible
    interface (__len__, __getitem__ -> (x[C,L], y), plus .class_weights()/.sampler()).
    data_contract: the ECGDataConfig fields that fix tensor shape/preprocessing,
    persisted into the checkpoint so eval/TTA scripts reconstruct the same pipeline.
    """
    torch.manual_seed(cfg.seed)
    ds_train, ds_val = datasets["train"], datasets["val"]
    if cfg.use_rare_class_augment:
        # NEVER wrap ds_val/ds_test -- augmentation must stay out of model
        # selection and the reported inter-patient metrics (see augment.py).
        ds_train = RareClassAugmentedDataset(ds_train, AugmentConfig(), seed=cfg.seed)

    train_kw = (dict(sampler=ds_train.sampler(seed=cfg.seed)) if cfg.balanced_sampling
               else dict(shuffle=True))
    if cfg.samples_per_epoch > 0 and "sampler" in train_kw:      # cap draws per epoch (sampler is with-replacement, class-balanced)
        train_kw["sampler"].num_samples = int(cfg.samples_per_epoch)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, drop_last=True,
                              num_workers=cfg.num_workers, **train_kw)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    class_weights = ds_train.class_weights(power=cfg.class_weight_power).to(cfg.device)  # [K]
    class_counts = ds_train.class_counts()                                                # [K] raw, for cb_* losses
    criterion = build_criterion(cfg, class_weights, class_counts).to(cfg.device)

    model = PhysioGuardCNN(cnn_cfg).to(cfg.device)
    trainable = list(model.backbone.parameters()) + list(model.adaptable_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # manual warmup+cosine schedule (lr_at_epoch), not torch's CosineAnnealingLR, so
    # warmup and decay share one function and one epoch counter -- see lr_at_epoch().
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    _loss_weights = (class_balanced_weights(class_counts, beta=cfg.cb_beta)
                    if cfg.loss_type in ("cb_focal", "cb_ce") else class_weights.cpu())
    print(f"[train_stage1] device={cfg.device}  train_beats={len(ds_train)}  "
         f"val_beats={len(ds_val)}  loss={cfg.loss_type}  use_ema={cfg.use_ema}  "
         f"lr_schedule={cfg.lr_schedule} (warmup_epochs={cfg.warmup_epochs})  "
         f"use_rare_class_augment={cfg.use_rare_class_augment}  decouple={cfg.decouple}  "
         f"loss_weights={dict(zip(CLASS_NAMES, [round(w, 2) for w in _loss_weights.tolist()]))}  "
         f"class_counts={dict(zip(CLASS_NAMES, class_counts.tolist()))}")

    def eval_for_selection(m: PhysioGuardCNN) -> Dict:
        """Evaluate on EMA-smoothed weights when enabled (loaded into a
        temporary copy so the live training weights in `model` are never
        disturbed), else evaluate the raw model directly. Model selection
        AND early stopping both use this, so the noisy raw per-epoch
        weights never drive either decision when EMA is on."""
        if ema is None:
            return run_eval_epoch(m, val_loader, criterion, cfg)
        shadow_model = copy.deepcopy(m)
        shadow_model.load_state_dict(ema.state_dict())
        return run_eval_epoch(shadow_model, val_loader, criterion, cfg)

    best_f1, best_epoch, best_state, epochs_since_best = -1.0, -1, None, 0
    # Secondary, SEPARATELY tracked checkpoint: best 5-class accuracy, kept purely so
    # a real, reproducible high-accuracy operating point can be reported ALONGSIDE the
    # macro-F1 one (which is what we actually ship/select on) -- never in place of it,
    # and never without its own sensitivity figure next to it. This is the honest
    # version of what a lot of published work does implicitly by only ever reporting
    # accuracy: here both ends of the accuracy<->sensitivity tradeoff are on the record.
    best_acc, best_acc_epoch, best_acc_state, best_acc_report = -1.0, -1, None, None
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if cfg.lr_schedule:
            for g in optimizer.param_groups:
                g["lr"] = lr_at_epoch(epoch, cfg, cfg.epochs)
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, cfg, ema=ema)
        val = eval_for_selection(model)

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            # save EMA weights (the ones actually evaluated) as the checkpoint,
            # not the raw model weights, when EMA is on.
            best_state = copy.deepcopy(ema.state_dict()) if ema is not None \
                else copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if val["accuracy"] > best_acc:
            best_acc, best_acc_epoch = val["accuracy"], epoch
            best_acc_state = copy.deepcopy(ema.state_dict()) if ema is not None \
                else copy.deepcopy(model.state_dict())
            best_acc_report = val

        if epoch % cfg.log_every == 0 or improved:
            per_cls = " ".join(f"{c}={val['per_class'][c]['f1']:.2f}" for c in CLASS_NAMES)
            f1_exF = macro_f1_excluding(val, exclude=("F",))
            prec_macro = macro_precision(val)
            rec_macro = macro_recall(val)
            bin_m = binary_screening_metrics(val)
            mark = " *" if improved else ""
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:3d}/{cfg.epochs}  lr={cur_lr:.2e}  train_loss={train_loss:.4f}  "
                 f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
                 f"val_macroF1={val['macro_f1']:.4f}  macroF1_exF={f1_exF:.4f}  "
                 f"macroPrecision={prec_macro:.4f}  macroRecall={rec_macro:.4f}  "
                 f"bin_acc={bin_m['accuracy']:.4f}  bin_sens={bin_m['sensitivity_arrhythmic']:.4f}  "
                 f"bin_spec={bin_m['specificity_normal']:.4f}  [{per_cls}]{mark}")

        min_stop_epoch = max(cfg.min_epochs, int(cfg.min_stop_frac * cfg.epochs))
        if cfg.fixed_epochs:
            last_state = copy.deepcopy(ema.state_dict()) if ema is not None else copy.deepcopy(model.state_dict())
            last_epoch, last_val = epoch, val
        elif epoch >= min_stop_epoch and epochs_since_best >= cfg.patience:
            print(f"[train_stage1] early stop at epoch {epoch} "
                 f"(no val macro-F1 improvement for {cfg.patience} epochs)")
            break

    if cfg.fixed_epochs:
        best_state, best_epoch, best_f1 = last_state, last_epoch, last_val["macro_f1"]
        print(f"[train_stage1] fixed_epochs: using the LAST epoch ({last_epoch}), no validation-based selection")
    if best_state is None:
        raise RuntimeError("training produced no valid epoch (check data/config)")
    model.load_state_dict(best_state)

    if cfg.decouple:
        # Decoupled classifier re-training (Kang et al., ICLR 2020 "cRT"):
        # freeze the backbone at its best normally-trained weights, then
        # re-train ONLY adaptable_head for a few epochs on a fully
        # class-balanced sampler (power=1.0, not the sqrt-power=0.5 used
        # for the main loop). The backbone already learned good general
        # features under the natural (long-tailed) distribution; this phase
        # only re-fits the decision boundary so it stops being dominated by
        # N, without re-biasing the features themselves.
        print(f"[train_stage1] --- decoupled classifier re-training "
             f"({cfg.decouple_epochs} epochs, backbone frozen) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False
        decouple_loader = DataLoader(
            ds_train, batch_size=cfg.batch_size, drop_last=True,
            sampler=ds_train.sampler(power=1.0, seed=cfg.seed + 1))
        head_optimizer = torch.optim.AdamW(
            model.adaptable_head.parameters(), lr=cfg.decouple_lr, weight_decay=cfg.weight_decay)
        head_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer, T_max=cfg.decouple_epochs) if cfg.lr_schedule else None)

        for d_epoch in range(1, cfg.decouple_epochs + 1):
            epoch = cfg.epochs + d_epoch                       # keep a single monotonic epoch counter
            train_loss = run_train_epoch(model, decouple_loader, criterion, head_optimizer, cfg)
            val = run_eval_epoch(model, val_loader, criterion, cfg)
            if head_scheduler is not None:
                head_scheduler.step()

            improved = val["macro_f1"] > best_f1
            if improved:
                best_f1, best_epoch = val["macro_f1"], epoch
                best_state = copy.deepcopy(model.state_dict())

            per_cls = " ".join(f"{c}={val['per_class'][c]['f1']:.2f}" for c in CLASS_NAMES)
            mark = " *" if improved else ""
            print(f"[decouple] epoch {d_epoch:3d}/{cfg.decouple_epochs}  "
                 f"train_loss={train_loss:.4f}  val_macroF1={val['macro_f1']:.4f}  "
                 f"[{per_cls}]{mark}")

        model.load_state_dict(best_state)
        for p in model.backbone.parameters():
            p.requires_grad = True              # restore, in case caller reuses `model`

    model.freeze_anchor()                       # theta_f := best theta_a; snapshot TTA source
    model.eval()

    elapsed = time.time() - t0
    print(f"[train_stage1] done in {elapsed/60:.1f} min | best epoch={best_epoch} "
         f"| best val macro-F1={best_f1:.4f}")

    final_report = run_eval_epoch(model, val_loader, criterion, cfg)
    macro_f1_ex_f = macro_f1_excluding(final_report, exclude=("F",))
    macro_f1_ex_fq = macro_f1_excluding(final_report, exclude=("F", "Q"))
    macro_prec = macro_precision(final_report)
    macro_rec = macro_recall(final_report)
    binary = binary_screening_metrics(final_report)
    print(f"[train_stage1] supplementary metrics (report ALL of these ALONGSIDE the full "
         f"5-class macro-F1, never in place of it): macroF1_exF={macro_f1_ex_f:.4f}  "
         f"macroF1_exFQ={macro_f1_ex_fq:.4f}")
    print(f"[train_stage1] macro-averaged precision/recall (the numbers directly comparable "
         f"to a paper reporting 'average precision'/'average sensitivity' -- Romdhane et al. "
         f"PeerJ CS 2020, same DS1/DS2 protocol: precision=64.68%, sensitivity=68.55%; "
         f"MBE 2024, inter-patient: precision=68.23%, sensitivity=89.91%): "
         f"macroPrecision={macro_prec:.4f}  macroRecall={macro_rec:.4f}")
    print(f"[train_stage1] binary screening (Normal vs Arrhythmic, a DIFFERENT clinically "
         f"real task -- label it explicitly as this, not as 5-class accuracy): "
         f"accuracy={binary['accuracy']:.4f}  sensitivity(catch-arrhythmia)="
         f"{binary['sensitivity_arrhythmic']:.4f}  specificity(normal)="
         f"{binary['specificity_normal']:.4f}  f1={binary['f1']:.4f}  "
         f"[TN={binary['tn']} FP={binary['fp']} FN={binary['fn']} TP={binary['tp']}]")

    # -- report the high-accuracy operating point ALONGSIDE the shipped (macro-F1)
    # checkpoint -- same protocol, same val split, just a different selection rule.
    # NEVER report accuracy here without the sensitivity/specificity next to it: that
    # omission is exactly the mechanism that makes some published accuracy numbers
    # look stronger than they are.
    if best_acc_report is not None:
        acc_bin = binary_screening_metrics(best_acc_report)
        acc_per_cls = " ".join(f"{c}={best_acc_report['per_class'][c]['f1']:.2f}" for c in CLASS_NAMES)
        print(f"[train_stage1] HIGH-ACCURACY operating point (epoch {best_acc_epoch}, "
             f"selected by accuracy instead of macro-F1 -- report ONLY together with its "
             f"sensitivity, never accuracy alone): accuracy={best_acc:.4f}  "
             f"macroF1={best_acc_report['macro_f1']:.4f}  "
             f"macroRecall={macro_recall(best_acc_report):.4f}  "
             f"sensitivity(binary, catch-arrhythmia)={acc_bin['sensitivity_arrhythmic']:.4f}  "
             f"specificity(normal)={acc_bin['specificity_normal']:.4f}  [{acc_per_cls}]")

    ckpt = {
        "model_state": model.state_dict(),
        "cnn_cfg": asdict(cnn_cfg),
        "data_cfg": data_contract,
        "epoch": best_epoch,
        "val_macro_f1": best_f1,
        "val_macro_f1_ex_f": macro_f1_ex_f,
        "val_macro_f1_ex_fq": macro_f1_ex_fq,
        "val_macro_precision": macro_prec,
        "val_macro_recall": macro_rec,
        "val_binary_screening": binary,
        "val_report": final_report,
        "train_cfg": asdict(cfg),
        # metadata only (not weights) for the high-accuracy operating point, so it's
        # on record in the SAME checkpoint file without shipping two sets of weights
        # or silently overwriting the macro-F1-selected model this run actually uses.
        "high_accuracy_operating_point": (
            {"epoch": best_acc_epoch, "accuracy": best_acc,
             "macro_f1": best_acc_report["macro_f1"],
             "binary_screening": binary_screening_metrics(best_acc_report)}
            if best_acc_report is not None else None),
    }
    torch.save(ckpt, cfg.checkpoint_path)
    print(f"[train_stage1] saved {cfg.checkpoint_path}")

    if best_acc_state is not None and cfg.save_high_accuracy_checkpoint:
        acc_ckpt_path = str(Path(cfg.checkpoint_path).with_name(
            Path(cfg.checkpoint_path).stem + "_high_accuracy.pt"))
        acc_bin = binary_screening_metrics(best_acc_report)
        torch.save({
            "model_state": best_acc_state,
            "cnn_cfg": asdict(cnn_cfg),
            "data_cfg": data_contract,
            "epoch": best_acc_epoch,
            "val_accuracy": best_acc,
            "val_macro_f1": best_acc_report["macro_f1"],
            "val_binary_screening": acc_bin,
            "val_report": best_acc_report,
            "train_cfg": asdict(cfg),
            "note": "selected by accuracy, not macro-F1 -- report sensitivity alongside "
                    "accuracy from this checkpoint, never accuracy alone",
        }, acc_ckpt_path)
        print(f"[train_stage1] saved high-accuracy operating-point checkpoint to "
             f"{acc_ckpt_path} (epoch {best_acc_epoch}, accuracy={best_acc:.4f}, "
             f"sensitivity={acc_bin['sensitivity_arrhythmic']:.4f} -- report both together)")

    return ckpt


# --------------------------------------------------------------------------- #
# CLI (real MIT-BIH data)
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_leads", type=int, default=1, choices=[1, 2])
    p.add_argument("--no_rr_features", action="store_true",
                   help="disable the 3 RR-interval timing channels (pre_rr, post_rr, "
                        "local_avg_rr) appended to the waveform input; on by default. "
                        "De Chazal et al. 2004 (the paper defining the DS1/DS2 split used "
                        "throughout this project) use RR timing specifically to separate "
                        "S-class beats, which are often morphologically ~normal.")
    p.add_argument("--n_rr_features", type=int, default=3, choices=[3, 5, 7],
                   help="7 = 5 + causal log(pre/mean of previous 80) and log(pre/mean of previous 400) beats; 3 = (pre_rr, post_rr, local_avg_rr) [default, unchanged]; 5 adds two "
                        "scale-free prematurity ratios log(pre/prev_avg), log(post/pre) -- aimed at S beats")
    p.add_argument("--extra_dir", default="", help="folder with extra TRAIN-ONLY records (e.g. SVDB)")
    p.add_argument("--extra_records", default="", help="comma list of record ids, or 'all' = every *.hea in --extra_dir")
    p.add_argument("--extra2_dir", default="", help="INCART folder (files I01..I75)")
    p.add_argument("--extra2_records", default="", help="'all' or comma list of INCART numbers 1..75")
    p.add_argument("--extra3_dir", default="", help="Icentia11k subset folder (contains p00/p000NN/...)")
    p.add_argument("--extra3_segments", default="0,1,2", help="segment numbers to use per patient")
    p.add_argument("--extra3_max_normal", type=int, default=400, help="normal beats kept per segment")
    p.add_argument("--samples_per_epoch", type=int, default=0, help="cap on class-balanced draws per epoch (0 = whole set)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for augmentation")
    p.add_argument("--n_template", type=int, default=0,
                   help="K>0: add patient-relative channels (median of the previous K beats of the same record, and "
                        "beat minus that median). Causal. 0 = off.")
    p.add_argument("--narrow", action="store_true",
                   help="use the ORIGINAL narrow architecture (stem_ch=16, b1/b2_branch_ch="
                        "16/32, feat_dim=64, head_hidden=32, no SE blocks; 18K params) instead "
                        "of the current default (SE+wide, 52K params). The SE+wide change "
                        "regressed DS2-test accuracy once already in isolation (88.61% vs "
                        "90.41%) -- this flag exists to re-test whether it's actually helping "
                        "now that RR features + rare-class augment are also in the mix, or "
                        "whether those two are carrying all the improvement while SE+wide is "
                        "quietly costing accuracy.")
    p.add_argument("--val_records", default=None,
                   help="comma-separated DS1 records to hold out as validation INSTEAD of the default "
                        "DS1_VAL_RECORDS (leave-record-group-out CV, see cv_folds.py). All other DS1 "
                        "records train. DS2 is never used.")
    p.add_argument("--corrupt_aug", type=float, default=0.0,
                   help="probability of corrupting a TRAINING beat (simulator components and/or real NSTDB "
                        "noise, random severity); 0 = off (previous behaviour). See robust_aug.py.")
    p.add_argument("--corrupt_max_severity", type=float, default=0.6)
    p.add_argument("--nstdb_dir", default=None, help="NSTDB folder (bw/em/ma); enables real-noise augmentation")
    p.add_argument("--real_noise_prob", type=float, default=0.4)
    p.add_argument("--exclude_corruption", default="",
                   help="comma list of families never used in training (leave-one-out): "
                        "microgravity,wander,radiation,bw,em,ma")
    p.add_argument("--checkpoint_path", default="checkpoint_stage1.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--no_lr_schedule", action="store_true",
                   help="disable warmup+cosine LR schedule (flat LR throughout)")
    p.add_argument("--no_ema", action="store_true",
                   help="disable EMA weight averaging (evaluate/select on raw weights)")
    p.add_argument("--ema_decay", type=float, default=0.98)
    p.add_argument("--loss_type", default="cb_focal",
                   choices=["weighted_ce", "focal", "cb_focal", "cb_ce"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--cb_beta", type=float, default=0.9999)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--balanced_sampling", action="store_true")
    p.add_argument("--fixed_epochs", action="store_true",
                   help="train all epochs and keep the last EMA weights (no early stop, no best-epoch selection)")
    p.add_argument("--zero_q_weight", action="store_true", help="give class Q zero loss weight (it has ~4 training beats)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_stop_frac", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_rare_class_augment", action="store_true",
                   help="apply jitter/scale/shift/same-class-mixup augmentation, weighted "
                        "toward S/V/F, on the train split only")
    p.add_argument("--decouple", action="store_true",
                   help="after normal training, freeze the backbone and re-train only the "
                        "classifier head on a fully class-balanced sampler (Kang et al. cRT)")
    p.add_argument("--decouple_epochs", type=int, default=5)
    p.add_argument("--decouple_lr", type=float, default=3e-4)
    p.add_argument("--no_high_accuracy_checkpoint", action="store_true",
                   help="skip saving the secondary best-accuracy checkpoint file")
    a = p.parse_args(argv)

    cfg = TrainConfig(data_dir=a.data_dir, cache_dir=a.cache_dir, n_leads=a.n_leads,
                      use_rr_features=not a.no_rr_features,
                      checkpoint_path=a.checkpoint_path, epochs=a.epochs,
                      batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
                      warmup_epochs=a.warmup_epochs, lr_schedule=not a.no_lr_schedule,
                      use_ema=not a.no_ema, ema_decay=a.ema_decay,
                      loss_type=a.loss_type, focal_gamma=a.focal_gamma, cb_beta=a.cb_beta,
                      label_smoothing=a.label_smoothing, balanced_sampling=a.balanced_sampling,
                      patience=a.patience, min_stop_frac=a.min_stop_frac, device=a.device, seed=a.seed,
                      use_rare_class_augment=a.use_rare_class_augment, decouple=a.decouple,
                      decouple_epochs=a.decouple_epochs, decouple_lr=a.decouple_lr,
                      save_high_accuracy_checkpoint=not a.no_high_accuracy_checkpoint,
                      fixed_epochs=a.fixed_epochs, zero_q_weight=a.zero_q_weight,
                      samples_per_epoch=a.samples_per_epoch, num_workers=a.num_workers)
    data_cfg = ECGDataConfig(data_dir=cfg.data_dir, cache_dir=cfg.cache_dir, n_leads=cfg.n_leads,
                             use_rr_features=cfg.use_rr_features, n_rr_features=a.n_rr_features, n_template=a.n_template,
                             batch_size=cfg.batch_size, balanced_sampling=cfg.balanced_sampling,
                             seed=cfg.seed)
    if a.extra_dir and a.extra_records:
        if a.extra_records == "all":
            ids = sorted(int(f.stem) for f in Path(a.extra_dir).glob("*.hea") if f.stem.isdigit())
        else:
            ids = [int(r) for r in a.extra_records.split(",")]
        data_cfg.extra_dir, data_cfg.extra_records = a.extra_dir, tuple(ids)
        print(f"[train_stage1] +{len(ids)} extra TRAIN-ONLY records from {a.extra_dir}", flush=True)
    if a.extra2_dir and a.extra2_records:
        if a.extra2_records == "all":
            ids2 = sorted(900 + int(f.stem[1:]) for f in Path(a.extra2_dir).glob("I*.hea") if f.stem[1:].isdigit())
        else:
            ids2 = [900 + int(r) for r in a.extra2_records.split(",")]
        data_cfg.extra2_dir, data_cfg.extra2_records = a.extra2_dir, tuple(ids2)
        print(f"[train_stage1] +{len(ids2)} extra TRAIN-ONLY INCART records from {a.extra2_dir}", flush=True)
    if a.extra3_dir:
        segs = [int(v) for v in a.extra3_segments.split(",")]
        ids3 = []
        for f in sorted(Path(a.extra3_dir).glob("p*/p*/p*_s*.atr")):
            pat, sg = int(f.stem[1:6]), int(f.stem.split("_s")[1])
            if sg in segs and f.with_suffix(".dat").exists() and f.with_suffix(".hea").exists():
                ids3.append(1_000_000 + pat * 100 + sg)
        data_cfg.extra3_dir, data_cfg.extra3_records, data_cfg.extra3_max_normal = a.extra3_dir, tuple(ids3), a.extra3_max_normal
        print(f"[train_stage1] +{len(ids3)} extra TRAIN-ONLY Icentia11k segments from {a.extra3_dir}", flush=True)
    if a.narrow:
        cnn_cfg = CNNConfig(in_channels=data_cfg.in_channels, aux_feat_dim=data_cfg.aux_feat_dim,
                            stem_ch=16, b1_branch_ch=16, b2_branch_ch=32,
                            feat_dim=64, head_hidden=32, use_se=False)
    else:
        cnn_cfg = CNNConfig(in_channels=data_cfg.in_channels, aux_feat_dim=data_cfg.aux_feat_dim)
    print(f"[train_stage1] architecture={'narrow (18K params, no SE)' if a.narrow else 'SE+wide (52K params)'}")
    print(f"[train_stage1] use_rr_features={data_cfg.use_rr_features}  "
          f"conv in_channels={data_cfg.in_channels}  late-fused aux_feat_dim={data_cfg.aux_feat_dim} "
          f"(RR, when on, is concatenated after pooling -- see base_cnn.CNNConfig.aux_feat_dim -- "
          f"not run through the conv stem)")
    val_recs = [int(r) for r in a.val_records.split(",")] if a.val_records else None
    train_tf = None
    if a.corrupt_aug > 0:
        from robust_aug import CorruptionAugment
        from real_shift import NoiseBank
        bank = NoiseBank.from_physionet(a.nstdb_dir).half("first") if a.nstdb_dir else None
        train_tf = CorruptionAugment(p=a.corrupt_aug, max_severity=a.corrupt_max_severity, bank=bank,
                                     p_real_noise=a.real_noise_prob, seed=cfg.seed,
                                     exclude=tuple(s for s in a.exclude_corruption.split(",") if s))
        print(f"[train_stage1] corruption-aware training: p={a.corrupt_aug} max_severity={a.corrupt_max_severity} "
              f"real_noise={'first half of NSTDB' if bank is not None else 'off'} (val stays CLEAN)")
    if train_tf is not None and a.n_template > 0:
        # corrupt only the CURRENT beat (channel 0); keep the patient template as is and recompute (beat - template).
        # (The simulator's 2-channel branch would otherwise treat channels 0/1 as two leads and rotate them.)
        class _TemplateAware:
            def __init__(self, aug): self.aug = aug
            def __call__(self, x):
                beat = self.aug(x[0:1])
                return torch.cat([beat, x[1:2], beat - x[1:2]], 0)
        train_tf = _TemplateAware(train_tf)
        print("[train_stage1] template channels: noise applied to the current beat only, difference channel recomputed")
    datasets = build_datasets(data_cfg, train_transform=train_tf, val_records=val_recs)   # {"train", "val", "test"} ; test unused here
    data_contract = {k: getattr(data_cfg, k) for k in _DATA_CONTRACT_FIELDS}
    return train(cfg, cnn_cfg, {"train": datasets["train"], "val": datasets["val"]}, data_contract)


# --------------------------------------------------------------------------- #
# Self-test: synthetic, class-imbalanced, MITBIHDataset-compatible dataset
# (no PhysioNet download needed; exercises the exact `train()` function above).
# --------------------------------------------------------------------------- #
class _SyntheticImbalancedECG(Dataset):
    """
    Each class gets a distinct fixed template + noise, with AAMI-like imbalance
    (N dominant, S/F/Q rare) so weighted-loss / macro-F1 code paths are exercised.
    """
    CLASS_COUNTS = (300, 40, 60, 15, 15)          # N, S, V, F, Q  (sums to 430)

    def __init__(self, c: int = 1, l: int = 256, seed: int = 0, split: str = "train"):
        g = torch.Generator().manual_seed(seed + (0 if split == "train" else 999))
        t = torch.linspace(-1, 1, l)
        templates = torch.stack([torch.sin((k + 1) * 3.0 * t) for k in range(NUM_CLASSES)])  # [K, L]
        xs, ys = [], []
        for k, n in enumerate(self.CLASS_COUNTS):
            n_split = max(1, n // 4) if split == "val" else n
            noise = 0.3 * torch.randn(n_split, c, l, generator=g)
            xs.append(templates[k].view(1, 1, l).repeat(n_split, c, 1) + noise)
            ys.append(torch.full((n_split,), k, dtype=torch.long))
        self.x = torch.cat(xs)                                       # [N, C, L]
        self.y = torch.cat(ys)                                       # [N]
        perm = torch.randperm(len(self.y), generator=g)
        self.x, self.y = self.x[perm], self.y[perm]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(self.y, minlength=NUM_CLASSES)

    def class_weights(self, power: float = 0.5) -> torch.Tensor:
        cnt = self.class_counts().float().clamp_min(1.0)
        w = (cnt.sum() / cnt) ** power
        return w / w.mean()

    def sampler(self, power: float = 0.5, seed: int = 0):
        from torch.utils.data import WeightedRandomSampler
        per_class = 1.0 / self.class_counts().float().clamp_min(1.0) ** power
        w = per_class[self.y]
        gen = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(w, num_samples=len(self), replacement=True, generator=gen)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("[train_stage1] no CLI args given -> running a fast synthetic self-test "
             "(no PhysioNet download needed).")
        cfg = TrainConfig(epochs=8, batch_size=32, patience=8, min_epochs=1,
                          checkpoint_path="./results/_selftest_stage1/checkpoint_stage1.pt",
                          device="cpu")
        Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        cnn_cfg = CNNConfig(in_channels=1)
        ds = {"train": _SyntheticImbalancedECG(c=1, l=256, seed=cfg.seed, split="train"),
             "val": _SyntheticImbalancedECG(c=1, l=256, seed=cfg.seed, split="val")}
        data_contract = {"n_leads": 1, "pre": 96, "post": 160, "norm": "record",
                        "bandpass": True, "band": (0.5, 40.0),
                        "use_rr_features": False, "n_rr_features": 0}
        ckpt = train(cfg, cnn_cfg, ds, data_contract)

        assert ckpt["val_macro_f1"] > 0.0, "macro-F1 should be > 0 on a learnable synthetic task"
        assert Path(cfg.checkpoint_path).exists()

        # round-trip through eval_cross_domain's own loader to guarantee contract compatibility
        from eval_cross_domain import load_checkpoint
        model2, cnn_cfg2, data_cfg2, is_trained = load_checkpoint(cfg.checkpoint_path, "cpu")
        assert is_trained and bool(model2.anchor_frozen)
        with torch.no_grad():
            out = model2(ds["val"].x[:4], use_anchor=True)
        assert out.logits.shape == (4, NUM_CLASSES) and out.anchor_logits.shape == (4, NUM_CLASSES)
        assert torch.allclose(out.logits, out.anchor_logits), \
            "freeze_anchor() must make anchor == adaptable at export time"
        print("Self-test passed: checkpoint trains, saves, and reloads via "
             "eval_cross_domain.load_checkpoint with matching heads.")

        # -- second pass: exercise use_rare_class_augment + decouple together ---- #
        print("[train_stage1] running second self-test with "
             "use_rare_class_augment=True, decouple=True ...")
        cfg2 = TrainConfig(epochs=6, batch_size=32, patience=8, min_epochs=1,
                           checkpoint_path="./results/_selftest_stage1/checkpoint_stage1_rc.pt",
                           device="cpu", loss_type="focal",
                           use_rare_class_augment=True, decouple=True,
                           decouple_epochs=3, decouple_lr=3e-4)
        ds2 = {"train": _SyntheticImbalancedECG(c=1, l=256, seed=cfg2.seed, split="train"),
              "val": _SyntheticImbalancedECG(c=1, l=256, seed=cfg2.seed, split="val")}
        ckpt2 = train(cfg2, cnn_cfg, ds2, data_contract)
        assert ckpt2["val_macro_f1"] > 0.0
        assert Path(cfg2.checkpoint_path).exists()
        model3, _, _, is_trained3 = load_checkpoint(cfg2.checkpoint_path, "cpu")
        assert is_trained3 and bool(model3.anchor_frozen)
        print(f"Self-test passed: augment+decouple run completed, "
             f"final val_macro_f1={ckpt2['val_macro_f1']:.4f}, "
             f"best_epoch={ckpt2['epoch']} (>{cfg2.epochs} means the decouple phase won).")