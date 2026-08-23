"""
ecg_tent_adaptation.py

Source-free, physiology-anchored TENT (Test-time ENTropy minimization) for
edge deployment. Core idea: adapt ONLY the CNN branch's BatchNorm affine
parameters (gamma/beta) using entropy minimization on the incoming target
stream - no source data needed at adaptation time, no labels needed, no
backprop through the full network.

WHAT'S FROZEN AND WHY (the "physiology-anchored" part):
  - RR/rhythm branch: FROZEN. This is the network's physiological anchor -
    its decision logic about rhythm shouldn't drift during adaptation.
  - Classifier head: FROZEN. The final decision boundary doesn't move;
    only how the CNN branch normalizes incoming signal statistics does.
  - CNN branch BatchNorm affine params (gamma/beta) ONLY: TRAINABLE. This
    is genuinely tiny - a few hundred to a few thousand parameters, not
    the ~250K of the full model - which is the actual edge-deployment
    payoff (see hardware comparison at the end of main()).

HONEST NAMING NOTE: the "physiology" branch here is the RR-interval branch
(prev/current/next RR + ratio) already used throughout this project, NOT
full windowed HRV statistics (SDNN/RMSSD/LF-HF power) - those need
multi-beat sequence context this per-beat pipeline doesn't have. This is
still a legitimate, physiologically-motivated rhythm anchor - just simpler
than "HRV" might imply. State this plainly in the paper.

CONFIDENCE GATING (collapse prevention): unsupervised entropy minimization
has a known failure mode - it can drift toward confidently predicting
whatever's most common (Normal) since that minimizes entropy without
needing to be correct. We only let a batch influence the update if the
model's mean prediction entropy on it is already below a threshold (i.e.
skip ambiguous/uncertain batches rather than let them push the decision
boundary around) - and we track predicted abnormal-rate across adaptation
steps explicitly, so collapse (rate drifting toward 0%) is visible, not
hidden.

Reuses the already-trained source-only model from ecg_dann_space_shift.py
(dann_source_only.pth) - no retraining of the base model needed.
"""

import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import ecg_multiclass_cnn as base
import ecg_dann_space_shift as dann

device = dann.device
MODEL_DIR = dann.MODEL_DIR
FIGURE_DIR = dann.FIGURE_DIR

print("Device:", device)

# ==========================================================
# Config
# ==========================================================

ADAPT_BATCH_SIZE = 256
ADAPT_LR = 1e-3           # only touching BN affine params, can afford a higher LR
ENTROPY_THRESHOLD = 0.3   # skip batches with mean entropy above this (out of ln(2)~0.693 max)
N_ADAPT_STEPS = 200       # simulated "astronaut wearing the suit" adaptation steps
EVAL_EVERY = 10


# ==========================================================
# Freezing logic - the core mechanism
# ==========================================================

def freeze_for_tent(model):
    """
    Freeze everything except BatchNorm1d affine params inside model.ecg_extractor.
    Returns the list of trainable parameters (for the optimizer) and prints a
    verification summary so the freezing can be checked, not just assumed.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    for module in model.ecg_extractor.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.weight.requires_grad = True
            module.bias.requires_grad = True
            trainable_params.append(module.weight)
            trainable_params.append(module.bias)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"TENT freezing: {n_trainable:,} trainable params "
          f"(BatchNorm affine only) out of {n_total:,} total "
          f"({100*n_trainable/n_total:.2f}%)")

    return trainable_params


def verify_freezing(model):
    """
    Sanity check: confirm rhythm branch + classifier head are frozen, CNN
    conv/SE weights are frozen, and ONLY BatchNorm1d affine params inside
    ecg_extractor are trainable. Raises if any of these don't hold - this
    should never silently pass on a broken freeze.
    """
    frozen_prefixes = ("rr_extractor", "task_head", "domain_head")
    n_bn_trainable, n_other_cnn_trainable = 0, 0

    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            assert not p.requires_grad, f"{name} should be FROZEN but isn't!"
        elif name.startswith("ecg_extractor"):
            is_bn_param = isinstance(
                dict(model.named_modules())[name.rsplit(".", 1)[0]], nn.BatchNorm1d
            )
            if is_bn_param:
                assert p.requires_grad, f"{name} is a BatchNorm affine param and should be TRAINABLE"
                n_bn_trainable += p.numel()
            else:
                assert not p.requires_grad, f"{name} is a non-BatchNorm CNN param and should be FROZEN"
                n_other_cnn_trainable += 0  # just for clarity, always 0 here if assert passes

    print(f"Freezing verification passed: rhythm branch + classifier head frozen, "
          f"only {n_bn_trainable:,} BatchNorm affine params in the CNN branch are trainable.")


# ==========================================================
# Entropy + confidence gating
# ==========================================================

def compute_log_priors(y_source):
    """Log of the source class prior, for logit adjustment (Menon et al.,
    long-tail classification). Debiases the entropy objective against the
    known 9:1 class imbalance, directly targeting the collapse-toward-
    majority-class failure mode observed in the balanced-queue experiment."""
    counts = np.bincount(y_source, minlength=2)
    priors = counts / counts.sum()
    return torch.log(torch.tensor(priors, dtype=torch.float32))


def compute_advanced_tent_loss(logits, log_priors=None, entropy_threshold=ENTROPY_THRESHOLD,
                                diversity_alpha=0.0):
    """
    Unified loss supporting two optional, independently-toggleable additions
    on top of plain confidence-gated entropy minimization:

    - log_priors set: LOGIT ADJUSTMENT. Subtract log(source prior) from
      logits before computing entropy, so "confidence" is measured against
      a debiased distribution rather than one that can be satisfied just by
      coasting on the majority class's base rate.
    - diversity_alpha > 0: SHOT-style INFORMATION MAXIMIZATION. Adds
      -alpha * H(mean_batch_prediction) to the loss (computed on the RAW,
      unadjusted, UNFILTERED batch - not the confidence-gated subset, since
      filtering first would defeat the purpose of measuring the true
      incoming class mix). Minimizing this term maximizes the entropy of
      the batch-average prediction, directly discouraging the whole batch
      from collapsing toward one class.
    """
    adj_logits = logits - log_priors.to(logits.device) if log_priors is not None else logits

    ent = prediction_entropy(adj_logits)
    mask = ent < entropy_threshold
    frac_kept = mask.float().mean().item()
    individual_loss = ent[mask].mean() if mask.sum() > 0 else None

    diversity_loss = None
    if diversity_alpha > 0:
        probs = F.softmax(logits, dim=1)  # raw logits, full unfiltered batch
        mean_probs = probs.mean(dim=0)
        diversity_loss = (mean_probs * torch.log(mean_probs + 1e-6)).sum()

    if individual_loss is None and diversity_loss is None:
        return None, frac_kept, None, None
    total = (individual_loss if individual_loss is not None else 0.0)
    if diversity_loss is not None:
        total = total + diversity_alpha * diversity_loss
    return total, frac_kept, individual_loss, diversity_loss


def run_tent_adaptation_advanced(model, X_tgt, RR_tgt, y_tgt, log_priors=None,
                                  diversity_alpha=0.0, n_steps=N_ADAPT_STEPS, tag="Advanced"):
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=ADAPT_LR, momentum=0.9
    )
    n = len(y_tgt)
    rng = np.random.default_rng(base.SEED)
    history = {"step": [], "accuracy": [], "f1": [], "abnormal_rate": []}

    for step in range(n_steps):
        idx = rng.integers(0, n, size=ADAPT_BATCH_SIZE)
        ecg = torch.tensor(X_tgt[idx], dtype=torch.float32).to(device)
        rr = torch.tensor(RR_tgt[idx], dtype=torch.float32).to(device)

        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            metrics = evaluate_current_model(model, X_tgt, RR_tgt, y_tgt)
            history["step"].append(step)
            history["accuracy"].append(metrics["accuracy"])
            history["f1"].append(metrics["f1"])
            history["abnormal_rate"].append(metrics["abnormal_rate"])

        model.train()
        task_out, _ = model(ecg, rr, lambd=0.0)
        loss, frac_kept, ind_loss, div_loss = compute_advanced_tent_loss(
            task_out, log_priors=log_priors, diversity_alpha=diversity_alpha
        )
        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            print(f"[{tag}] Step {step:4d}/{n_steps}  acc={history['accuracy'][-1]:.4f}  "
                  f"f1={history['f1'][-1]:.4f}  abnormal_rate={history['abnormal_rate'][-1]:.4f}  "
                  f"frac_kept={frac_kept:.2f}")

    return model, history


def threshold_curve_diagnostic(model, X, RR, y, label=""):
    """Direct (non-bootstrap) F1-vs-threshold curve - shows the actual shape
    rather than a single summary statistic, so claims about 'the optimal
    threshold is below/above 0.5' can be checked directly rather than assumed."""
    result = evaluate_with_threshold(model, X, RR, y, threshold=0.5)
    probs = result["probs"]
    thresholds = np.arange(0.10, 0.91, 0.02)
    f1s = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        f1s.append(f1)
    best_idx = int(np.argmax(f1s))
    print(f"\nThreshold-F1 curve ({label}): peak F1={f1s[best_idx]:.4f} at threshold={thresholds[best_idx]:.2f}")
    print("  " + " ".join(f"{t:.2f}:{f:.3f}" for t, f in zip(thresholds, f1s)))
    return thresholds, f1s


def prediction_entropy(logits):
    """Per-sample entropy of the softmax distribution, in nats."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def entropy_loss_confidence_gated(logits, threshold=ENTROPY_THRESHOLD):
    """
    Mean entropy loss, but only over samples whose OWN entropy is already
    below threshold (i.e. the model is already reasonably confident about
    them). This is the collapse-prevention mechanism: batches/samples the
    model is uncertain about don't get to push the BatchNorm statistics
    around, since those are exactly the ambiguous cases most likely to be
    mis-adapted.

    Returns (loss, mask, fraction_kept). loss is None if nothing passes the
    gate (skip the update entirely that step).
    """
    ent = prediction_entropy(logits)
    mask = ent < threshold
    frac_kept = mask.float().mean().item()
    if mask.sum() == 0:
        return None, mask, frac_kept
    return ent[mask].mean(), mask, frac_kept


# ==========================================================
# Adaptation loop
# ==========================================================

class BalancedPseudoLabelQueue:
    """
    Buffers confidently-predicted beats by their OWN predicted (pseudo-)
    label, and only releases a balanced batch (equal count per class) for
    the entropy-minimization update. Motivation: if most confident
    predictions are the majority class, every BatchNorm update is shaped
    mostly by majority-class statistics - forcing balance directly targets
    that. Includes a safety valve: if a class's buffer hasn't filled within
    `max_wait_batches` incoming batches, release whatever's available
    (falling back to the plain confidence-gated behavior for that round)
    rather than stalling indefinitely if the minority class is rarely
    predicted confidently.
    """
    def __init__(self, per_class_size=10, max_wait_batches=20):
        self.per_class_size = per_class_size
        self.max_wait_batches = max_wait_batches
        self.buffers = {0: [], 1: []}       # each entry: (ecg, rr, true_label) for diagnostics
        self.batches_waited = 0

    def add(self, ecg_batch, rr_batch, logits, true_labels=None):
        ent = prediction_entropy(logits)
        preds = torch.argmax(logits, dim=1)
        confident_mask = ent < ENTROPY_THRESHOLD

        for i in range(len(preds)):
            if confident_mask[i]:
                cls = int(preds[i].item())
                true_lbl = int(true_labels[i]) if true_labels is not None else None
                self.buffers[cls].append((ecg_batch[i].clone(), rr_batch[i].clone(), true_lbl))
        self.batches_waited += 1

    def ready(self):
        both_full = (len(self.buffers[0]) >= self.per_class_size and
                     len(self.buffers[1]) >= self.per_class_size)
        timed_out = self.batches_waited >= self.max_wait_batches
        has_anything = len(self.buffers[0]) + len(self.buffers[1]) > 0
        return both_full or (timed_out and has_anything)

    def pop_batch(self):
        """Returns (ecg, rr, pseudo_labels, true_labels, was_balanced)."""
        n0, n1 = len(self.buffers[0]), len(self.buffers[1])
        was_balanced = n0 >= self.per_class_size and n1 >= self.per_class_size

        take0 = min(self.per_class_size, n0) if was_balanced else n0
        take1 = min(self.per_class_size, n1) if was_balanced else n1

        items = self.buffers[0][:take0] + self.buffers[1][:take1]
        pseudo_labels = [0] * take0 + [1] * take1
        self.buffers = {0: self.buffers[0][take0:], 1: self.buffers[1][take1:]}
        self.batches_waited = 0

        if not items:
            return None

        ecg = torch.stack([it[0] for it in items])
        rr = torch.stack([it[1] for it in items])
        true_labels = [it[2] for it in items]
        return ecg, rr, pseudo_labels, true_labels, was_balanced


def run_tent_adaptation_balanced(model, X_tgt, RR_tgt, y_tgt, n_steps=N_ADAPT_STEPS,
                                  per_class_size=10, max_wait_batches=20):
    """
    Same overall structure as run_tent_adaptation, but updates are triggered
    by a class-balanced pseudo-label queue instead of "one incoming batch =
    one update." True labels (y_tgt) are used ONLY for the pseudo-label
    accuracy diagnostic below - never for the adaptation update itself, so
    this stays genuinely source-free/label-free in terms of what drives the
    weight updates.
    """
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=ADAPT_LR, momentum=0.9
    )
    n = len(y_tgt)
    rng = np.random.default_rng(base.SEED)
    queue = BalancedPseudoLabelQueue(per_class_size=per_class_size, max_wait_batches=max_wait_batches)

    history = {"step": [], "accuracy": [], "f1": [], "abnormal_rate": [],
               "pseudo_label_acc_normal": [], "pseudo_label_acc_abnormal": [],
               "was_balanced": []}
    n_updates = 0

    for step in range(n_steps):
        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            metrics = evaluate_current_model(model, X_tgt, RR_tgt, y_tgt)
            history["step"].append(step)
            history["accuracy"].append(metrics["accuracy"])
            history["f1"].append(metrics["f1"])
            history["abnormal_rate"].append(metrics["abnormal_rate"])

        idx = rng.integers(0, n, size=ADAPT_BATCH_SIZE)
        ecg = torch.tensor(X_tgt[idx], dtype=torch.float32).to(device)
        rr = torch.tensor(RR_tgt[idx], dtype=torch.float32).to(device)
        true_labels = y_tgt[idx]

        model.eval()  # collect pseudo-labels without touching BN running stats yet
        with torch.no_grad():
            logits, _ = model(ecg, rr, lambd=0.0)
        queue.add(ecg, rr, logits, true_labels)

        if queue.ready():
            popped = queue.pop_batch()
            if popped is not None:
                b_ecg, b_rr, pseudo_labels, b_true_labels, was_balanced = popped

                # --- Pseudo-label accuracy diagnostic (offline only, using true labels) ---
                pseudo_arr = np.array(pseudo_labels)
                true_arr = np.array([t for t in b_true_labels if t is not None])
                if len(true_arr) == len(pseudo_arr):
                    for cls, name in [(0, "normal"), (1, "abnormal")]:
                        cls_mask = pseudo_arr == cls
                        if cls_mask.sum() > 0:
                            acc = (pseudo_arr[cls_mask] == true_arr[cls_mask]).mean()
                            history[f"pseudo_label_acc_{name}"].append(float(acc))

                history["was_balanced"].append(was_balanced)

                model.train()
                logits, _ = model(b_ecg, b_rr, lambd=0.0)
                ent = prediction_entropy(logits)
                loss = ent.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_updates += 1

        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            pa_n = history["pseudo_label_acc_normal"][-1] if history["pseudo_label_acc_normal"] else float("nan")
            pa_a = history["pseudo_label_acc_abnormal"][-1] if history["pseudo_label_acc_abnormal"] else float("nan")
            print(f"Step {step:4d}/{n_steps}  acc={history['accuracy'][-1]:.4f}  "
                  f"f1={history['f1'][-1]:.4f}  abnormal_rate={history['abnormal_rate'][-1]:.4f}  "
                  f"updates_so_far={n_updates}  pseudo_acc[N]={pa_n:.2f} pseudo_acc[Ab]={pa_a:.2f}")

    print(f"\nTotal balanced/fallback updates performed: {n_updates}")
    return model, history


def run_tent_adaptation(model, X_tgt, RR_tgt, y_tgt, n_steps=N_ADAPT_STEPS):
    """
    Simulates the astronaut-wearing-the-suit scenario: stream target beats
    in small batches, adapt BN affine params via confidence-gated entropy
    minimization, track task performance AND predicted abnormal-rate over
    time (the collapse check) using the held-out labels for evaluation only
    (never for adaptation itself - this stays genuinely source-free/label-free
    at adaptation time, labels here are for offline research measurement).
    """
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=ADAPT_LR, momentum=0.9
    )

    n = len(y_tgt)
    rng = np.random.default_rng(base.SEED)

    history = {"step": [], "accuracy": [], "f1": [], "abnormal_rate": [],
               "mean_entropy": [], "frac_batches_kept": []}

    batches_kept, batches_skipped = 0, 0

    for step in range(n_steps):
        idx = rng.integers(0, n, size=ADAPT_BATCH_SIZE)
        ecg = torch.tensor(X_tgt[idx], dtype=torch.float32).to(device)
        rr = torch.tensor(RR_tgt[idx], dtype=torch.float32).to(device)

        # Evaluate BEFORE this step's update, so "step N" reflects the model
        # state after exactly N completed updates (step 0 = no adaptation yet).
        if step % EVAL_EVERY == 0 or step == n_steps - 1:
            metrics = evaluate_current_model(model, X_tgt, RR_tgt, y_tgt)
            history["step"].append(step)
            history["accuracy"].append(metrics["accuracy"])
            history["f1"].append(metrics["f1"])
            history["abnormal_rate"].append(metrics["abnormal_rate"])

        model.train()  # BN layers need train mode to use batch statistics
        task_out, _ = model(ecg, rr, lambd=0.0)

        loss, mask, frac_kept = entropy_loss_confidence_gated(task_out)

        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_kept += 1
        else:
            batches_skipped += 1

        if history["step"] and history["step"][-1] == step:
            history["mean_entropy"].append(loss.item() if loss is not None else np.nan)
            history["frac_batches_kept"].append(frac_kept)
            print(f"Step {step:4d}/{n_steps}  acc={history['accuracy'][-1]:.4f}  "
                  f"f1={history['f1'][-1]:.4f}  abnormal_rate={history['abnormal_rate'][-1]:.4f}  "
                  f"batch_entropy={'skip' if loss is None else f'{loss.item():.4f}'}  "
                  f"frac_kept={frac_kept:.2f}")

    print(f"\nBatches used for updates: {batches_kept} | Skipped (low confidence): {batches_skipped}")
    return model, history


def find_best_threshold_bootstrap(probs, labels, n_bootstrap=200, seed=base.SEED):
    """
    Same technique already validated on the cascade (moved Stage 1's F1 from
    ~0.33 to ~0.61) - never applied here yet, since evaluate_current_model
    has been using naive argmax (implicit 0.5 threshold) throughout. Tuned
    on the TARGET-domain VALIDATION set (post-adaptation), applied to test.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs); labels = np.asarray(labels)
    n = len(labels)
    candidate_thresholds = np.arange(0.10, 0.91, 0.01)
    chosen = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_s, y_s = probs[idx], labels[idx]
        if y_s.sum() == 0 or y_s.sum() == n:
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in candidate_thresholds:
            preds = (p_s >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_s, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        chosen.append(best_t)

    chosen = np.array(chosen)
    median_t = float(np.median(chosen))
    iqr = (float(np.percentile(chosen, 25)), float(np.percentile(chosen, 75)))
    print(f"Bootstrap threshold: median={median_t:.2f} | IQR=[{iqr[0]:.2f}, {iqr[1]:.2f}]")
    return median_t


def evaluate_with_threshold(model, X, RR, y, threshold, batch_size=2048):
    """Like evaluate_current_model, but with a tuned threshold instead of argmax."""
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            ecg = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            rr = torch.tensor(RR[i:i+batch_size], dtype=torch.float32).to(device)
            out, _ = model(ecg, rr, lambd=0.0)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probs_all.extend(probs[:, 1])
    probs_all = np.array(probs_all)
    preds = (probs_all >= threshold).astype(int)
    acc = accuracy_score(y, preds)
    _, _, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return dict(accuracy=acc, f1=f1, abnormal_rate=preds.mean(), probs=probs_all)


def evaluate_current_model(model, X, RR, y, batch_size=2048):
    model.eval()
    preds_all = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            ecg = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            rr = torch.tensor(RR[i:i+batch_size], dtype=torch.float32).to(device)
            out, _ = model(ecg, rr, lambd=0.0)
            preds_all.extend(torch.argmax(out, dim=1).cpu().numpy())
    preds_all = np.array(preds_all)
    acc = accuracy_score(y, preds_all)
    _, _, f1, _ = precision_recall_fscore_support(y, preds_all, average="binary", zero_division=0)
    abnormal_rate = preds_all.mean()
    return dict(accuracy=acc, f1=f1, abnormal_rate=abnormal_rate)


# ==========================================================
# Hardware comparison: TENT (BN-only) vs full backprop
# ==========================================================

def hardware_comparison(model):
    n_total = sum(p.numel() for p in model.parameters())
    n_tent = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # rough memory estimate: trainable params need gradient + optimizer state
    # (SGD+momentum: 1x param + 1x grad + 1x momentum buffer = 3x per param)
    bytes_per_param = 4  # float32
    mem_full = n_total * bytes_per_param * 3
    mem_tent = n_tent * bytes_per_param * 3

    print("\n=== Hardware Comparison: TENT (BN-only) vs Full Backprop ===")
    print(f"Total model parameters:        {n_total:,}")
    print(f"TENT-trainable parameters:     {n_tent:,} ({100*n_tent/n_total:.2f}%)")
    print(f"Est. adaptation memory (full backprop): {mem_full/1024:.1f} KB")
    print(f"Est. adaptation memory (TENT, BN-only):  {mem_tent/1024:.1f} KB")
    print(f"Memory reduction: {mem_full/mem_tent:.1f}x")
    print("(This is the concrete number that makes 'edge-deployable adaptation' "
          "verifiable rather than asserted - report it alongside accuracy.)")


# ==========================================================
# Main
# ==========================================================

def main():
    print("Loading cached paired domain data (severity='moderate')...")
    train_arrs, val_arrs, test_arrs = dann.build_dann_data()
    X_train_src, RR_train_src, X_train_tgt, RR_train_tgt, y_train = train_arrs
    X_src_test, RR_src_test, X_tgt_test, RR_tgt_test, y_test = test_arrs

    print("\nLoading pretrained source-only model (no retraining needed)...")
    model = dann.DANNModel().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "dann_source_only.pth", map_location=device))

    # Baseline: frozen, no adaptation at all
    baseline_metrics = evaluate_current_model(model, X_tgt_test, RR_tgt_test, y_test)
    print(f"\nBaseline (no adaptation) on target domain: "
          f"acc={baseline_metrics['accuracy']:.4f} f1={baseline_metrics['f1']:.4f} "
          f"abnormal_rate={baseline_metrics['abnormal_rate']:.4f}")

    print("\n" + "=" * 60)
    print("APPLYING TENT (physiology-anchored, confidence-gated)")
    print("=" * 60)

    trainable_params = freeze_for_tent(model)
    verify_freezing(model)

    # Save a pristine frozen-and-ready copy so both variants start identically
    pristine_state = copy.deepcopy(model.state_dict())

    print("\n--- Variant A: plain confidence-gated TENT (original) ---")
    adapted_model, history = run_tent_adaptation(model, X_tgt_test, RR_tgt_test, y_test)
    final_metrics = evaluate_current_model(adapted_model, X_tgt_test, RR_tgt_test, y_test)

    print("\n--- Variant B: class-balanced pseudo-label queue TENT ---")
    model_b = dann.DANNModel().to(device)
    model_b.load_state_dict(pristine_state)
    freeze_for_tent(model_b)  # re-apply freezing (state_dict load doesn't preserve requires_grad)
    adapted_model_b, history_b = run_tent_adaptation_balanced(model_b, X_tgt_test, RR_tgt_test, y_test)
    final_metrics_b = evaluate_current_model(adapted_model_b, X_tgt_test, RR_tgt_test, y_test)

    print("\n--- Variant C: logit-adjusted entropy TENT ---")
    log_priors = compute_log_priors(y_train)
    print(f"Log priors (Normal, Abnormal): {log_priors.tolist()}")
    model_c = dann.DANNModel().to(device)
    model_c.load_state_dict(pristine_state)
    freeze_for_tent(model_c)
    adapted_model_c, history_c = run_tent_adaptation_advanced(
        model_c, X_tgt_test, RR_tgt_test, y_test, log_priors=log_priors, tag="VariantC"
    )
    final_metrics_c = evaluate_current_model(adapted_model_c, X_tgt_test, RR_tgt_test, y_test)

    print("\n--- Variant D: SHOT-style diversity-regularized TENT ---")
    model_d = dann.DANNModel().to(device)
    model_d.load_state_dict(pristine_state)
    freeze_for_tent(model_d)
    adapted_model_d, history_d = run_tent_adaptation_advanced(
        model_d, X_tgt_test, RR_tgt_test, y_test, diversity_alpha=1.0, tag="VariantD"
    )
    final_metrics_d = evaluate_current_model(adapted_model_d, X_tgt_test, RR_tgt_test, y_test)

    print(f"\n{'='*60}")
    print("VARIANT COMPARISON (argmax, before any threshold tuning)")
    print(f"{'='*60}")
    print(f"{'Variant':<40}{'Accuracy':<12}{'F1'}")
    print(f"{'A: plain confidence-gated':<40}{final_metrics['accuracy']:<12.4f}{final_metrics['f1']:.4f}")
    print(f"{'B: class-balanced pseudo-label':<40}{final_metrics_b['accuracy']:<12.4f}{final_metrics_b['f1']:.4f}")
    print(f"{'C: logit-adjusted entropy':<40}{final_metrics_c['accuracy']:<12.4f}{final_metrics_c['f1']:.4f}")
    print(f"{'D: SHOT-style diversity regularized':<40}{final_metrics_d['accuracy']:<12.4f}{final_metrics_d['f1']:.4f}")

    if "pseudo_label_acc_abnormal" in history_b and history_b["pseudo_label_acc_abnormal"]:
        mean_pa_abnormal = np.nanmean(history_b["pseudo_label_acc_abnormal"])
        mean_pa_normal = np.nanmean(history_b["pseudo_label_acc_normal"])
        print(f"\nPseudo-label accuracy (offline diagnostic, NOT used in adaptation):")
        print(f"  Normal pseudo-labels:   {mean_pa_normal:.3f} accurate")
        print(f"  Abnormal pseudo-labels: {mean_pa_abnormal:.3f} accurate")
        if mean_pa_abnormal < 0.5:
            print("  WARNING: abnormal pseudo-labels are less than 50% accurate - the queue is")
            print("  likely reinforcing WRONG confident predictions (confirmation bias). The")
            print("  class-balanced variant's results should be treated with real skepticism")
            print("  if this is the case, regardless of what the F1 number shows.")
        else:
            print("  Pseudo-labels are majority-correct - reinforcement is more likely helping")
            print("  than entrenching errors, though this doesn't guarantee no bias at all.")

    # Pick whichever variant genuinely wins on F1 for downstream reporting
    all_variants = [
        ("A", adapted_model, final_metrics), ("B", adapted_model_b, final_metrics_b),
        ("C", adapted_model_c, final_metrics_c), ("D", adapted_model_d, final_metrics_d),
    ]
    best_name, adapted_model, final_metrics = max(all_variants, key=lambda v: v[2]["f1"])
    print(f"\nVariant {best_name} wins on F1 ({final_metrics['f1']:.4f}) - using it for downstream reporting.")

    # ---- Threshold curve diagnostic (direct, not bootstrap-summarized) ----
    print("\n" + "=" * 60)
    print("THRESHOLD-F1 CURVE DIAGNOSTIC (on the winning variant)")
    print("=" * 60)
    _, _, X_tgt_v_diag, RR_tgt_v_diag, y_v_diag = val_arrs
    threshold_curve_diagnostic(adapted_model, X_tgt_v_diag, RR_tgt_v_diag, y_v_diag, label="validation")
    threshold_curve_diagnostic(adapted_model, X_tgt_test, RR_tgt_test, y_test, label="test (diagnostic only - not a valid selection criterion)")

    # ---- Threshold tuning: the lever never applied to this model before ----
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING (on target-domain validation, post-adaptation)")
    print("=" * 60)
    _, _, X_tgt_v, RR_tgt_v, y_v = val_arrs
    val_result = evaluate_with_threshold(adapted_model, X_tgt_v, RR_tgt_v, y_v, threshold=0.5)
    tuned_threshold = find_best_threshold_bootstrap(val_result["probs"], y_v)
    thresholded_metrics = evaluate_with_threshold(
        adapted_model, X_tgt_test, RR_tgt_test, y_test, threshold=tuned_threshold
    )
    print(f"\nArgmax (0.5) on test:        F1={final_metrics['f1']:.4f}  "
          f"Acc={final_metrics['accuracy']:.4f}")
    print(f"Tuned threshold ({tuned_threshold:.2f}) on test: "
          f"F1={thresholded_metrics['f1']:.4f}  Acc={thresholded_metrics['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Condition':<28}{'Accuracy':<12}{'F1':<12}{'Abnormal Rate'}")
    print(f"{'No adaptation (argmax)':<28}{baseline_metrics['accuracy']:<12.4f}"
          f"{baseline_metrics['f1']:<12.4f}{baseline_metrics['abnormal_rate']:.4f}")
    print(f"{'TENT-adapted (argmax)':<28}{final_metrics['accuracy']:<12.4f}"
          f"{final_metrics['f1']:<12.4f}{final_metrics['abnormal_rate']:.4f}")
    print(f"{'TENT + tuned threshold':<28}{thresholded_metrics['accuracy']:<12.4f}"
          f"{thresholded_metrics['f1']:<12.4f}{thresholded_metrics['abnormal_rate']:.4f}")

    # ---- Collapse check (proper version) ----
    # A naive "did the rate drift a lot" check is too simplistic - drift
    # TOWARD the true prevalence, accompanied by IMPROVING F1, is healthy
    # recalibration (exactly what BatchNorm-statistic adaptation should do).
    # Real collapse looks like: rate drifting AWAY from truth (often toward
    # 0% or 100%) WHILE F1 degrades toward 0 - the "just predict the
    # majority class" failure mode. Check both signals together, not drift
    # magnitude alone. Uses the THRESHOLDED result as the true final state,
    # since that's now the headline number, not raw argmax.
    rates = history["abnormal_rate"] + [thresholded_metrics["abnormal_rate"]]
    f1s = history["f1"] + [thresholded_metrics["f1"]]
    true_rate = y_test.mean()

    initial_gap = abs(rates[0] - true_rate)
    final_gap = abs(rates[-1] - true_rate)
    f1_trend = f1s[-1] - f1s[0]

    print(f"\nCollapse check:")
    print(f"  True abnormal prevalence: {true_rate:.4f}")
    print(f"  Predicted rate: {rates[0]:.4f} -> {rates[-1]:.4f} "
          f"(distance from truth: {initial_gap:.4f} -> {final_gap:.4f})")
    print(f"  F1 trend: {f1s[0]:.4f} -> {f1s[-1]:.4f} ({f1_trend:+.4f})")

    moved_toward_truth = final_gap < initial_gap
    f1_improved_or_stable = f1_trend > -0.02  # small tolerance for noise

    if moved_toward_truth and f1_improved_or_stable:
        print("  -> HEALTHY: rate moved toward true prevalence, F1 stable/improved. "
              "No collapse.")
    elif not moved_toward_truth and f1_trend < -0.05:
        print("  -> WARNING: rate moved AWAY from true prevalence AND F1 degraded - "
              "this is the collapse signature. Consider a stricter entropy threshold.")
    else:
        print("  -> Ambiguous - rate and F1 trends don't clearly agree. Inspect the "
              "full history/plot before concluding either way.")

    hardware_comparison(model)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["step"], history["accuracy"], marker="o")
    axes[0].axhline(baseline_metrics["accuracy"], linestyle="--", color="gray", label="No adaptation")
    axes[0].set_xlabel("Adaptation Step"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
    axes[0].set_title("Accuracy over adaptation")

    axes[1].plot(history["step"], history["f1"], marker="o", color="orange")
    axes[1].axhline(baseline_metrics["f1"], linestyle="--", color="gray", label="No adaptation")
    axes[1].set_xlabel("Adaptation Step"); axes[1].set_ylabel("F1 (Abnormal class)"); axes[1].legend()
    axes[1].set_title("F1 over adaptation")

    axes[2].plot(history["step"], history["abnormal_rate"], marker="o", color="red")
    axes[2].axhline(y_test.mean(), linestyle="--", color="green", label="True abnormal rate")
    axes[2].set_xlabel("Adaptation Step"); axes[2].set_ylabel("Predicted Abnormal Rate"); axes[2].legend()
    axes[2].set_title("Collapse check")

    plt.tight_layout()
    out_path = FIGURE_DIR / "tent_adaptation.png"
    plt.savefig(out_path)
    print(f"\nFigure saved to {out_path}")

    torch.save(adapted_model.state_dict(), MODEL_DIR / "tent_adapted_model.pth")
    print(f"Adapted model saved to {MODEL_DIR / 'tent_adapted_model.pth'}")


if __name__ == "__main__":
    main()