"""
diagnose_domain_gap.py

Answers one question before building anything else: is the domain
discriminator sitting at chance accuracy (~0.50) because the physiological
shift is genuinely too subtle for this feature space to notice, or because
gradient reversal was suppressing it too fast/hard to ever get a foothold?

Method: train the SAME feature extractor + domain head, on the SAME paired
source/target data, but with NO gradient reversal (lambd=0 throughout) -
i.e. an undefended domain classifier with nothing fighting it. This is the
maximum domain-separability the current feature representation can support.

Interpretation:
  - Stays near chance (~0.50-0.55): genuine finding - the shift, as
    physiologically calibrated, doesn't leave an easily-linearly-separable
    signature at this feature abstraction. Reportable as-is; also tells you
    TENT's entropy-minimization won't have much to adapt to under FEATURE
    DISTRIBUTION shift specifically (though it may still help via other
    mechanisms - worth knowing before building the full TENT pipeline).
  - Climbs meaningfully above chance (>0.65): the shift IS separable, and
    DANN's lambda schedule (which reaches ~0.55 by just 10% of training)
    was suppressing the extractor before the domain head could exploit it -
    a tunable hyperparameter issue, not a fundamental limitation.

Uses the SAME cached paired dataset already built by ecg_dann_space_shift.py
- no reprocessing needed if you've already run that script.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ecg_multiclass_cnn as base
import ecg_dann_space_shift as dann

device = dann.device
EPOCHS = 15  # time-boxed - this is a diagnostic, not a full training run


def main():
    print("Loading cached paired domain data (severity='moderate')...")
    train_arrs, val_arrs, test_arrs = dann.build_dann_data()
    X_src, RR_src, X_tgt, RR_tgt, y = train_arrs
    X_src_v, RR_src_v, X_tgt_v, RR_tgt_v, y_v = val_arrs

    train_ds = dann.PairedDomainDataset(X_src, RR_src, X_tgt, RR_tgt, y)
    val_ds = dann.PairedDomainDataset(X_src_v, RR_src_v, X_tgt_v, RR_tgt_v, y_v)
    train_loader = DataLoader(train_ds, batch_size=dann.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=dann.BATCH_SIZE, shuffle=False)

    model = dann.DANNModel().to(device)
    domain_criterion = nn.CrossEntropyLoss()
    # only train the extractor + domain head - task head is irrelevant here
    params = (list(model.ecg_extractor.parameters()) +
              list(model.rr_extractor.parameters()) +
              list(model.domain_head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)

    print(f"\nTraining UNDEFENDED domain classifier (lambd=0, no reversal) "
          f"for {EPOCHS} epochs...\n")

    history = []
    for epoch in range(EPOCHS):
        model.train()
        running_acc, running_loss, n_batches = 0.0, 0.0, 0
        for ecg_src, rr_src, ecg_tgt, rr_tgt, _ in train_loader:
            ecg_src, rr_src = ecg_src.to(device), rr_src.to(device)
            ecg_tgt, rr_tgt = ecg_tgt.to(device), rr_tgt.to(device)

            optimizer.zero_grad()
            _, domain_out_src = model(ecg_src, rr_src, lambd=0.0)  # NO reversal
            _, domain_out_tgt = model(ecg_tgt, rr_tgt, lambd=0.0)

            domain_logits = torch.cat([domain_out_src, domain_out_tgt], dim=0)
            domain_labels = torch.cat([
                torch.zeros(len(rr_src), dtype=torch.long, device=device),
                torch.ones(len(rr_tgt), dtype=torch.long, device=device)
            ])
            loss = domain_criterion(domain_logits, domain_labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(domain_logits, dim=1)
            running_acc += (preds == domain_labels).float().mean().item()
            running_loss += loss.item()
            n_batches += 1

        train_acc = running_acc / n_batches
        train_loss = running_loss / n_batches

        # validation
        model.eval()
        val_acc_sum, val_n = 0.0, 0
        with torch.no_grad():
            for ecg_src, rr_src, ecg_tgt, rr_tgt, _ in val_loader:
                ecg_src, rr_src = ecg_src.to(device), rr_src.to(device)
                ecg_tgt, rr_tgt = ecg_tgt.to(device), rr_tgt.to(device)
                _, domain_out_src = model(ecg_src, rr_src, lambd=0.0)
                _, domain_out_tgt = model(ecg_tgt, rr_tgt, lambd=0.0)
                domain_logits = torch.cat([domain_out_src, domain_out_tgt], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(len(rr_src), dtype=torch.long, device=device),
                    torch.ones(len(rr_tgt), dtype=torch.long, device=device)
                ])
                preds = torch.argmax(domain_logits, dim=1)
                val_acc_sum += (preds == domain_labels).float().sum().item()
                val_n += len(domain_labels)
        val_acc = val_acc_sum / val_n

        history.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  "
              f"train_domain_acc={train_acc:.4f}  val_domain_acc={val_acc:.4f}")

    best_val_acc = max(history)
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC RESULT: best val domain accuracy = {best_val_acc:.4f} "
          f"(chance = 0.50)")
    print(f"{'='*60}")

    if best_val_acc < 0.55:
        print("\n-> Domain gap is essentially UNDETECTABLE even without adversarial")
        print("   suppression. This is a genuine finding: at this feature")
        print("   abstraction, the physiologically-calibrated 'moderate' shift")
        print("   does not leave a linearly/nonlinearly separable signature the")
        print("   CNN+RR features can exploit. Report this directly - it's evidence")
        print("   the shift is subtle/realistic, not evidence anything is broken.")
        print("   Implication for TENT: entropy minimization on FEATURE shift alone")
        print("   may have limited room to help here specifically; consider whether")
        print("   'severe' severity (or a stronger real perturbation) shows a")
        print("   detectable gap before committing further engineering effort.")
    elif best_val_acc < 0.65:
        print("\n-> Weak but present domain signal. DANN's lambda schedule (reaching")
        print("   ~0.55 by just 10% of training) may be suppressing it before the")
        print("   extractor/domain head can exploit it fully. Worth trying a slower")
        print("   lambda ramp before concluding adaptation isn't needed.")
    else:
        print("\n-> Clear domain signal exists (>0.65 undefended accuracy). DANN's")
        print("   lambda schedule was very likely too aggressive too early. This is")
        print("   a fixable hyperparameter issue, not a fundamental limitation -")
        print("   worth re-tuning the ramp schedule before moving to TENT.")


if __name__ == "__main__":
    main()
