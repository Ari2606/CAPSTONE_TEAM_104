"""
ecg_tent_edge_profile.py

SCOPE AND RATIONALE - read this before presenting the numbers:

No ONNX/TFLite/CoreML conversion happens anywhere in this script, and
that is deliberate, not an oversight. Those formats are frozen inference
graphs with no autograd engine - converting to them would make continuous
TENT adaptation impossible in that runtime regardless of whether
conversion happens before or after adaptation. Since this project's
actual contribution is CONTINUOUS on-device adaptation (not one-time
inference), the right proof stays entirely inside PyTorch's live
autograd graph - nothing here is a frozen snapshot.

What this proves instead: the TENT adaptation step (forward through the
FROZEN backbone + backward restricted to ~1,344 BatchNorm affine params
+ optimizer.step()) has a small, MEASURED (not estimated) memory and
latency footprint, under conditions that approximate edge-class hardware
without requiring physical access to it:

  - torch.set_num_threads(1)   - simulates a single embedded CPU core
  - device forced to "cpu"      - most edge boards have no GPU
  - batch size 1 tested         - the realistic streaming scenario
    (one incoming beat at a time), not a training-sized batch
  - torch.profiler with profile_memory=True - MEASURED peak allocator
    activity, not the params*4bytes*3 back-of-envelope formula

A full fine-tuning step (ALL params trainable) is profiled alongside
TENT as a contrast baseline - this is what makes the comparison land:
not just "TENT uses N KB" in isolation, but "TENT uses N KB vs M KB for
naive full fine-tuning on the same hardware-like conditions."

WHAT THIS DOES NOT CLAIM: no physical edge device was used. Present
these as measured, resource-constrained SOFTWARE profiling results, not
hardware-validated deployment numbers. That is an honest and, per
current on-device/TinyML learning literature (e.g. BN-affine-only
adaptation schemes like TinyTL), an accepted form of evidence when
physical hardware isn't available.

Run AFTER ecg_dann_space_shift.py (needs dann_source_only.pth).
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ecg_multiclass_cnn as base
import ecg_dann_space_shift as dann
import ecg_tent_adaptation as tent

MODEL_DIR = dann.MODEL_DIR
FIGURE_DIR = dann.FIGURE_DIR

# ---- Force edge-like software conditions BEFORE any profiling ----
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

BATCH_SIZES_TO_TEST = [1, 8, 32]
N_TIMED_STEPS = 100

# Reference points for the presentation table - not a claim that any of
# these specific boards were tested, just a scale for the reader to judge
# the measured footprint against.
EDGE_RAM_REFERENCE = [
    ("Cortex-M0+ (e.g. low-end MCU)", 32),
    ("Cortex-M4 (e.g. STM32F4)", 256),
    ("Cortex-M7 (e.g. STM32H7)", 1024),
    ("Raspberry Pi Zero 2 W", 512 * 1024),
    ("Raspberry Pi 4 (1GB variant)", 1024 * 1024),
]


def build_full_finetune_model():
    """Contrast baseline: identical architecture, but ALL parameters
    trainable (the naive approach TENT is explicitly designed to avoid)."""
    model = dann.DANNModel().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "dann_source_only.pth", map_location=DEVICE)
    )
    for p in model.parameters():
        p.requires_grad = True
    return model


def build_tent_model():
    model = dann.DANNModel().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "dann_source_only.pth", map_location=DEVICE)
    )
    tent.freeze_for_tent(model)
    tent.verify_freezing(model)
    return model


def profile_one_step(model, batch_size, tag):
    """
    Profiles ONE full adaptation step - forward, backward, optimizer.step()
    - under torch.profiler with profile_memory=True, so the reported
    memory is measured allocator activity, not a parameter-count estimate.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=tent.ADAPT_LR, momentum=0.9)

    ecg = torch.randn(batch_size, 2, 361, device=DEVICE)
    rr = torch.randn(batch_size, 4, device=DEVICE)

    def one_step():
        model.train()
        optimizer.zero_grad()
        task_out, _ = model(ecg, rr, lambd=0.0)
        loss, _, _ = tent.entropy_loss_confidence_gated(task_out)
        if loss is None:
            loss = task_out.mean() * 0.0  # keep graph alive if gate skips this random batch
        loss.backward()
        optimizer.step()

    # warmup (also lets CPU allocator caches settle before measuring)
    for _ in range(5):
        one_step()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        one_step()

    peak_bytes = max(
        (evt.cpu_memory_usage for evt in prof.key_averages() if evt.cpu_memory_usage > 0),
        default=0,
    )
    total_alloc = sum(
        evt.cpu_memory_usage for evt in prof.key_averages() if evt.cpu_memory_usage > 0
    )

    # separately timed latency (profiler overhead skews raw timing, so time cleanly)
    start = time.perf_counter()
    for _ in range(N_TIMED_STEPS):
        one_step()
    latency_ms = (time.perf_counter() - start) / N_TIMED_STEPS * 1000

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"[{tag}] batch={batch_size:<4} trainable_params={n_trainable:<8,} "
          f"({100*n_trainable/n_total:.2f}% of {n_total:,}) "
          f"peak_cpu_mem={peak_bytes/1024:.2f} KB  "
          f"total_alloc_activity={total_alloc/1024:.2f} KB  "
          f"latency={latency_ms:.3f} ms/step")

    return dict(tag=tag, batch_size=batch_size, n_trainable=n_trainable, n_total=n_total,
                peak_kb=peak_bytes/1024, total_alloc_kb=total_alloc/1024, latency_ms=latency_ms)


def main():
    print(f"torch.get_num_threads() = {torch.get_num_threads()} "
          f"(forced to 1 to simulate a single embedded CPU core)")
    print(f"Device: {DEVICE} (forced CPU - most edge boards have no GPU)\n")

    tent_model = build_tent_model()
    full_model = build_full_finetune_model()

    print("=" * 70)
    print("MEASURED PROFILING: TENT (BN-affine only) vs full fine-tuning")
    print("=" * 70)

    results = []
    for bs in BATCH_SIZES_TO_TEST:
        results.append(profile_one_step(copy.deepcopy(tent_model), bs, tag="TENT"))
    for bs in BATCH_SIZES_TO_TEST:
        results.append(profile_one_step(copy.deepcopy(full_model), bs, tag="FullFinetune"))

    # ---- Streaming (batch=1) headline numbers ----
    tent_b1 = next(r for r in results if r["tag"] == "TENT" and r["batch_size"] == 1)
    full_b1 = next(r for r in results if r["tag"] == "FullFinetune" and r["batch_size"] == 1)

    print("\n" + "=" * 70)
    print("HEADLINE COMPARISON (batch=1, the realistic streaming-beat scenario)")
    print("=" * 70)
    print(f"{'Metric':<30}{'TENT':<18}{'Full fine-tune':<18}{'Reduction'}")
    print(f"{'Trainable params':<30}{tent_b1['n_trainable']:<18,}{full_b1['n_trainable']:<18,}"
          f"{100*(1-tent_b1['n_trainable']/full_b1['n_trainable']):.2f}%")
    print(f"{'Measured peak CPU mem (KB)':<30}{tent_b1['peak_kb']:<18.2f}{full_b1['peak_kb']:<18.2f}"
          f"{100*(1-tent_b1['peak_kb']/full_b1['peak_kb']):.2f}%")
    print(f"{'Latency (ms/step)':<30}{tent_b1['latency_ms']:<18.3f}{full_b1['latency_ms']:<18.3f}"
          f"{100*(1-tent_b1['latency_ms']/full_b1['latency_ms']):.2f}%")

    # ---- Reference table: where does this footprint sit vs known edge RAM budgets? ----
    print("\n" + "=" * 70)
    print("CONTEXT: measured TENT footprint vs typical edge/embedded RAM budgets")
    print("(reference only - no specific board was tested)")
    print("=" * 70)
    for name, ram_kb in EDGE_RAM_REFERENCE:
        fits = "fits comfortably" if tent_b1["peak_kb"] < ram_kb * 0.1 else \
               ("fits" if tent_b1["peak_kb"] < ram_kb else "exceeds budget")
        print(f"{name:<32}{ram_kb:>10,} KB RAM   TENT step uses "
              f"{100*tent_b1['peak_kb']/ram_kb:.4f}% of it -> {fits}")

    # ---- Plot: peak memory vs batch size, TENT vs full fine-tune ----
    plt.figure(figsize=(8, 5))
    for tag, marker in [("TENT", "o"), ("FullFinetune", "s")]:
        sub = [r for r in results if r["tag"] == tag]
        plt.plot([r["batch_size"] for r in sub], [r["peak_kb"] for r in sub],
                  marker=marker, label=tag)
    plt.xlabel("Batch size")
    plt.ylabel("Measured peak CPU memory (KB)")
    plt.title("TENT vs full fine-tuning: measured adaptation-step memory (torch.profiler)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = FIGURE_DIR / "tent_edge_profile.png"
    plt.savefig(out_path)
    print(f"\nFigure saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY FOR PRESENTATION")
    print("=" * 70)
    print("Claim: TENT's adaptation step is a measured, software-profiled small")
    print("footprint under single-core-CPU, batch=1 streaming conditions -")
    print("evidence for functional edge-compatibility of the ALGORITHM.")
    print("Explicitly NOT claimed: performance on any specific physical device.")
    print("No ONNX/TFLite conversion was used, because those formats have no")
    print("autograd engine and would make the continuous-adaptation claim moot -")
    print("staying in native PyTorch is what lets this profiling reflect the")
    print("real, live training computation rather than a frozen inference graph.")


if __name__ == "__main__":
    main()
