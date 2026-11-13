"""
quant_eval.py  --  what does int8 deployment cost in accuracy, and how big is the model?

SIMULATED post-training quantisation (PTQ); it is NOT a measurement on int8 kernels:
  * weights of every Conv1d / Linear: symmetric int8, per-output-channel scale (quantise->dequantise)
  * activations: the OUTPUT of every Conv1d / Linear / BatchNorm1d / ReLU-like module and the model
    input are fake-quantised to symmetric int8 with a per-tensor scale from calibration
    (99.99th percentile of |activation| over --n_calib TRAIN beats, never test beats)
  * the late-fused RR features (5 numbers) stay float
  Quantising at every module output is more noise than a fused int8 kernel would add (BatchNorm
  folds into the preceding conv on a real device), so the reported drop is a CONSERVATIVE estimate.

Reports, on the full clean DS2: fp32 vs simulated-int8 accuracy / macro-F1 (N/S/V) / per-class
Se & PPV, plus parameter count and model size in KB (fp32 and int8 weights + fp32 scales).

  !python quant_eval.py --checkpoint ckpt.pt --data_dir ... --cache_dir ... --out_dir ./results/quant
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

_ACT_TYPES = (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU)


def fq_tensor(t: torch.Tensor, scale: float) -> torch.Tensor:
    s = max(float(scale), 1e-12)
    return torch.clamp(torch.round(t / s), -127, 127) * s


@torch.no_grad()
def quantise_weights_(model: nn.Module) -> Dict[str, float]:
    """In-place per-output-channel symmetric int8 quantise->dequantise. Returns size stats."""
    n_w, n_scales = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().max(dim=1).values.clamp_min(1e-12) / 127.0
            q = torch.clamp(torch.round(flat / scale[:, None]), -127, 127)
            m.weight.data = (q * scale[:, None]).reshape_as(w)
            n_w += w.numel()
            n_scales += w.shape[0]
    return {"int8_weight_params": n_w, "scales": n_scales}


class ActivationQuantiser:
    """Calibrate on a handful of train beats, then fake-quantise module outputs."""

    def __init__(self, model: nn.Module, percentile: float = 99.99):
        self.model, self.pct = model, percentile
        self.mods = [m for m in model.modules() if isinstance(m, _ACT_TYPES)]
        self.samples: Dict[int, List[float]] = {id(m): [] for m in self.mods}
        self.scale: Dict[int, float] = {}
        self.input_samples: List[float] = []
        self.input_scale: Optional[float] = None
        self._hooks: list = []
        self.mode = "off"

    def _hook(self, mod, inp, out):
        if self.mode == "calib":
            self.samples[id(mod)].append(float(np.percentile(out.detach().abs().flatten().numpy(), self.pct)))
            return None
        if self.mode == "quant" and id(mod) in self.scale:
            return fq_tensor(out, self.scale[id(mod)] / 127.0)
        return None

    def attach(self):
        for m in self.mods:
            self._hooks.append(m.register_forward_hook(self._hook))

    def finish_calibration(self):
        for m in self.mods:
            v = self.samples[id(m)]
            self.scale[id(m)] = float(np.max(v)) if v else 1.0
        self.input_scale = float(np.max(self.input_samples)) if self.input_samples else 1.0
        self.mode = "quant"

    def quantise_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "calib":
            self.input_samples.append(float(np.percentile(x.abs().flatten().numpy(), self.pct)))
            return x
        if self.mode == "quant" and self.input_scale:
            return fq_tensor(x, self.input_scale / 127.0)
        return x

    def close(self):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def forward_batches(model, x, rr, bs, qa: Optional[ActivationQuantiser] = None) -> np.ndarray:
    model.eval()
    use_aux = model.cfg.aux_feat_dim > 0
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        if qa is not None:
            xb = qa.quantise_input(xb)
        aux = rr[i:i + bs] if use_aux else None
        out.append(model(xb, aux=aux).logits.cpu())
    return torch.cat(out).numpy()


def quantise_and_predict(model, x_calib, rr_calib, x_eval, rr_eval, bs: int = 512):
    """Returns (fp32_logits, int8_sim_logits, size_stats)."""
    fp = forward_batches(model, x_eval, rr_eval, bs)
    qm = copy.deepcopy(model).eval()
    stats = quantise_weights_(qm)
    qa = ActivationQuantiser(qm)
    qa.attach()
    qa.mode = "calib"
    forward_batches(qm, x_calib, rr_calib, bs, qa)
    qa.finish_calibration()
    q = forward_batches(qm, x_eval, rr_eval, bs, qa)
    qa.close()
    n_params = sum(p.numel() for p in model.parameters())
    stats.update({"total_params": n_params,
                  "fp32_kb": 4.0 * n_params / 1024.0,
                  "int8_kb": (stats["int8_weight_params"] + 4.0 * stats["scales"]
                              + 4.0 * (n_params - stats["int8_weight_params"])) / 1024.0})
    return fp, q, stats


def main(argv: Optional[Sequence[str]] = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--n_calib", type=int, default=2000)
    p.add_argument("--out_dir", default="./results/quant")
    a = p.parse_args(argv)
    from dataset import CLASS_NAMES, ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    from eval_source_model import summarize_predictions
    model, _, data_cfg, trained = load_checkpoint(a.checkpoint, "cpu", 1)
    assert trained, "not a trained checkpoint"
    data_cfg = ECGDataConfig(**{**dataclasses.asdict(data_cfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
    ds = build_datasets(data_cfg)
    tr, te = ds["train"], ds["test"]
    g = torch.Generator().manual_seed(0)
    ix = torch.randperm(len(tr.y), generator=g)[:a.n_calib]
    rr_tr = tr.rr[ix] if tr.rr.shape[-1] > 0 else None
    rr_te = te.rr if te.rr.shape[-1] > 0 else None
    fp, q, stats = quantise_and_predict(model, tr.x[ix], rr_tr, te.x, rr_te)
    y, rid = te.y.numpy(), te.record_ids.numpy()
    r_fp = summarize_predictions("fp32", y, fp.argmax(-1), rid, 500)
    r_q = summarize_predictions("int8-sim", y, q.argmax(-1), rid, 500)
    agree = float((fp.argmax(-1) == q.argmax(-1)).mean())
    L = [f"Model: {stats['total_params']} parameters; fp32 {stats['fp32_kb']:.1f} KB; "
         f"int8 weights + fp32 scales/biases/BN {stats['int8_kb']:.1f} KB (BN not folded).",
         f"Prediction agreement fp32 vs int8-sim: {100*agree:.2f}% of {len(y)} beats.", "",
         "| | accuracy | macro-F1 (N/S/V) | " + " | ".join(f"{c} Se/PPV" for c in CLASS_NAMES[:3]) + " |",
         "|---|---|---|---|---|---|"]
    for r in (r_fp, r_q):
        L.append(f"| {r['ckpt']} | {100*r['accuracy']:.2f}% | {r['macro_f1_ex_fq']:.3f} | " + " | ".join(
            f"{r['per_class'][c]['Se']:.1f} / {r['per_class'][c]['PPV']:.1f}" for c in CLASS_NAMES[:3]) + " |")
    L += ["", "SIMULATED PTQ (fake-quant at every module output, conservative); not measured on int8 kernels."]
    md = "\n".join(L) + "\n"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "quant_eval.md").write_text(md)
    (out / "quant_eval.json").write_text(json.dumps({"stats": stats, "agreement": agree,
                                                      "fp32": r_fp, "int8_sim": r_q}, indent=1))
    print(md)
    return {"stats": stats, "agreement": agree}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        sys.exit(0)
    from base_cnn import CNNConfig, PhysioGuardCNN
    torch.manual_seed(0)
    cfg = CNNConfig(in_channels=1, aux_feat_dim=5, stem_ch=16, b1_branch_ch=16, b2_branch_ch=32,
                    feat_dim=64, head_hidden=32, use_se=False)
    model = PhysioGuardCNN(cfg).eval()
    for m in model.modules():                                  # give BN non-trivial stats
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    xc, xe = torch.randn(300, 1, 256), torch.randn(400, 1, 256)
    rc, re = torch.randn(300, 5), torch.randn(400, 5)
    fp, q, st = quantise_and_predict(model, xc, rc, xe, re, bs=128)
    corr = float(np.corrcoef(fp.flatten(), q.flatten())[0, 1])
    agree = float((fp.argmax(-1) == q.argmax(-1)).mean())
    print(f"params={st['total_params']} fp32={st['fp32_kb']:.1f}KB int8={st['int8_kb']:.1f}KB  "
          f"logit corr={corr:.4f}  argmax agreement={agree:.3f}")
    assert corr > 0.95 and st["int8_kb"] < st["fp32_kb"]
    # weight quantisation must be exactly representable (<=255 levels per channel) and non-destructive
    qm = copy.deepcopy(model)
    quantise_weights_(qm)
    for m in qm.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            flat = m.weight.data.reshape(m.weight.shape[0], -1)
            assert all(len(torch.unique(r)) <= 255 for r in flat)
    # the original model must be untouched by quantise_and_predict
    fp2 = forward_batches(model, xe, re, 128)
    assert np.allclose(fp, fp2)
    print("Self-test passed.")
