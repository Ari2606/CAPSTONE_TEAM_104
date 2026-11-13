"""
Export a trained PhysioGuard checkpoint to ONNX (fp32) and true int8 (QDQ, static), and check parity.
This is the first step of the on-device deployment: it replaces the *simulated* int8 numbers in
quant_eval.py with a real quantised artefact that runs in ONNX Runtime (Raspberry Pi, phone, PC).

  python export_onnx.py --checkpoint ckpts/robust5_seed0.pt --data_dir .../mitdb --cache_dir .../cache \
         --out_dir deploy/seed0
  python export_onnx.py --smoke --out_dir /tmp/deploy_smoke        # random model, pipeline test only

Writes: model_fp32.onnx, model_int8.onnx, eval_beats.npz (600 class-balanced test beats + labels, used by the
Pi benchmark), export_report.md.   Batch size is fixed to 1 (streaming, one beat at a time).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class _LogitsOnly(nn.Module):
    def __init__(self, model, use_aux: bool):
        super().__init__()
        self.m, self.use_aux = model, use_aux

    def forward(self, x, aux=None):
        out = self.m(x, aux=aux) if self.use_aux else self.m(x)
        return out.logits


def _balanced(y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(n, int((y == c).sum())), replace=False)
                           for c in (0, 1, 2)])


def _bal_recall(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in (0, 1, 2) if (y == c).any()]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_dir", default="./data/mitdb")
    ap.add_argument("--cache_dir", default="./data/cache")
    ap.add_argument("--out_dir", default="./deploy")
    ap.add_argument("--n_calib", type=int, default=300, help="beats per class for int8 calibration")
    ap.add_argument("--n_eval", type=int, default=200, help="beats per class for parity/eval + Pi benchmark")
    ap.add_argument("--calib_train", action="store_true", help="calibrate int8 on DS1 train beats")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        from eval_cross_domain import load_checkpoint
        model, cnn_cfg, _, _ = load_checkpoint(None, "cpu")
        cfg = model.cfg
        L = 256
        n = 3 * a.n_eval
        x = torch.randn(n, cfg.in_channels, L)
        rr = torch.rand(n, cfg.aux_feat_dim) if cfg.aux_feat_dim > 0 else None
        y = np.repeat([0, 1, 2], a.n_eval)
        xc, rrc = x, rr
    else:
        if not a.checkpoint:
            raise SystemExit("ERROR: give --checkpoint (or --smoke).")
        from run_experiments import load_source_model_and_stream
        model, ds, is_real = load_source_model_and_stream(a.checkpoint, a.data_dir, a.cache_dir, 1, 3, 400, 0)
        if not is_real:
            raise SystemExit("ERROR: real checkpoint/data unusable (mount Drive, pip install wfdb, check paths).")
        cfg = model.cfg
        y_all = ds.y.numpy()
        has_rr = getattr(ds, "rr", None) is not None and ds.rr.shape[-1] > 0
        sel = _balanced(y_all, a.n_eval, seed=0)
        rest = np.setdiff1d(np.arange(len(y_all)), sel)
        csel = _balanced(y_all[rest], a.n_calib, seed=1)
        csel = rest[csel]
        x, y = ds.x[sel], y_all[sel]
        rr = ds.rr[sel] if has_rr else None
        xc = ds.x[csel]
        rrc = ds.rr[csel] if has_rr else None
        if a.calib_train:       # calibrate int8 on DS1 TRAIN beats, never on the evaluation stream
            import dataclasses
            from dataset import ECGDataConfig, build_datasets
            from eval_cross_domain import load_checkpoint
            _, _, dcfg, _ = load_checkpoint(a.checkpoint, "cpu", 1)
            tcfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
            tr = build_datasets(tcfg)["train"]
            ci = _balanced(tr.y.numpy(), a.n_calib, seed=1)
            xc = tr.x[ci]
            rrc = tr.rr[ci] if has_rr else None
            print(f"[export] int8 calibration: {len(ci)} DS1-train beats", flush=True)

    model.eval()
    use_aux = cfg.aux_feat_dim > 0
    # The model stores the frozen-source snapshot as a dict "extra state"; the ONNX tracer expects tensors only.
    # Export from a copy with that hook neutralised (the snapshot is not part of the inference graph).
    import copy
    export_model = copy.deepcopy(model).eval()
    export_model.get_extra_state = lambda: torch.zeros(1)
    wrap = _LogitsOnly(export_model, use_aux).eval()
    ex_x = x[:1]
    inputs = (ex_x, rr[:1]) if use_aux else (ex_x,)
    names = ["beat", "aux"] if use_aux else ["beat"]
    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(wrap, inputs, str(fp32), input_names=names, output_names=["logits"],
                      opset_version=17, dynamo=False)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(fp32)))

    def run(path, xs, rs):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outl = []
        for i in range(len(xs)):
            feed = {"beat": xs[i:i + 1].numpy().astype(np.float32)}
            if use_aux:
                feed["aux"] = rs[i:i + 1].numpy().astype(np.float32)
            outl.append(sess.run(None, feed)[0][0])
        return np.stack(outl)

    with torch.no_grad():
        ref = torch.cat([(wrap(x[i:i + 1], rr[i:i + 1]) if use_aux else wrap(x[i:i + 1])) for i in range(len(x))]).numpy()
    onx = run(fp32, x, rr)
    max_diff = float(np.abs(ref - onx).max())

    # ---- int8 static quantisation (QDQ, per-channel weights) ----
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(xc):
                return None
            feed = {"beat": xc[self.i:self.i + 1].numpy().astype(np.float32)}
            if use_aux:
                feed["aux"] = rrc[self.i:self.i + 1].numpy().astype(np.float32)
            self.i += 1
            return feed

    int8 = out / "model_int8.onnx"
    quantize_static(str(fp32), str(int8), _Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    q = run(int8, x, rr)

    p_ref, p_fp, p_q = ref.argmax(1), onx.argmax(1), q.argmax(1)
    agree_fp = float((p_fp == p_ref).mean())
    agree_q = float((p_q == p_fp).mean())
    rec_fp, rec_q = _bal_recall(y, p_fp), _bal_recall(y, p_q)
    kb = lambda p: os.path.getsize(p) / 1024.0
    np.savez(out / "eval_beats.npz", x=x.numpy().astype(np.float32), y=y,
             **({"aux": rr.numpy().astype(np.float32)} if use_aux else {}))

    rep = [
        "# ONNX export report", "",
        f"- inputs: beat [1,{cfg.in_channels},256]" + (f", aux [1,{cfg.aux_feat_dim}]" if use_aux else ""),
        f"- fp32 file: {kb(fp32):.1f} KB; int8 (QDQ) file: {kb(int8):.1f} KB",
        f"- torch vs onnx fp32: max |logit diff| = {max_diff:.2e}; prediction agreement = {100*agree_fp:.2f}%",
        f"- onnx fp32 vs onnx int8 prediction agreement = {100*agree_q:.2f}% on {len(x)} beats",
        f"- balanced recall (N/S/V) on these beats: fp32 {100*rec_fp:.1f}%, int8 {100*rec_q:.1f}%",
        "", "Note: file size includes graph metadata; the weight payload is smaller. Report the runtime memory "
        "measured on the target device, not this file size.",
    ]
    (out / "export_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())