"""
protocol_eval.py  --  the SAME checkpoints scored under the protocols the literature uses, each labelled.

For each checkpoint on the full clean DS2:
  A. 5-class AAMI (our strict row): argmax over N/S/V/F/Q, all beats scored.
  B. 3-class N/S/V (Farag / Zhang style): F and Q beats REMOVED from scoring, logits restricted to N/S/V.
     Reported with all DS2 records, and again with record 232 removed (it holds ~75% of DS2 S beats).
  C. Binary screening (normal vs any arrhythmia: S, V, F, Q): sensitivity, specificity, PPV, on all beats.
No retraining, no threshold tuning. Everything is deterministic given the checkpoint.

  python protocol_eval.py --checkpoint a.pt,b.pt --data_dir ... --cache_dir ... --out_dir ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    se = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    return {"Se": 100 * se, "PPV": 100 * ppv, "F1": 100 * 2 * se * ppv / max(se + ppv, 1e-12)}


def class_table(y: np.ndarray, pred: np.ndarray, classes) -> Dict:
    out = {}
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        out[c] = prf(tp, int(((pred == c) & (y != c)).sum()), int(((pred != c) & (y == c)).sum()))
    out["acc"] = 100 * float((pred == y).mean())
    out["macro_F1"] = float(np.mean([out[c]["F1"] for c in classes]))
    return out


def evaluate(logits: np.ndarray, y: np.ndarray, rid: np.ndarray) -> Dict:
    r = {"A_5class": class_table(y, logits.argmax(1), (0, 1, 2))}
    keep = y <= 2                                                 # drop F (3) and Q (4) from scoring
    p3 = logits[:, :3].argmax(1)                                  # restrict outputs to N/S/V
    r["B_3class_all_records"] = class_table(y[keep], p3[keep], (0, 1, 2))
    k2 = keep & (rid != 232)
    r["B_3class_without_232"] = class_table(y[k2], p3[k2], (0, 1, 2))
    pa = logits.argmax(1) != 0
    ya = y != 0
    tp, fp, fn, tn = int((pa & ya).sum()), int((pa & ~ya).sum()), int((~pa & ya).sum()), int((~pa & ~ya).sum())
    r["C_binary_screening"] = {**prf(tp, fp, fn), "Spec": 100 * tn / max(tn + fp, 1), "acc": 100 * (tp + tn) / len(y)}
    return r


def render(per_ckpt: Dict[str, Dict]) -> str:
    keys = ["A_5class", "B_3class_all_records", "B_3class_without_232"]
    names = {"A_5class": "A. 5-class, all beats (strict)", "B_3class_all_records": "B. 3-class N/S/V, F/Q removed",
             "B_3class_without_232": "B'. as B, record 232 excluded"}
    f = lambda t: f"{t['Se']:.1f}/{t['PPV']:.1f}/{t['F1']:.1f}"
    L = ["| protocol | acc % | N Se/PPV/F1 | S Se/PPV/F1 | V Se/PPV/F1 | macro-F1 |", "|---|---|---|---|---|---|"]
    for k in keys:
        rs = [v[k] for v in per_ckpt.values()]
        m = lambda c, q: float(np.mean([r[c][q] for r in rs]))
        row = lambda c: f"{m(c,'Se'):.1f}/{m(c,'PPV'):.1f}/{m(c,'F1'):.1f}"
        L.append(f"| {names[k]} | {np.mean([r['acc'] for r in rs]):.1f} | {row(0)} | {row(1)} | {row(2)} | "
                 f"{np.mean([r['macro_F1'] for r in rs]):.1f} |")
    b = [v["C_binary_screening"] for v in per_ckpt.values()]
    L += ["", f"C. Binary screening (normal vs any arrhythmia, all beats): sensitivity {np.mean([x['Se'] for x in b]):.1f}, "
          f"specificity {np.mean([x['Spec'] for x in b]):.1f}, PPV {np.mean([x['PPV'] for x in b]):.1f}, "
          f"accuracy {np.mean([x['acc'] for x in b]):.1f}  (mean of {len(b)} checkpoints)",
          "", f"Means over {len(per_ckpt)} checkpoint(s). Row A is the strict protocol; B is the protocol used by some 3-class "
          "papers (not directly comparable to 5-class); C is a different task and must not be mixed with S/V rows."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="./data/mitdb")
    ap.add_argument("--cache_dir", default="./data/cache")
    ap.add_argument("--out_dir", default="./results/protocols")
    a = ap.parse_args(argv)
    from dataset import ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    from operating_points import logits_of
    per = {}
    for ck in [c.strip() for c in a.checkpoint.split(",")]:
        model, _, dcfg, ok = load_checkpoint(ck, "cpu", 1)
        assert ok, ck
        cfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
        ds = build_datasets(cfg)["test"]
        per[Path(ck).stem] = evaluate(logits_of(model, ds), ds.y.numpy(), ds.record_ids.numpy())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "protocols.json").write_text(json.dumps(per, indent=1))
    md = render(per)
    (out / "protocols.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.choice(5, n, p=[.8, .06, .1, .03, .01]); rid = rng.choice([100, 232, 233], n)
    lg = rng.normal(size=(n, 5)); lg[np.arange(n), y] += 2.5
    r = evaluate(lg, y, rid)
    assert r["B_3class_without_232"][1]["Se"] >= 0 and 0 < r["C_binary_screening"]["Spec"] <= 100
    keep = y <= 2
    assert abs(r["B_3class_all_records"]["acc"] - 100 * float((lg[:, :3].argmax(1)[keep] == y[keep]).mean())) < 1e-9
    assert "B'. as B" in render({"a": r, "b": r})
    print("Self-test passed.")
