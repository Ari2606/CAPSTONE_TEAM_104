"""
ci_eval.py  --  record-level bootstrap CIs on DS2 and a full-DS2 int8 score.

Strict 5-class scoring (argmax over N/S/V/F/Q, all beats). DS2 records are resampled WITH replacement (the unit of
independence is the patient, not the beat); the same resample is applied to every seed, per-seed metrics are
computed from summed per-record counts, and the CI is for the MEAN OVER SEEDS. Also reported with record 232 removed
from the pool. Optionally scores an int8 ONNX model on every DS2 beat against the fp32 checkpoint.

  python ci_eval.py --checkpoint a.pt,b.pt --data_dir ... --cache_dir ... --out_dir ... [--int8 model_int8.onnx]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

CLS = {"S": 1, "V": 2}


def record_counts(logits: np.ndarray, y: np.ndarray, rid: np.ndarray, recs: np.ndarray) -> np.ndarray:
    """[n_rec, 2(S,V), 3(tp,fp,fn)] strict 5-class counts per record."""
    pred = logits.argmax(1)
    out = np.zeros((len(recs), 2, 3))
    for i, r in enumerate(recs):
        m = rid == r
        for j, c in enumerate((1, 2)):
            out[i, j] = [((pred[m] == c) & (y[m] == c)).sum(), ((pred[m] == c) & (y[m] != c)).sum(),
                         ((pred[m] != c) & (y[m] == c)).sum()]
    return out


def _prf(cnt: np.ndarray) -> np.ndarray:
    """cnt [..., 3] -> [..., 3] (Se, PPV, F1) in %."""
    tp, fp, fn = cnt[..., 0], cnt[..., 1], cnt[..., 2]
    se, ppv = tp / np.maximum(tp + fn, 1), tp / np.maximum(tp + fp, 1)
    return 100 * np.stack([se, ppv, 2 * se * ppv / np.maximum(se + ppv, 1e-12)], -1)


def bootstrap(counts: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict:
    """counts [n_seeds, n_rec, 2, 3]. Returns point estimate and 95% CI of the seed-mean (Se,PPV,F1) per class."""
    rng = np.random.default_rng(seed)
    n_rec = counts.shape[1]
    point = _prf(counts.sum(1)).mean(0)                                   # [2,3]
    reps = np.empty((n_boot, 2, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n_rec, n_rec)
        reps[b] = _prf(counts[:, idx].sum(1)).mean(0)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return {nm: {m: (float(point[j, k]), float(lo[j, k]), float(hi[j, k])) for k, m in enumerate(("Se", "PPV", "F1"))}
            for j, nm in enumerate(("S", "V"))}


def analyse(per_seed_logits: List[np.ndarray], y, rid, n_boot: int = 2000) -> Dict:
    recs = np.unique(rid)
    c_all = np.stack([record_counts(l, y, rid, recs) for l in per_seed_logits])
    keep = recs != 232
    return {"all_records": bootstrap(c_all, n_boot), "without_232": bootstrap(c_all[:, keep], n_boot)}


def render(res: Dict) -> str:
    L = ["| records | class | Se | PPV | F1 |", "|---|---|---|---|---|"]
    f = lambda t: f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"
    for k, nm in (("all_records", "all DS2"), ("without_232", "232 removed")):
        for c in ("S", "V"):
            r = res[k][c]
            L.append(f"| {nm} | {c} | {f(r['Se'])} | {f(r['PPV'])} | {f(r['F1'])} |")
    return "\n".join(L) + "\n\nMean over seeds with 95% CI from a record-level bootstrap of DS2 records (22 records, " \
           "so CIs are wide by construction). Strict 5-class scoring.\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--int8", default=None, help="int8 ONNX for the FIRST checkpoint; scored on all DS2 beats")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args(argv)
    from dataset import ECGDataConfig, build_datasets
    from eval_cross_domain import load_checkpoint
    from operating_points import logits_of
    from protocol_eval import evaluate
    lgs, y, rid, first_ds = [], None, None, None
    for ck in [c.strip() for c in a.checkpoint.split(",")]:
        model, _, dcfg, ok = load_checkpoint(ck, "cpu", 1)
        assert ok, ck
        cfg = ECGDataConfig(**{**dataclasses.asdict(dcfg), "data_dir": a.data_dir, "cache_dir": a.cache_dir})
        ds = build_datasets(cfg)["test"]
        lgs.append(logits_of(model, ds))
        y, rid = ds.y.numpy(), ds.record_ids.numpy()
        first_ds = first_ds or ds
    res = analyse(lgs, y, rid, a.n_boot)
    md = render(res)
    if a.int8:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.int8, providers=["CPUExecutionProvider"])
        ds, q = first_ds, []
        use_aux = any(i.name == "aux" for i in sess.get_inputs())
        for i in range(len(ds)):
            feed = {"beat": ds.x[i:i + 1].numpy().astype(np.float32)}
            if use_aux:
                feed["aux"] = ds.rr[i:i + 1].numpy().astype(np.float32)
            q.append(sess.run(None, feed)[0][0])
        q = np.stack(q)
        f = lambda t, c: f"{t[c]['Se']:.1f}/{t[c]['PPV']:.1f}/{t[c]['F1']:.1f}"
        ef, eq = evaluate(lgs[0], y, rid)["A_5class"], evaluate(q, y, rid)["A_5class"]
        md += ("\n| first checkpoint, all DS2 beats | acc | S Se/PPV/F1 | V Se/PPV/F1 |\n|---|---|---|---|\n"
               f"| fp32 | {ef['acc']:.1f} | {f(ef, 1)} | {f(ef, 2)} |\n| int8 (ONNX QDQ) | {eq['acc']:.1f} | {f(eq, 1)} | {f(eq, 2)} |\n"
               f"\nint8 vs fp32 argmax agreement on all DS2 beats: {100 * float((q.argmax(1) == lgs[0].argmax(1)).mean()):.2f}%\n")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "ci.json").write_text(json.dumps(res, indent=1)); (out / "ci.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main())
    rng = np.random.default_rng(0)
    n = 8000
    y = rng.choice(5, n, p=[.8, .06, .1, .03, .01]); rid = rng.choice([100, 101, 232, 233, 234], n)
    L = []
    for _ in range(3):
        lg = rng.normal(size=(n, 5)); lg[np.arange(n), y] += 2.0; L.append(lg)
    res = analyse(L, y, rid, 200)
    from protocol_eval import evaluate
    pt = np.mean([evaluate(l, y, rid)["A_5class"][2]["F1"] for l in L])    # pooled point estimate must match
    assert abs(res["all_records"]["V"]["F1"][0] - pt) < 1e-6, (res["all_records"]["V"]["F1"], pt)
    for k in res:
        for c in res[k]:
            for m in res[k][c]:
                p, lo, hi = res[k][c][m]
                assert lo <= hi and 0 <= lo and hi <= 100
    print(render(res)); print("Self-test passed.")
