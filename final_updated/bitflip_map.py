"""
bitflip_map.py  --  exhaustive single-bit-flip vulnerability map of the ADAPTED parameter set.

For every element of the adapted set (dw_bn weight and bias, 384 values in this model) and every
bit position in --bits, flip that ONE bit, run a fixed stratified set of DS2 beats through the
model, and record the damage; then repeat with the deployed guard applied after the flip.
This replaces "random injection" with a complete enumeration: it answers "what is the WORST a
single upset can do", with and without the guard, on the evaluated beats.

Damage metrics (on a class-balanced sample of --n_per_class N, S and V beats, so they do not depend
on the 90% N prior):  pred_change = fraction of beats whose predicted class differs from the
unflipped model;  bal_recall = mean of N/S/V recall;  drop = bal_recall(unflipped) - bal_recall(flipped).
catastrophic = drop > 0.20.   Non-finite outputs count as all-N predictions (drop is computed from that).

Scope: single-bit upsets of float32 values in the adapted set only (see fault_injection.py for
what is not modelled). The guard assumes the stored source copy is intact.

  !python bitflip_map.py --checkpoint ckpt.pt --data_dir .. --cache_dir .. --out_dir ./results/bitflip
Resumable (partial results are appended to bitflip_rows.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from fault_injection import flip_bit_
from safety_controller import PhysioGuardTTA, SafetyControllerConfig

DEFAULT_BITS = "31,30,29,28,27,26,25,24,23,22,18,12"


def balanced_sample(y: np.ndarray, n_per_class: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1, 2):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False))
    return np.concatenate(idx)


@torch.no_grad()
def predict(model, x, rr) -> np.ndarray:
    out = model(x, aux=rr).logits
    bad = ~torch.isfinite(out).all(dim=-1)
    pred = out.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).argmax(-1)
    pred[bad] = 0                                              # non-finite output -> reported as N
    return pred.numpy()


def bal_recall(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1, 2)]))


def run_map(model, x, rr, y, bits: Sequence[int], out: Path, max_flips: int = 0, log_every: int = 200,
            guard_k: float = 4.0) -> List[dict]:
    model.eval()
    model.configure_for_tta()
    ctrl = PhysioGuardTTA(SafetyControllerConfig(fault_guard=True, guard_k=guard_k))
    ctrl.reset(model)
    base_pred = predict(model, x, rr)
    base_rec = bal_recall(y, base_pred)
    rows_path = out / "bitflip_rows.csv"
    done = set()
    rows: List[dict] = []
    if rows_path.exists():
        with open(rows_path) as f:
            for r in csv.DictReader(f):
                done.add((r["param"], int(r["index"]), int(r["bit"])))
                rows.append({**r, "index": int(r["index"]), "bit": int(r["bit"]),
                             "drop": float(r["drop"]), "drop_guarded": float(r["drop_guarded"]),
                             "pred_change": float(r["pred_change"]), "pred_change_guarded": float(r["pred_change_guarded"])})
    cols = ["param", "index", "bit", "old", "new", "drop", "pred_change", "guard_active", "drop_guarded", "pred_change_guarded"]
    new_file = not rows_path.exists()
    fh = open(rows_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new_file:
        w.writeheader()
    t0, n_run = time.time(), 0
    for name, p in model.tta_named_parameters():
        for idx in range(p.numel()):
            for bit in bits:
                if (name, idx, bit) in done:
                    continue
                if max_flips and n_run >= max_flips:
                    fh.close()
                    return rows
                orig = p.data.clone()
                old, new = flip_bit_(p, idx, bit)
                pred = predict(model, x, rr)
                drop, chg = base_rec - bal_recall(y, pred), float((pred != base_pred).mean())
                ctrl.guard(model)                                   # deployed guard, applied after the upset
                # guard only matters if it changed the flipped value
                flipped_val = new
                after_val = float(p.data.view(-1)[idx])
                guard_active = int((not math.isfinite(flipped_val)) or abs(after_val - flipped_val) > 0)
                if guard_active:
                    predg = predict(model, x, rr)
                    dropg, chgg = base_rec - bal_recall(y, predg), float((predg != base_pred).mean())
                else:
                    dropg, chgg = drop, chg
                p.data.copy_(orig)
                row = {"param": name, "index": idx, "bit": bit, "old": old, "new": new, "drop": drop,
                       "pred_change": chg, "guard_active": guard_active, "drop_guarded": dropg, "pred_change_guarded": chgg}
                w.writerow(row)
                rows.append(row)
                n_run += 1
                if n_run % log_every == 0:
                    fh.flush()
                    print(f"  {n_run} flips done ({time.time()-t0:.0f}s)  worst drop so far: "
                          f"unguarded {max(r['drop'] for r in rows):.3f}  guarded {max(r['drop_guarded'] for r in rows):.3f}", flush=True)
    fh.close()
    return rows


def summarise(rows: List[dict], base_rec: float, n_beats: int) -> str:
    cat = 0.20
    L = [f"Exhaustive single-bit-flip map: {len(rows)} flips, evaluated on {n_beats} class-balanced beats "
         f"(unflipped balanced recall {100*base_rec:.1f}%).", "",
         f"Overall: catastrophic (drop>{cat:.0%}) unguarded {100*np.mean([r['drop']>cat for r in rows]):.1f}% of flips, "
         f"guarded {100*np.mean([r['drop_guarded']>cat for r in rows]):.2f}%. Worst single flip: unguarded drop "
         f"{max(r['drop'] for r in rows):.3f}, guarded {max(r['drop_guarded'] for r in rows):.3f}. "
         f"Mean prediction change: unguarded {100*np.mean([r['pred_change'] for r in rows]):.2f}%, guarded "
         f"{100*np.mean([r['pred_change_guarded'] for r in rows]):.2f}%.", "",
         "| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for bit in sorted({r["bit"] for r in rows}, reverse=True):
        g = [r for r in rows if r["bit"] == bit]
        field = "sign" if bit == 31 else ("exponent" if bit >= 23 else "mantissa")
        L.append(f"| {bit} | {field} | {len(g)} | {np.mean([r['drop'] for r in g]):.4f} | {max(r['drop'] for r in g):.3f} | "
                 f"{100*np.mean([r['drop']>cat for r in g]):.1f} | {np.mean([r['drop_guarded'] for r in g]):.4f} | "
                 f"{max(r['drop_guarded'] for r in g):.3f} | {100*np.mean([r['drop_guarded']>cat for r in g]):.2f} |")
    L += ["", "By parameter type:", "", "| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |", "|---|---|---|---|---|"]
    for t in ("weight", "bias"):
        g = [r for r in rows if r["param"].endswith(t)]
        if g:
            L.append(f"| dw_bn.{t} | {len(g)} | {100*np.mean([r['drop']>cat for r in g]):.1f} | {max(r['drop'] for r in g):.3f} | {max(r['drop_guarded'] for r in g):.3f} |")
    worst = sorted(rows, key=lambda r: -r["drop"])[:8]
    L += ["", "Most damaging unguarded flips:", "", "| param | element | bit | old -> new | drop | guarded drop |", "|---|---|---|---|---|---|"]
    for r in worst:
        L.append(f"| {r['param']} | {r['index']} | {r['bit']} | {float(r['old']):.3g} -> {float(r['new']):.3g} | {r['drop']:.3f} | {r['drop_guarded']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--bits", default=DEFAULT_BITS)
    p.add_argument("--n_per_class", type=int, default=200)
    p.add_argument("--max_flips", type=int, default=0, help="stop after N new flips (0 = all)")
    p.add_argument("--out_dir", default="./results/bitflip")
    p.add_argument("--guard_k", type=float, default=4.0, help="guard box half-width in std units (default 4.0)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    from run_experiments import load_source_model_and_stream
    model, ds, is_real = load_source_model_and_stream(a.checkpoint, a.data_dir, a.cache_dir, 1, 3, 400, 0)
    if a.checkpoint and not is_real:
        raise SystemExit("ERROR: real checkpoint/data unusable (mount Drive, pip install wfdb, check paths).")
    if not a.checkpoint and not a.smoke:
        raise SystemExit("ERROR: give --checkpoint (or --smoke).")
    y_all = ds.y.numpy()
    sel = balanced_sample(y_all, a.n_per_class)
    x = ds.x[sel]
    rr = ds.rr[sel] if getattr(ds, "rr", None) is not None and ds.rr.shape[-1] > 0 else None
    y = y_all[sel]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bits = [int(b) for b in a.bits.split(",")]
    model.eval()
    base_rec = bal_recall(y, predict(model, x, rr))
    print(f"[bitflip_map] {len(sel)} beats; balanced recall {100*base_rec:.1f}%; "
          f"{sum(q.numel() for _, q in model.tta_named_parameters())} adapted values x {len(bits)} bits"
          f"{'  [SYNTHETIC]' if not is_real else ''}", flush=True)
    rows = run_map(model, x, rr, y, bits, out, a.max_flips, guard_k=a.guard_k)
    md = summarise(rows, base_rec, len(sel))
    (out / "bitflip_summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    import shutil
    d = "./results/_selftest_bitflip"
    shutil.rmtree(d, ignore_errors=True)
    assert main(["--smoke", "--bits", "31,30,29,23,22,0", "--n_per_class", "40", "--max_flips", "200", "--out_dir", d]) == 0
    assert main(["--smoke", "--bits", "31,30,29,23,22,0", "--n_per_class", "40", "--max_flips", "100", "--out_dir", d]) == 0   # resume appends
    n = sum(1 for _ in open(Path(d) / "bitflip_rows.csv")) - 1
    assert n == 300, n
    rows = [{k: v for k, v in r.items()} for r in csv.DictReader(open(Path(d) / "bitflip_rows.csv"))]
    assert all(float(r["drop_guarded"]) <= float(r["drop"]) + 0.5 for r in rows)
    shutil.rmtree(d, ignore_errors=True)
    print("Self-test passed.")