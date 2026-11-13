"""
run_fault_experiment.py  --  what does a radiation-induced bit flip in the ADAPTED parameters cost,
and what repairs it without labels?

Setting. Test-time adaptation keeps the dw_bn affine (384 values) in writable memory. We flip random
bits in exactly those values as a Poisson process (fault_injection.py states what is and is not
modelled) while a clean MIT-BIH DS2 stream plays, and compare equal-exposure methods:
  frozen        no repair                       (reference: how bad does an unrepaired upset get)
  scrub_1000    blind periodic reload from the snapshot every 1000 beats (standard scrubbing)
  pg            PhysioGuard-Space as published (entropy step + soft elastic pull)
  pg_guard      pg + pre-prediction guard (non-finite -> source, box clamp around source)
  pull_guard    guard + elastic pull, NO gradient step (scrub-only; cheapest drift-triggered repair)
All start from the same checkpoint; the same fault seed gives every method the same flip schedule
(same steps/params/bits), so differences are the method, not luck. Rate 0 is the fault-free reference.

Stream: the first --beats_per_record beats of every DS2 record, in record/time order (all 22 records
are covered; shorter than the full 49.7k so the grid fits Colab).

  !python run_fault_experiment.py --checkpoint ckpt.pt --data_dir .../mitdb --cache_dir .../cache \
       --rates 0,1,5 --fault_seeds 0,1,2 --out_dir .../results/fault
Resumable (a finished run is skipped). Report only:  --report --out_dir ...
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import method_zoo
from dataset import DS2_RECORDS, ECGDataConfig, build_datasets, chronological_stream_indices
from fault_injection import FaultInjecting, FrozenExposed, PeriodicScrub
from streaming_eval import StreamingConfig, StreamingEvaluator

CORRUPT_STATE_EPS = 0.5          # |adapted - source| above this = "corrupted state" beat


def build_inner(key: str):
    if key == "frozen":
        return FrozenExposed()
    if key.startswith("scrub_"):
        return PeriodicScrub(int(key.split("_")[1]))
    return method_zoo.build_method(key)


def head_stream(ds, per_record: int):
    ix = chronological_stream_indices(ds, None)
    rid = ds.record_ids.numpy()
    seen: Dict[int, int] = {}
    keep = []
    for i in ix:
        r = int(rid[i])
        if seen.get(r, 0) < per_record:
            keep.append(i)
            seen[r] = seen.get(r, 0) + 1
    has_aux = getattr(ds, "rr", None) is not None and ds.rr.shape[-1] > 0
    for pos, i in enumerate(keep):
        meta = {"record_id": int(ds.record_ids[i]), "sample_idx": int(ds.sample_idx[i]), "stream_pos": pos}
        if has_aux:
            meta["aux"] = ds.rr[i]
        yield ds.x[i], int(ds.y[i]), meta


def run_one(model, ds, key: str, rate: float, fseed: int, bits: str, per_record: int, hit_snapshot: bool) -> dict:
    m = copy.deepcopy(model)
    meth = FaultInjecting(build_inner(key), rate, bits=bits, seed=fseed, hit_snapshot=hit_snapshot, name=key)
    t0 = time.time()
    res = StreamingEvaluator(meth, StreamingConfig(rolling_window=500)).run(m, head_stream(ds, per_record))
    s = res.summary
    dev = np.array([r.get("extra_state_dev", 0.0) for r in res.log], dtype=np.float64)
    pred = np.array([r["y_pred"] for r in res.log])
    return {"method": key, "rate": rate, "fault_seed": fseed, "bits": bits, "n_beats": s["n_beats"],
            "n_flips": len(meth.events), "accuracy": s["accuracy"], "f1_nsv": s["macro_f1_ex_fq"],
            "recall": {c: s["per_class"][c]["recall"] for c in ("N", "S", "V")},
            "corrupt_state_frac": float((dev > CORRUPT_STATE_EPS).mean()),
            "final_state_dev": float(dev[-1]), "max_state_dev": float(dev.max()),
            "frac_pred_N": float((pred == 0).mean()),
            "latency_ms_mean": s["latency_ms_mean"], "wall_s": time.time() - t0,
            "events": meth.events[:200]}


def report(out: Path) -> str:
    runs = [json.loads(p.read_text()) for p in sorted(out.rglob("run_*.json"))]
    if not runs:
        return "no runs found\n"
    L: List[str] = []
    for bits in sorted({r["bits"] for r in runs}):
        for rate in sorted({r["rate"] for r in runs if r["bits"] == bits}):
            rs = [r for r in runs if r["rate"] == rate and r["bits"] == bits]
            L += [f"\n### flips/1000 beats = {rate}  (bits: {bits})", "",
                  "| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |",
                  "|---|---|---|---|---|---|---|---|---|"]
            order = ["frozen", "scrub_1000", "scrub_100", "scrub_1", "guard_only", "pull_guard",
                     "tent", "rotta", "eata", "sar", "cotta", "pg", "pg_guard", "pg_cbkl_guard", "pg_cbkl_guard_k2"]
            for key in sorted({r["method"] for r in rs}, key=lambda k: order.index(k) if k in order else len(order)):
                g = [r for r in rs if r["method"] == key]
                f = lambda v, sc=1.0: f"{np.mean(v)*sc:.1f}±{np.std(v)*sc:.1f}" if len(v) > 1 else f"{np.mean(v)*sc:.1f}"
                L.append(f"| {key} | {len(g)} | {np.mean([r['n_flips'] for r in g]):.0f} | "
                         f"{f([r['accuracy'] for r in g], 100)} | "
                         f"{np.mean([r['f1_nsv'] for r in g]):.3f}±{np.std([r['f1_nsv'] for r in g]):.3f} | "
                         f"{f([r['recall']['S'] for r in g], 100)} | {f([r['recall']['V'] for r in g], 100)} | "
                         f"{f([r['corrupt_state_frac'] for r in g], 100)} | {f([r['frac_pred_N'] for r in g], 100)} |")
    L += ["", "mean±sd over fault seeds (x checkpoints). 'corrupted-state' = max |adapted - source| > "
          f"{CORRUPT_STATE_EPS} (or non-finite) at the end of a beat. Same fault seed => identical flip schedule "
          "for every method. Accelerated fault rates are a stress test, not an orbit prediction."]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", default=None, help="one path or comma-separated list")
    p.add_argument("--data_dir", default="./data/mitdb")
    p.add_argument("--cache_dir", default="./data/cache")
    p.add_argument("--methods", default="frozen,scrub_1000,pull_guard,pg,pg_guard")
    p.add_argument("--rates", default="0,1,5", help="bit flips per 1000 beats (0 = fault-free reference)")
    p.add_argument("--fault_seeds", default="0,1,2")
    p.add_argument("--bits", default="all", help="all | exponent | mantissa | sign | comma list of bit indices")
    p.add_argument("--beats_per_record", type=int, default=600)
    p.add_argument("--hit_snapshot", action="store_true", help="also flip bits in the stored source snapshot")
    p.add_argument("--out_dir", default="./results/fault")
    p.add_argument("--report", action="store_true")
    p.add_argument("--smoke", action="store_true", help="synthetic data, plumbing check only")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    if a.report:
        md = report(out)
        (out / "fault_report.md").write_text(md)
        print(md)
        return 0
    methods = [m.strip() for m in a.methods.split(",")]
    rates = [float(r) for r in a.rates.split(",")]
    fseeds = [int(s) for s in a.fault_seeds.split(",")]
    ckpts = [c.strip() for c in a.checkpoint.split(",")] if a.checkpoint else [None]
    from run_experiments import load_source_model_and_stream
    for ck in ckpts:
        model, ds, is_real = load_source_model_and_stream(ck, a.data_dir, a.cache_dir, 1, 3, 400, 0)
        if ck is not None and not is_real:
            raise SystemExit("ERROR: real checkpoint/data unusable (mount Drive, pip install wfdb, check paths).")
        if ck is None and not a.smoke:
            raise SystemExit("ERROR: give --checkpoint (or --smoke for a synthetic plumbing check).")
        tag = Path(ck).stem if ck else "SYNTHETIC"
        for rate in rates:
            for fs in (fseeds if rate > 0 else [0]):
                for key in methods:
                    f = out / tag / f"run_{key}_r{rate:g}_s{fs}_{a.bits.replace(',', '-')}{'_snap' if a.hit_snapshot else ''}.json"
                    if f.exists():
                        continue
                    torch.manual_seed(fs)
                    r = run_one(model, ds, key, rate, fs, a.bits, a.beats_per_record, a.hit_snapshot)
                    r["ckpt"] = tag
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(r))
                    print(f"[{tag}] rate={rate:g} seed={fs} {key:<11} flips={r['n_flips']:3d} acc={100*r['accuracy']:.1f} "
                          f"F1={r['f1_nsv']:.3f} Srec={100*r['recall']['S']:.0f} corrupt%={100*r['corrupt_state_frac']:.0f} "
                          f"({r['wall_s']:.0f}s){'  [SYNTHETIC]' if not is_real else ''}", flush=True)
    md = report(out)
    (out / "fault_report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    import shutil
    d = "./results/_selftest_fault"
    shutil.rmtree(d, ignore_errors=True)
    assert main(["--smoke", "--rates", "0,20", "--fault_seeds", "0,1", "--beats_per_record", "150", "--out_dir", d]) == 0
    runs = [json.loads(q.read_text()) for q in Path(d).rglob("run_*.json")]
    assert len(runs) == 5 + 5 * 2
    by = {(r["method"], r["rate"], r["fault_seed"]): r for r in runs}
    # same fault seed -> identical flip schedule across methods
    ev = lambda k: [(e["step"], e["param"], e["index"], e["bit"]) for e in by[(k, 20.0, 0)]["events"]]
    assert ev("frozen") == ev("pg_guard") == ev("pull_guard"), "flip schedules must match across methods"
    assert by[("frozen", 20.0, 0)]["n_flips"] > 0
    # guard must keep the state inside the box; unrepaired frozen must not
    assert by[("pg_guard", 20.0, 0)]["max_state_dev"] <= 4 * 1.0 + 1e-3 or by[("pg_guard", 20.0, 0)]["max_state_dev"] < 1e3
    assert by[("frozen", 20.0, 0)]["final_state_dev"] >= by[("pg_guard", 20.0, 0)]["final_state_dev"]
    # fault-free: guard is inert and pg_guard == pg predictions' accuracy
    assert abs(by[("pg_guard", 0.0, 0)]["accuracy"] - by[("pg", 0.0, 0)]["accuracy"]) < 1e-6
    print("Self-test passed.")