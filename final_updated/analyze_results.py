"""
analyze_results.py  --  statistics for the paper, from run_benchmark.py outputs.

Two sources of variability matter and are reported separately, because the earlier
3-seed analysis (corruption draw only) understated one of them:

 (a) WHICH PATIENTS: DS2 has only 22 records; a few patients dominate S and V.
     -> record-level (cluster) bootstrap: resample records with replacement, recompute
        every metric from the pooled confusion matrix of the resampled records, B draws.
        The SAME resampled records are used for every method and every (checkpoint, seed)
        unit, so differences between methods are PAIRED. 95% percentile intervals.
 (b) WHICH MODEL / WHICH NOISE DRAW: independent source checkpoints (different training
     seeds) and independent corruption/noise seeds.
     -> mean +- sd over units (checkpoint x seed), and a paired sign-flip permutation test
        on per-unit differences vs the reference method (exact when <= 14 units, otherwise
        200k Monte-Carlo flips). Where >= 5 checkpoints exist, a second test is run on
        checkpoint-level means (seeds averaged within a checkpoint) because seeds sharing
        a checkpoint are not independent.
 Holm-Bonferroni adjustment across the comparisons of one family
 (reference vs each other method, one metric, one scenario).

Metrics come from the project's own `classification_report` (same definitions as every
earlier number): accuracy, macro-F1 (present classes), macro-F1 ex-F/Q (N,S,V),
per-class recall, plus mean Jensen-Shannon drift from the source anchor and the
fraction of beats on which an update was applied.

Usage:
  python analyze_results.py --bench_dir <out_dir of run_benchmark> --ref pg \
      --primary macro_f1_ex_fq --out_dir ./results/analysis
Writes: <out_dir>/<scenario>.md (+ .csv) and a LaTeX table `main_table.tex` for the
scenarios listed in --latex_scenarios.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset import CLASS_NAMES, NUM_CLASSES
from eval_cross_domain import classification_report
from run_benchmark import load_compact

K = NUM_CLASSES
METRICS = ["accuracy", "macro_f1", "macro_f1_ex_fq", "recall_S", "recall_V", "drift", "adapt_rate"]
PRETTY = {"accuracy": "Acc.", "macro_f1": "Macro-F1", "macro_f1_ex_fq": "Macro-F1 (N/S/V)",
          "recall_S": "Recall S", "recall_V": "Recall V", "drift": "Drift $D_{JS}$",
          "adapt_rate": "Adapt. rate"}


# --------------------------------------------------------------------------- #
# per-job aggregation
# --------------------------------------------------------------------------- #
class Job:
    """Per-record confusion matrices + per-record drift sums for one (method, unit)."""

    def __init__(self, cols: Dict[str, np.ndarray], rec_index: Dict[int, int]):
        R = len(rec_index)
        self.cm = np.zeros((R, K, K), dtype=np.int64)
        self.js_sum = np.zeros(R)
        self.ad_sum = np.zeros(R)
        self.n = np.zeros(R)
        ri = np.array([rec_index[int(r)] for r in cols["record_id"]])
        np.add.at(self.cm, (ri, cols["y_true"].astype(int), cols["y_pred"].astype(int)), 1)
        js = np.nan_to_num(cols["js_div"].astype(np.float64), nan=0.0)
        np.add.at(self.js_sum, ri, js)
        np.add.at(self.ad_sum, ri, cols["adapted"].astype(np.float64))
        np.add.at(self.n, ri, 1.0)


def metrics_from_cms(cms: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised over a leading batch axis. cms [B,K,K] -> dict of [B] arrays.
    Mirrors eval_cross_domain.classification_report (present classes only)."""
    cms = cms.astype(np.float64)
    tp = np.einsum("bkk->bk", cms)
    sup = cms.sum(2)
    pred = cms.sum(1)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    present = sup > 0
    macro = np.where(present.any(1), (f1 * present).sum(1) / np.maximum(present.sum(1), 1), 0.0)
    nsv = [0, 1, 2]                                           # N, S, V
    pv = present[:, nsv]
    ex = np.where(pv.any(1), (f1[:, nsv] * pv).sum(1) / np.maximum(pv.sum(1), 1), 0.0)
    return {"accuracy": tp.sum(1) / np.maximum(cms.sum((1, 2)), 1), "macro_f1": macro,
            "macro_f1_ex_fq": ex, "recall_S": rec[:, 1], "recall_V": rec[:, 2]}


def job_metrics(job: Job, w: np.ndarray) -> Dict[str, np.ndarray]:
    """w [B,R] record weights (counts). Returns metric -> [B]."""
    cms = np.einsum("br,rij->bij", w, job.cm)
    m = metrics_from_cms(cms)
    n = np.maximum(w @ job.n, 1.0)
    m["drift"] = (w @ job.js_sum) / n
    m["adapt_rate"] = (w @ job.ad_sum) / n
    return m


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def signflip_p(d: np.ndarray, rng: np.random.Generator, mc: int = 200_000) -> float:
    """Two-sided paired sign-flip permutation test on differences d (H0: symmetric about 0)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if n <= 14:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        stat = np.abs((signs * d).mean(1))
    else:
        signs = rng.choice([-1.0, 1.0], size=(mc, n))
        stat = np.abs((signs * d).mean(1))
    return float((np.sum(stat >= obs - 1e-12) + (0 if n <= 14 else 1)) / (len(stat) + (0 if n <= 14 else 1)))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> Tuple[float, float]:
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


# --------------------------------------------------------------------------- #
# per-scenario analysis
# --------------------------------------------------------------------------- #
def analyze_scenario(scn_dir: Path, ref: str, primary: str, n_boot: int, seed: int
                     ) -> Optional[dict]:
    files = sorted(scn_dir.rglob("*.npz"))
    if not files:
        return None
    rng = np.random.default_rng(seed)
    raw: Dict[Tuple[str, str, int], Tuple[dict, dict]] = {}          # (method, ckpt, seed) -> (meta, cols)
    rec_ids = set()
    for f in files:
        meta, cols = load_compact(f)
        raw[(meta["method"], meta["ckpt"], int(meta["seed"]))] = (meta, cols)
        rec_ids |= set(int(r) for r in np.unique(cols["record_id"]))
    rec_index = {r: i for i, r in enumerate(sorted(rec_ids))}
    R = len(rec_index)
    methods = sorted({k[0] for k in raw})
    units = sorted({(k[1], k[2]) for k in raw})
    # units where EVERY method is present (needed for paired comparison)
    units = [u for u in units if all((m, *u) in raw for m in methods)]
    if ref not in methods or not units:
        print(f"[analyze] {scn_dir.name}: reference '{ref}' missing or no complete units -> skipped")
        return None
    is_real = all(raw[(m, *u)][0]["is_real"] for m in methods for u in units)
    jobs = {(m, *u): Job(raw[(m, *u)][1], rec_index) for m in methods for u in units}

    full_w = np.ones((1, R))
    boot_w = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(np.float64)   # shared across all

    # per (method, metric): unit-level point values, bootstrap draws averaged over units
    unit_vals: Dict[Tuple[str, str], np.ndarray] = {}
    boot_vals: Dict[Tuple[str, str], np.ndarray] = {}
    for m in methods:
        pu = {k: [] for k in METRICS}
        bu = {k: [] for k in METRICS}
        for u in units:
            j = jobs[(m, *u)]
            full = job_metrics(j, full_w)
            bt = job_metrics(j, boot_w)
            for k in METRICS:
                pu[k].append(float(full[k][0])); bu[k].append(bt[k])
        for k in METRICS:
            unit_vals[(m, k)] = np.array(pu[k])
            boot_vals[(m, k)] = np.mean(np.stack(bu[k]), axis=0)          # [B] mean over units

    ckpts = sorted({u[0] for u in units})
    ck_of = np.array([ckpts.index(u[0]) for u in units])

    rows = []
    for m in methods:
        row = {"method": m}
        for k in METRICS:
            uv = unit_vals[(m, k)]
            lo, hi = ci(boot_vals[(m, k)])
            row[k] = float(uv.mean()); row[k + "_ci"] = (lo, hi)
            row[k + "_sd_units"] = float(uv.std(ddof=1)) if len(uv) > 1 else float("nan")
        rows.append(row)

    # paired comparison vs reference, every metric in {primary, accuracy, macro_f1, recall_V, recall_S}
    comps: Dict[str, Dict[str, dict]] = {}
    for k in ["macro_f1_ex_fq", "macro_f1", "accuracy", "recall_S", "recall_V", "drift"]:
        pvals_perm, pvals_ck, entries = {}, {}, {}
        for m in methods:
            if m == ref:
                continue
            d_units = unit_vals[(ref, k)] - unit_vals[(m, k)]
            d_boot = boot_vals[(ref, k)] - boot_vals[(m, k)]
            lo, hi = ci(d_boot)
            p_boot = float(min(1.0, 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())))
            p_perm = signflip_p(d_units, rng)
            p_ck = float("nan")
            if len(ckpts) >= 5:
                d_ck = np.array([d_units[ck_of == c].mean() for c in range(len(ckpts))])
                p_ck = signflip_p(d_ck, rng)
            pvals_perm[m] = p_perm
            pvals_ck[m] = p_ck if not np.isnan(p_ck) else 1.0
            entries[m] = {"delta": float(d_units.mean()), "delta_ci": (lo, hi), "p_boot_records": p_boot,
                          "p_units": p_perm, "p_ckpt": p_ck}
        hp = holm(pvals_perm)
        hc = holm(pvals_ck) if len(ckpts) >= 5 else {}
        for m, e in entries.items():
            e["p_units_holm"] = hp[m]
            e["p_ckpt_holm"] = hc.get(m, float("nan"))
        comps[k] = entries
    return {"scenario": scn_dir.name, "ref": ref, "primary": primary, "rows": rows, "comps": comps,
            "n_units": len(units), "n_ckpts": len(ckpts), "n_records": R, "is_real": bool(is_real),
            "n_boot": n_boot, "methods": methods}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _p(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def render_md(res: dict) -> str:
    ref, prim = res["ref"], res["primary"]
    L = [f"## {res['scenario']}  ({'REAL data' if res['is_real'] else 'SYNTHETIC smoke test -- not a result'})",
         f"{res['n_units']} units ({res['n_ckpts']} checkpoints x seeds), {res['n_records']} records, "
         f"{res['n_boot']} record-bootstrap draws. Intervals: 95% record-bootstrap (patients resampled, paired "
         f"across methods). Delta = {ref} - method; p from paired sign-flip test over units "
         f"(Holm-adjusted within the family of comparisons to {ref}); p_ckpt over checkpoint means when >=5 ckpts.", ""]
    cols = ["accuracy", "macro_f1", "macro_f1_ex_fq", "recall_S", "recall_V", "drift", "adapt_rate"]
    L.append("| method | " + " | ".join(PRETTY[c].replace("$", "") for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for r in res["rows"]:
        cells = []
        for c in cols:
            lo, hi = r[c + "_ci"]
            cells.append(f"{r[c]:.3f} [{lo:.3f}, {hi:.3f}]" if c in ("macro_f1_ex_fq", "macro_f1", "recall_V", "recall_S")
                         else f"{r[c]:.3f}")
        name = f"**{r['method']}**" if r["method"] == ref else r["method"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    for k in ("macro_f1_ex_fq", "recall_V", "accuracy", "drift"):
        L += ["", f"### {ref} minus method, {PRETTY[k].replace('$','')}", "",
              "| method | delta | 95% CI (records) | p (units, Holm) | p (ckpts, Holm) | p (records bootstrap) |",
              "|---|---|---|---|---|---|"]
        for m, e in res["comps"][k].items():
            L.append(f"| {m} | {e['delta']:+.3f} | [{e['delta_ci'][0]:+.3f}, {e['delta_ci'][1]:+.3f}] | "
                     f"{_p(e['p_units_holm'])} | {_p(e['p_ckpt_holm'])} | {_p(e['p_boot_records'])} |")
    return "\n".join(L) + "\n"


def _tex(s: str) -> str:
    return s.replace("_", "\\_")


def render_latex(results: List[dict], ref: str, methods_order: Optional[List[str]] = None) -> str:
    """Rows = methods; one macro-F1 (N/S/V) [95% CI] column per scenario, then Recall V and
    drift for the first scenario."""
    res0 = results[0]
    methods = methods_order or res0["methods"]
    heads = ["Method"] + [_tex(r["scenario"]) for r in results] + [PRETTY["recall_V"], PRETTY["drift"]]
    lines = ["\\begin{tabular}{l" + "c" * (len(results) + 2) + "}", "\\toprule",
             " & ".join(heads) + " \\\\", "\\midrule"]
    by_scn = {r["scenario"]: {x["method"]: x for x in r["rows"]} for r in results}
    for m in methods:
        cells = [_tex(m)]
        for r in results:
            x = by_scn[r["scenario"]].get(m)
            if x is None:
                cells.append("-")
                continue
            lo, hi = x["macro_f1_ex_fq_ci"]
            cell = f"{x['macro_f1_ex_fq']:.3f} [{lo:.3f},{hi:.3f}]"
            cells.append("\\textbf{" + cell + "}" if m == ref else cell)
        x = by_scn[results[0]["scenario"]].get(m)
        cells += [f"{x['recall_V']:.3f}", f"{x['drift']:.3f}"] if x else ["-", "-"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--ref", default="pg")
    p.add_argument("--primary", default="macro_f1_ex_fq")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="./results/analysis")
    p.add_argument("--latex_scenarios", default="")
    a = p.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    allres: Dict[str, dict] = {}
    for d in sorted(Path(a.bench_dir).iterdir()):
        if not d.is_dir():
            continue
        r = analyze_scenario(d, a.ref, a.primary, a.n_boot, a.seed)
        if r is None:
            continue
        allres[d.name] = r
        md = render_md(r)
        (out / f"{d.name}.md").write_text(md)
        (out / f"{d.name}.json").write_text(json.dumps(r, indent=1, default=str))
        print(md)
    sel = [allres[s] for s in a.latex_scenarios.split(",") if s in allres]
    if sel:
        (out / "main_table.tex").write_text(render_latex(sel, a.ref))
        print(f"[analyze] wrote {out/'main_table.tex'}")
    return allres


# --------------------------------------------------------------------------- #
# Self-test: simulated logs with KNOWN truth -> the statistics must behave
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import tempfile
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    # 1. metrics_from_cms must equal the project's classification_report
    rng = np.random.default_rng(0)
    cm = rng.integers(0, 40, size=(K, K)); cm[3] = 0; cm[:, 3] = 0        # empty class F
    ref_rep = classification_report(cm)
    got = {k: v[0] for k, v in metrics_from_cms(cm[None]).items()}
    for k in ("accuracy", "macro_f1", "macro_f1_ex_fq"):
        assert abs(got[k] - ref_rep[k]) < 1e-9, (k, got[k], ref_rep[k])
    assert abs(got["recall_V"] - ref_rep["per_class"]["V"]["recall"]) < 1e-9
    # 2. Holm, sign-flip
    assert holm({"a": 0.01, "b": 0.04, "c": 0.5}) == {"a": 0.03, "b": 0.08, "c": 0.5}
    assert signflip_p(np.zeros(5), rng) == 1.0
    assert signflip_p(np.full(8, 0.1), rng) < 0.02                        # all same sign, exact p = 2/256
    # 3. end-to-end on simulated benchmark files with a known effect
    from run_benchmark import LOG_COLS
    from method_zoo import GROUPS  # noqa: F401  (import check only)
    with tempfile.TemporaryDirectory() as td:
        n_rec, per = 12, 400
        p_correct = {"ref": 0.80, "worse": 0.72, "same": 0.80}
        sim_rng = np.random.default_rng(1)
        rec_difficulty = sim_rng.normal(0, 0.04, n_rec)                   # patients differ
        for ck in ("c0", "c1", "c2", "c3", "c4", "c5"):
            ck_shift = sim_rng.normal(0, 0.01)
            for sd in range(3):
                base_y = sim_rng.choice(K, size=n_rec * per, p=[0.7, 0.1, 0.15, 0.03, 0.02])
                rid = np.repeat(np.arange(n_rec), per)
                for name, pc in p_correct.items():
                    prob = np.clip(pc + ck_shift + rec_difficulty[rid], 0.05, 0.99)
                    ok = sim_rng.random(len(base_y)) < prob
                    pred = np.where(ok, base_y, (base_y + 1 + sim_rng.integers(0, K - 1, len(base_y))) % K)
                    cols = {k: np.full(len(base_y), np.nan, np.float32) for k in LOG_COLS}
                    cols.update(step=np.arange(len(base_y)).astype(np.int32), record_id=rid.astype(np.int32),
                                y_true=base_y.astype(np.int32), y_pred=pred.astype(np.int32),
                                js_div=np.full(len(base_y), {"ref": 0.02, "worse": 0.3, "same": 0.02}[name], np.float32),
                                adapted=np.ones(len(base_y), np.int8))
                    pth = Path(td) / "scn" / ck / f"seed{sd}" / f"{name}.npz"
                    pth.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pth, meta=json.dumps({"method": name, "ckpt": ck, "seed": sd,
                                                              "is_real": True, "scenario": "scn"}), **cols)
        res = analyze_scenario(Path(td) / "scn", "ref", "macro_f1_ex_fq", 400, 0)
        assert res["n_units"] == 18 and res["n_ckpts"] == 6 and res["n_records"] == 12
        c_worse = res["comps"]["accuracy"]["worse"]
        c_same = res["comps"]["accuracy"]["same"]
        print(f"  known effect +0.08: delta={c_worse['delta']:+.3f} CI={c_worse['delta_ci']} "
              f"p_units={c_worse['p_units']:.4f} p_ckpt={c_worse['p_ckpt']:.3f}")
        print(f"  null (same skill) : delta={c_same['delta']:+.3f} CI={c_same['delta_ci']} "
              f"p_units={c_same['p_units']:.3f}")
        assert 0.06 < c_worse["delta"] < 0.10 and c_worse["delta_ci"][0] > 0
        assert c_worse["p_units"] < 0.001 and c_worse["p_ckpt"] < 0.05
        assert c_same["delta_ci"][0] < 0 < c_same["delta_ci"][1] and c_same["p_units"] > 0.05
        assert abs(res["comps"]["drift"]["worse"]["delta"] - (0.02 - 0.3)) < 1e-6
        out = Path(td) / "analysis"
        allres = main(["--bench_dir", td, "--ref", "ref", "--out_dir", str(out), "--n_boot", "200",
                       "--latex_scenarios", "scn"])
        assert (out / "scn.md").exists() and (out / "main_table.tex").exists()
        assert "\\toprule" in (out / "main_table.tex").read_text()
    print("Self-test passed (metrics match classification_report; CI/p behave on known truth).")
