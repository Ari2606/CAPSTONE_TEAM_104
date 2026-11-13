"""
streaming_eval.py  --  PhysioGuard-Space, shared streaming-TTA evaluation harness

Used by BOTH Stage 4 (`baselines_tta.py`: Source-Only, TENT, SQI-gated TENT)
and Stage 5 (`safety_controller.py`: PhysioGuard-Space) so every method is scored under
the exact same protocol -- this is what makes their numbers comparable.

Protocol (strict online TTA, N = 1)
------------------------------------
For beat t, a method's `step()`:
  1. runs ONE forward pass with the model's CURRENT (pre-update) weights,
  2. returns those logits as beat t's prediction (this is what gets scored), and
  3. may then perform an unsupervised, gradient-based update using only that beat's
     own signal (entropy, an SQI mask, JS-divergence vs. the frozen anchor, ...) --
     NEVER the true label. Weight `y` is passed to `StreamingEvaluator.run()` for
     scoring ONLY; it is never exposed to `TTAMethod.step()`.
This "predict-with-current-weights, then adapt" order is what makes the reported
accuracy/F1 a genuine streaming (causal) number rather than a post-hoc one.

Design: this module owns the LOOP, timing, and metrics. It knows nothing about how
any method decides to adapt -- that is entirely inside each `TTAMethod.step()`. The
only thing every method is asked to also compute is `anchor_logits` from
`model.forward_anchor(x)` (see `base_cnn.py`'s module docstring for why that -- and
NOT `model(x, use_anchor=True).anchor_logits`, which shares the adapting backbone's
features and is therefore always ~0 divergence -- is the real drift signal), so
Jensen-Shannon divergence against the frozen source anchor is reported uniformly for
every method, including Source-Only, whose exactly-0 D_JS curve is the reference every
adapting method is compared against.

Shape legend: C = leads, L = 256 samples, K = NUM_CLASSES = 5.
"""
from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, NUM_CLASSES, MITBIHDataset, chronological_stream_indices
from eval_cross_domain import classification_report, confusion_matrix


# --------------------------------------------------------------------------- #
# Shared numeric utilities
# --------------------------------------------------------------------------- #
def predictive_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """logits: [..., K] -> entropy H(softmax(logits)) in nats, shape [...]."""
    p = F.softmax(logits, dim=-1).clamp_min(eps)
    return -(p * p.log()).sum(-1)


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two logit distributions, in nats.
    D_JS(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m), m = 0.5*(p+q). Symmetric, bounded by ln(2).
    logits_p, logits_q: [..., K] -> [...]
    """
    p = F.softmax(logits_p, dim=-1).clamp_min(eps)
    q = F.softmax(logits_q, dim=-1).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(-1)
    kl_qm = (q * (q / m).log()).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def step_aux(meta: Dict, device: str) -> Optional[torch.Tensor]:
    """
    Pulls this beat's auxiliary (RR late-fusion) features out of `meta` for a
    `TTAMethod.step()` call, or returns None when the stream has none.

    `make_beat_stream()` puts the beat's RR vector under `meta["aux"]` (shape
    [n_rr_features]) whenever the underlying dataset was built with
    `use_rr_features=True` (dataset.py); it is absent (key missing) for a
    no-RR dataset/checkpoint, in which case every method must call
    `model(x)`/`model.forward_anchor(x)` with no `aux` at all, exactly as
    before RR existed. This helper is the single place that distinction is
    handled, so every Stage 4/5 `step()` does:
        aux = step_aux(meta, x.device)
        out = model(x, aux=aux)
        anchor = model.forward_anchor(x, aux=aux)
    and `base_cnn.PhysioGuardCNN.forward()`'s own validation (requires `aux`
    iff `cfg.aux_feat_dim>0`) is what actually enforces the contract -- this
    helper only shapes/places whatever is there.
    """
    aux = meta.get("aux")
    if aux is None:
        return None
    if not torch.is_tensor(aux):
        aux = torch.as_tensor(aux)
    if aux.dim() == 1:
        aux = aux.unsqueeze(0)                 # [n_rr_features] -> [1, n_rr_features]
    return aux.to(device)


# --------------------------------------------------------------------------- #
# Method protocol
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    """What a `TTAMethod.step()` call returns for a single beat."""
    logits: torch.Tensor                          # [1, K] pre-update prediction (required)
    anchor_logits: Optional[torch.Tensor] = None   # [1, K] frozen-anchor logits, SAME forward
    adapted: bool = False                          # a gradient update was applied this beat
    rejected: bool = False                         # an update was considered but withheld/gated
    reset_triggered: bool = False                  # an elastic/hard reset fired this beat
    sqi: Optional[float] = None                    # method-defined scalar quality/telemetry value
    extra: Dict[str, float] = field(default_factory=dict)  # method-specific extras (e.g. adapt_loss)


class TTAMethod(ABC):
    """Base class every Stage-4/5 method implements. `name` is used in logs/tables."""
    name: str = "unnamed"

    @abstractmethod
    def reset(self, model: torch.nn.Module) -> None:
        """Called once before a stream starts: put `model` into this method's expected
        state (frozen / configure_for_tta / fresh optimizer, ...). Must NOT assume the
        model is in any particular state from a previous method's run on the same object."""

    @abstractmethod
    def step(self, model: torch.nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        """x: [1, C, L]. meta: per-beat info (record_id, sample_idx, corrupt_mask [L], and
        "aux" [1, n_rr_features] when the checkpoint uses RR late-fusion -- pull it via
        `step_aux(meta, x.device)` and pass `aux=...` into every `model(x, ...)` /
        `model.forward_anchor(x, ...)` call; see that helper's docstring).
        MUST NOT read a ground-truth label -- none is passed in."""


# --------------------------------------------------------------------------- #
# Stream construction (shared convenience; fully duck-typed, no hard import of
# space_shift_simulator so this module has zero dependency on Stage 2 internals)
# --------------------------------------------------------------------------- #
def make_beat_stream(ds: MITBIHDataset, n_beats: Optional[int] = 10_000,
                     records: Optional[Sequence[int]] = None,
                     sim: Optional[object] = None,
                     severity: "float | Callable[[int], float]" = 1.0,
                     ) -> Iterator[Tuple[torch.Tensor, int, Dict]]:
    """
    Yields (x [C, L], y int, meta dict) in true chronological (record, then time) order.

    sim: optional object with the `space_shift_simulator.SpaceShiftSimulator` call
    signature: sim(x[C,L], record_id=, beat_key=, t0_sec=, severity=, return_mask=True)
    -> (x_shifted[C,L], mask[L] bool). Duck-typed on purpose: any object with that
    signature works, so this module never imports Stage 2.
    severity: constant, or a callable(stream_index) -> float (e.g. `linear_ramp`).

    When `ds` carries RR late-fusion features (`ds.rr`, set by MITBIHDataset when
    built with `use_rr_features=True` -- see dataset.py), each beat's RR vector is
    attached as `meta["aux"]` ([n_rr_features]), bypassing `ds.__getitem__()` just
    like `ds.x`/`ds.y` already do. RR is beat-timing information (pre/post/local-avg
    R-R interval), untouched by `sim`'s waveform-only corruption, so it is read once
    here rather than threaded through the simulator call below.
    """
    idx = chronological_stream_indices(ds, n_beats, records)
    fs = getattr(getattr(sim, "cfg", None), "fs", None)
    r_index = getattr(getattr(sim, "cfg", None), "r_index", None)
    has_aux = getattr(ds, "rr", None) is not None and ds.rr.shape[-1] > 0
    for i, ix in enumerate(idx):
        x = ds.x[ix]                                          # [C, L]
        y = int(ds.y[ix])
        rid, sidx = int(ds.record_ids[ix]), int(ds.sample_idx[ix])
        meta: Dict = {"record_id": rid, "sample_idx": sidx, "stream_pos": i}
        if has_aux:
            meta["aux"] = ds.rr[ix]                            # [n_rr_features]
        if sim is not None:
            sev = float(severity(i)) if callable(severity) else float(severity)
            t0 = (sidx - r_index) / fs if (fs and r_index is not None) else None
            x, mask = sim(x, record_id=rid, beat_key=ix, t0_sec=t0, severity=sev, return_mask=True)
            meta["corrupt_mask"] = mask                        # [L] bool
            meta["severity"] = sev
        yield x, y, meta


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
@dataclass
class StreamingConfig:
    device: str = "cpu"
    rolling_window: int = 500          # beats in the rolling accuracy/F1 window
    log_every: int = 0                 # 0 = silent; else print every N beats
    latency_budget_ms: float = 5.0     # edge target (reported as frac-over-budget, NOT enforced)


@dataclass
class StreamingResult:
    method_name: str
    log: List[Dict]                    # one dict per beat, see StreamingEvaluator._record()
    summary: Dict

    def save_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.log:
            return
        cols = list(self.log[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.log)

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"method_name": self.method_name, "summary": self.summary}, f, indent=2)

    def head(self, n: int = 5) -> List[Dict]:
        return self.log[:n]


class StreamingEvaluator:
    """Runs one `TTAMethod` over one chronological beat stream, N=1 throughout."""

    def __init__(self, method: TTAMethod, cfg: Optional[StreamingConfig] = None):
        self.method = method
        self.cfg = cfg or StreamingConfig()

    def run(self, model: torch.nn.Module, stream: Iterable[Tuple[torch.Tensor, int, Dict]]
           ) -> StreamingResult:
        cfg = self.cfg
        self.method.reset(model)

        cum_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        roll_buf: deque = deque()                              # (y, pred) pairs, manual eviction
        roll_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        log: List[Dict] = []
        latencies: List[float] = []
        t_run0 = time.perf_counter()

        for i, (x, y, meta) in enumerate(stream):
            xb = x.unsqueeze(0).to(cfg.device) if x.dim() == 2 else x.to(cfg.device)  # [1,C,L]

            t0 = time.perf_counter()
            step_res = self.method.step(model, xb, meta)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            logits = step_res.logits.detach()                                        # [1, K]
            pred = int(logits.argmax(-1).item())
            entropy = float(predictive_entropy(logits).item())

            js_div: Optional[float] = None
            if step_res.anchor_logits is not None:
                js_div = float(js_divergence(logits, step_res.anchor_logits.detach()).item())
            elif hasattr(model, "forward_anchor"):
                # Fallback only: an extra no_grad forward AFTER the method's own step. Since
                # `forward_anchor()` never touches the live parameters (functional_call against
                # a substituted param dict), this fallback is exact regardless of call order --
                # not an approximation -- it is just an extra forward a method could have
                # avoided by computing anchor_logits itself. All Stage 4/5 methods here do.
                with torch.no_grad():
                    js_div = float(js_divergence(logits, model.forward_anchor(
                        xb, aux=step_aux(meta, xb.device))).item())

            loss = float(F.cross_entropy(logits, torch.tensor([y], device=logits.device)).item())
            correct = int(pred == y)

            cum_cm[y, pred] += 1
            roll_buf.append((y, pred)); roll_cm[y, pred] += 1
            if len(roll_buf) > cfg.rolling_window:
                ey, ep = roll_buf.popleft()
                roll_cm[ey, ep] -= 1
            cum_rep = classification_report(cum_cm)
            roll_rep = classification_report(roll_cm)

            corrupt_frac = None
            if "corrupt_mask" in meta and meta["corrupt_mask"] is not None:
                cm = meta["corrupt_mask"]
                corrupt_frac = float(cm.float().mean().item()) if torch.is_tensor(cm) else float(np.mean(cm))

            record = {
                "step": i, "record_id": meta.get("record_id"), "sample_idx": meta.get("sample_idx"),
                "severity": meta.get("severity"), "corrupt_frac": corrupt_frac,
                "y_true": y, "y_pred": pred, "correct": correct, "loss": loss,
                "entropy": entropy, "js_div": js_div, "sqi": step_res.sqi,
                "adapted": step_res.adapted, "rejected": step_res.rejected,
                "reset_triggered": step_res.reset_triggered,
                "latency_ms": latency_ms,
                "cum_accuracy": cum_rep["accuracy"], "cum_macro_f1": cum_rep["macro_f1"],
                "roll_accuracy": roll_rep["accuracy"], "roll_macro_f1": roll_rep["macro_f1"],
            }
            for k, v in step_res.extra.items():
                record[f"extra_{k}"] = v
            log.append(record)

            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                print(f"[{self.method.name}] beat {i+1:6d}  acc(cum)={cum_rep['accuracy']:.3f}  "
                     f"F1(cum)={cum_rep['macro_f1']:.3f}  acc(roll{cfg.rolling_window})={roll_rep['accuracy']:.3f}  "
                     f"H={entropy:.3f}  D_JS={js_div if js_div is None else round(js_div,4)}  "
                     f"adapt={step_res.adapted}  reject={step_res.rejected}  reset={step_res.reset_triggered}")

        total_wall_s = time.perf_counter() - t_run0
        summary = self._summarize(log, cum_cm, latencies, total_wall_s, cfg)
        return StreamingResult(method_name=self.method.name, log=log, summary=summary)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarize(log: List[Dict], cum_cm: np.ndarray, latencies: List[float],
                   total_wall_s: float, cfg: StreamingConfig) -> Dict:
        n = len(log)
        final = classification_report(cum_cm)
        lat = np.asarray(latencies, dtype=np.float64) if latencies else np.zeros(1)
        js_vals = np.array([r["js_div"] for r in log if r["js_div"] is not None], dtype=np.float64)
        ent_vals = np.array([r["entropy"] for r in log], dtype=np.float64)
        n_adapt = sum(1 for r in log if r["adapted"])
        n_reject = sum(1 for r in log if r["rejected"])
        n_reset = sum(1 for r in log if r["reset_triggered"])
        return {
            "n_beats": n,
            "accuracy": final["accuracy"], "macro_f1": final["macro_f1"],
            # exclude F/Q (tiny-support classes, e.g. 3/5 beats out of 20,000 here -- noise-level
            # at that n, and equal-weighted against them in macro_f1 dilutes real N/S/V differences)
            # -- the metric actually comparable to Romdhane et al.'s reported number, see
            # eval_cross_domain.classification_report's module comment for the verified match.
            "macro_f1_ex_f": final.get("macro_f1_ex_f"), "macro_f1_ex_fq": final.get("macro_f1_ex_fq"),
            "weighted_f1": final["weighted_f1"], "per_class": final["per_class"],
            "confusion_matrix": final["confusion_matrix"],
            "mean_entropy": float(ent_vals.mean()) if n else float("nan"),
            "mean_js_div": float(js_vals.mean()) if js_vals.size else None,
            "max_js_div": float(js_vals.max()) if js_vals.size else None,
            "adaptation_rate": n_adapt / max(n, 1),
            "rejection_rate": n_reject / max(n, 1),
            "reset_count": n_reset, "reset_rate": n_reset / max(n, 1),
            "latency_ms_mean": float(lat.mean()), "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)), "latency_ms_p99": float(np.percentile(lat, 99)),
            "latency_ms_max": float(lat.max()),
            "frac_beats_over_latency_budget": float((lat > cfg.latency_budget_ms).mean()),
            "total_wall_s": total_wall_s, "throughput_beats_per_s": n / max(total_wall_s, 1e-9),
        }


# --------------------------------------------------------------------------- #
# Cross-method comparison table (used by Stage 4's sweep and later Stage 5/7)
# --------------------------------------------------------------------------- #
def compare_methods(results: Dict[str, StreamingResult]) -> str:
    cols = [("method", 16), ("n", 7), ("acc", 7), ("macroF1", 8), ("F1exFQ", 8), ("meanH", 7),
           ("meanD_JS", 11), ("adapt%", 7), ("reject%", 8), ("resets", 7), ("lat_p95ms", 10)]
    header = "".join(f"{name:<{w}}" for name, w in cols)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        s = r.summary
        djs = s["mean_js_div"]
        exfq = s.get("macro_f1_ex_fq")
        # scientific notation: D_JS legitimately spans ~1e-9 (Source-Only, fp noise
        # only) to ~1e-1 (a badly drifted method) -- fixed decimals would hide that range.
        row = [name, s["n_beats"], f"{s['accuracy']:.3f}", f"{s['macro_f1']:.3f}",
              "n/a" if exfq is None else f"{exfq:.3f}",
              f"{s['mean_entropy']:.3f}", "n/a" if djs is None else f"{djs:.3e}",
              f"{100*s['adaptation_rate']:.1f}", f"{100*s['rejection_rate']:.1f}",
              s["reset_count"], f"{s['latency_ms_p95']:.3f}"]
        lines.append("".join(f"{str(v):<{w}}" for v, (_, w) in zip(row, cols)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test: a trivial Source-Only-like and a real entropy-min "dummy TENT" method
# on a synthetic stream, to exercise the loop/metrics/logging without any other
# project file (baselines_tta.py exercises the real Stage-4 methods).
# --------------------------------------------------------------------------- #
class _DummySourceOnly(TTAMethod):
    name = "dummy_source_only"

    def reset(self, model):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    def step(self, model, x, meta):
        aux = step_aux(meta, x.device)
        with torch.no_grad():
            out = model(x, aux=aux)
            anchor_logits = model.forward_anchor(x, aux=aux)
        return StepResult(logits=out.logits, anchor_logits=anchor_logits, adapted=False)


class _DummyEntropyTTA(TTAMethod):
    name = "dummy_entropy_tta"

    def __init__(self, lr: float = 1e-3):
        self.lr = lr
        self.opt = None

    def reset(self, model):
        params = model.configure_for_tta()                    # dw_bn affine only, eval mode
        self.opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

    def step(self, model, x, meta):
        aux = step_aux(meta, x.device)
        out = model(x, aux=aux)                                 # adaptable head, WITH grad
        loss = predictive_entropy(out.logits).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            anchor_logits = model.forward_anchor(x, aux=aux)      # source-dw_bn backbone state
        return StepResult(logits=out.logits.detach(), anchor_logits=anchor_logits,
                          adapted=True, extra={"adapt_loss": float(loss.detach())})


def _synthetic_stream(n: int = 300, c: int = 1, l: int = 256, seed: int = 0
                      ) -> Iterator[Tuple[torch.Tensor, int, Dict]]:
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1, 1, l)
    templates = torch.stack([torch.sin((k + 1) * 3.0 * t) for k in range(NUM_CLASSES)])  # [K, L]
    for i in range(n):
        k = int(torch.randint(0, NUM_CLASSES, (1,), generator=g))
        drift = 1.0 + 0.5 * (i / n)                             # slow amplitude drift -> rising H
        x = drift * templates[k].view(1, l).repeat(c, 1) + 0.25 * torch.randn(c, l, generator=g)
        yield x, k, {"record_id": 0, "sample_idx": i * 300, "stream_pos": i}


if __name__ == "__main__":
    from base_cnn import CNNConfig, PhysioGuardCNN

    print("[streaming_eval] self-test: dummy Source-Only vs. dummy entropy-TTA on a "
         "synthetic drifting stream.")
    torch.manual_seed(0)
    model = PhysioGuardCNN(CNNConfig(in_channels=1))

    # Brief supervised warm-up BEFORE freeze_anchor() (mirrors train_stage1.py's real
    # order: theta_f is a snapshot of a TRAINED model, not a random-init one). Skipping
    # this would leave the model's outputs near-uniform, whose entropy gradient is
    # near-zero almost everywhere -- the drift signal would be technically correct but
    # numerically negligible, which is not a convincing self-test of the mechanism.
    # Mini-batched (not batch-size-1 online SGD, which does not converge here).
    warm_data = list(_synthetic_stream(n=300, seed=0))
    Xw = torch.stack([d[0] for d in warm_data]); Yw = torch.tensor([d[1] for d in warm_data])
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.adaptable_head.parameters()),
                           lr=1e-3)
    model.train()
    for _ in range(15):
        perm = torch.randperm(len(Yw))
        for i in range(0, len(Yw), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(model(Xw[idx]).logits, Yw[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    model.freeze_anchor()                       # NOW lock the warmed-up state as theta_f

    cfg = StreamingConfig(device="cpu", rolling_window=50, log_every=0)
    results: Dict[str, StreamingResult] = {}
    for method in (_DummySourceOnly(), _DummyEntropyTTA(lr=5e-3)):
        m = __import__("copy").deepcopy(model)                 # same source weights per method
        ev = StreamingEvaluator(method, cfg)
        res = ev.run(m, _synthetic_stream(n=300, seed=0))
        results[method.name] = res
        assert res.summary["n_beats"] == 300
        assert 0.0 <= res.summary["accuracy"] <= 1.0
        assert len(res.log) == 300 and set(res.log[0].keys()) >= {
            "y_true", "y_pred", "entropy", "js_div", "adapted", "cum_macro_f1", "roll_macro_f1"}

    print(compare_methods(results))
    assert results["dummy_entropy_tta"].summary["adaptation_rate"] == 1.0
    assert results["dummy_source_only"].summary["adaptation_rate"] == 0.0
    assert results["dummy_source_only"].summary["mean_js_div"] < 1e-6, \
        "Source-Only must show ~0 divergence from its own anchor"
    assert results["dummy_entropy_tta"].summary["mean_js_div"] > 1e-4, \
        "an adapting method on a warmed-up (confident) model must show non-negligible drift"
    out_dir = "./results/_selftest_streaming_eval"
    for name, res in results.items():
        res.save_csv(f"{out_dir}/{name}.csv")
        res.save_json(f"{out_dir}/{name}.json")
    print(f"[streaming_eval] wrote per-beat logs to {out_dir}/")
    print("Self-test passed.")