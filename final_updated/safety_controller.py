"""
safety_controller.py  --  PhysioGuard-Space, Stage 5 (Elastic Trust Region Safety Controller)

Implements `PhysioGuardTTA`, this paper's method, combining:
  1. TS-SQI masked entropy adaptation (`ts_sqi_masker.py`) -- soft, sub-window
     masking of the ADAPTATION SIGNAL'S input, not a whole-beat gate like
     `baselines_tta.SQIGatedTent`.
  2. Elastic Anchor Divergence trust region -- a continuous restoring force
         alpha = gamma * sigmoid(kappa * (D_JS - tau))
     pulling the adapted dw_bn affine parameters back toward the frozen source
     snapshot, applied EVERY beat (not a discrete "if exceeds threshold, hard-reset"
     event), sized by how far D_JS currently sits from source. D_JS itself comes
     from `model.forward_anchor()` (`base_cnn.py`) -- the real backbone-state drift
     signal, not the same-features head comparison that is always ~0.
  3. Sub-512KB sparse backprop -- reuses the SAME `model.configure_for_tta()`
     (dw_bn affine only, <5% of weights) as every Stage-4 baseline, for a
     budget-fair comparison.

Two design choices worth being explicit about
------------------------------------------------
* PREDICTION vs. ADAPTATION SIGNAL are deliberately different inputs. The reported
  prediction (`StepResult.logits`, what gets scored for accuracy/F1) comes from a
  forward pass on the RAW beat x, computed BEFORE this beat's own update -- exactly
  like every other Stage-4 method, consistent with the streaming protocol. The
  entropy loss that drives the gradient step is instead computed on `x_masked`
  (from `TSSQIMasker`), a SEPARATE forward pass. A misfiring SQI mask can, at
  worst, corrupt one beat's adaptation signal -- it can never alter what is
  reported as the device's diagnostic output. Deliberate and conservative for a
  medical/astronaut-monitoring context, not an oversight.

* ORDER OF OPERATIONS each beat: (a) masked-entropy gradient step, if the beat has
  enough trustworthy time-frequency content (b) measure D_JS on the JUST-UPDATED
  state (c) apply the elastic pull sized by that measurement. This makes the
  elastic force a predictor-corrector: entropy-min proposes a small update, then
  the trust region immediately reins in whatever net drift resulted. It guards
  against COMPOUNDING drift over many beats -- what makes TENT collapse in Stage
  4 -- not any single beat's update in isolation. The pull runs even on beats
  where no entropy update happened (near-zero mask coverage): past drift can still
  need correcting when this beat contributes nothing new.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from streaming_eval import StepResult, TTAMethod, js_divergence, predictive_entropy, step_aux
from ts_sqi_masker import TSSQIConfig, TSSQIMasker


def _stable_sigmoid(z: float) -> float:
    """Overflow-safe sigmoid for python floats (kappa*(D_JS - tau) can be large)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class SafetyControllerConfig:
    # masked-entropy adaptation
    lr: float = 1e-3
    momentum: float = 0.9
    min_mask_coverage: float = 0.05   # degenerate-case floor (near-zero trustworthy content this
                                      # beat) -- NOT a SQI-gate-style primary gate; see module docstring

    # elastic trust region: alpha = gamma * sigmoid(kappa * (D_JS - tau))
    tau: float = 0.05                 # nats; D_JS around which the restoring force ramps up
    kappa: float = 40.0                # sigmoid steepness (1/nats)
    gamma: float = 0.3                 # max per-beat restoring strength, alpha upper bound in [0,1]
    reset_log_threshold: float = 0.5   # fraction of gamma -> logged as reset_triggered=True (diagnostic only)

    # V-recall experiment (default OFF = original behaviour, results unchanged).
    # "none": adapt on every beat. "skip_abnormal": do not adapt on a beat whose FROZEN-ANCHOR
    # prediction is not class 0 (N). Motivation: DS2 diagnostics showed V beats drifting to S
    # (V->S 281 -> 522 vs source); entropy minimization on beats the source already calls
    # abnormal sharpens those (often wrong) calls. Uses no label: anchor prediction only.
    adapt_gate: str = "none"

    # Minority-class protection (defaults OFF = original behaviour, results unchanged).
    # Observed on SVDB: entropy minimisation on a stream that is ~90% N moves the shared
    # dw_bn affine toward "N" (S recall 0.64 -> 0.21), and skipping abnormal beats cannot fix it
    # (it removes exactly the beats that would push back). Two label-free remedies:
    #  class_balance: weight each beat's entropy by 1/(K * f_c), f_c = EMA frequency of the
    #     FROZEN-ANCHOR's predicted class c (clipped to [cb_w_min, cb_w_max]), so every
    #     class contributes comparable gradient and the majority class cannot dominate.
    #  anchor_kl: + lambda * KL(p_anchor || p_adapted) on the same beat (per-beat distillation to
    #     the frozen source; uses the anchor pass the elastic pull already needs).
    class_balance: bool = False
    cb_momentum: float = 0.99
    cb_w_min: float = 0.2
    cb_w_max: float = 5.0
    anchor_kl: float = 0.0

    # Ablation switches (defaults = the published method, results unchanged).
    use_mask: bool = True             # False: entropy on the RAW window (no TS-SQI mask)
    fixed_alpha: Optional[float] = None   # not None: constant pull strength per beat (no D_JS
                                          # dependence) -- isolates the *elastic* part of the TR

    # Parameter-fault protection (defaults OFF = original behaviour, results unchanged).
    # The adapted dw_bn affine lives in writable memory (SRAM), so a single-event upset there is a
    # new failure mode that a frozen model stored in flash/ECC does not have. fault_guard runs
    # BEFORE the prediction each beat: any non-finite parameter is replaced by its source value and
    # every parameter is clamped to within guard_k * max(std(source tensor), guard_floor) of its
    # source value (a box around the source = a hard trust region; cost: 384 compares per beat).
    # The soft elastic pull below then removes remaining in-box deviations in proportion to D_JS.
    # adapt_enabled=False gives a "scrub-only" controller: guard + elastic pull, no gradient step.
    fault_guard: bool = False
    guard_k: float = 4.0
    guard_floor: float = 0.05
    adapt_enabled: bool = True

    sqi_cfg: TSSQIConfig = field(default_factory=TSSQIConfig)


class PhysioGuardTTA(TTAMethod):
    """PhysioGuard-Space: TS-SQI masked entropy adaptation + Elastic Anchor Divergence trust region."""
    name = "physioguard_space"

    def __init__(self, cfg: Optional[SafetyControllerConfig] = None):
        self.cfg = cfg or SafetyControllerConfig()
        self.masker = TSSQIMasker(self.cfg.sqi_cfg)
        self.opt: Optional[torch.optim.Optimizer] = None
        # In-band-only coverage bins, computed once from config (cheap, static given
        # `sqi_cfg`). `coverage` (== mask.mean() over the FULL [F,T] grid, `apply()`'s
        # own diagnostic) is dominated by STFT bins that sit outside the ECG passband
        # by construction regardless of content -- at n_fft=32/fs=360, frequency
        # resolution is fs/n_fft=11.25 Hz, so only ~3 of 17 rfft bins fall inside
        # 0.5-40 Hz -- so full-grid `coverage` barely moves between a clean and a
        # corrupted beat (confirmed empirically: ~0.16 either way on a synthetic
        # stream) and is unsuitable as a per-beat "how much was trusted" signal for
        # external evaluation (Stage 7's HURR/BUAR). It remains exactly right for its
        # ORIGINAL purpose here -- `min_mask_coverage`'s coarse "is there anything
        # left to learn from at all" floor -- so that gate is untouched; this adds an
        # ADDITIONAL diagnostic restricted to in-band bins (same convention as
        # `ts_sqi_masker.py`'s own self-test), which is far more discriminative
        # (~0.80 clean vs. ~0.76 corrupted in that same empirical check) and costs
        # nothing extra to compute since `apply()` already returns the full mask.
        sqi = self.cfg.sqi_cfg
        freqs = torch.fft.rfftfreq(sqi.n_fft, d=1.0 / sqi.fs)
        self._inband_bins = (freqs >= sqi.low_hz + 2 * sqi.low_softness_hz) & \
                            (freqs <= sqi.high_hz - sqi.high_softness_hz)

    def reset(self, model: nn.Module) -> None:
        params = model.configure_for_tta()                 # dw_bn affine only, same budget as Stage 4
        self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self._freq: Optional[torch.Tensor] = None          # EMA of anchor-predicted class frequency
        self._bounds: Dict[str, torch.Tensor] = {}
        if self.cfg.fault_guard:
            for n, _ in model.tta_named_parameters():
                src = model._source_tta[n]
                self._bounds[n] = self.cfg.guard_k * max(float(src.std()) if src.numel() > 1 else 0.0,
                                                          self.cfg.guard_floor)

    def _cb_weight(self, anchor_logits: torch.Tensor) -> float:
        """Inverse-frequency weight of this beat's anchor-predicted class (see config)."""
        cfg = self.cfg
        k = anchor_logits.shape[-1]
        if getattr(self, "_freq", None) is None:
            self._freq = torch.full((k,), 1.0 / k)
        c = int(anchor_logits.argmax(-1).item())
        onehot = torch.zeros(k)
        onehot[c] = 1.0
        self._freq = cfg.cb_momentum * self._freq + (1.0 - cfg.cb_momentum) * onehot
        w = 1.0 / (k * max(float(self._freq[c]), 1e-6))
        return float(min(max(w, cfg.cb_w_min), cfg.cb_w_max))

    @torch.no_grad()
    def guard(self, model: nn.Module) -> int:
        """Replace non-finite adapted params by their source value; clamp the rest into the box.
        Returns the number of parameter elements that were changed."""
        changed = 0
        for n, p in model.tta_named_parameters():
            src = model._source_tta[n].to(p.device)
            bad = ~torch.isfinite(p)
            if bool(bad.any()):
                p.copy_(torch.where(bad, src, p))
            b = self._bounds[n]
            lo, hi = src - b, src + b
            clamped = torch.minimum(torch.maximum(p, lo), hi)
            changed += int((clamped != p).sum()) + int(bad.sum())
            p.copy_(clamped)
        return changed

    def elastic_alpha(self, djs: float) -> float:
        """alpha = gamma * sigmoid(kappa * (D_JS - tau)), in [0, gamma]."""
        if self.cfg.fixed_alpha is not None:
            return float(self.cfg.fixed_alpha)
        if not math.isfinite(djs):                          # NaN/inf divergence = maximal divergence
            return float(self.cfg.gamma)
        return self.cfg.gamma * _stable_sigmoid(self.cfg.kappa * (djs - self.cfg.tau))

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        cfg = self.cfg
        aux = step_aux(meta, x.device)                       # RR late-fusion features, if any (unchanged
                                                              # by masking: beat TIMING, not waveform content)

        n_guarded = self.guard(model) if cfg.fault_guard else 0
        with torch.no_grad():
            pred_logits = model(x, aux=aux).logits           # pre-update, RAW-input: the reported prediction

        if self.cfg.use_mask:
            x_masked, mask, coverage = self.masker.apply(x)  # soft, sub-window-masked reconstruction
        else:                                                # ablation: no quality mask at all
            x_masked, mask, coverage = x, torch.ones(1, len(self._inband_bins)), torch.tensor(1.0)
        coverage_val = float(coverage.item())
        inband_coverage_val = float(mask[:, self._inband_bins].mean().item())

        adapted = False
        adapt_loss = float("nan")
        gate_ok = True
        anchor_logits = None                                  # anchor_logits only depend on x (frozen snapshot)
        if cfg.adapt_gate == "skip_abnormal" or cfg.class_balance or cfg.anchor_kl > 0.0:
            with torch.no_grad():
                anchor_logits = model.forward_anchor(x, aux=aux)
            if cfg.adapt_gate == "skip_abnormal":
                gate_ok = int(anchor_logits.argmax(-1).item()) == 0
        cb_w = self._cb_weight(anchor_logits) if cfg.class_balance else 1.0
        if cfg.adapt_enabled and gate_ok and coverage_val >= cfg.min_mask_coverage:
            out_masked = model(x_masked, aux=aux)             # WITH grad, on the masked input only
            loss = cb_w * predictive_entropy(out_masked.logits).mean()
            if cfg.anchor_kl > 0.0:
                p_a = F.softmax(anchor_logits, dim=-1)
                kl = (p_a * (torch.log(p_a + 1e-8) - F.log_softmax(out_masked.logits, dim=-1))).sum(-1).mean()
                loss = loss + cfg.anchor_kl * kl
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            adapted = True
            adapt_loss = float(loss.detach())

        with torch.no_grad():
            # POST-update (if any), PRE-elastic-pull. With no update the parameters are unchanged, so
            # this is exactly pred_logits (eval mode, deterministic) -- reuse it instead of a 2nd pass.
            post_logits = model(x, aux=aux).logits if adapted else pred_logits
            if anchor_logits is None:
                anchor_logits = model.forward_anchor(x, aux=aux)  # backbone-state anchor (source dw_bn snapshot)
            djs = float(js_divergence(post_logits, anchor_logits).item())
            alpha = self.elastic_alpha(djs)
            model.elastic_reset(alpha, reset_head=False, reset_backbone_affine=True)

        reset_triggered = alpha >= cfg.reset_log_threshold * cfg.gamma   # diagnostic count only

        return StepResult(
            logits=pred_logits, anchor_logits=anchor_logits,
            adapted=adapted, rejected=not adapted, reset_triggered=reset_triggered,
            sqi=coverage_val,
            extra={"adapt_loss": adapt_loss, "js_div_post_update": djs,
                  "elastic_alpha": alpha, "mask_coverage": coverage_val,
                  "inband_mask_coverage": inband_coverage_val, "guard_changed": float(n_guarded)},
        )


# --------------------------------------------------------------------------- #
# Self-tests
#   1. elastic_alpha(): monotonic, bounded, centered correctly at tau.
#   2. Parameter restoration dynamics: artificially drifted dw_bn affine must be
#      pulled back toward source under repeated elastic pulls, converging (not
#      diverging/oscillating), with the pull strength itself shrinking as it
#      approaches source (self-decelerating, not overshooting).
#   3. Full comparison vs. Stage 4's three baselines on the SAME synthetic
#      space-shifted stream, warmed-up model, and severity schedule.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import torch.nn.functional as F

    from base_cnn import CNNConfig, PhysioGuardCNN
    from baselines_tta import SQIGatedTent, SourceOnlyTTA, TentConfig, TentTTA, _SyntheticStreamDataset

    # ---- 1. elastic_alpha() shape ------------------------------------------
    print("[safety_controller] self-test 1/3: elastic_alpha() shape.")
    ctrl = PhysioGuardTTA(SafetyControllerConfig(tau=0.05, kappa=40.0, gamma=0.3))
    a_zero = ctrl.elastic_alpha(0.0)
    a_tau = ctrl.elastic_alpha(0.05)
    a_big = ctrl.elastic_alpha(1.0)
    a_neg_extreme = ctrl.elastic_alpha(-50.0)   # must not overflow/NaN
    a_pos_extreme = ctrl.elastic_alpha(50.0)
    print(f"  alpha(0)={a_zero:.4f}  alpha(tau=0.05)={a_tau:.4f}  alpha(1.0)={a_big:.4f}  "
         f"alpha(-50)={a_neg_extreme:.6f}  alpha(50)={a_pos_extreme:.6f}")
    assert math.isfinite(a_neg_extreme) and math.isfinite(a_pos_extreme), "sigmoid must not overflow"
    assert abs(a_tau - ctrl.cfg.gamma / 2) < 1e-9, "alpha(tau) must equal gamma/2 (sigmoid(0)=0.5)"
    assert 0.0 <= a_zero < a_tau < a_big <= ctrl.cfg.gamma + 1e-9, "alpha must be monotonic and bounded in [0, gamma]"
    xs = [-0.2, -0.05, 0.0, 0.02, 0.05, 0.08, 0.2, 1.0]
    alphas = [ctrl.elastic_alpha(v) for v in xs]
    assert all(alphas[i] <= alphas[i + 1] + 1e-12 for i in range(len(alphas) - 1)), "alpha must be monotonic in D_JS"
    print("  monotonic, bounded in [0, gamma], correctly centered at tau -> OK")

    # ---- 2. parameter restoration dynamics ---------------------------------
    print("\n[safety_controller] self-test 2/3: parameter restoration dynamics under repeated elastic pulls.")
    # Uses a BRIEFLY SUPERVISED-WARMED-UP model (mini-batched, mirroring train_stage1.py's
    # real order: theta_f is a snapshot of a TRAINED model), not a random-init one -- a
    # random-init backbone's final logits turn out to be nearly insensitive to dw_bn affine
    # perturbations (confirmed empirically: djs stayed ~1e-4 even at perturbation=3.0),
    # which would make this test start already-degenerate rather than showing real dynamics.
    torch.manual_seed(0)
    warm_model = PhysioGuardCNN(CNNConfig(in_channels=1))
    warm_ds = _SyntheticStreamDataset(n_records=3, beats_per_record=200, c=1, l=256, seed=0)
    warm_opt = torch.optim.Adam(list(warm_model.backbone.parameters())
                                + list(warm_model.adaptable_head.parameters()), lr=1e-3)
    warm_model.train()
    for _ in range(15):
        perm = torch.randperm(len(warm_ds.y))
        for i in range(0, len(warm_ds.y), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(warm_model(warm_ds.x[idx]).logits, warm_ds.y[idx])
            warm_opt.zero_grad(set_to_none=True); loss.backward(); warm_opt.step()
    warm_model.eval()
    warm_model.freeze_anchor()

    model = copy.deepcopy(warm_model)
    x_probe = warm_ds.x[:40]

    with torch.no_grad():
        source_vals = {n: p.clone() for n, p in model.tta_named_parameters()}
        for p in model.tta_parameters():                    # simulate catastrophic drift
            p.add_(1.0)
    with torch.no_grad():
        param_dist_0 = float(sum((p.detach() - source_vals[n]).pow(2).sum()
                                 for n, p in model.tta_named_parameters()) ** 0.5)

    djs_trace, alpha_trace = [], []
    for i in range(60):
        with torch.no_grad():
            cur = model(x_probe).logits
            anchor = model.forward_anchor(x_probe)
            djs = float(js_divergence(cur, anchor).mean().item())   # mean over the probe batch:
            # a SINGLE beat's D_JS is idiosyncratic (can plateau at a small, input-specific
            # value well before the model has actually been pulled back -- confirmed
            # empirically), so this diagnostic averages over `x_probe`'s batch of beats for
            # a smooth, representative signal. The real per-beat controller (`step()` above)
            # still reacts to one beat's own D_JS, as it must in a true streaming setting.
            alpha = ctrl.elastic_alpha(djs)
            model.elastic_reset(alpha, reset_head=False, reset_backbone_affine=True)
        djs_trace.append(djs)
        alpha_trace.append(alpha)

    with torch.no_grad():
        param_dist_final = float(sum((p.detach() - source_vals[n]).pow(2).sum()
                                     for n, p in model.tta_named_parameters()) ** 0.5)
    print(f"  D_JS trace: {[round(v, 4) for v in djs_trace[:5]]} ... {[round(v, 5) for v in djs_trace[-5:]]}")
    print(f"  alpha trace: {[round(v, 4) for v in alpha_trace[:5]]} ... {[round(v, 5) for v in alpha_trace[-5:]]}")
    print(f"  parameter L2 distance from source: {param_dist_0:.3f} -> {param_dist_final:.5f}")
    assert all(math.isfinite(v) for v in djs_trace), "D_JS must stay finite (no divergence/blow-up)"
    assert djs_trace[-1] < djs_trace[0] * 0.05, "repeated elastic pulls must substantially reduce divergence"
    assert param_dist_final < 0.05 * param_dist_0, "dw_bn affine must be pulled back close to the source snapshot"
    # self-decelerating: the pull weakens as parameters approach source (no overshoot/oscillation)
    assert alpha_trace[0] > alpha_trace[len(alpha_trace) // 2] > alpha_trace[-1], \
        "pull strength must shrink monotonically as divergence shrinks (stable, not oscillating)"
    print("  converges, stays finite, self-decelerates (no overshoot) -> OK")

    # ---- 3. full comparison vs. Stage 4 baselines ---------------------------
    print("\n[safety_controller] self-test 3/3: PhysioGuard-Space vs. Source-Only / TENT / SQI-gate "
         "on the same synthetic space-shifted stream.")
    from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator, linear_ramp
    from streaming_eval import StreamingConfig, StreamingEvaluator, compare_methods, make_beat_stream

    source_model = warm_model                    # reuse test 2's warmed-up, frozen-anchor model
    ds = warm_ds

    sim = SpaceShiftSimulator(SpaceShiftConfig(seed=0))
    severity_schedule = linear_ramp(start_idx=100, n_ramp=300)
    stream_cfg = StreamingConfig(device="cpu", rolling_window=100, log_every=0)

    methods = [SourceOnlyTTA(), TentTTA(TentConfig(scope="sparse")), SQIGatedTent(),
              PhysioGuardTTA(SafetyControllerConfig())]
    results = {}
    for method in methods:
        m = copy.deepcopy(source_model)
        stream = make_beat_stream(ds, n_beats=None, sim=sim, severity=severity_schedule)
        ev = StreamingEvaluator(method, stream_cfg)
        res = ev.run(m, stream)
        results[method.name] = res
        res.save_csv(f"./results/_selftest_safety_controller/{method.name}.csv")
        res.save_json(f"./results/_selftest_safety_controller/{method.name}.json")

    print()
    print(compare_methods(results))

    pg = results["physioguard_space"].summary
    tent = results["tent_sparse"].summary
    src = results["source_only"].summary
    br = results["sqi_gated_tent"].summary

    print(f"\nPhysioGuard-Space: accuracy {src['accuracy']:.3f} (source-only) -> {pg['accuracy']:.3f}, "
         f"mean D_JS {pg['mean_js_div']:.2e} vs. TENT's {tent['mean_js_div']:.2e}, "
         f"adaptation_rate {100*pg['adaptation_rate']:.1f}% vs. SQI-gate's {100*br['adaptation_rate']:.1f}%.")

    assert pg["mean_js_div"] < 0.5 * tent["mean_js_div"], \
        "the elastic trust region must keep PhysioGuard's drift well below unconstrained TENT's"
    assert pg["accuracy"] >= src["accuracy"] - 0.02, \
        "PhysioGuard must not collapse accuracy below the Source-Only floor"
    assert pg["adaptation_rate"] > br["adaptation_rate"], \
        "soft sub-window masking should adapt on more beats than SQI-gate's whole-beat gate"
    print("Self-test passed.")