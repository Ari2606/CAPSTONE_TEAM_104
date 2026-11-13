"""
fault_injection.py  --  single-event-upset (SEU) model for the adapted parameters.

What is modelled (and what is not)
  * A fault is ONE bit flip in the float32 representation of ONE element of the TTA-adapted
    parameter set (the dw_bn affine: 384 values = 12,288 bits in this model). This is the state a
    test-time-adapting device must keep in writable memory; a frozen model's weights can sit in
    flash/ECC memory, so this exposure is NEW with adaptation.
  * Flips arrive as a Poisson process with a stated rate (flips per 1000 beats). Real orbital upset
    rates are far lower than anything simulated here and depend on orbit, shielding and device; the
    rate is a stress parameter, not a prediction, and no real rate is claimed.
  * The frozen source snapshot (the reference the elastic pull and the guard restore to) is assumed
    intact, i.e. kept in protected memory (flash/ECC). `hit_snapshot=True` drops that assumption.
  * Not modelled: flips in the other ~97% of weights, in BN running statistics, in activations, in
    code, or multi-bit upsets.

Wrappers (all are TTAMethods so they run in the unchanged StreamingEvaluator):
  FaultInjecting(inner, rate, bits, seed)  : before every inner.step, inject Poisson(rate) flips;
                                             afterwards log state deviation from the source snapshot.
  FrozenExposed()                          : frozen model, faults hit it, nothing repairs them.
  PeriodicScrub(period)                    : frozen model; every `period` beats the adapted set is
                                             reloaded from the snapshot (the standard blind scrubber).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from streaming_eval import StepResult, TTAMethod, step_aux

BIT_SETS = {                       # float32: bit 31 sign, 30..23 exponent, 22..0 mantissa
    "all": tuple(range(32)),
    "exponent": tuple(range(23, 31)),
    "mantissa": tuple(range(0, 23)),
    "sign": (31,),
}


@torch.no_grad()
def flip_bit_(p: torch.Tensor, flat_index: int, bit: int) -> Tuple[float, float]:
    """In-place flip of `bit` of element `flat_index` of a float32 tensor. Returns (old, new)."""
    assert p.dtype == torch.float32 and 0 <= bit < 32
    iv = p.data.view(-1).view(torch.int32)
    old = float(p.data.view(-1)[flat_index])
    mask = (1 << bit) if bit < 31 else -(2 ** 31)
    iv[flat_index] = torch.bitwise_xor(iv[flat_index], torch.tensor(mask, dtype=torch.int32))
    return old, float(p.data.view(-1)[flat_index])


def state_deviation(model: nn.Module) -> float:
    """max |adapted - source| over the TTA parameter set; inf if anything is non-finite."""
    dev = 0.0
    for n, p in model.tta_named_parameters():
        d = (p.detach() - model._source_tta[n].to(p.device)).abs()
        if not bool(torch.isfinite(d).all()):
            return float("inf")
        dev = max(dev, float(d.max()))
    return dev


class FaultInjecting(TTAMethod):
    def __init__(self, inner: TTAMethod, rate_per_1000: float, bits: str = "all", seed: int = 0,
                 hit_snapshot: bool = False, name: Optional[str] = None):
        self.inner, self.rate = inner, rate_per_1000 / 1000.0
        self.bits = BIT_SETS[bits] if bits in BIT_SETS else tuple(int(b) for b in bits.split(","))
        self.seed, self.hit_snapshot = seed, hit_snapshot
        self.name = name or inner.name
        self.events: List[dict] = []

    def reset(self, model: nn.Module) -> None:
        self.inner.reset(model)
        self.rng = np.random.default_rng(self.seed)
        self.events, self.t = [], 0

    def _inject(self, model: nn.Module) -> int:
        n = int(self.rng.poisson(self.rate))
        tta = model.tta_named_parameters()
        sizes = np.array([p.numel() for _, p in tta])
        for _ in range(n):
            if self.hit_snapshot and self.rng.random() < 0.5:        # snapshot copy holds the same bits
                names = list(model._source_tta)
                k = int(self.rng.integers(len(names)))
                tgt, nm, size = model._source_tta[names[k]], names[k], model._source_tta[names[k]].numel()
            else:
                k = int(self.rng.choice(len(tta), p=sizes / sizes.sum()))
                nm, tgt, size = tta[k][0], tta[k][1], tta[k][1].numel()
            idx, bit = int(self.rng.integers(size)), int(self.rng.choice(self.bits))
            old, new = flip_bit_(tgt, idx, bit)
            self.events.append({"step": self.t, "param": nm, "index": idx, "bit": bit, "old": old, "new": new})
        return n

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        n = self._inject(model)
        res = self.inner.step(model, x, meta)
        res.extra = dict(res.extra)
        dev = state_deviation(model)
        res.extra["n_flips"] = float(n)
        res.extra["state_dev"] = dev if math.isfinite(dev) else 1e30
        self.t += 1
        return res


class FrozenExposed(TTAMethod):
    name = "frozen"

    def reset(self, model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        aux = step_aux(meta, x.device)
        with torch.no_grad():
            logits = model(x, aux=aux).logits
            anchor = model.forward_anchor(x, aux=aux)
        return StepResult(logits=logits, anchor_logits=anchor)


class PeriodicScrub(TTAMethod):
    def __init__(self, period: int = 1000):
        self.period, self.name = period, f"scrub_{period}"

    def reset(self, model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.t = 0

    @torch.no_grad()
    def step(self, model: nn.Module, x: torch.Tensor, meta: Dict) -> StepResult:
        if self.t % self.period == 0:
            for n, p in model.tta_named_parameters():
                p.copy_(model._source_tta[n].to(p.device))
        self.t += 1
        aux = step_aux(meta, x.device)
        logits = model(x, aux=aux).logits
        anchor = model.forward_anchor(x, aux=aux)
        return StepResult(logits=logits, anchor_logits=anchor)


if __name__ == "__main__":
    # bit-flip primitive: exact, involutive, correct field behaviour
    t = torch.tensor([1.0, -2.5, 0.0])
    o, n = flip_bit_(t, 0, 31); assert n == -1.0 and t[0] == -1.0
    flip_bit_(t, 0, 31); assert t[0] == 1.0
    o, n = flip_bit_(t, 0, 30); assert not math.isfinite(n) or abs(n) > 1e30, n   # 1.0 -> exponent MSB -> inf
    t = torch.tensor([0.5]); o, n = flip_bit_(t, 0, 30); assert abs(n) > 1e37
    t = torch.tensor([1.0]); o, n = flip_bit_(t, 0, 0); assert 0 < abs(n - 1.0) < 1e-6   # low mantissa: benign
    flip_bit_(t, 0, 0); assert t[0] == 1.0
    print("bit-flip primitive OK")
    # injection statistics: Poisson mean and bit-set restriction on a toy model
    from base_cnn import CNNConfig, PhysioGuardCNN
    m = PhysioGuardCNN(CNNConfig(in_channels=1)).eval(); m.freeze_anchor()
    fi = FaultInjecting(FrozenExposed(), rate_per_1000=50, bits="exponent", seed=1)
    fi.reset(m)
    tot = sum(fi._inject(m) for _ in range(4000))
    assert 150 < tot < 250, tot                                 # 4000 beats * 0.05 = 200
    assert all(23 <= e["bit"] <= 30 for e in fi.events)
    assert state_deviation(m) > 0
    print(f"injection OK: {tot} flips in 4000 beats at 50/1000 (expect ~200)")
    print("Self-test passed.")