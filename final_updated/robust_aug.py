"""
robust_aug.py  --  corruption-aware TRAINING augmentation for the source model.

Motivation (diag_corruption.py): the source model never sees corruption during training, so
it loses most of its accuracy on shifted streams; every adaptation method then starts from
a brittle model. This transform exposes the model to the same FAMILIES of corruption, with
random severity and random "subject" draws, at training time (DS1-train only):

  * parametric simulator components (axis tilt, morphology, baseline wander, radiation), the
    full mix or a single component, severity ~ U(0, max_severity);
  * optionally REAL noise (NSTDB bw / em / ma) at SNR ~ U(snr_lo, snr_hi) dB, drawn from the
    FIRST half of each noise recording only. Evaluation (run_benchmark.py, diag_corruption.py)
    uses the SECOND half, so a model trained on real noise is never tested on the same stretch.

Evaluation-side honesty: results on the parametric simulator are then IN-DISTRIBUTION for the
augmented model and must be labelled so; the real-noise scenarios (held-out half) and the
cross-database scenarios (INCART/SVDB) are the out-of-distribution evidence.
Use only on the training split.

  !python train_stage1.py ... --corrupt_aug 0.6 --corrupt_max_severity 0.6 \
        --nstdb_dir $D/data/nstdb --real_noise_prob 0.4
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import torch

from real_shift import NOISE_KINDS, NoiseBank, RealNoiseConfig, RealNoiseSimulator
from space_shift_simulator import SpaceShiftConfig, SpaceShiftSimulator

_SIM_KW = {
    "all": dict(),
    "microgravity": dict(enable_wander=False, enable_radiation=False),
    "wander": dict(enable_microgravity=False, enable_radiation=False),
    "radiation": dict(enable_microgravity=False, enable_wander=False),
}


class CorruptionAugment:
    """Callable [C,L] -> [C,L], suitable as MITBIHDataset(transform=...)."""

    def __init__(self, p: float = 0.5, max_severity: float = 0.6, bank: Optional[NoiseBank] = None,
                 p_real_noise: float = 0.0, snr_range: Sequence[float] = (0.0, 24.0), seed: int = 0,
                 fs: float = 360.0, r_index: int = 96, exclude: Sequence[str] = ()):
        # exclude: corruption families the model must NEVER see in training (leave-one-out test).
        # Simulator families: microgravity, wander, radiation ("all" is dropped if any of them is excluded);
        # real NSTDB kinds: bw, em, ma.
        self.exclude = set(exclude)
        bad = self.exclude - set(_SIM_KW) - set(NOISE_KINDS)
        assert not bad, f"unknown corruption families {sorted(bad)}"
        self.p, self.max_sev, self.p_real = p, max_severity, (p_real_noise if bank is not None else 0.0)
        self.snr_lo, self.snr_hi = snr_range
        self.rng = random.Random(seed)
        sim_excluded = bool(self.exclude & {"microgravity", "wander", "radiation"})
        self.sims: Dict[str, SpaceShiftSimulator] = {
            k: SpaceShiftSimulator(SpaceShiftConfig(seed=seed, fs=fs, r_index=r_index, **kw))
            for k, kw in _SIM_KW.items() if k not in self.exclude and not (k == "all" and sim_excluded)}
        self.noise_kinds = tuple(k for k in NOISE_KINDS if k not in self.exclude)
        assert self.sims or self.noise_kinds, "everything excluded"
        self.p_real = self.p_real if self.noise_kinds else 0.0
        self.p_real = self.p_real if self.sims else (1.0 if self.noise_kinds and bank is not None else 0.0)
        self.bank = bank
        self.noise_sims: Dict[str, RealNoiseSimulator] = {}
        if bank is not None:                       # snr_db=0 -> unit amplitude; per-call severity sets the SNR
            self.noise_sims = {k: RealNoiseSimulator(bank, RealNoiseConfig(
                snr_db=0.0, kinds=(k,), clean_start_s=0.0, always_on=True, seed=seed, fs=fs, r_index=r_index))
                for k in self.noise_kinds}
        self.fs = fs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r = self.rng
        if r.random() >= self.p:
            return x
        if self.noise_sims and r.random() < self.p_real:
            kind = r.choice(self.noise_kinds)
            snr = r.uniform(self.snr_lo, self.snr_hi)
            t0 = r.uniform(0.0, len(self.bank) / self.fs)
            return self.noise_sims[kind](x, record_id=r.randrange(10**6), beat_key=0, t0_sec=t0,
                                         severity=10.0 ** (-snr / 20.0), return_mask=False)
        name = r.choice(list(self.sims))
        sev = r.uniform(0.0, self.max_sev)
        return self.sims[name](x, record_id=r.randrange(10**6), beat_key=r.randrange(10**9),
                               t0_sec=r.uniform(0.0, 1800.0), severity=sev, return_mask=False)


if __name__ == "__main__":
    from space_shift_simulator import _synthetic_beat
    x0 = _synthetic_beat(256, 96, 360.0).unsqueeze(0) if _synthetic_beat(256, 96, 360.0).dim() == 1 else _synthetic_beat(256, 96, 360.0)
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    bank = NoiseBank.synthetic(120)
    aug = CorruptionAugment(p=1.0, max_severity=0.6, bank=bank, p_real_noise=0.5, seed=1)
    outs = [aug(x0.clone()) for _ in range(200)]
    assert all(o.shape == x0.shape and torch.isfinite(o).all() for o in outs)
    changed = sum(not torch.equal(o, x0) for o in outs)
    assert changed >= 190, f"p=1 should corrupt nearly every call, changed {changed}/200"
    assert len({round(float((o - x0).abs().mean()), 6) for o in outs}) > 100, "draws must differ"
    off = CorruptionAugment(p=0.0, seed=1)
    assert all(torch.equal(off(x0.clone()), x0) for _ in range(20)), "p=0 must be the identity"
    half = CorruptionAugment(p=0.5, seed=2)
    frac = sum(not torch.equal(half(x0.clone()), x0) for _ in range(400)) / 400
    assert 0.35 < frac < 0.65, frac
    # first-half / second-half noise banks must be disjoint stretches of the recording
    full = NoiseBank.synthetic(120)
    a, b = full.half("first"), full.half("second")
    assert len(a) + len(b) <= len(full) and len(a) == len(b)
    # leave-one-out: excluded families are never drawn
    ex = CorruptionAugment(p=1.0, bank=bank, p_real_noise=0.5, seed=3, exclude=("ma", "radiation"))
    assert set(ex.sims) == {"microgravity", "wander"} and ex.noise_kinds == ("bw", "em")
    seen = set()
    for _ in range(300):
        ex(x0.clone())
    ex_ma = CorruptionAugment(p=1.0, bank=bank, p_real_noise=1.0, seed=3, exclude=("bw", "em"))
    assert ex_ma.noise_kinds == ("ma",)
    print(f"  p=1 corrupts {changed}/200, p=0 identity, p=0.5 -> {frac:.2f}; noise halves disjoint ({len(a)}+{len(b)} samples)")
    print("Self-test passed.")