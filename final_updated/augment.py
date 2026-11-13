"""
augment.py  --  PhysioGuard-Space, minority-class-targeted data augmentation.

Wraps an existing MITBIHDataset-compatible TRAIN split (never val/test -- see
warning below) and applies randomized signal augmentation more aggressively
to the rare AAMI classes (S, V, F), stacking on top of the sqrt-inverse-
frequency `balanced_sampling` sampler already in dataset.py. Q is
deliberately excluded from extra augmentation: DS1-train has only a handful
of real Q beats (paced records 102/104/107/217 are excluded from the de
Chazal split entirely), so synthetically inflating Q would mostly be
manufacturing signal rather than adding real information.

Techniques (standard, cheap, run per-beat on CPU inside __getitem__ so they
add negligible overhead to the existing DataLoader):
  * amplitude scaling  -- random gain in [1-a, 1+a]
  * Gaussian jitter     -- additive noise, sigma scaled to the beat's own
                            std (so it scales sensibly whether norm="record"
                            or norm="beat" was used upstream)
  * small time shift    -- reflect-padded shift of the R-peak inside the
                            window (never wraps the QRS around the edges)
  * same-class mixup    -- linear blend with ANOTHER beat of the identical
                            label, minority classes only. Never mixes two
                            different classes: that would manufacture an
                            example with an ambiguous label instead of a
                            genuinely new example of the same class.

WARNING: only ever wrap the DS1-train split with this. Wrapping val or the
DS2 test split would inject synthetic variation into model selection /
reported metrics and break the project's inter-patient, leakage-free
evaluation protocol -- do not do it, and do not add a flag that makes it
convenient to do so by accident.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class AugmentConfig:
    minority_classes: Sequence[int] = (1, 2, 3)   # AAMI indices: S, V, F
    minority_prob: float = 0.8       # prob. ANY augmentation fires on a minority beat
    base_prob: float = 0.15          # prob. on non-minority beats (light regularization)
    amp_scale: float = 0.15          # gain in [1-amp_scale, 1+amp_scale]
    jitter_sigma: float = 0.03       # relative to the beat's own std
    max_shift: int = 8               # samples (~22 ms @ 360 Hz, out of L=256)
    mixup_prob: float = 0.3          # of augmented beats, fraction that also get same-class mixup
    mixup_max_lam: float = 0.3       # partner beat's weight is drawn from [0, mixup_max_lam] -> x stays dominant


class RareClassAugmentedDataset(Dataset):
    """
    Thin wrapper: same (x, y) contract as MITBIHDataset, plus the same
    class_counts()/class_weights()/sampler() utilities (passed through to the
    base dataset, since augmentation must not change the class distribution
    the sampler/loss are computed against -- only what each drawn beat looks
    like).
    """

    def __init__(self, base: Dataset, cfg: AugmentConfig = None, seed: int = 0):
        self.base = base
        self.cfg = cfg or AugmentConfig()
        y = base.y if hasattr(base, "y") else torch.tensor([int(base[i][1]) for i in range(len(base))])
        self.by_class: Dict[int, List[int]] = {}
        for k in range(int(y.max().item()) + 1):
            idx = (y == k).nonzero(as_tuple=True)[0].tolist()
            if idx:
                self.by_class[k] = idx
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def _augment(self, x: torch.Tensor, y: int) -> torch.Tensor:
        c = self.cfg
        prob = c.minority_prob if y in c.minority_classes else c.base_prob
        if self.rng.random() > prob:
            return x

        gain = 1.0 + self.rng.uniform(-c.amp_scale, c.amp_scale)
        x = x * gain

        sigma = c.jitter_sigma * (x.std() + 1e-6)
        x = x + torch.randn_like(x) * sigma

        shift = self.rng.randint(-c.max_shift, c.max_shift)
        if shift != 0:
            pad = abs(shift)
            xp = torch.nn.functional.pad(x.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
            start = pad - shift
            x = xp[:, start:start + x.shape[-1]]

        if (y in c.minority_classes and self.rng.random() < c.mixup_prob
                and len(self.by_class.get(y, [])) > 1):
            j = self.rng.choice(self.by_class[y])
            x2 = self.base[j][0]                        # waveform is always element 0
            lam = self.rng.uniform(0.0, c.mixup_max_lam)
            x = (1 - lam) * x + lam * x2

        return x

    def __getitem__(self, i: int):
        """
        Passes through whatever arity `self.base` yields -- (x, y) normally, or
        (x, rr, y) when the base MITBIHDataset has use_rr_features=True (see
        dataset.py). Only x (the waveform) is ever augmented; rr (beat timing) is
        passed through UNCHANGED -- these signal-level perturbations (amplitude/
        jitter/shift/mixup) don't change when the beat actually occurred, so
        perturbing rr alongside x would be fabricating timing information, not
        augmenting real information.
        """
        item = self.base[i]
        if len(item) == 3:
            x, rr, y = item
            x = self._augment(x, int(y))
            return x, rr, y
        x, y = item
        x = self._augment(x, int(y))
        return x, y

    # -- pass-throughs the training script relies on ------------------------ #
    def class_counts(self):
        return self.base.class_counts()

    def class_weights(self, power: float = 0.5):
        return self.base.class_weights(power=power)

    def sampler(self, power: float = 0.5, seed: int = 0):
        return self.base.sampler(power=power, seed=seed)


# --------------------------------------------------------------------------- #
# Self-test: augmentation changes minority-class beats more than N, preserves
# shape/dtype, and never touches the underlying label.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from dataset import NUM_CLASSES

    class _Toy(Dataset):
        CLASS_COUNTS = (300, 40, 60, 15, 15)

        def __init__(self):
            t = torch.linspace(-1, 1, 256)
            templates = torch.stack([torch.sin((k + 1) * 3.0 * t) for k in range(NUM_CLASSES)])
            xs, ys = [], []
            g = torch.Generator().manual_seed(0)
            for k, n in enumerate(self.CLASS_COUNTS):
                xs.append(templates[k].view(1, 1, 256).repeat(n, 1, 1)
                          + 0.05 * torch.randn(n, 1, 256, generator=g))
                ys.append(torch.full((n,), k, dtype=torch.long))
            self.x, self.y = torch.cat(xs), torch.cat(ys)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

        def class_counts(self):
            return torch.bincount(self.y, minlength=NUM_CLASSES)

        def class_weights(self, power=0.5):
            cnt = self.class_counts().float().clamp_min(1.0)
            w = (cnt.sum() / cnt) ** power
            return w / w.mean()

        def sampler(self, power=0.5, seed=0):
            from torch.utils.data import WeightedRandomSampler
            per_class = 1.0 / self.class_counts().float().clamp_min(1.0) ** power
            w = per_class[self.y]
            return WeightedRandomSampler(w, num_samples=len(self), replacement=True,
                                          generator=torch.Generator().manual_seed(seed))

    base = _Toy()
    aug = RareClassAugmentedDataset(base, AugmentConfig(), seed=0)

    assert len(aug) == len(base)
    assert aug.class_counts().tolist() == base.class_counts().tolist(), \
        "augmentation must not change reported class counts (sampler/loss weights depend on this)"

    # shape/dtype/label integrity
    for i in [0, 300, 340, 400, 415]:
        x, y = aug[i]
        x0, y0 = base[i]
        assert x.shape == x0.shape and x.dtype == x0.dtype
        assert int(y) == int(y0), "augmentation must never change the label"

    # minority beats (S/V/F, indices 300-414) should differ from the raw
    # beat noticeably more often / more strongly than N beats (indices 0-299)
    random.seed(0)
    torch.manual_seed(0)
    n_diff = sum(float((aug[i][0] - base[i][0]).abs().mean()) for i in range(0, 50))
    minority_diff = sum(float((aug[i][0] - base[i][0]).abs().mean()) for i in range(340, 390))
    assert minority_diff > n_diff, (
        f"expected minority-class beats to be perturbed more on average "
        f"(minority={minority_diff:.4f} vs N={n_diff:.4f})")

    print(f"Self-test passed: shapes/labels preserved, class_counts unchanged, "
          f"minority perturbation ({minority_diff:.4f}) > N perturbation ({n_diff:.4f}).")