"""
base_cnn.py  --  PhysioGuard-Space, Stage 1 (Earth Baseline)

Multi-scale depthwise-separable 1D-CNN with a decoupled dual-head design:

    x [B, C, 256] -> backbone -> features [B, 64]
                                   |-- adaptable_head   (theta_a, may drift during TTA)
                                   |-- frozen_anchor_head (theta_f, permanently frozen)

Sparse-TTA contract (<5 % of weights, tiny SRAM footprint)
----------------------------------------------------------
Only the affine (gamma, beta) of the BatchNorm that directly follows every
*depthwise* convolution ("dw_bn") is eligible for TTA. Everything else
(depthwise/pointwise kernels, pointwise-BN, stem, heads) is frozen, and BN
running statistics are never updated (model is kept in eval mode).
Because backprop only has to reach dw_bn, gradient buffers exist for
~2 * (#depthwise channels) scalars.

Elastic reset primitives (`snapshot_source`, `elastic_reset`) live here so the
controller (Stage 5) only decides *when* to reset, not *how*.

Two anchor comparisons, not one (read this before computing D_JS anywhere)
----------------------------------------------------------------------------
`forward(x, use_anchor=True)` runs `frozen_anchor_head` on the SAME backbone
features as `adaptable_head`. Since only `dw_bn` affine (inside the backbone)
is ever TTA-trainable and `adaptable_head`'s own weights never move,
`adaptable_head` and `frozen_anchor_head` stay bit-identical functions applied
to the same input for as long as TTA only touches the backbone -- so
`anchor_logits` from `forward()` is USELESS as a drift signal (D_JS against it
is identically 0 no matter how far dw_bn drifts; verified empirically in
`streaming_eval.py`'s self-test). It is kept only as a cheap same-features
head-diff diagnostic (non-zero only if a future variant also adapts head
weights).

`forward_anchor(x)` is the real drift signal: it reruns the BACKBONE with its
current (possibly adapted) `dw_bn` affine replaced by the frozen SOURCE
snapshot (`_source_tta`, captured in `freeze_anchor()`), via
`torch.func.functional_call` so the live parameters are never touched (no
in-place swap, hence no autograd-versioning interaction with an adaptation
step's forward/backward -- safe to call before, interleaved with, or after
one). `D_JS(forward(x).logits, forward_anchor(x))` is what Stage 5's Elastic
Anchor Divergence Reset must use.

Shape legend: B = batch, C = in-channels, L = 256 samples, K = 5 classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

TTA_SUFFIXES: Tuple[str, ...] = ("dw_bn.weight", "dw_bn.bias")


# --------------------------------------------------------------------------- #
# Config / output container
# --------------------------------------------------------------------------- #
@dataclass
class CNNConfig:
    in_channels: int = 1
    num_classes: int = 5
    stem_ch: int = 24                      # widened from 16 -- see module docstring
    b1_branch_ch: int = 24                 # widened from 16; block-1 out = 3 * 24 = 72
    b1_kernels: Tuple[int, ...] = (3, 7, 15)
    b2_branch_ch: int = 48                 # widened from 32; block-2 out = 3 * 48 = 144
    b2_kernels: Tuple[int, ...] = (3, 5, 9)
    feat_dim: int = 96                     # widened from 64
    head_hidden: int = 48                  # widened from 32
    head_dropout: float = 0.2
    use_se: bool = True                    # Squeeze-and-Excitation (Hu et al., CVPR 2018)
                                            # channel attention after each MultiScaleBlock --
                                            # near-zero param cost, targets per-class feature
                                            # discrimination directly rather than just adding
                                            # raw capacity. See module docstring for rationale
                                            # and the resulting parameter budget.
    se_reduction: int = 4
    aux_feat_dim: int = 0                  # extra scalar features (e.g. RR-interval timing,
                                            # see dataset.py) concatenated onto the pooled
                                            # backbone embedding, AFTER global-average-pool and
                                            # BEFORE the classifier heads -- i.e. late fusion,
                                            # not run through the conv stem. 0 = no aux features
                                            # (fully backward compatible: existing checkpoints
                                            # and every synthetic self-test in this project use
                                            # the default and are unaffected).


class ModelOutput(NamedTuple):
    logits: torch.Tensor                       # [B, K]  adaptable head
    anchor_logits: Optional[torch.Tensor]      # [B, K]  frozen anchor head (or None)
    features: Optional[torch.Tensor]           # [B, F]  shared embedding (or None)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class DWSeparableBranch(nn.Module):
    """
    Depthwise(k) -> dw_bn (TTA-eligible affine) -> ReLU -> Pointwise -> BN -> ReLU
    [B, Cin, L] -> [B, Cout, L]   ('same' padding, stride 1)
    """

    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=k // 2, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_ch)                    # <- sparse-TTA site
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, Cin, L]
        x = F.relu(self.dw_bn(self.dw(x)))                    # [B, Cin, L]
        x = F.relu(self.pw_bn(self.pw(x)))                    # [B, Cout, L]
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu, Shen & Sun, CVPR 2018,
    "Squeeze-and-Excitation Networks"): global-average-pool each channel to a
    scalar ("squeeze"), pass through a tiny 2-layer bottleneck MLP to learn a
    per-channel gate in [0,1] ("excitation"), then rescale the channels by
    that gate. Cost is ~2*C*(C/reduction) params -- for C=144, reduction=4,
    that's ~10K params, versus the ~10x-larger cost of doubling every conv
    channel count to get a comparable capacity increase. Chosen specifically
    because our problem (per published diagnosis) looks like a feature-
    discrimination gap, not a raw-capacity gap: SE lets the network learn
    which of the multi-scale branches' channels matter most for a given beat,
    which plain width cannot do.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, C, L]
        s = x.mean(dim=-1)                                    # [B, C]  squeeze
        s = F.relu(self.fc1(s))                               # [B, C//r]
        s = torch.sigmoid(self.fc2(s))                        # [B, C]  excitation gate
        return x * s.unsqueeze(-1)                            # [B, C, L]  rescale channels


class MultiScaleBlock(nn.Module):
    """
    Parallel branches with different receptive fields (QRS-scale vs. T/P-wave scale),
    concatenated on channels, optionally SE-gated, then MaxPool(2).
    [B, Cin, L] -> [B, n_branch*Cout, L/2]
    """

    def __init__(self, in_ch: int, branch_ch: int, kernels: Sequence[int],
                use_se: bool = True, se_reduction: int = 4):
        super().__init__()
        self.branches = nn.ModuleList(
            [DWSeparableBranch(in_ch, branch_ch, k) for k in kernels])
        self.out_ch = branch_ch * len(kernels)
        self.se = SEBlock(self.out_ch, reduction=se_reduction) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, Cin, L]
        x = torch.cat([b(x) for b in self.branches], dim=1)   # [B, n*Cout, L]
        if self.se is not None:
            x = self.se(x)
        return F.max_pool1d(x, 2)                             # [B, n*Cout, L/2]


class ClassifierHead(nn.Module):
    """[B, F] -> [B, K]  (tiny 2-layer MLP so theta_a / theta_f are non-trivial)."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:      # f: [B, F]
        return self.fc2(self.drop(F.relu(self.fc1(f))))       # [B, K]


class Backbone(nn.Module):
    """[B, C, 256] -> [B, feat_dim]"""

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.stem = nn.Sequential(                            # [B,C,256] -> [B,16,128]
            nn.Conv1d(cfg.in_channels, cfg.stem_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(cfg.stem_ch),
            nn.ReLU(inplace=True),
        )
        self.block1 = MultiScaleBlock(cfg.stem_ch, cfg.b1_branch_ch, cfg.b1_kernels,
                                     use_se=cfg.use_se, se_reduction=cfg.se_reduction)
        self.block2 = MultiScaleBlock(self.block1.out_ch, cfg.b2_branch_ch, cfg.b2_kernels,
                                     use_se=cfg.use_se, se_reduction=cfg.se_reduction)
        self.proj = nn.Sequential(                            # [B,96,32] -> [B,64,32]
            nn.Conv1d(self.block2.out_ch, cfg.feat_dim, 1, bias=False),
            nn.BatchNorm1d(cfg.feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, C, 256]
        x = self.stem(x)                                      # [B, 16, 128]
        x = self.block1(x)                                    # [B, 48, 64]   (branches at L=128, pooled)
        x = self.block2(x)                                    # [B, 96, 32]   (branches at L=64, pooled)
        x = self.proj(x)                                      # [B, 64, 32]
        return x.mean(dim=-1)                                 # [B, 64]  global average pool


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class PhysioGuardCNN(nn.Module):
    """
    Multi-scale 1D-CNN + decoupled adaptable / frozen-anchor heads.

    Typical life-cycle
    ------------------
    Stage 1 : train backbone + adaptable_head on DS1 (anchor unused).
    end     : model.freeze_anchor()  -> anchor := copy(adaptable), requires_grad=False,
                                        and source snapshot of TTA affine params is stored.
    TTA     : model.configure_for_tta() -> only dw_bn affine params get gradients.
    Reset   : model.elastic_reset(alpha) -> theta_a <- (1-a) theta_a + a theta_f
                                            (and dw_bn affine <- source snapshot, same rule).
    """

    def __init__(self, cfg: Optional[CNNConfig] = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        c = self.cfg
        self.backbone = Backbone(c)
        head_in = c.feat_dim + c.aux_feat_dim
        self.adaptable_head = ClassifierHead(head_in, c.head_hidden, c.num_classes, c.head_dropout)
        self.frozen_anchor_head = ClassifierHead(head_in, c.head_hidden, c.num_classes, 0.0)
        self.frozen_anchor_head.load_state_dict(self.adaptable_head.state_dict())
        for p in self.frozen_anchor_head.parameters():
            p.requires_grad_(False)
        self.register_buffer("anchor_frozen", torch.tensor(False))
        self._source_tta: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None,
                use_anchor: bool = False,
                return_features: bool = False) -> ModelOutput:
        """
        x: [B, C, 256]
        aux: [B, cfg.aux_feat_dim] extra scalar features (e.g. RR-interval timing),
             concatenated onto the pooled backbone embedding before the classifier heads.
             Required iff cfg.aux_feat_dim > 0.
        returns ModelOutput(logits [B,K], anchor_logits [B,K]|None, features [B,64]|None)
        `features` is the backbone embedding BEFORE aux concatenation (for anything that
        inspects it as a pure waveform representation, e.g. drift diagnostics).
        The anchor path is computed under no_grad on detached features: it is a
        reference signal (for JS-divergence), never a gradient source.
        """
        if x.dim() != 3 or x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"expected [B,{self.cfg.in_channels},L], got {tuple(x.shape)}")
        if self.cfg.aux_feat_dim > 0:
            if aux is None or aux.shape[-1] != self.cfg.aux_feat_dim:
                raise ValueError(f"expected aux [B,{self.cfg.aux_feat_dim}], got "
                                 f"{None if aux is None else tuple(aux.shape)}")
        feat = self.backbone(x)                               # [B, feat_dim]
        feat_in = torch.cat([feat, aux], dim=-1) if self.cfg.aux_feat_dim > 0 else feat
        logits = self.adaptable_head(feat_in)                 # [B, K]
        anchor = None
        if use_anchor:
            with torch.no_grad():
                anchor = self.frozen_anchor_head(feat_in.detach())   # [B, K]
        return ModelOutput(logits, anchor, feat if return_features else None)

    @torch.no_grad()
    def forward_anchor(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backbone-state anchor forward (the real theta_f drift reference; see the
        module docstring for why `forward(..., use_anchor=True)` is NOT this).

        Runs the backbone with its dw_bn affine parameters substituted by the
        frozen SOURCE snapshot (via `functional_call` -- the live, possibly
        adapting parameters are never read from or written to), then applies
        frozen_anchor_head. x: [B, C, 256] -> anchor_logits [B, K].

        Inference-only, no_grad. Raises if called before freeze_anchor()/
        snapshot_source(). BatchNorm running stats are shared with the live
        backbone (they are never updated during TTA either way, so this is
        exact, not an approximation).
        """
        if not self._source_tta:
            raise RuntimeError("call freeze_anchor()/snapshot_source() before forward_anchor()")
        src_params = dict(self.backbone.named_parameters())        # names relative to `backbone`
        for full_name, src_val in self._source_tta.items():        # full_name e.g. "backbone.block1...dw_bn.weight"
            key = full_name[len("backbone."):] if full_name.startswith("backbone.") else full_name
            if key in src_params:
                src_params[key] = src_val.to(x.device)
        src_buffers = dict(self.backbone.named_buffers())          # running stats: shared, unmodified
        feat_src = functional_call(self.backbone, (src_params, src_buffers), (x,))  # [B, feat_dim]
        if self.cfg.aux_feat_dim > 0:
            if aux is None or aux.shape[-1] != self.cfg.aux_feat_dim:
                raise ValueError(f"expected aux [B,{self.cfg.aux_feat_dim}], got "
                                 f"{None if aux is None else tuple(aux.shape)}")
            feat_src = torch.cat([feat_src, aux], dim=-1)
        return self.frozen_anchor_head(feat_src)                                    # [B, K]

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_anchor_head.eval()                        # anchor never uses dropout
        return self

    # ------------------------------------------------------------------ #
    # Anchor / source management
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def freeze_anchor(self) -> None:
        """Call once after Stage-1 training: theta_f := theta_a, frozen forever."""
        self.frozen_anchor_head.load_state_dict(self.adaptable_head.state_dict())
        for p in self.frozen_anchor_head.parameters():
            p.requires_grad_(False)
        self.anchor_frozen.fill_(True)
        self.snapshot_source()

    @torch.no_grad()
    def snapshot_source(self) -> None:
        """Store source-domain values of all TTA-eligible affine params."""
        self._source_tta = {n: p.detach().clone() for n, p in self.tta_named_parameters()}

    @torch.no_grad()
    def elastic_reset(self, alpha: float, reset_head: bool = True,
                      reset_backbone_affine: bool = True) -> None:
        """
        Elastic Anchor Divergence Reset:  theta_a <- (1 - alpha) theta_a + alpha theta_f.
          * head        : blended toward frozen_anchor_head parameters
          * dw_bn affine: blended toward the stored source snapshot
        alpha=1 is a hard reset to the source model.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if reset_head:
            for pa, pf in zip(self.adaptable_head.parameters(),
                              self.frozen_anchor_head.parameters()):
                pa.mul_(1.0 - alpha).add_(pf, alpha=alpha)
        if reset_backbone_affine:
            if not self._source_tta:
                raise RuntimeError("call freeze_anchor()/snapshot_source() before elastic_reset")
            for n, p in self.tta_named_parameters():
                p.mul_(1.0 - alpha).add_(self._source_tta[n].to(p.device), alpha=alpha)

    # Persist the source snapshot inside state_dict (checkpoint-safe).
    def get_extra_state(self):
        return {k: v.cpu() for k, v in self._source_tta.items()}

    def set_extra_state(self, state) -> None:
        self._source_tta = {k: v.clone() for k, v in (state or {}).items()}

    # ------------------------------------------------------------------ #
    # Sparse-TTA parameter API
    # ------------------------------------------------------------------ #
    def tta_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """(name, param) for dw_bn.{weight,bias} only."""
        return [(n, p) for n, p in self.named_parameters() if n.endswith(TTA_SUFFIXES)]

    def tta_parameters(self) -> List[nn.Parameter]:
        return [p for _, p in self.tta_named_parameters()]

    def configure_for_tta(self) -> List[nn.Parameter]:
        """
        Freeze everything except dw_bn affine params; keep BN running stats fixed
        (eval mode, so also no batch-stat dependence at batch size 1).
        Returns the list of trainable TTA params for the optimizer.
        """
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        params = self.tta_parameters()
        for p in params:
            p.requires_grad_(True)
        return params

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        anchor = sum(p.numel() for p in self.frozen_anchor_head.parameters())
        tta = sum(p.numel() for p in self.tta_parameters())
        deployed = total - anchor                # anchor is a duplicate of theta_a at deploy time,
        return {                                 # reported separately (flash, not SRAM-resident grads)
            "total": total,
            "anchor_head": anchor,
            "tta_trainable": tta,
            "tta_fraction_pct": 100.0 * tta / max(deployed, 1),
        }

    def weight_size_kb(self, bits: int = 8) -> float:
        """Static weight storage (flash) at a given quantization width."""
        return sum(p.numel() for p in self.parameters()) * bits / 8 / 1024


if __name__ == "__main__":
    m = PhysioGuardCNN().eval()
    m.freeze_anchor()
    x = torch.randn(4, 1, 256)
    out = m(x, use_anchor=True, return_features=True)
    print("logits", tuple(out.logits.shape), "anchor", tuple(out.anchor_logits.shape),
          "feat", tuple(out.features.shape))
    print(m.count_parameters(), f"| int8 weights: {m.weight_size_kb(8):.1f} KB")

    # forward_anchor() sanity: identical to forward() right after freeze_anchor()
    # (adapted == source), then must diverge once dw_bn affine is perturbed.
    anchor_logits = m.forward_anchor(x)
    assert torch.allclose(m(x).logits, anchor_logits, atol=1e-5), \
        "forward_anchor() must match forward() before any TTA drift"
    with torch.no_grad():
        for p in m.tta_parameters():
            p.add_(1.0)                                    # simulate drift
    drifted_logits = m(x).logits
    drifted_anchor = m.forward_anchor(x)
    assert not torch.allclose(drifted_logits, drifted_anchor, atol=1e-3), \
        "forward_anchor() must diverge from forward() once dw_bn affine drifts"
    assert torch.allclose(drifted_anchor, anchor_logits, atol=1e-5), \
        "forward_anchor() must be invariant to live-parameter drift (uses the frozen snapshot)"
    print("forward_anchor(): matches pre-drift, diverges post-drift, invariant to live params -> OK")