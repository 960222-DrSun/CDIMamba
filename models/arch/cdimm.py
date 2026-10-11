"""CDIMM: Cross-Directional Interactive Mamba Module.

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP), Sec. 2.2 and Fig. 3.

Design summary
--------------
1. The input feature ``F_in`` is split along the channel dimension into ``m=4``
   sub-features, each traversed by an ``SS2D`` operator under a *distinct* scan
   order (left-to-right, right-to-left, top-to-bottom, bottom-to-top).
2. The sub-features are fused by channel concatenation + ``1x1`` conv.
3. A *directional expert gate* (parallel max/avg pooling -> ``3x3`` conv ->
   TopK + SoftMax, k=2) produces the internal-expert weights.
4. Each internal expert ``E_j`` is **bound to one scan direction** and its
   parameters are **shared across all scales and all CDIMM instances**
   (``weight_sharing`` flag). This is the key inductive bias of the paper: a
   compact, scale-invariant directional bias space that is transferred rather
   than re-learned at each level.
5. A second split + SS2D + directional expert gating captures the second-order
   horizontal-vertical interaction, **conditioned on the first-stage expert
   weighting** (``Z_exp`` is the input of the second stage).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from .ss2d import SelectiveScan2D

__all__ = ['CDIMM', 'DirectionalExpertGate', 'build_directional_expert_bank']


class DirectionalExpertGate(nn.Module):
    """Directional Expert Gate.

    Pools global statistics through **parallel max and average pooling** with
    ``3x3`` convolutions, then generates internal expert weights via
    TopK + SoftMax (k = 2).

    Args:
        dim (int): channel count of the fused feature ``Z``.
        num_experts (int): number of internal (directional) experts ``n``.
        topk (int): number of experts actually activated per sample.
    """

    def __init__(self, dim, num_experts, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.max_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # channel mixer, kept light: the gate must not dominate the parameter budget
        self.mixer = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, num_experts, kernel_size=1, bias=True),
        )

    def forward(self, z):
        """Returns ``(weights, logits)``.

        weights: (B, num_experts) sparse, rows sum to 1 (used by the DLB loss).
        logits : (B, num_experts) pre-activation scores (dense, kept for logging).
        """
        b = z.shape[0]
        g = self.avg_conv(self.avg_pool(z)) + self.max_conv(self.max_pool(z))
        logits = self.mixer(g).flatten(1)  # (B, n)

        # TopK + SoftMax: only the top-k experts receive non-zero mass
        topk_val, topk_idx = torch.topk(logits, self.topk, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_val, dim=-1))
        return weights, logits


@ARCH_REGISTRY.register()
class CDIMM(nn.Module):
    """Cross-Directional Interactive Mamba Module.

    Args:
        dim (int): channel count ``C`` of ``F_in``.
        num_directions (int): ``m``, number of scan directions / sub-features.
        num_experts (int): ``n``, number of directional internal experts.
        topk (int): ``k``, number of activated experts in the gate.
        expert_bank (nn.ModuleList | None): shared weight bank produced by
            :func:`build_directional_expert_bank`. When ``None`` the module
            owns a private bank (used for single-scale ablation).
        weight_sharing (bool): if True, ``expert_bank`` is supplied externally
            and its parameters are shared across all CDIMM instances.
        second_order (bool): enable the second split + SS2D stage.
    """

    def __init__(self, dim, num_directions=4, num_experts=4, topk=2,
                 expert_bank=None, weight_sharing=True, second_order=True):
        super().__init__()
        assert num_directions in (2, 4), 'CDIMM supports 2 or 4 directional scans'
        self.dim = dim
        self.m = num_directions
        self.n = num_experts
        self.weight_sharing = weight_sharing
        self.second_order = second_order

        # ---- internal (direction-bound) experts -------------------------------
        # Each expert is bound to one scan direction. When weight_sharing is on,
        # the expert bank is owned by the *backbone*, not by this module, so the
        # very same parameters are reused at every scale.
        if expert_bank is None:
            # private bank: pick the widest width that still splits ``dim`` by m
            sub_dim = dim // self.m
            assert dim % self.m == 0, f'channels {dim} not divisible by m={self.m}'
            expert_bank = build_directional_expert_bank(self.m, sub_dim)
        else:
            # the shared bank fixes the operator width; the gate is free to feed
            # it ``d / sub_dim`` direction groups (see ``_directional_scan``).
            sub_dim = expert_bank[0].d_model
            assert dim % sub_dim == 0, (
                f'channels {dim} not divisible by the shared bank width {sub_dim}')
        self.experts = expert_bank
        self.sub_dim = sub_dim
        # number of direction groups at this level: >= m when the level is wider
        # than the bank width (which is how the single bank survives all scales)
        self.num_groups = max(self.m, dim // sub_dim)

        self.split_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.fusion_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Fallback projection used only when the gate/bank width does not divide
        # ``dim`` (ablation configurations). Parameter-free in the default setup.
        self._mix_proj = nn.Conv2d(len(self.experts) * self.sub_dim, dim, 1, bias=False)

        self.gate = DirectionalExpertGate(dim, num_experts, topk)

        # ---- second-order cross-directional interaction -----------------------
        if self.second_order:
            self.split_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            self.fusion_conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            self.gate2 = DirectionalExpertGate(dim, num_experts, topk)
            self.refine = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            )
        else:
            self.split_conv2 = self.fusion_conv2 = self.gate2 = self.refine = None

        self.last_gating = None  # (B, n) tensor cached for the DLB loss

    def _directional_scan(self, x, split_conv):
        """Split -> per-direction SS2D -> concat -> 1x1 fusion.

        The feature is split into ``G = dim / sub_dim`` channel groups (``G >= m``
        when the level is wider than the shared bank width). Group ``g`` is
        traversed by the bank operator ``E_{g mod m}`` under scan order
        ``g mod m``, so direction and operator remain coupled by construction
        while the *same* fixed-width bank serves every scale.

        Returns:
            (Tensor, Tensor, int): ``(Z, E_stack, G)`` where ``Z`` is the fused
            feature and ``E_stack`` is ``(B, n_e * sub_dim, H, W)`` holding the
            first ``n_e = min(G, len(bank))`` direction-bound responses in
            expert-major order (reused by Eq. 3, so no second SS2D evaluation).
        """
        # (B, C, H, W) -> G groups of (B, sub_dim, H, W)
        chunks = split_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        bound = []
        n_e = min(self.num_groups, len(self.experts))
        for i, chunk in enumerate(chunks):
            d = i % self.m                       # scan direction for this group
            ei = self.experts[d]
            # scan order d: 0 = L->R, 1 = R->L, 2 = T->B, 3 = B->T
            if d == 1:
                chunk = torch.flip(chunk, dims=[-1])
            elif d == 2:
                chunk = torch.flip(chunk, dims=[-2])
            elif d == 3:
                chunk = torch.flip(chunk, dims=[-2, -1])
            out = ei(chunk)
            if d == 1:
                out = torch.flip(out, dims=[-1])
            elif d == 2:
                out = torch.flip(out, dims=[-2])
            elif d == 3:
                out = torch.flip(out, dims=[-2, -1])
            outs.append(out)
            if i < n_e:
                bound.append(out)
        fused = self.fusion_conv(torch.cat(outs, dim=1))

        b, _, h, w = fused.shape
        if bound:
            cat = torch.cat(bound, dim=1)                       # (B, n_e*sub, H, W)
            sub = cat.shape[1] // n_e
            # (B, n_e, sub, H, W) -> (B, n_e*sub, H, W): keep expert-major order
            e_stack = cat.view(b, n_e, sub, h, w).reshape(b, n_e * sub, h, w)
        else:
            e_stack = fused
            n_e = 1
        return fused, e_stack, n_e

    def _mix_experts(self, e_stack, n_e, weights):
        """Eq. (3): ``Z_exp = sum_j w_j * E_j(F_j)`` applied per expert slice.

        ``E_j(F_j)`` for every ``j`` already lives in a distinct ``sub_dim``-wide
        slice of ``e_stack`` (expert-major order), so the per-sample gate weight
        ``w_j`` only has to be broadcast over that slice. The channel count is
        preserved, keeping the module a drop-in residual block.

        Args:
            e_stack (Tensor): (B, n_e * sub_dim, H, W) interleaved responses.
            n_e (int): number of experts in the stack.
            weights (Tensor): (B, n) gating weights, rows sum to 1.
        """
        b = e_stack.shape[0]
        sub = e_stack.shape[1] // n_e
        w = weights
        if w.shape[1] != n_e:
            # ablation: gate wider/narrower than the bank -> renormalise onto n_e
            w = w[:, :n_e]
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        stacked = e_stack.view(b, n_e, sub, *e_stack.shape[-2:])
        mixed = (stacked * w.view(b, n_e, 1, 1, 1)).reshape(b, n_e * sub, *e_stack.shape[-2:])
        if mixed.shape[1] != self.dim:
            mixed = self._mix_proj(mixed)
        return mixed

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        z, e_stack, n_e = self._directional_scan(x, self.split_conv)  # Eq. (2)
        weights, logits = self.gate(z)                                # directional gate
        z_exp = self._mix_experts(e_stack, n_e, weights)              # Eq. (3)

        if not self.second_order:
            self.last_gating = weights
            return z_exp

        # second-order horizontal-vertical interaction, conditioned on Z_exp
        z2, e_stack2, n_e2 = self._directional_scan(z_exp, self.split_conv2)
        weights2, logits2 = self.gate2(z2)
        z2_exp = self._mix_experts(e_stack2, n_e2, weights2)

        out = self.refine(z2_exp) + z_exp
        # the DLB loss is applied to *both* gates of every CDIMM layer; we expose
        # the concatenated gating distribution.
        self.last_gating = torch.cat([weights, weights2], dim=-1)
        return out


def build_directional_expert_bank(num_directions, sub_dim, d_state=16, expand=2):
    """Create the weight-shared directional expert bank.

    Each expert is a select-scan operator bound to one scan direction. The bank
    is instantiated **once** per backbone (see :class:`CDIMamba`) so that its
    parameters are shared across scales and across all CDIMM instances, which is
    the "weight sharing" arrow in Fig. 3 of the paper.
    """
    return nn.ModuleList([
        SelectiveScan2D(sub_dim, d_state=d_state, expand=expand)
        for _ in range(num_directions)
    ])
