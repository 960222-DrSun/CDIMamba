"""CDCEM: Cross-Directional Expert Aggregation Module.

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP), Sec. 2.3 and Fig. 4.

Takes the direction-aware feature ``F_in`` produced by the preceding CDIMM and
adaptively aggregates it through **external spatial experts**.

* Gating branch: parallel max / average pooling, each followed by a ``3x3``
  depth-wise convolution, combined by element-wise summation:
  ``G = DWConv(MaxPool(F_in)) + DWConv(AvgPool(F_in))``.
* Expert branch: each external expert ``E_j`` is a lightweight **depth-wise
  separable convolution with independent parameters**. Unlike CDIMM's
  direction-bound *shared* internal experts, these are parameter-independent and
  specialise on different blur patterns.
* Combination: TopK + SoftMax gating, then weighted summation.

The per-sample gating distribution is cached in ``self.last_gating`` so that the
Directional Load-Balancing (DLB) loss can be applied to every CDCEM layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['CDCEM']


class DepthwiseSeparableConv(nn.Module):
    """Lightweight depth-wise separable convolution used as an external expert."""

    def __init__(self, dim, kernel_size=3, expansion=2):
        super().__init__()
        hidden = int(dim * expansion)
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                            groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        return self.proj(self.act(self.pw(self.dw(x))))


@ARCH_REGISTRY.register()
class CDCEM(nn.Module):
    """Cross-Directional Expert Aggregation Module.

    Args:
        dim (int): channel count ``C``.
        num_experts (int): ``n``, number of external spatial experts.
        topk (int): ``k``, number of activated experts.
        expert_kernel (int): kernel size of each external expert.
    """

    def __init__(self, dim, num_experts=4, topk=2, expert_kernel=3):
        super().__init__()
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # ---- gating branch ---------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dw_avg = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw_max = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, num_experts, 1, bias=True),
        )

        # ---- external spatial experts (parameter-independent) ----------------
        self.experts = nn.ModuleList([
            DepthwiseSeparableConv(dim, kernel_size=expert_kernel) for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, dim)
        self.last_gating = None

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        b, c, h, w = x.shape

        # gating: parallel max / avg pooling + 3x3 DWConv, summed
        g = self.dw_avg(self.avg_pool(x)) + self.dw_max(self.max_pool(x))
        logits = self.gate_mlp(g).flatten(1)  # (B, n)

        topk_val, topk_idx = torch.topk(logits, self.topk, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_val, dim=-1))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.last_gating = weights  # consumed by the DLB loss

        # weighted summation of the external experts
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n, C, H, W)
        out = (stacked * weights.view(b, self.num_experts, 1, 1, 1)).sum(dim=1)

        return self.norm(out) + x
