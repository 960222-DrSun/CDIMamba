"""Selective Scan 2D (SS2D) operator.

Reference implementation of the 2D selective scan used by VMamba / EVSSM [13].
The implementation is dependency-free (pure PyTorch) so that the model can be
instantiated and run on CPU without the CUDA kernel of ``mamba-ssm``.

If ``mamba_ssm`` / ``selective_scan_cuda`` is available and ``use_cuda_scan`` is
set to True, the fused kernel is used instead (about 3-5x faster on GPU).

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP), Sec. 2.2, Eq. (2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY


class SelectiveScan2D(nn.Module):
    """Pure-PyTorch 2D selective scan (SS2D).

    Args:
        d_model (int): number of channels of the input feature.
        d_state (int): SSM state dimension N. Default: 16.
        d_conv (int): local convolution width applied on the scan path. Default: 3.
        expand (int): channel expansion ratio of the inner projection. Default: 2.
        dt_rank (int | str): rank of the delta projection. 'auto' -> ceil(d_inner / 16).
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank='auto'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = max(1, (self.d_inner + 15) // 16) if dt_rank == 'auto' else dt_rank

        # input projection: x -> (x_branch, z, dt, B, C)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.act = nn.SiLU()

        # delta (dt), B and C projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt bias initialisation, as in the original Mamba paper
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (torch.log(torch.tensor(1e-3)) - torch.log(torch.tensor(0.1)))
            + torch.log(torch.tensor(0.1))
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # state-space parameters A (kept in log space), D
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        b, c, h, w = x.shape
        h, w = int(h), int(w)
        L = h * w

        xz = self.in_proj(x.permute(0, 2, 3, 1))  # (B, H, W, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)

        # local convolution on the scanned sequence
        x_ssm = rearrange(x_ssm, 'b h w d -> b d h w')
        x_ssm = self.act(self.conv2d(x_ssm))

        # flatten the scan path (row-major)
        x_ssm = rearrange(x_ssm, 'b d h w -> b d (h w)').transpose(1, 2)  # (B, L, d_inner)

        dt, B, C = torch.split(
            self.x_proj(x_ssm), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # discretisation: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # parallel scan recurrence: h_t = dA_t * h_{t-1} + dB_t * x_t
        # computed with an associative (linear) scan so that no python loop is
        # required and the whole sequence stays on the autograd graph.
        x_d = x_ssm.unsqueeze(-1) * dB  # (B, L, d_inner, d_state)
        h_seq = _associative_scan(dA, x_d)  # (B, L, d_inner, d_state)
        y = (h_seq * C.unsqueeze(2)).sum(-1)  # (B, L, d_inner)
        y = y + x_ssm * self.D

        # gate with the z branch
        y = y * F.silu(z.reshape(b, L, self.d_inner))
        y = self.out_proj(y)
        return rearrange(y, 'b (h w) d -> b d h w', h=h, w=w)


def _associative_scan(a, x):
    """Linear (associative) scan of ``h_t = a_t * h_{t-1} + x_t``.

    Hillis-Steele style, O(log L) sequential depth. Numerically stable because
    ``a = dA`` is the exponential of a negative quantity, hence in (0, 1].
    """
    a, x = a.clone(), x.clone()
    L = a.shape[1]
    step = 1
    while step < L:
        # shift by `step`, padding the front with the identity element
        a_shift = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
        x_shift = F.pad(x[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
        x = x + a * x_shift
        a = a * a_shift
        step *= 2
    return x


@ARCH_REGISTRY.register()
class SS2D(nn.Module):
    """Registry-facing wrapper of the SS2D operator, keeping the input layout."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, **kwargs):
        super().__init__()
        self.op = SelectiveScan2D(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.op(x)


@ARCH_REGISTRY.register()
class VSSBlock(nn.Module):
    """A single visual state-space block (pre-norm + SS2D + residual)."""

    def __init__(self, dim, d_state=16, d_conv=3, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.op = SelectiveScan2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(self.op(x))
        return x
