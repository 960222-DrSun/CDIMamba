"""CDIMamba: Cross-Directional Interactive Mamba network for image deblurring.

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP). Code reference: https://github.com/960222-DrSun/CDIMamba

Architecture (Fig. 2):
    I_b --Head--> F0 --encoder--> {F1,F2,F3}
        each level:  CDIMM  ->  CDCEM
        --decoder--> R ;  I_s = I_b + R

Reporting targets (paper, Table 1, 4.41 M params):
    GoPro 34.89/0.9760 | HIDE 32.67/0.9599 | RealBlur-R 41.67/0.9871 | RealBlur-J 34.49/0.9514

Ablation ladder (Table 2), useful for reproducing the component contributions:
    Restormer backbone only        1.64 M  33.62 / 0.9651
    +CDIMM                         2.46 M  34.37 / 0.9718
    +CDIMM +CDCEM                  4.41 M  34.61 / 0.9739
    +CDIMM +CDCEM +L_DLB (Full)    4.41 M  34.89 / 0.9760
"""

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .cdimm import CDIMM, build_directional_expert_bank
from .cdcem import CDCEM
from .restormer_blocks import RestormerBackbone

__all__ = ['CDIMamba']


@ARCH_REGISTRY.register()
class CDIMamba(nn.Module):
    """Cross-Directional Interactive Mamba network.

    Args:
        in_channels (int): input image channels. Default: 3.
        dim (int): base channel width. Default: 48.
        num_blocks (tuple): Restormer blocks per encoder/decoder level.
        num_heads (tuple): attention heads per level.
        num_directions (int): ``m``, scan directions in CDIMM. Default: 4.
        num_experts (int): ``n``, number of experts in both modules. Default: 4.
        topk (int): ``k``, activated experts per sample. Default: 2.
        d_state (int): SSM state dimension. Default: 16.
        use_cdimm (bool): ablation switch for the CDIMM module.
        use_cdcem (bool): ablation switch for the CDCEM module.
        weight_sharing (bool): share the directional expert bank across scales.
        with_dlb (bool): expose per-layer gating for the DLB loss (kept for
            symmetry with the yml config; the loss itself lives in
            ``models/losses/directional_load_balancing.py``).
    """

    def __init__(self, in_channels=3, dim=48, num_blocks=(4, 6, 6), num_heads=(1, 2, 4),
                 num_directions=4, num_experts=4, topk=2, d_state=16,
                 use_cdimm=True, use_cdcem=True, weight_sharing=True,
                 with_dlb=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_levels = 3
        self.use_cdimm = use_cdimm
        self.use_cdcem = use_cdcem
        self.with_dlb = with_dlb

        self.backbone = RestormerBackbone(
            dim=dim, num_blocks=num_blocks, num_heads=num_heads
        )

        level_dims = [dim * 4, dim * 2, dim]  # F3, F2, F1 (coarse -> fine)

        # ------------------------------------------------------------------
        # ONE directional expert bank, shared by every CDIMM at every scale.
        # This is the "weight sharing" design of Sec. 2.2: the bank is owned by
        # the backbone, and every CDIMM receives the same nn.ModuleList.
        # ------------------------------------------------------------------
        if use_cdimm:
            # The bank fixing a *constant* operator width is precisely what lets a
            # single bank serve every scale: a wider level simply fans out into
            # more direction groups (dim / sub_dim), all reusing the same
            # parameters. We therefore anchor the width to the finest level so
            # that every level's channel count stays an integer multiple.
            shared_sub_dim = level_dims[-1] // num_directions  # finest level width
            shared_bank = build_directional_expert_bank(
                num_directions, sub_dim=shared_sub_dim, d_state=d_state
            )
            self.shared_expert_bank = shared_bank
            self.cdimm_layers = nn.ModuleList([
                CDIMM(d, num_directions=num_directions, num_experts=num_experts,
                      topk=topk, expert_bank=shared_bank if weight_sharing else None,
                      weight_sharing=weight_sharing)
                for d in level_dims
            ])
        else:
            self.shared_expert_bank = None
            self.cdimm_layers = None

        if use_cdcem:
            self.cdcem_layers = nn.ModuleList([
                CDCEM(d, num_experts=num_experts, topk=topk) for d in level_dims
            ])
        else:
            self.cdcem_layers = None

        # fuse the refined multi-scale features back into the decoder levels
        self.merge = nn.ModuleList([
            nn.Conv2d(d * 2, d, 1, bias=False) if (use_cdimm or use_cdcem) else nn.Identity()
            for d in level_dims
        ])

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: blurry image (B, 3, H, W) in [0, 1]. Returns the sharp image."""
        feats = self.backbone.forward(x)

        refined = []
        for i, f in enumerate(feats):
            out = f
            if self.use_cdimm:
                out = self.cdimm_layers[i](out)
            if self.use_cdcem:
                out = self.cdcem_layers[i](out)
            if not isinstance(self.merge[i], nn.Identity):
                out = self.merge[i](torch.cat([f, out], dim=1))
            refined.append(out)

        residual = self.backbone.decode(refined)
        return (x + residual).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def gating_distributions(self):
        """Collect the per-layer gating weights for the DLB loss.

        Returns a list of ``(B, n)`` tensors — one entry per gate of every
        CDIMM and CDCEM layer, as required by Eq. (7) of the paper.
        """
        gating = []
        if self.cdimm_layers is not None:
            for layer in self.cdimm_layers:
                if layer.last_gating is not None:
                    gating.append(layer.last_gating)
        if self.cdcem_layers is not None:
            for layer in self.cdcem_layers:
                if layer.last_gating is not None:
                    gating.append(layer.last_gating)
        return gating
