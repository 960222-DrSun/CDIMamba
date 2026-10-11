"""Directional Load-Balancing (DLB) loss.

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP), Sec. 2.4, Eq. (7).

    L_DLB^l = 1/log(n) * D_KL(p^l || U),   p^l_j = (1/B) * sum_b W_j^{(b,l)}

where ``W^{(b,l)}`` is the gating distribution of sample ``b`` at layer ``l``,
``U = {1/n, ..., 1/n}`` is the uniform distribution, and ``B`` is the
mini-batch size.

Because ``D_KL(p || U) in [0, log n]``, the normalised loss lies in ``[0, 1]``:
it is 0 for a perfectly balanced workload and 1 when every sample is routed to a
single expert. The loss constrains the **batch-level workload**, not each sample
individually — forcing per-sample uniformity would destroy the intended sparsity
of the routing.

The corresponding normalised *entropy* acts in the opposite direction (it
rewards concentrated routing and therefore reinforces expert collapse); it is
provided here only for diagnostic logging, never as a training objective.
"""

import torch
import torch.nn as nn

from basicsr.models.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

__all__ = ['DirectionalLoadBalancingLoss', 'directional_load_balancing_loss',
           'expert_utilization_entropy']


def _dlb_per_layer(gating, eps=1e-8):
    """KL divergence from the batch-level expert workload to the uniform prior.

    Args:
        gating (Tensor): (B, n) per-sample gating distribution, rows sum to 1.
    Returns:
        Tensor: scalar loss in [0, 1].
    """
    if gating.dim() != 2:
        raise ValueError(f'expected (B, n) gating tensor, got shape {tuple(gating.shape)}')
    num_experts = gating.shape[1]
    if num_experts < 2:
        return gating.new_zeros(())

    # importance of expert j over the mini-batch (Eq. 7)
    p = gating.mean(dim=0).clamp_min(eps)
    p = p / p.sum().clamp_min(eps)

    log_n = torch.log(gating.new_tensor(float(num_experts)))
    kl = (p * torch.log(num_experts * p + eps)).sum()
    return kl / log_n


@LOSS_REGISTRY.register()
class DirectionalLoadBalancingLoss(nn.Module):
    """DLB loss over all gated layers.

    ``model.gating_distributions()`` returns one ``(B, n)`` tensor per gate of
    every CDCEM layer (and, by construction, every CDIMM gate as well). The loss
    averages the per-layer terms: ``L_DLB = (1/L) * sum_l L_DLB^l``.

    Usage in the training loop::

        L_total = L_rec + lambda_dlb * loss_dlb(model.gating_distributions())

    Args:
        loss_weight (float): overall multiplier, i.e. ``lambda`` in Eq. (7).
        reduction (str): 'mean' averages over layers, 'sum' accumulates.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, gating_list, weight=None, **kwargs):
        """Args:
            gating_list (list[Tensor] | Tensor): list of (B, n) gating tensors,
                or a single stacked tensor of shape (L, B, n).
        """
        if gating_list is None:
            return torch.zeros((), requires_grad=True)
        if isinstance(gating_list, torch.Tensor):
            if gating_list.dim() == 3:
                gating_list = list(gating_list.unbind(0))
            elif gating_list.dim() == 2:
                gating_list = [gating_list]
            else:
                raise ValueError(
                    f'unsupported gating shape {tuple(gating_list.shape)}')
        if len(gating_list) == 0:
            return torch.zeros((), requires_grad=True)

        terms = torch.stack([_dlb_per_layer(g) for g in gating_list])
        loss = terms.sum() if self.reduction == 'sum' else terms.mean()
        if weight is not None:
            loss = loss * weight
        return loss * self.loss_weight


def directional_load_balancing_loss(gating_list, loss_weight=1.0, reduction='mean'):
    """Functional form of :class:`DirectionalLoadBalancingLoss`."""
    return DirectionalLoadBalancingLoss(loss_weight, reduction)(gating_list)


def expert_utilization_entropy(gating):
    """Normalised entropy of the batch-level expert workload.

    **Diagnostic only.** Its normalised form is ``-1/log(n) * sum_j p_j log p_j``;
    minimising it *rewards concentration* and therefore reinforces expert
    collapse, which is precisely what the DLB loss is designed to prevent.
    Use it to monitor routing diversity, never as a training term.
    """
    num_experts = gating.shape[1]
    p = gating.mean(dim=0).clamp_min(1e-8)
    p = p / p.sum()
    return -(p * torch.log(p)).sum() / torch.log(gating.new_tensor(float(num_experts)))
