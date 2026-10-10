"""CDIMamba model wrapper.

Extends ``basicsr.models.sr_model.SRModel`` so that the whole BasicSR training
loop (optimisers, schedulers, EMA, logging, checkpointing, ``val``) is reused
verbatim. Only two things are added:

1. the Directional Load-Balancing term is appended to the reconstruction loss;
2. ``feed_data`` accepts the paired blur/sharp dictionary of the deblurring
   dataset.

Total objective (Eq. 7): ``L_total = L_rec + lambda * L_DLB``
"""

import torch
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

from models.arch import CDIMamba as CDIMambaArch  # noqa: F401  ensures registration
from models.losses.directional_load_balancing import DirectionalLoadBalancingLoss

__all__ = ['CDIMambaModel']


@MODEL_REGISTRY.register()
class CDIMambaModel(SRModel):
    """SRModel with the directional load-balancing regulariser.

    ``opt['train']['dlb_opt']`` controls the term::

        train:
          dlb_opt:
            loss_weight: 0.01     # lambda in L_total = L_rec + lambda * L_DLB
            reduction: mean
    """

    def init_training_settings(self):
        super().init_training_settings()

        dlb_opt = self.opt['train'].get('dlb_opt')
        if dlb_opt is None:
            dlb_opt = {'loss_weight': self.opt['train'].get('lambda_dlb', 0.01)}
        self.dlb_loss = DirectionalLoadBalancingLoss(
            loss_weight=dlb_opt.get('loss_weight', 0.01),
            reduction=dlb_opt.get('reduction', 'mean'),
        ).to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """One optimisation step.

        Note: this override deliberately does **not** call
        ``super().optimize_parameters`` because the pixel loss is recomputed
        here with an explicit ``tensor + tensor`` accumulation so that the DLB
        term stays on the autograd graph.
        """
        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq)

        # ---- reconstruction loss ------------------------------------------
        # ``cri_pix`` is normally an nn.ModuleList built by
        # SRModel.init_training_settings, but a bare nn.Module is also accepted
        # (single-loss configurations), so normalise it to a tuple first.
        cri_pix = self.cri_pix
        if not isinstance(cri_pix, (list, tuple, torch.nn.ModuleList)):
            cri_pix = [cri_pix]
        l_pix = 0.0
        for loss_term in cri_pix:
            l_pix = l_pix + loss_term(self.output, self.gt)

        # ---- directional load-balancing loss over all gated layers ---------
        gating = []
        if hasattr(self.net_g, 'gating_distributions'):
            gating = self.net_g.gating_distributions()
        l_dlb = self.dlb_loss(gating)

        l_total = l_pix + l_dlb

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(
            {'l_pix': float(l_pix.detach()), 'l_dlb': float(l_dlb.detach()),
             'l_total': float(l_total.detach())}
        )

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    @torch.no_grad()
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, **kwargs):
        """Single-GPU validation, delegating to the BasicSR implementation."""
        if self.opt.get('rank', 0) != 0:
            return None
        return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _forward_impl(self, x):
        """Single-image helper used by the inference script."""
        return self.net_g(x)
