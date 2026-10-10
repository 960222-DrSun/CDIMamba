"""CDIMamba architecture registry.

Importing this package registers every architecture in ``ARCH_REGISTRY``.

Note: we deliberately do **not** overwrite ``basicsr.models.archs.__init__``.
Instead the training / inference entry points import this module explicitly, so
that the official ``basicsr`` installation stays untouched and the project can
co-exist with other BasicSR-based repositories.
"""

from .ss2d import SS2D, VSSBlock, SelectiveScan2D
from .cdimm import CDIMM, DirectionalExpertGate, build_directional_expert_bank
from .cdcem import CDCEM
from .restormer_blocks import (
    RestormerBackbone, RestormerBlock, Attention, FeedForward, LayerNorm,
    Downsample, Upsample,
)
from .cdimmaba import CDIMamba

__all__ = [
    'SS2D', 'VSSBlock', 'SelectiveScan2D',
    'CDIMM', 'DirectionalExpertGate', 'build_directional_expert_bank',
    'CDCEM',
    'RestormerBackbone', 'RestormerBlock', 'Attention', 'FeedForward',
    'LayerNorm', 'Downsample', 'Upsample',
    'CDIMamba',
]
