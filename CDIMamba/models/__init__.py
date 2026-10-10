"""CDIMamba package.

Importing ``models`` registers everything this project needs:

* architectures (``models.arch``)          -> ARCH_REGISTRY
* datasets      (``models.data``)           -> DATASET_REGISTRY
* losses        (``models.losses``)         -> LOSS_REGISTRY
* models        (``models.cdimmaba_model``) -> MODEL_REGISTRY

The official ``basicsr`` package is used as an external dependency; nothing in
``site-packages/basicsr`` is patched. Training / inference scripts simply
``import models`` before calling ``basicsr.train``.
"""

from . import arch       # noqa: F401  registers architectures
from . import data       # noqa: F401  registers datasets
from . import losses     # noqa: F401  registers losses
from .cdimmaba_model import CDIMambaModel  # noqa: F401  registers the model
from .arch import CDIMamba

__all__ = ['CDIMamba', 'CDIMambaModel']
