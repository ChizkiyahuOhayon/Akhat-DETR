"""Akhat-DETR = DDQ-DETR + an integrated dehazing front-end.

The dehazing module is switched by a single config flag so that the "with" and
"without" arms the reviewer asked for differ in exactly one thing.
"""
from typing import Tuple

from torch import Tensor

from mmdet.models.detectors.ddq_detr import DDQDETR
from mmdet.registry import MODELS

from .dehazing import Dehazing  # noqa: F401  (registers the module)


@MODELS.register_module()
class AkhatDETR(DDQDETR):
    """DDQ-DETR with an optional AOD-Net dehazing module before the backbone.

    Args:
        use_dehazing (bool): If True, insert the dehazing module between the
            data preprocessor and the backbone. Setting it to False is the
            "No Dehazing Module" ablation arm and changes nothing else.
        dehazing_init (str): 'default' (released code), 'identity' or
            'pretrained' (see Dehazing).
        dehazing_checkpoint (str): weights for dehazing_init='pretrained'.
        dehazing_frozen (bool): keep the dehazing module fixed during training.
    """

    def __init__(self, *args, use_dehazing: bool = True,
                 dehazing_init: str = 'default', dehazing_checkpoint: str = None,
                 dehazing_frozen: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dehazing = use_dehazing
        self.dehazing = Dehazing(init=dehazing_init, checkpoint=dehazing_checkpoint,
                                 frozen=dehazing_frozen) if use_dehazing else None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Dehaze, then run the standard backbone/neck feature extraction.

        ``batch_inputs`` arrives ImageNet-standardised from the data
        preprocessor. The dehazing network's closing ReLU only makes sense on
        non-negative input, so we invert the standardisation to get [0, 1]
        images, dehaze there, and re-standardise. This is a pure change of
        variables -- with ``use_dehazing=False`` the tensor reaching the
        backbone is bit-identical to stock DDQ-DETR.
        """
        if self.dehazing is not None:
            mean = self.data_preprocessor.mean
            std = self.data_preprocessor.std
            img01 = (batch_inputs * std + mean) / 255.0
            clean01 = self.dehazing(img01)
            batch_inputs = (clean01 * 255.0 - mean) / std

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
