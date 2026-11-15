"""AOD-Net style dehazing module.

Reproduction of `dehazing.py` recovered from the paper's public repo
(https://github.com/ChizkiyahuOhayon/Akhat-DETR), with the MODELS registry
decorator added so it can be built from a config. The forward pass is unchanged;
the optional ``init``, ``checkpoint`` and ``frozen`` arguments are additions.

Implements Eq. 3 of the manuscript: a single CNN estimates K(x), and the clean
image is recovered as J(x) = K(x)*I(x) - K(x) + b with b = 1.

NOTE ON INPUT RANGE: the final ReLU means this module is only meaningful on
non-negative input, i.e. images in [0, 1]. See AkhatDETR.extract_feat, which
de-normalises the ImageNet-standardised batch back to [0, 1] before calling
this, then re-normalises. Feeding it standardised input would clamp away every
negative value and destroy roughly half the input distribution.
"""
import torch
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class Dehazing(nn.Module):

    def __init__(self, init: str = 'default', checkpoint: str = None, frozen: bool = False):
        """init='default' is the released code (PyTorch default initialisation).
        init='identity' zeroes conv5's weights and sets its bias to 1, so K(x) = 1 and
        J(x) = I(x) exactly at the start of training.
        init='pretrained' loads weights trained for dehazing on paired synthetic images
        (tools/pretrain_dehaze.py) from ``checkpoint``.
        frozen=True keeps the weights fixed during detector training (restore-then-detect);
        otherwise the module is fine-tuned through the detection loss.
        """
        super(Dehazing, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)
        assert init in ('default', 'identity', 'pretrained'), init
        if init in ('identity', 'pretrained'):
            nn.init.zeros_(self.conv5.weight)
            nn.init.ones_(self.conv5.bias)
        if init == 'pretrained':
            assert checkpoint, 'init="pretrained" needs a checkpoint'
            sd = torch.load(checkpoint, map_location='cpu')
            self.load_state_dict(sd.get('state_dict', sd), strict=True)
        self.frozen = frozen
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.conv5(concat3))
        restored_image = self.relu((x5 * x) - x5 + 1)

        return restored_image
