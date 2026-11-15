"""Does the image operator start dead?

The operator is K = ReLU(conv(...)) and J = ReLU(K*I - K + 1). If the chain of ReLUs outputs
K = 0 on every pixel, then J = 1 everywhere: a blank white image reaches the backbone and the
operator receives no gradient, so it cannot recover. This script measures how often that
happens and, for the exact seeds and arms used in the study, whether it happens.
"""
import glob
import sys

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

import akhat  # noqa: F401
from akhat.dehazing import Dehazing

ROOT = '/path/to/akhat'


def images(n=24):
    out = []
    for f in sorted(glob.glob(f'{ROOT}/data/rtts/images/val2017/*'))[:n]:
        im = cv2.imread(f)[:, :, ::-1]
        im = cv2.resize(im, (640, 480)).astype(np.float32) / 255.0
        out.append(torch.from_numpy(im.copy()).permute(2, 0, 1))
    return torch.stack(out)


def stats(mod, x):
    x = x.clone()
    mod = mod.to(x.device)
    for p in mod.parameters():
        p.grad = None
    j = mod(x)
    blank = (j == 1).float().mean().item()
    zero = (j == 0).float().mean().item()
    jm, js = j.mean().item(), j.std().item()
    j.mean().backward()
    g = sum(p.grad.norm().item() ** 2 for p in mod.parameters() if p.grad is not None) ** 0.5
    return blank, zero, jm, js, g


def main():
    register_all_modules()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = images().to(dev)
    print('images', x.shape, 'mean', round(x.mean().item(), 3))

    res = []
    for s in range(500):
        torch.manual_seed(s)
        res.append(stats(Dehazing(), x))
    a = np.array(res)
    print(f'random inits n=500: blank (J==1) pixel fraction mean {a[:,0].mean():.3f}; J==0 fraction mean {a[:,1].mean():.3f}')
    print(f'  fully blank (J==1 on >=99.9% of pixels): {(a[:,0] >= 0.999).sum()}/500')
    print(f'  zero gradient: {(a[:,4] == 0).sum()}/500')
    print(f'  J mean: median {np.median(a[:,2]):.3f}, 5-95% {np.percentile(a[:,2],5):.3f} to {np.percentile(a[:,2],95):.3f}')

    print('\nExact study initialisations (same seed call and build order as the training runner):')
    arms = ['a3_r50_fpn_dehaze', 'a4_r50_ffta_dehaze', 'a5_r50_ffta_attn_dehaze',
            'b3_swinl_fpn_dehaze', 'b4_swinl_ffta_dehaze']
    for arm in arms:
        seeds = [0, 1, 2] if arm.startswith('a') else [0]
        for seed in seeds:
            set_random_seed(seed, deterministic=False)
            m = MODELS.build(Config.fromfile(f'{ROOT}/repro/configs/{arm}.py').model)
            bl, z, jm, js, g = stats(m.dehazing, x)
            print(f'{arm:28s} seed {seed}: blank {bl:.3f}  J==0 {z:.3f}  J mean {jm:.3f}  J std {js:.3f}  grad norm {g:.3e}')
            del m


if __name__ == '__main__':
    sys.exit(main())
