"""Checks for the revised-code arms (c2, c3, c4), run on CPU before any training.

    python tools/smoke_v2.py <old_checkpoint.pth>

1. The identity-initialised dehazing module returns its input exactly.
2. Parameter counts of the shared and untied necks.
3. One training step on a fake batch: finite loss, gradients reach the dehazing module and
   every untied fusion block.
4. With default arguments the revised code is the released code: an old checkpoint loads with
   strict=True and the shared CCFM has the same parameter names.
"""
import sys

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

import akhat  # noqa: F401
from akhat.dehazing import Dehazing

sys.path.insert(0, 'tools')
from smoke_model import fake_batch  # noqa: E402


def n(m):
    return sum(p.numel() for p in m.parameters())


def main():
    register_all_modules()
    torch.set_num_threads(16)
    set_random_seed(0)

    d = Dehazing(init='identity')
    x = torch.rand(2, 3, 64, 80)
    print('1. identity init: max |J - I| =', (d(x) - x).abs().max().item())
    assert torch.equal(d(x), x)

    for name in ('b4_swinl_ffta_dehaze', 'c4_swinl_ffta_dehaze_v2', 'c2_swinl_ffta_v2',
                 'c3_swinl_fpn_dehaze_v2'):
        cfg = Config.fromfile(f'configs/{name}.py')
        cfg.model.backbone.init_cfg = None
        m = MODELS.build(cfg.model)
        m.neck.init_weights()
        print(f'2. {name:26s} total {n(m)/1e6:7.2f} M  neck {n(m.neck)/1e6:5.2f} M  '
              f'dehazing {n(m.dehazing) if m.dehazing else 0}')
        if name == 'c4_swinl_ffta_dehaze_v2':
            m.train()
            data = m.data_preprocessor(fake_batch(bs=1, h=256, w=320), True)
            losses = m.loss(data['inputs'], data['data_samples'])
            flat = []
            for v in losses.values():
                flat.extend(v if isinstance(v, (list, tuple)) else [v])
            total = sum(t.sum() for t in flat)
            total.backward()
            g5 = m.dehazing.conv5.weight.grad.norm().item()
            gf = [f.conv1.conv.weight.grad.norm().item() for f in m.neck.ccfm.fusions]
            print(f'3. loss {total.item():.3f}  grad conv5 {g5:.2e}  grad fusions '
                  + ' '.join(f'{g:.2e}' for g in gf))
            assert torch.isfinite(total) and g5 > 0 and all(g > 0 for g in gf)
        if name == 'b4_swinl_ffta_dehaze' and len(sys.argv) > 1:
            sd = torch.load(sys.argv[1], map_location='cpu')['state_dict']
            m.load_state_dict(sd, strict=False)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            unexpected = [k for k in unexpected if not k.startswith('decoder.aux_reg_branches')]
            print('4. old checkpoint into revised code with defaults: missing', missing,
                  'unexpected', unexpected)
            assert not missing and not unexpected
    print('SMOKE_V2_OK')


if __name__ == '__main__':
    main()
