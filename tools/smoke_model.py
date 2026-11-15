"""Build each ablation arm and run one train-mode forward/backward on fake data.

Catches config errors, CCFM shape misalignment and broken gradient paths without
needing the dataset. Run BEFORE launching any real training.

    python tools/smoke_model.py configs/a1_r50_fpn.py [more configs...]
"""
import argparse
import sys

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

import akhat  # noqa: F401  registers Dehazing / FFTA / AkhatDETR


def fake_batch(bs=2, h=512, w=640):
    inputs = [
        torch.randint(0, 255, (3, h, w), dtype=torch.uint8) for _ in range(bs)
    ]
    samples = []
    for _ in range(bs):
        ds = DetDataSample()
        ds.set_metainfo(
            dict(
                img_shape=(h, w),
                ori_shape=(h, w),
                pad_shape=(h, w),
                scale_factor=(1.0, 1.0)))
        gt = InstanceData()
        gt.bboxes = torch.tensor([[10., 10., 100., 100.],
                                  [50., 60., 300., 320.]])
        gt.labels = torch.tensor([0, 2])
        ds.gt_instances = gt
        samples.append(ds)
    return dict(inputs=inputs, data_samples=samples)


def check(cfg_path):
    print('=' * 78)
    print('CONFIG:', cfg_path)
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.cuda().train()

    n_all = sum(p.numel() for p in model.parameters())
    n_dehaze = sum(p.numel()
                   for p in model.dehazing.parameters()) if model.dehazing else 0
    n_neck = sum(p.numel() for p in model.neck.parameters())
    print(f'  detector      : {type(model).__name__}')
    print(f'  backbone      : {type(model.backbone).__name__}')
    print(f'  neck          : {type(model.neck).__name__}  ({n_neck/1e6:.2f} M)')
    print(f'  dehazing      : {"ON " if model.dehazing else "OFF"}'
          f'  ({n_dehaze/1e3:.1f} K params)')
    print(f'  num_queries   : {model.num_queries}')
    print(f'  total params  : {n_all/1e6:.1f} M')

    data = fake_batch()
    data = model.data_preprocessor(data, True)
    feats = model.extract_feat(data['inputs'])
    print('  neck outputs  :', [tuple(f.shape) for f in feats])
    assert len(feats) == cfg.model.encoder.layer_cfg.self_attn_cfg.num_levels, \
        (f'neck emits {len(feats)} levels but encoder expects '
         f'{cfg.model.encoder.layer_cfg.self_attn_cfg.num_levels}')

    losses = model.loss(data['inputs'], data['data_samples'])
    flat = []
    for v in losses.values():
        flat.extend(v if isinstance(v, (list, tuple)) else [v])
    total = sum(x.sum() for x in flat)
    print(f'  total loss    : {total.item():.4f}')
    assert torch.isfinite(total), 'loss is not finite'

    total.backward()
    if model.dehazing is not None:
        gn = sum(p.grad.norm().item()**2 for p in model.dehazing.parameters()
                 if p.grad is not None)**0.5
        got = sum(1 for p in model.dehazing.parameters() if p.grad is not None)
        want = len(list(model.dehazing.parameters()))
        print(f'  dehaze grads  : {got}/{want} params, grad norm {gn:.4e}')
        assert got == want and gn > 0, 'dehazing module receives NO gradient'

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'  peak VRAM     : {peak:.2f} GiB (bs=2 @ 512x640)')
    print('  RESULT        : PASS')
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    args = ap.parse_args()
    register_all_modules()
    failed = []
    for c in args.configs:
        try:
            check(c)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'  RESULT        : FAIL -- {type(e).__name__}: {e}')
            failed.append(c)
    print('=' * 78)
    print(f'{len(args.configs) - len(failed)}/{len(args.configs)} passed')
    if failed:
        print('FAILED:', *failed, sep='\n  ')
        sys.exit(1)
    print('ALL_SMOKE_PASS')
