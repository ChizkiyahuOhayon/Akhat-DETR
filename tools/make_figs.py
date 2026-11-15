"""Figures 1 and 5 of the revised manuscript.

Runs a trained Akhat-DETR checkpoint on openly licensed photographs and writes, per image:
  <name>_det.png        input with predicted boxes (score >= --thr)
  <name>_det.json       all predictions above --thr
  <name>_enc_obj.png    encoder feature similarity for a reference point on an object
  <name>_enc_bg.png     the same for a reference point on the background
  <name>_enc.json       reference points and map statistics

Encoder similarity: the encoder uses deformable attention, which only weights a few sampled
points per query, so there is no dense attention map to show. Instead we take the encoder
output (memory) at the finest level and show the cosine similarity between the feature at a
reference location and the features at every other location of that level.

Reference points are chosen by rule, not by hand: the object point is the centre of the
highest-scoring car box; the background point is fixed at (0.12 W, 0.15 H).
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

import akhat  # noqa: F401  registers AkhatDETR, Dehazing, FFTA
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

CLASSES = ('person', 'bus', 'car', 'motorbike', 'bicycle')
COLORS = {0: (60, 60, 230), 1: (230, 160, 30), 2: (40, 200, 60), 3: (220, 60, 220), 4: (30, 200, 230)}  # BGR


def draw_boxes(img, boxes, scores, labels):
    out = img.copy()
    t = max(3, round(img.shape[1] / 350))
    fs = img.shape[1] / 900
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        c = COLORS[int(l)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, t)
        txt = f'{CLASSES[int(l)]} {s:.2f}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, t // 2), cv2.LINE_AA)
    return out


def overlay(img, sim, ref_xy):
    """Cosine similarity at image size, colour scale stretched to the map's own min and max
    (stated in the figure caption); brighter = more similar."""
    lo, hi = float(sim.min()), float(sim.max())
    s = (sim - lo) / max(hi - lo, 1e-6)
    heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(img, 0.35, heat, 0.65, 0)
    x, y = int(ref_xy[0]), int(ref_xy[1])
    r = max(10, img.shape[1] // 60)
    cv2.line(out, (x - r, y), (x + r, y), (255, 255, 255), max(2, r // 4))
    cv2.line(out, (x, y - r), (x, y + r), (255, 255, 255), max(2, r // 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('out_dir')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    register_all_modules()
    cfg = Config.fromfile(args.config)
    pipe = cfg.test_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.pipeline = [t for t in pipe if t['type'] != 'LoadAnnotations']
    model = init_detector(cfg, args.checkpoint, device=args.device)

    cache = {}

    def hook(mod, a, kw, out):
        cache['memory'] = out.detach()
        cache['shapes'] = kw['spatial_shapes'].detach().cpu()

    h = model.encoder.register_forward_hook(hook, with_kwargs=True)

    for path in args.images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        H, W = img.shape[:2]
        with torch.no_grad():
            res = inference_detector(model, path)
        inst = res.pred_instances
        keep = inst.scores >= args.thr
        boxes = inst.bboxes[keep].cpu().numpy()
        scores = inst.scores[keep].cpu().numpy()
        labels = inst.labels[keep].cpu().numpy()
        cv2.imwrite(os.path.join(args.out_dir, f'{name}_det.png'), draw_boxes(img, boxes, scores, labels))
        json.dump(dict(image=os.path.basename(path), size=[W, H], thr=args.thr,
                       detections=[dict(label=CLASSES[int(l)], score=round(float(s), 4),
                                        box=[round(float(v), 1) for v in b])
                                   for b, s, l in zip(boxes, scores, labels)]),
                  open(os.path.join(args.out_dir, f'{name}_det.json'), 'w'), indent=1)

        # encoder similarity at the finest level
        mem = cache['memory'][0]                       # (tokens, C)
        h0, w0 = [int(v) for v in cache['shapes'][0]]
        f0 = F.normalize(mem[:h0 * w0].float(), dim=-1).reshape(h0, w0, -1)
        meta = res.metainfo
        sh, sw = meta['img_shape']                     # resized, unpadded
        sx, sy = sw / W, sh / H                        # original -> resized
        stride_y = meta['batch_input_shape'][0] / h0
        stride_x = meta['batch_input_shape'][1] / w0
        vh, vw = int(round(sh / stride_y)), int(round(sw / stride_x))

        cars = [(s, b) for b, s, l in zip(boxes, scores, labels) if CLASSES[int(l)] == 'car']
        refs = {}
        if cars:
            b = max(cars, key=lambda t: t[0])[1]
            refs['obj'] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        refs['bg'] = (0.12 * W, 0.15 * H)
        info = dict(image=os.path.basename(path), level0=[h0, w0], valid=[vh, vw], refs={})
        for k, (x, y) in refs.items():
            fx = min(int(x * sx / stride_x), vw - 1)
            fy = min(int(y * sy / stride_y), vh - 1)
            sim = (f0[:vh, :vw] @ f0[fy, fx]).cpu().numpy()
            sim_full = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(args.out_dir, f'{name}_enc_{k}.png'), overlay(img, sim_full, (x, y)))
            info['refs'][k] = dict(xy=[round(x, 1), round(y, 1)], feat=[fy, fx],
                                   sim_min=round(float(sim.min()), 4),
                                   sim_max=round(float(sim.max()), 4),
                                   sim_mean=round(float(sim.mean()), 4),
                                   sim_p95=round(float(np.percentile(sim, 95)), 4))
        json.dump(info, open(os.path.join(args.out_dir, f'{name}_enc.json'), 'w'), indent=1)
        print(name, 'dets', len(scores), info['refs'])
    h.remove()


if __name__ == '__main__':
    main()
