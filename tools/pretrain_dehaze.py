"""Pretrain the dehazing module for dehazing on paired synthetic images, as AOD-Net is trained.

    python tools/pretrain_dehaze.py <clear_dir> <haze_dir> <out.pth> [--iters 20000]

Pairs come from RESIDE OTS: a hazy file <id>_<A>_<beta>.jpg belongs to the clear file <id>.jpg.
The module starts from the identity (J = I) and is trained with the mean squared error between J
and the clear image, on random 256 x 256 RGB crops in [0, 1]. Five per cent of the scenes are
held out, and the PSNR of the input and of the output against the clear image is reported on
them before and after training. The detector later loads the saved weights with
dehazing_init='pretrained'.
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from akhat.dehazing import Dehazing


class Pairs(Dataset):

    def __init__(self, clear_dir, haze_files, crop, train):
        self.clear_dir, self.files, self.crop, self.train = clear_dir, haze_files, crop, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hp = self.files[i]
        cid = os.path.basename(hp).split('_')[0]
        hz = Image.open(hp).convert('RGB')
        cl = Image.open(os.path.join(self.clear_dir, cid + '.jpg')).convert('RGB')
        if cl.size != hz.size:
            cl = cl.resize(hz.size, Image.BICUBIC)
        w, h = hz.size
        c = self.crop
        if min(w, h) < c:  # a few OTS images are smaller than the crop: enlarge both
            s = c / min(w, h)
            size = (max(c, round(w * s)), max(c, round(h * s)))
            hz, cl = hz.resize(size, Image.BICUBIC), cl.resize(size, Image.BICUBIC)
            w, h = size
        if self.train:
            x, y = random.randint(0, w - c), random.randint(0, h - c)
        else:
            x, y = (w - c) // 2, (h - c) // 2
        hz, cl = hz.crop((x, y, x + c, y + c)), cl.crop((x, y, x + c, y + c))
        if self.train and random.random() < 0.5:
            hz, cl = hz.transpose(Image.FLIP_LEFT_RIGHT), cl.transpose(Image.FLIP_LEFT_RIGHT)
        t = lambda im: torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)
        return t(hz), t(cl)


def psnr(model, loader, dev):
    se_in = se_out = n = 0.
    with torch.no_grad():
        for hz, cl in loader:
            hz, cl = hz.to(dev), cl.to(dev)
            out = model(hz).clamp(0, 1)
            se_in += F.mse_loss(hz, cl, reduction='sum').item()
            se_out += F.mse_loss(out, cl, reduction='sum').item()
            n += cl.numel()
    return 10 * math.log10(n / se_in), 10 * math.log10(n / se_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clear_dir')
    ap.add_argument('haze_dir')
    ap.add_argument('out')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(os.path.join(args.haze_dir, f) for f in os.listdir(args.haze_dir))
    ids = sorted({os.path.basename(f).split('_')[0] for f in files})
    held = set(random.Random(args.seed).sample(ids, max(1, len(ids) // 20)))
    tr = [f for f in files if os.path.basename(f).split('_')[0] not in held]
    va = [f for f in files if os.path.basename(f).split('_')[0] in held]
    print(f'{len(ids)} scenes, {len(tr)} training pairs, {len(va)} held-out pairs ({len(held)} scenes)')
    tl = DataLoader(Pairs(args.clear_dir, tr, args.crop, True), args.bs, shuffle=True,
                    num_workers=8, drop_last=True, persistent_workers=True)
    vl = DataLoader(Pairs(args.clear_dir, va, args.crop, False), 64, num_workers=8)

    model = Dehazing(init='identity').to(dev)
    p_in, p0 = psnr(model, vl, dev)
    print(f'held-out PSNR: input {p_in:.2f} dB, module at identity init {p0:.2f} dB')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    it = 0
    while it < args.iters:
        for hz, cl in tl:
            hz, cl = hz.to(dev, non_blocking=True), cl.to(dev, non_blocking=True)
            loss = F.mse_loss(model(hz), cl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % 1000 == 0:
                print(f'iter {it} loss {loss.item():.5f}', flush=True)
            if it >= args.iters:
                break
    p_in, p1 = psnr(model, vl, dev)
    print(f'held-out PSNR after training: input {p_in:.2f} dB, module output {p1:.2f} dB')
    torch.save({'state_dict': model.cpu().state_dict(),
                'meta': dict(iters=args.iters, held_out_psnr_input=p_in, held_out_psnr_output=p1,
                             data='RESIDE OTS part2, ' + str(len(tr)) + ' training pairs')}, args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
