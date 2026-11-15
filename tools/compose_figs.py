"""Compose Fig. 1 and Fig. 5 from the outputs of repro/tools/make_figs.py.

    python compose_figs.py <make_figs_out_dir> <latex/figures dir>

Fig. 1: four photographs, input on top, predictions below. Boxes are redrawn here from
<name>_det.json at the final print size, so line width and colours are the same in every
panel; colour encodes the class (legend under the figure), threshold 0.3 as in make_figs.
No image is cropped; all panels share one height.

Fig. 5: the two encoder-similarity overlays of commons_11 side by side, with a colour bar.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, 'photos')
DPI = 600
TEXT_W_IN = 372 / 72.27            # sn-jnl text width
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
FONT_B = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
# Okabe-Ito, readable for colour-blind readers and on grey haze
CLASS_RGB = {'person': (213, 94, 0), 'bus': (0, 114, 178), 'car': (0, 158, 115),
             'motorbike': (204, 121, 167), 'bicycle': (230, 159, 0)}
FIG1 = [('commons_06', 'commons_06.jpg'), ('commons_04', 'commons_04.jpg'),
        ('commons_02', 'commons_02.jpg'),
        ('commons_19_plates_blurred', 'commons_19_plates_blurred.jpg')]


def pt(p):
    return int(round(p / 72 * DPI))


def fig1(det_dir, out):
    gap = pt(2.5)
    label_h = pt(9)
    legend_h = pt(11)
    total_w = int(TEXT_W_IN * DPI)
    ims = [Image.open(os.path.join(PHOTOS, f)).convert('RGB') for _, f in FIG1]
    aspect = [im.width / im.height for im in ims]
    h = int((total_w - gap * (len(ims) - 1)) / sum(aspect))
    widths = [int(round(h * a)) for a in aspect]
    widths[-1] = total_w - gap * (len(ims) - 1) - sum(widths[:-1])
    H = label_h + 2 * h + gap + legend_h
    canvas = Image.new('RGB', (total_w, H), 'white')
    d = ImageDraw.Draw(canvas)
    f_lab = ImageFont.truetype(FONT_B, pt(8))
    f_leg = ImageFont.truetype(FONT, pt(6.5))
    lw = max(2, pt(0.7))
    x = 0
    used = set()
    for (name, _), im, w, letter in zip(FIG1, ims, widths, 'abcd'):
        small = im.resize((w, h), Image.LANCZOS)
        canvas.paste(small, (x, label_h))
        det = json.load(open(os.path.join(det_dir, f'{name}_det.json')))
        assert det['size'] == [im.width, im.height], (name, det['size'], im.size)
        s = w / im.width
        pred = small.copy()
        pd = ImageDraw.Draw(pred)
        for o in sorted(det['detections'], key=lambda o: o['score']):
            x1, y1, x2, y2 = [v * s for v in o['box']]
            pd.rectangle([x1, y1, x2, y2], outline=CLASS_RGB[o['label']], width=lw)
            used.add(o['label'])
        canvas.paste(pred, (x, label_h + h + gap))
        d.text((x, label_h // 2), letter, font=f_lab, fill='black', anchor='lm')
        x += w + gap
    # legend: only the classes that appear
    lx, ly = 0, label_h + 2 * h + gap + legend_h // 2
    sw = pt(6)
    for c in CLASS_RGB:
        if c not in used:
            continue
        d.rectangle([lx, ly - sw // 2, lx + sw, ly + sw // 2], outline=CLASS_RGB[c], width=lw)
        d.text((lx + sw + pt(2), ly), c, font=f_leg, fill='black', anchor='lm')
        lx += sw + pt(2) + int(d.textlength(c, font=f_leg)) + pt(8)
    canvas.save(out, dpi=(DPI, DPI))
    print('fig1', canvas.size, 'panel height', h, 'classes', sorted(used))


def fig5(det_dir, out, name='commons_11'):
    gap = pt(3)
    label_h = pt(9)
    bar_h = pt(4)
    bar_lab = pt(9)
    total_w = int(TEXT_W_IN * DPI)
    ims = [Image.open(os.path.join(det_dir, f'{name}_enc_{k}.png')).convert('RGB')
           for k in ('obj', 'bg')]
    w = (total_w - gap) // 2
    h = int(round(w * ims[0].height / ims[0].width))
    H = label_h + h + pt(3) + bar_h + bar_lab
    canvas = Image.new('RGB', (total_w, H), 'white')
    d = ImageDraw.Draw(canvas)
    f_lab = ImageFont.truetype(FONT_B, pt(8))
    f_txt = ImageFont.truetype(FONT, pt(6.5))
    for i, (im, letter) in enumerate(zip(ims, 'ab')):
        x = i * (w + gap)
        canvas.paste(im.resize((w, h), Image.LANCZOS), (x, label_h))
        d.text((x, label_h // 2), letter, font=f_lab, fill='black', anchor='lm')
    # colour bar: inferno, per-map min to max (the overlays blend it 65:35 with the photo)
    import numpy as np
    import matplotlib
    cmap = matplotlib.colormaps['inferno']
    bw = w
    bx, by = (total_w - bw) // 2, label_h + h + pt(3)
    ramp = (np.array([cmap(t)[:3] for t in np.linspace(0, 1, bw)]) * 255).astype('uint8')
    bar = Image.fromarray(np.repeat(ramp[None], bar_h, axis=0))
    canvas.paste(bar, (bx, by))
    ty = by + bar_h + bar_lab // 2
    d.text((bx, ty), 'lowest', font=f_txt, fill='black', anchor='lm')
    d.text((bx + bw, ty), 'highest', font=f_txt, fill='black', anchor='rm')
    d.text((bx + bw // 2, ty), 'similarity to the reference point (per map)', font=f_txt,
           fill='black', anchor='mm')
    canvas.save(out, dpi=(DPI, DPI))
    print('fig5', canvas.size)


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    fig1(src, os.path.join(dst, 'fig1_examples.png'))
    fig5(src, os.path.join(dst, 'fig5_encoder.png'))
