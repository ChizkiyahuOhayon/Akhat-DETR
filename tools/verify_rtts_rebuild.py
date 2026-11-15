"""Check that annotation files rebuilt by rtts_voc_to_coco.py match the ones used in the paper.

    python tools/verify_rtts_rebuild.py <rebuilt_root> <reference_root>

Both roots hold annotations/instances_{train,val}2017.json. Images are compared by file name,
size and the multiset of (class, box) pairs; ids and ordering are ignored. If
<reference_root>/images exists, images are also compared pixel by pixel (the PNG encoding of our
copy differs from the RESIDE files, so the bytes differ although the pixels are identical).
"""
import collections
import json
import os
import sys

import numpy as np
from PIL import Image


def load(root, split):
    d = json.load(open(os.path.join(root, 'annotations', f'instances_{split}2017.json')))
    cats = {c['id']: c['name'] for c in d['categories']}
    imgs = {i['id']: i for i in d['images']}
    boxes = collections.defaultdict(list)
    for a in d['annotations']:
        boxes[imgs[a['image_id']]['file_name']].append(
            (cats[a['category_id']], tuple(round(float(v), 2) for v in a['bbox'])))
    return {i['file_name']: ((i['width'], i['height']), sorted(boxes[i['file_name']]))
            for i in imgs.values()}


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im)


def main():
    new, ref = sys.argv[1], sys.argv[2]
    ok = True
    for split in ('train', 'val'):
        a, b = load(new, split), load(ref, split)
        if set(a) != set(b):
            ok = False
            print(split, 'file sets differ:', len(set(a) - set(b)), 'only rebuilt,',
                  len(set(b) - set(a)), 'only reference')
        size_diff = [f for f in a.keys() & b.keys() if a[f][0] != b[f][0]]
        box_diff = [f for f in a.keys() & b.keys() if a[f][1] != b[f][1]]
        n_box = sum(len(v[1]) for v in a.values())
        print(f'{split}: {len(a)} images, {n_box} boxes; size mismatches {len(size_diff)}, '
              f'box mismatches {len(box_diff)}')
        for f in box_diff[:3]:
            print('  e.g.', f, '\n   rebuilt  ', a[f][1][:4], '\n   reference', b[f][1][:4])
        ok &= not size_diff and not box_diff
        img_dir_new = os.path.join(new, 'images', f'{split}2017')
        img_dir_ref = os.path.join(ref, 'images', f'{split}2017')
        if os.path.isdir(img_dir_ref):
            bad = [f for f in sorted(a.keys() & b.keys())
                   if not np.array_equal(pixels(os.path.join(img_dir_new, f)),
                                         pixels(os.path.join(img_dir_ref, f)))]
            print(f'  pixels differ for {len(bad)} images', bad[:3])
            ok &= not bad
    print('REBUILD_MATCHES' if ok else 'REBUILD_DIFFERS')


if __name__ == '__main__':
    main()
