"""Hold out a validation set from the RTTS training split for design selection.

    python tools/make_val_split.py <rtts_coco_root> [--n 389] [--seed 0]

Writes <root>/annotations_valsplit/instances_{trainsub,valsub}.json. The images stay in
images/train2017/. Design choices are made on valsub only; the 433 test images are not used
until the final runs.
"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--n', type=int, default=389)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

d = json.load(open(os.path.join(args.root, 'annotations', 'instances_train2017.json')))
ids = sorted(i['id'] for i in d['images'])
val = set(random.Random(args.seed).sample(ids, args.n))
out = os.path.join(args.root, 'annotations_valsplit')
os.makedirs(out, exist_ok=True)
for name, keep in (('trainsub', lambda i: i not in val), ('valsub', lambda i: i in val)):
    sub = dict(d, images=[i for i in d['images'] if keep(i['id'])],
               annotations=[a for a in d['annotations'] if keep(a['image_id'])])
    json.dump(sub, open(os.path.join(out, f'instances_{name}.json'), 'w'))
    print(name, len(sub['images']), 'images', len(sub['annotations']), 'boxes')
