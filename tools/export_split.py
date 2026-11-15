"""Write the train/test split used in the paper as plain file lists, without the annotations.

    python tools/export_split.py <rtts_coco_root> <out_dir>

<rtts_coco_root> holds annotations/instances_{train,val}2017.json (the COCO-format copy used for
all experiments; "val2017" is the test split). Writes:
  rtts_train.txt, rtts_test.txt   RTTS file names, one per line, in annotation-file order
  rtts_split_manifest.csv         split, file name, width, height, boxes per class
  rtts_split_md5.txt              md5 of the two annotation files we trained and tested on
The manifest lets anyone check that tools/rtts_voc_to_coco.py rebuilds the same files from
the RESIDE download, without us redistributing RTTS.
"""
import csv
import hashlib
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
md5 = []
for split, tag in (('train', 'train'), ('val', 'test')):
    path = os.path.join(root, 'annotations', f'instances_{split}2017.json')
    md5.append(f'{hashlib.md5(open(path, "rb").read()).hexdigest()}  instances_{split}2017.json')
    d = json.load(open(path))
    names = [c['name'] for c in sorted(d['categories'], key=lambda c: c['id'])]
    counts = {i['id']: [0] * len(names) for i in d['images']}
    for a in d['annotations']:
        counts[a['image_id']][a['category_id']] += 1
    with open(os.path.join(out, f'rtts_{tag}.txt'), 'w') as f:
        for i in d['images']:
            f.write(i['file_name'] + '\n')
            rows.append([tag, i['file_name'], i['width'], i['height']] + counts[i['id']])
    print(tag, len(d['images']), 'images', len(d['annotations']), 'boxes')
with open(os.path.join(out, 'rtts_split_manifest.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['split', 'file_name', 'width', 'height'] + names)
    w.writerows(rows)
with open(os.path.join(out, 'rtts_split_md5.txt'), 'w') as f:
    f.write('\n'.join(md5) + '\n')
print('classes', names)
