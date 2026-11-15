"""Rebuild the COCO-format RTTS split used in the paper from the RESIDE download.

    python tools/rtts_voc_to_coco.py <RTTS_dir> <split_dir> <out_dir>

<RTTS_dir>   the RTTS folder from the RESIDE page, with JPEGImages/ and Annotations/ (VOC XML)
<split_dir>  folder with rtts_train.txt and rtts_test.txt (in this archive under splits/)
<out_dir>    receives annotations/instances_{train,val}2017.json and
             images/{train2017,val2017}/ (symbolic links to the RTTS images)

The format matches the files we trained on (checked with tools/verify_rtts_rebuild.py against the
RESIDE download: same images, boxes and classes): classes person, bus, car, motorbike, bicycle with
ids 0-4; bbox = [xmin, ymin, xmax - xmin, ymax - ymin] taken directly from the VOC coordinates;
area = w * h; iscrowd 0; segmentation holds the bbox, as in our files (it is not used, because
the pipeline loads boxes only). The test split is stored as "val2017" because that is the name
the configs expect. Check the result against splits/rtts_split_manifest.csv.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image

CLASSES = ['person', 'bus', 'car', 'motorbike', 'bicycle']


def convert(rtts, names, img_out):
    images, anns = [], []
    os.makedirs(img_out, exist_ok=True)
    for img_id, fname in enumerate(names):
        stem = os.path.splitext(fname)[0]
        root = ET.parse(os.path.join(rtts, 'Annotations', stem + '.xml')).getroot()
        # image size from the image itself: for four RTTS images (SGP_Google_024, JL_Baidu_146,
        # MLS_Bing_517, hv3_106) the <size> field of the XML does not match the file
        with Image.open(os.path.join(rtts, 'JPEGImages', fname)) as im:
            w, h = im.size
        images.append(dict(height=h, width=w, id=img_id, file_name=fname))
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name not in CLASSES:
                raise ValueError(f'{fname}: unknown class {name}')
            b = obj.find('bndbox')
            x1, y1, x2, y2 = (int(float(b.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
            box = [x1, y1, x2 - x1, y2 - y1]
            anns.append(dict(id=len(anns), image_id=img_id, category_id=CLASSES.index(name),
                             segmentation=[box], bbox=box, iscrowd=0, area=box[2] * box[3]))
        src = os.path.abspath(os.path.join(rtts, 'JPEGImages', fname))
        dst = os.path.join(img_out, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    return dict(images=images, annotations=anns,
                categories=[dict(id=i, name=n) for i, n in enumerate(CLASSES)])


def main():
    rtts, split_dir, out = sys.argv[1:4]
    os.makedirs(os.path.join(out, 'annotations'), exist_ok=True)
    for tag, split in (('train', 'train'), ('test', 'val')):
        names = [l.strip() for l in open(os.path.join(split_dir, f'rtts_{tag}.txt')) if l.strip()]
        d = convert(rtts, names, os.path.join(out, 'images', f'{split}2017'))
        json.dump(d, open(os.path.join(out, 'annotations', f'instances_{split}2017.json'), 'w'))
        print(tag, len(d['images']), 'images', len(d['annotations']), 'boxes')


if __name__ == '__main__':
    main()
