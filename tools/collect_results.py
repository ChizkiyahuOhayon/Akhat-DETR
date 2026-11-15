"""Collect the ablation results into one table.

    python tools/collect_results.py /path/to/akhat/work_dirs [--epoch 30] [--json out.json]

The primary metric is the checkpoint at the final epoch (30), fixed before training. RTTS has
no separate validation split, so choosing the best epoch on the test set would inflate the
numbers; the best epoch is printed as a supplementary column only.

A run that was interrupted and resumed writes one log folder per start. All folders are read
and, if an epoch was evaluated twice, the later evaluation is kept.
"""
import argparse
import json
import os

METRICS = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
           'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
SHORT = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
CLASSES = ['person', 'bus', 'car', 'motorbike', 'bicycle']

# work dir prefix -> table row
ARMS = {
    'c2_swinl_ffta_v2':        '(a) Swin-L, no dehazing module',
    'c3_swinl_fpn_dehaze_v2':  '(b) Swin-L, dehazing + standard FPN',
    'c4_swinl_ffta_dehaze_v2': '(c) Swin-L, dehazing + FFTA (full)',
    'r50_a_v2':                '(d) R50, no dehazing module',
    'r50_b_v2':                '(e) R50, dehazing + standard FPN',
    'r50_c_v2':                '(f) R50, dehazing + FFTA (full)',
    # design screening: validation split, 12 epochs (use --epoch 12)
    's0_nodh_ffta':            '[screen] no dehazing, FFTA',
    's1_preft_ffta':           '[screen] pretrained dehazing, fine-tuned, FFTA',
    's2_prefrz_ffta':          '[screen] pretrained dehazing, frozen, FFTA',
    's3_nodh_fpn':             '[screen] no dehazing, FPN',
    's4_nodh_fftaattn':        '[screen] no dehazing, FFTA with attention',
}
# (full, ablated, what the difference measures)
PAIRS = [
    ('c4_swinl_ffta_dehaze_v2', 'c2_swinl_ffta_v2', 'Swin-L dehazing module, c - a'),
    ('c4_swinl_ffta_dehaze_v2', 'c3_swinl_fpn_dehaze_v2', 'Swin-L FFTA vs FPN, c - b'),
    ('r50_c_v2', 'r50_a_v2', 'R50 dehazing module, f - d'),
    ('r50_c_v2', 'r50_b_v2', 'R50 FFTA vs FPN, f - e'),
    ('s1_preft_ffta', 's0_nodh_ffta', '[screen] pretrained, fine-tuned'),
    ('s2_prefrz_ffta', 's0_nodh_ffta', '[screen] pretrained, frozen'),
    ('s0_nodh_ffta', 's3_nodh_fpn', '[screen] FFTA vs FPN'),
    ('s4_nodh_fftaattn', 's0_nodh_ffta', '[screen] attention on'),
]


def read_run(run_dir):
    """{epoch: record} from every scalars.json under run_dir, in start order."""
    recs = {}
    starts = sorted(d for d in os.listdir(run_dir)
                    if os.path.isfile(os.path.join(run_dir, d, 'vis_data', 'scalars.json')))
    for d in starts:
        for line in open(os.path.join(run_dir, d, 'vis_data', 'scalars.json')):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'coco/bbox_mAP' in rec:
                recs[int(rec['step'])] = rec
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work_dirs')
    ap.add_argument('--epoch', type=int, default=30)
    ap.add_argument('--json', help='also write the table to this file')
    args = ap.parse_args()

    out = {}
    print(f'{"row":48s} {"done":>4s} ' + ' '.join(f'{s:>6s}' for s in SHORT)
          + f'   {"best":>5s} {"ep":>3s}')
    for arm, label in ARMS.items():
        d = os.path.join(args.work_dirs, f'{arm}_seed0')
        if not os.path.isdir(d):
            continue
        recs = read_run(d)
        if not recs:
            continue
        final = recs.get(args.epoch)
        best_ep = max(recs, key=lambda e: recs[e]['coco/bbox_mAP'])
        cells = [f'{final[m]:6.3f}' if final else f'{"-":>6s}' for m in METRICS]
        print(f'{label:48s} {max(recs):4d} ' + ' '.join(cells)
              + f'   {recs[best_ep]["coco/bbox_mAP"]:5.3f} {best_ep:3d}')
        out[arm] = dict(
            label=label, epochs_done=max(recs),
            final={s: final[m] for s, m in zip(SHORT, METRICS)} if final else None,
            final_per_class={c: final.get(f'coco/{c}_precision') for c in CLASSES} if final else None,
            best_epoch=best_ep, best_AP=recs[best_ep]['coco/bbox_mAP'],
            curve={e: recs[e]['coco/bbox_mAP'] for e in sorted(recs)})

    print(f'\nDifferences at epoch {args.epoch}:')
    for hi, lo, text in PAIRS:
        a, b = out.get(hi), out.get(lo)
        if a and b and a['final'] and b['final']:
            print(f'  {text:34s} ' + ' '.join(f'{s} {a["final"][s] - b["final"][s]:+.3f}' for s in SHORT))
    print('\nOriginal manuscript, Akhat-DETR Swin-L: 0.606 AP at 30 epochs (Table 1), '
          '0.610 at 800 queries (Fig. 6b); logs not kept.')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
