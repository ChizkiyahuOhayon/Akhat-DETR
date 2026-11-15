# s0_nodh_ffta
#
# SCREENING (validation split, 12 epochs): no dehazing module, FFTA (untied); reference

_base_ = ['./c2_swinl_ffta_v2.py']


# ---- screening on a validation split held out from the training split (test images unused)
train_dataloader = dict(dataset=dict(
    ann_file='annotations_valsplit/instances_trainsub.json',
    data_prefix=dict(img='images/train2017/')))
val_dataloader = dict(dataset=dict(
    ann_file='annotations_valsplit/instances_valsub.json',
    data_prefix=dict(img='images/train2017/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='/path/to/akhat/data/rtts/annotations_valsplit/instances_valsub.json')
test_evaluator = val_evaluator
# shorter schedule for screening: 12 epochs, learning rate divided by 10 after epochs 8 and 11
train_cfg = dict(max_epochs=12)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)]
