"""Shared base for every Akhat-DETR ablation arm.

Derived from mmdetection v3.3.0 `configs/ddq/ddq-detr-4scale_swinl_8xb2-30e_coco.py`,
which the manuscript's hyperparameters match item for item:
  lr 2e-4, weight_decay 0.05, backbone lr_mult 0.05 -> 1e-5 (paper: "backbone's
  to 0.00001"), dqs IoU threshold 0.8 (paper Table 2), loss 5*L1 + 1*focal +
  2*GIoU (paper Eq. 4), 30 epochs, base batch size 16.

Deviations from that config, applied IDENTICALLY to all arms so the ablation
stays controlled:
  * num_queries 900 -> 800            (paper Fig. 7: 800 is optimal)
  * num_classes 80 -> 5               (RTTS)
  * neck num_outs 4 -> 3, encoder/decoder num_levels 4 -> 3
                                      (paper Fig. 6 uses exactly S1/S2/S3)
  * pad_size_divisor 1 -> 32          (FFTA's CCFM upsamples by scale_factor=2
                                       and concatenates, so each level must be
                                       an exact 2x of the next; odd sizes crash)

  * batch 2/GPU x accumulative_counts 8 = effective 16 on a single GPU

The neck defined here is the standard FPN baseline. FFTA arms replace it with
`_delete_=True`.

This config is deliberately SELF-CONTAINED: it inherits nothing. The `mmdet::`
cross-package inheritance used by the stock configs resolves through mmdet's
`.mim` directory, which an editable (`pip install -e .`) install never creates,
so it breaks. Inlining the two base files also makes this artifact portable --
it can be dropped next to any mmdet 3.3.0 checkout and just run.
"""
custom_imports = dict(imports=['akhat'], allow_failed_imports=False)

# ------------------------------------------------------- runtime (inlined
# from mmdetection/configs/_base_/default_runtime.py)
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

dataset_type = 'CocoDataset'
backend_args = None

data_root = 'data/rtts/'  # set to your copy of the data
classes = ('person', 'bus', 'car', 'motorbike', 'bicycle')
num_classes = 5
num_queries = 800
num_levels = 3

model = dict(
    type='AkhatDETR',
    use_dehazing=False,  # overridden per arm
    num_queries=num_queries,
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # Standard FPN baseline (Lin et al.). GroupNorm is enabled deliberately:
    # FFTA normalises internally (BatchNorm in ConvNormLayer/RepVggBlock) and
    # stock DDQ-DETR's ChannelMapper uses GN, so a norm-free FPN would be a
    # handicapped baseline and would bias the comparison in our favour.
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=num_levels,
        norm_cfg=dict(type='GN', num_groups=32)),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DDQDETRHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# ----------------------------------------------------------------- pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ------------------------------------------------------------------- dataset
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise=True)
test_evaluator = val_evaluator

# ------------------------------------------------------------------ schedule
max_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    # 2 (batch) x 8 (accumulation) = 16, the paper's effective batch size
    accumulative_counts=8,
    # paper: transformer & dehazing lr 2e-4, backbone 1e-5 == 2e-4 * 0.05
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 26],
        gamma=0.1)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # max_keep_ckpts=1: Swin-L checkpoints are ~2.5 GB and the data disk is 50 GB
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Fixed across every arm so the comparison is controlled. Override on the CLI
# with --cfg-options randomness.seed=... for the multi-seed runs.
randomness = dict(seed=0, deterministic=False)

auto_scale_lr = dict(enable=False, base_batch_size=16)
