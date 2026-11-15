# c3_swinl_fpn_dehaze_v2
#
# REVIEWER'S REQUEST, revised code: identity-initialised dehazing module + standard FPN

_base_ = ['./_base_rtts.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa: E501

model = dict(
    use_dehazing=True,
    dehazing_init='identity',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,  # gradient checkpointing, needed to fit Swin-L on 24GB
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[384, 768, 1536]),
)
