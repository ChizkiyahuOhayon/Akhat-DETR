# r50_c_v2
#
# Table 1 R50 block, row c, revised code: identity-initialised dehazing module, FFTA with untied
# CCFM weights (full model)
#   backbone = ResNet-50 (as in _base_rtts.py)

_base_ = ['./_base_rtts.py']

model = dict(
    use_dehazing=True,
    dehazing_init='identity',
    neck=dict(
        _delete_=True,
        type='FFTA',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
        use_attention=False,
        ccfm_shared=False),
)
