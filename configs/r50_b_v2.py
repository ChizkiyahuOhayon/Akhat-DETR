# r50_b_v2
#
# Table 1 R50 block, row b, revised code: identity-initialised dehazing module, standard FPN
#   backbone = ResNet-50 (as in _base_rtts.py)

_base_ = ['./_base_rtts.py']

model = dict(
    use_dehazing=True,
    dehazing_init='identity',
    neck=dict(in_channels=[512, 1024, 2048]),
)
