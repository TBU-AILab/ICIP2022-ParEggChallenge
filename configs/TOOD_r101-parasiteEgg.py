_base_ = 'configs/TOOD_r50-parasiteEgg.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

work_dir = 'models/TOOD_r101/fold-0/'
