dataset_type = 'CocoDataset'
data_root = 'Chula-ParasiteEgg-11/'#main folder with the data, change according to your filesystem
split_root = 'Chula-ParasiteEgg-11/5-fold-train-val/fold-0/' #folder with COCO json file (val and train split)
classes = ('Ascaris lumbricoides', 'Capillaria philippinensis', 'Enterobius vermicularis', 'Fasciolopsis buski',
           'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana', 'Opisthorchis viverrine',
           'Paragonimus spp', 'Taenia spp. egg', 'Trichuris trichiura')
img_norm_cfg = dict(
    mean=[126.09942512, 126.18320368, 111.28689804], std=[62.71579381, 60.8892706, 52.85799401], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280,960), keep_ratio=True), #(height, width)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280,960),#(1060, 800),#(1224,735),#
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=split_root + 'train.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=split_root + 'val.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=split_root + 'val.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
