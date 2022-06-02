_base_ = [
    'configs/_base_/models/faster_rcnn_r50_fpn.py',
    'configs/parasiteEgg/parasiteEgg.py',
    'configs/_base_/schedules/schedule_2x.py',
    'configs/_base_/default_runtime.py'
]
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=11)
    ),
    test_cfg=dict(
        rpn=dict(
            #nms_thr=0.5,
            nms_post=1000,
        ),
    ),
)
# activate logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# checkpoint saving
checkpoint_config = dict(interval=1)
# evaluation
evaluation = dict(interval=1)
# working diretory
work_dir = 'models/FasterRCNN_r50/fold-0/'
