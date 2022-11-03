_base_ = '../rpn/rpn_r50_caffe_fpn_1x_coco.py'
model = dict(
    rpn_head=dict(
        _delete_=True,
        type='CascadeRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2)),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                ),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                sampling=False,
                with_cls=True,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
        ]),
    train_cfg=dict(rpn=[
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            aux_assigner=None,
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            aux_assigner=None,
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='InstaBoost',
            action_candidate=('normal', 'horizontal', 'skip'),
            action_prob=(1, 0, 0),
            scale=(0.8, 1.2),
            dx=15,
            dy=15,
            theta=(-1, 1),
            color_prob=0.5,
            hflag=False,
            aug_ratio=0.5),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes',]),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)
