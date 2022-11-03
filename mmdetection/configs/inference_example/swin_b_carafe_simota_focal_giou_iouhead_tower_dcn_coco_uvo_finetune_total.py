_base_ = '../rpn/rpn_r50_caffe_fpn_1x_coco.py'
pretrained = None

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        load_like_mmseg=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
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
                dcn_on_last_conv=True,
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2)),
                use_tower_convs=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                ),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                dcn_on_last_conv=True,
                bridged_feature=False,
                sampling=False,
                with_cls=True,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                use_tower_convs=True,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                )
        ]),
    train_cfg=dict(rpn=[
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            #assigner=dict(type='ATSSAssigner', topk=9),
            #aux_assigner=dict(type='ATSSAssigner', topk=21),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            #assigner=dict(type='ATSSAssigner', topk=9),
            #aux_assigner=dict(type='ATSSAssigner', topk=21),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0)))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
data_root = 'data/uvo/'
uvo_frame_train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root='data/uvo/',
            ann_file='annotations/UVO_frame_train.json',
            img_prefix='uvo_videos_sparse_frames/',
            pipeline=train_pipeline,
        )
)
uvo_frame_val=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root='data/uvo/',
            ann_file='annotations/UVO_frame_val.json',
            img_prefix='uvo_videos_sparse_frames/',
            pipeline=train_pipeline,
        )
)
uvo_dense_train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root='data/uvo/',
            ann_file='annotations/UVO_dense_train_coco_format.json',
            img_prefix='uvo_videos_dense_frames/',
            pipeline=train_pipeline,
        )
)
uvo_dense_val=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root='data/uvo/',
            ann_file='annotations/UVO_dense_val_coco_format.json',
            img_prefix='uvo_videos_dense_frames/',
            pipeline=train_pipeline,
        )
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        uvo_frame_train,
        uvo_frame_val,
        uvo_dense_train,
        uvo_dense_val,
        ],
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017_clsagn.json',
        img_prefix='data/coco/val2017/',
        ),
    test=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017_clsagn.json',
        img_prefix='data/coco/val2017/',
        ))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00001 / 2,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

#load_from = '/tmp-network/user/ydu/mmdetection/work_dirs/swin_b_carafe_simota_focal_giou_iouhead_tower_dcn_object365/epoch_18.pth'
lr_config = dict(warmup_iters=1000, step=[3, 5])
runner = dict(max_epochs=6)

