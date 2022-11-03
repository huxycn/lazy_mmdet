# model settings
from lazyconfig import LazyCall as L
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.models.backbones.resnet import ResNet
from mmcv.cnn import PretrainedInit
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.core.anchor.anchor_generator import AnchorGenerator
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner
from mmcv.cnn.bricks.norm import NormBuilder
from torch import nn
from mmcv.ops.nms import nms

model = L(RetinaNet)(
    backbone=L(ResNet)(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_builder=L(NormBuilder)(norm_type=nn.BatchNorm2d, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_method=L(PretrainedInit)(checkpoint='torchvision://resnet50')),
    neck=L(FPN)(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=L(RetinaHead)(
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=L(AnchorGenerator)(
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=L(DeltaXYWHBBoxCoder)(
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=L(FocalLoss)(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=L(L1Loss)(loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=L(MaxIoUAssigner)(
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)
