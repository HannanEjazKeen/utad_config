# The new config inherits a base config to highlight the necessary modification
_base_ = '../yolox/yolox_l_8xb8-300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
        bbox_head=dict(num_classes=1))

# Modify dataset related settings
max_epochs = 15
interval = 5
data_root = '/mnt/public_work/Datasets/External/Object_Detection/Underwater_Detection/UATD/images/'
metainfo = {
    'classes': ('object', ),
    'palette': [
        (220, 20, 60),
    ]
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        dataset=dict(
        data_root=data_root,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline)))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
