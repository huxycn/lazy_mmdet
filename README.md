# lazy_mmdet
mmdet-2.25.2 &amp; mmcv-1.6.2 with lazy config

## Install

### Install mmcv-full

```shell
cd mmcv-1.6.2

pip install -r requirements/optional.txt

MMCV_WITH_OPS=1 pip install -e .

# check
python .dev_scripts/check_installation.py
```

### Install mmdet

```shell
cd mmdetection-2.25.2

pip install -r requirements/build.txt

pip install -v -e .

# check
# http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --out-file demo_result.jpg
```
