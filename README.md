# SmallObjectDetection

## Introduction
This is our final project of EI339 on small object detection. Our implementation references [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch). Several modifications on SSD are implemented:
+ Add more anchor boxes
+ Augment small objects with cut-and-paste
+ Add deconvolution layers (not work now)
+ Add non-local block
+ [Feature-Fused SSD](https://arxiv.org/abs/1709.05054)
+ [Receptive Field Block](https://arxiv.org/abs/1711.07767)



## Usage
To train ssd with added anchor boxes and non-local block:
```
python ssd/train.py
```
To train fssd or rfb:
```
python ssds.pytorch/train.py --cfg=ssds.pytorch/experiments/cfgs/fssd_vgg16_train_voc.yml
python ssds.pytorch/train.py --cfg=ssds.pytorch/experiments/cfgs/rfb_resnet50_train_voc.yml
```

To evaluate ssd with added anchor boxes and non-local block:
```
python ssd/eval.py
```
Evaluation on small objects:
```
python ssd/eval_small.py
```


## Performance
+ 在extras每一个conv前添加non local block， mAP 77.64。
+ 在multibox每一个conv前添加non local block, mAP 77.98。

