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

To visualize bounding box on VOC2007 test, `ssd/JPEGImages/` contains VOC2007 test images:
```
python ssd/viz_bb.py
```

## Performance
Here is performance comparison in our experiment.

| Method | mAP | mAP(small) | mREC | mREC(small)| FPS |
| :-----| :---- | :---- | :-----| :---- | :---- |
| SSD | 77.3 | 52.6 | 93.4 | 83.4 | 23 |
| SSD-ab | 77.7 | 53.2 | 92.9 | 82.3 | **24** |
| SSD-aug | 77.1 | 52.5 | 93.1 | 82.2 | 14 |
| SSD-nlb | 78.0 | 52.9 | 93.8 | 82.8 | **24** |
| FSSD | 77.6 | 56.8 | 92.1 | 82.5 | 21 |
| RFB | **80.9** | **59.8** | **95.6** | **88.2** | 7 |

where **SSD-ab, SSD-aug, SSD-nlb** denote SSD with added anchor boxes,  SSD with small object augmentatio, SSD with non-local blocks respectively.

## Visualization 
Small object augmentation:
![image](ssd/imgs/air.jpg)
![image](ssd/imgs/boat.jpg)
Predicted bounding box of SSD baseline:
![image](ssd/imgs/img1.jpg)
![image](ssd/imgs/img2.jpg)











