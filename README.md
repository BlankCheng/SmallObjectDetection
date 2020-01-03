# SmallObjectDetection
Final project of EI339

## TODO
+ ~~计算小物体的标签~~ 具体标准未统一
+ ~~增加attention~~
+ 增加anchor box数量
+ 增加反卷积层
+ 增加IoU loss
+ Data augmentation


## Original
mAP 77.43

## Attention
+ 在extras每一个conv前添加non local block， mAP 77.64。
+ 在multibox每一个conv前添加non local block, mAP 77.98。

## Deconvolution
到20w iterations，出现严重过拟合。
+ 在vgg16上添加6个deconv， 41w iterations，mAP 55.13。
+ 在vgg16上添加6个deconv， 15w iterations, mAP 54.67。
