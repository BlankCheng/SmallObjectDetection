# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, nonlocals, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)  # 8732 * 4
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.nonlocals = nn.ModuleList(nonlocals)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch, num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors, 4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list() # 连到输出层的每层数据, 之后要经过multibox layer
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        # print("VGG:")
        for k in range(23):
            x = self.vgg[k](x)
            # print(x.size())

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # print("VGG_FC:")
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # print("Extra:")
        for k, v in enumerate(self.extras):
            # x = self.nonlocals[k](x)
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        i = 0
        for (x, l, c) in zip(sources, self.loc, self.conf):
            x = self.nonlocals[i](x)
            i += 1
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # print("loc:")
        # for i in loc:
        #     print(i.size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # n * B * ? * ? * 4k -> B * 34928
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1) # n * B * ? * ? * 21k -> B * 183372
        # print("final loc:")
        # print(loc.size())
        # print("final conf:")
        # print(conf.size())

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),  # B * 8732 * 4, 每张图给了8732个proposal
                conf.view(conf.size(0), -1, self.num_classes),  # B * 8732 * 21
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    # 13 * conv，2 * new_conv
    layers = []
    in_channels = i # i = 3, rgb channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # ceil_mode=True表示奇数除法取向上取整
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) # padding=1, (a - 3 + 2 * padding)/stride + 1 = a,所以边长不变
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # B * 1024 * 38 * 38
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # B * 1024 * 19 * 19
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    # 8 * conv
    # input: B * 1024 * 19 * 19
    # output: B * 256 * 1 * 1

    layers = []
    in_channels = i  # i = 1024
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  # S表示shrink, 即stride=2,feature map缩小
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]  # flag表示交替取kernel_size=1或3
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    # 根据构建好的vgg layer和extra layer，构造multibox layer
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,  # e.g. B * 1024 * 19 * 19 -> B * (4*k) * 19 * 19
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,  # e.g. B * 1024 * 19 * 19 -> B * (num_classes*k) * 19 * 19
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def add_nonlocals():

    layers = []
    for k, v in enumerate(nonlocals['300']):
        layers += [NonLocalBlock(v, v, 1024)]

    return layers


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [10, 10, 10, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
nonlocals = {
    '300': [512, 1024, 512, 256, 256, 256],
    '512' : [],
}


def build_ssd(phase, size = 300, num_classes = 21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    nonlocals_ = add_nonlocals()
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, nonlocals_, head_, num_classes)


if __name__ == '__main__':
    batch_data = torch.randn((8, 3, 300, 300))
    model = build_ssd('train', 300, 21)
    output = model(batch_data)

