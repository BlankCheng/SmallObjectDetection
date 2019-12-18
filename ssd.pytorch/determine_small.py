# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os

root = '/media/data_2/VOCdevkit/VOC2007/'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_sizes = {}
class_small_thresh = {}
image_size = 300

if __name__ == '__main__':

    for cls in classes:
        class_sizes[cls] = []
        class_small_thresh[cls] = {}

    testfiles = open(os.path.join(root, 'ImageSets', 'Main', 'test.txt')).read().strip().split()
    for testfile in testfiles:
        annot = open(os.path.join(root, 'Annotations', '{}.xml'.format(testfile)))
        tree = ET.parse(annot)
        treeroot = tree.getroot()
        size = treeroot.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in treeroot.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')

            # 计算bb放缩到300*300图片后的area
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox_w = (b[1] - b[0])  # * image_size / w
            bbox_h = (b[3] - b[2])  # * image_size / h
            class_sizes[cls].append((bbox_w * bbox_h) ) #/ (w * h))

    f = open('small_thresh.txt', 'w')
    for cls in class_sizes:
        class_sizes[cls].sort()
        pos_xs = round(len(class_sizes[cls]) * 0.1)
        pos_s = round(len(class_sizes[cls]) * 0.3)
        class_small_thresh[cls]['xs'] = class_sizes[cls][pos_xs]
        class_small_thresh[cls]['s'] = class_sizes[cls][pos_s]

        print(cls, round(class_small_thresh[cls]['xs'], 3), round(class_small_thresh[cls]['s'], 3))
        f.write("{}    {:.3f}    {:.3f}\n".format(cls, class_small_thresh[cls]['xs'], class_small_thresh[cls]['s']))
    f.close()







