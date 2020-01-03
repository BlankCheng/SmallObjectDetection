# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import numpy as np
import torch
import shutil


root = '/home/xiezhihui/data/VOCdevkit/VOC2012/'
new_root = '/home/xiezhihui/data_oversampling_x2/VOCdevkit/VOC2012/'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_sizes = {}
image_size = 300


def create_oversampling_dataset(small_thresh='s', ratio=2):
    assert small_thresh == 's' or small_thresh == 'xs'

    for cls in classes:
        class_sizes[cls] = []
        
    f = open('small_thresh.txt', 'r')
    
    thresh = {}

    for line in f.readlines():
        line = line.split()

        thresh[line[0]] = {'xs': float(line[1]), 's': float(line[2])}

    f.close()


    imageAnnotPairs = []
    trainval = open(os.path.join(root, 'ImageSets', 'Main', 'trainval.txt'))
    for i, trainvalfile in enumerate(trainval.readlines()):
        trainvalfile = trainvalfile.rstrip('\n')

        imageAnnotPairs.append(('{}.jpg'.format(trainvalfile), '{}.xml'.format(trainvalfile)))

        annot = open(os.path.join(root, 'Annotations', '{}.xml'.format(trainvalfile)))
        tree = ET.parse(annot)
        treeroot = tree.getroot()
        size = treeroot.find('size')
        
        ##############################################
        filename = treeroot.find('filename').text
        base, extension = os.path.splitext(filename)
        ##############################################

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
            class_sizes[cls].append((bbox_w * bbox_h, i)) #/ (w * h))

    for cls in class_sizes:
        class_sizes[cls].sort()
        # print(class_sizes[cls], thresh[cls])

        idx_s = idx_xs = 0

        for i, object in enumerate(class_sizes[cls]):
            if object[0] < thresh[cls]['s']:
                idx_s = i

            if object[0] < thresh[cls]['xs']:
                idx_xs = i

        idx_s += 1
        idx_xs += 1

        print("idx_s ", idx_s, " idx_xs ", idx_xs)

        if small_thresh == 's':
            for i in range(idx_s):
                if imageAnnotPairs[class_sizes[cls][i][1]] != None:
                    imageFileName = imageAnnotPairs[class_sizes[cls][i][1]][0]
                    annotFileName = imageAnnotPairs[class_sizes[cls][i][1]][1]
                    print(imageFileName, annotFileName)
                    imageBase, imageExtension = os.path.splitext(imageFileName)
                    annotBase, annotExtension = os.path.splitext(annotFileName)

                    shutil.copy(os.path.join(root, "JPEGImages", imageFileName), os.path.join(new_root, "JPEGImages", imageBase + '_x' + str(int(ratio)) + imageExtension))
                    annot = open(os.path.join(root, 'Annotations', annotFileName))
                    tree = ET.parse(annot)
                    treeroot = tree.getroot()
                    
                    ##############################################
                    fileName = treeroot.find('filename').text
                    base, extension = os.path.splitext(fileName)
                    newFileName = base + '_x' + str(int(ratio)) + extension

                    treeroot.find('filename').text = newFileName

                    tree.write(os.path.join(new_root, "Annotations", base + '_x' + str(int(ratio)) + '.xml'), xml_declaration=False)
                    
                    ##############################################

                imageAnnotPairs[class_sizes[cls][i][1]] = None


def append_new_data_to_trainval():
    file = os.path.join(new_root, "ImageSets", "Main", "trainval_x2.txt")
    file = open(file, 'a')

    annotFiles = os.listdir(os.path.join(new_root, "Annotations"))
    
    for annotFile in annotFiles:
        base, extension = os.path.splitext(annotFile)

        if base[-3] == '_':
            file.write(base + '\n')

    file.close()

if __name__ == '__main__':
    create_oversampling_dataset()
    # append_new_data_to_trainval()
    # print(len(weights))
    # trainvalfiles = open(os.path.join(root, 'ImageSets', 'Main', 'trainval.txt')).read().strip().split()
    # print(trainvalfiles)
    # f = open('small_thresh.txt', 'r')
    
    # thresh = {}

    # for line in f.readlines():
    #     line = line.split()

    #     thresh[line[0]] = {'xs': float(line[1]), 's': float(line[2])}

    # f.close()

    # print(thresh)