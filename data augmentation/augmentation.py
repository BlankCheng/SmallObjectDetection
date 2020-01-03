# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import  cv2
from PIL import Image
import numpy as np
import random

root = 'VOC2007'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_sizes = {}
class_small_thresh = {}
image_size = 300
cnt = 0


class2smallObj = dict()

f = open('small_thresh.txt', 'r').read().strip().split()
for i in range(0, len(f), 3):
    class2smallObj[f[i]] = float(f[i+2])


def randomPos(w, h):
    return (random.randint(0,w-1), random.randint(0,h-1))


# bbox xmin ymin xmax ymax


def isOverlapped(bbox1, bbox2):
    xmin1, xmax1, ymin1, ymax1 = bbox1
    xmin2, xmax2, ymin2, ymax2 = bbox2
    
    if xmin1 < xmin2 < xmax1 and ymin1 < ymin2 < ymax1: 
        return True
    elif xmin1 < xmax2 < xmax1 and ymin1 < ymax2 < ymax1:
        return True
    elif xmin1 < xmin2 < xmax1 and ymin1 < ymax2 < ymax1:
        return True
    elif xmin1 < xmax2 < xmax1 and ymin1 < ymin2 < ymax1:
        return True
    
    return False


def isOverlappedSet(bbox, boxes):
    for box in boxes:
        print(bbox, box)
        if isOverlapped(bbox, box) or isOverlapped(box,bbox):
            return True
    return False


if __name__ == '__main__':

    for cls in classes:
        class_sizes[cls] = []
        class_small_thresh[cls] = {}

    testfiles = open(os.path.join('files.txt')).read().strip().split()
    for testfile in testfiles:
        #print(testfile)
        

        
        #origin = cv2.imread(os.path.join(root, 'JPEGImages','{0:06s}.jpg'.format(testfile)))
        #mask = cv2.imread(os.path.join(root, 'Segmentation','{0:06d}.png'.format(testfile)))
        origin = cv2.imread(os.path.join(root, 'JPEGImages','{}.jpg'.format(testfile)))
        mask = Image.open(os.path.join(root, 'SegmentationClass','{}.png'.format(testfile)))
        mask = np.array(mask)
        annot = open(os.path.join(root, 'Annotations', '{}.xml'.format(testfile)))
        tree = ET.parse(annot)
        treeroot = tree.getroot()
        size = treeroot.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        boxes_tmp = []
        for obj in treeroot.iter('object'):
            xmlbox = obj.find('bndbox')
            boxes_tmp.append((float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)))
        objs = list(treeroot.iter('object'))
        print(len(objs))
        count = 1
        for obj in objs:
            print(count)
            count+=1
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

            #center = ((b[1] + b[0])/2, (b[3] + b[2])/2)

            print(bbox_h * bbox_w, class2smallObj[cls])
            # 如果对象小于阈值， 则进行复制粘贴
            if bbox_h * bbox_w < class2smallObj[cls]:
                boxes = boxes_tmp.copy()
                origin_ = origin.copy()
                mask_ = mask[int(b[2]):int(b[3]),int(b[0]):int(b[1])]
                small = origin_[int(b[2]):int(b[3]),int(b[0]):int(b[1])]
                #small = origin[b[2]:b[3],b[0]:b[1]]
                
                small_h, small_w = mask_.shape
                #small_h, small_w = (b[3]-b[2], b[1]-b[0])
                refer = mask_[int(small_h / 2)][int(small_w / 2)]
                print("refer: ", refer)


                copy_xmin, copy_ymin = randomPos(w - bbox_w, h - bbox_h)
                copy_xmax, copy_ymax = (copy_xmin + bbox_w, copy_ymin + bbox_h)
                for i in range(100):
                    print(len(boxes))
                    if not isOverlappedSet((copy_xmin, copy_xmax, copy_ymin, copy_ymax), boxes):
                        for i in range(small_w):
                            for j in range(small_h):
                                if(mask_[j][i] == refer):
                                    #origin[copy_ymin+j][copy_xmin+i] = origin[b[2]+j][b[0]+i]
                                    origin_[copy_ymin+j][copy_xmin+i] = small[j][i]

                        boxes.append((copy_xmin, copy_xmax, copy_ymin, copy_ymax))
                        copy_xmin, copy_ymin = randomPos(w - bbox_w, h - bbox_h)
                        copy_xmax, copy_ymax = (copy_xmin + bbox_w, copy_ymin + bbox_h)
                        continue
                if len(boxes_tmp) == len(boxes):
                    continue
                annot_copy = open(os.path.join(root, 'Annotations', '{}.xml'.format(testfile)))
                tree_copy = ET.parse(annot_copy)
                #tree_copy = tree
                root_copy = tree_copy.getroot()
                cnt_obj = len(list(treeroot.iter('object')))
                print(list(treeroot.iter('object')))
                
                for i in range(cnt_obj, len(boxes)):
                    Obj = ET.Element('object')
                    name = ET.Element('name')
                    name.text = cls
                    pose = ET.Element('pose')
                    pose.text = obj.find('pose').text
                    truncated = ET.Element('truncated')
                    truncated.text = obj.find('truncated').text
                    difficult = ET.Element('difficult')
                    difficult.text = obj.find('difficult').text
                    bndbox = ET.Element('bndbox')
                    
                    Obj.append(name)
                    Obj.append(pose)
                    Obj.append(truncated)
                    Obj.append(difficult)
                    Obj.append(bndbox)


                    xmin_element = ET.Element('xmin')
                    xmin_element.text = str(int(boxes[i][0]))
                    xmax_element = ET.Element('xmax')
                    xmax_element.text = str(int(boxes[i][1]))
                    ymin_element = ET.Element('ymin')
                    ymin_element.text = str(int(boxes[i][2]))
                    ymax_element = ET.Element('ymax')
                    ymax_element.text = str(int(boxes[i][3]))
                    bndbox.append(xmin_element)
                    bndbox.append(ymin_element)
                    bndbox.append(xmax_element)
                    bndbox.append(ymax_element)

                    root_copy.append(Obj)

                root_copy.find('filename').text = '{}_aug_{}.jpg'.format(testfile, cnt)
                tree_copy.write('new_annos\{}_aug_{}.xml'.format(testfile, cnt))
                cv2.imwrite('new_jpg\{}_aug_{}.jpg'.format(testfile, cnt),origin_)
                print(testfile, cnt)
                cnt+=1
                


            
            #cv2.imshow('img',origin)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

                                    
                                

                        








        '''
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
        '''




