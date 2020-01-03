import os


root = '/home/xiezhihui/data_aug/VOCdevkit/VOC2012/'


if __name__ == "__main__":
    f = open('small_in_trainval_2012.txt', 'r')
    smallImages = f.readlines()
    f.close()

    trainval_aug = open(os.path.join(root, "ImageSets", "Main", "trainval_aug.txt"), 'a')

    images = os.listdir(os.path.join(root, "JPEGImages"))

    for image in images:
        if image.find("aug") > 0 and image.split('_aug')[0] + '\n' in smallImages:
            print(image.split('_aug')[0])

            trainval_aug.write(image[:-4] + '\n')


    trainval_aug.close()