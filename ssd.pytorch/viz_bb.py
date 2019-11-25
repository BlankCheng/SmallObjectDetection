import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'b', 'w']

if __name__ == '__main__':
    root = '../VOCdevkit'
    image_dir = os.listdir(os.path.join(root, 'JPEGImages'))
    annots = pickle.load(open(os.path.join(root, 'annotations_cache', 'annots.pkl'), "rb"))
    for file in image_dir:
        img = mpimg.imread(os.path.join(root, 'JPEGImages', file))
        annot = annots.get(file[:-4], -1)
        if annot == -1:
            continue

        plt.figure()
        plt.imshow(img)
        for i in range(len(annot)):
            currentAxis = plt.gca()
            obj, color = annot[i], COLORS[i]
            l, t, r, b = obj['bbox']
            currentAxis.add_patch(plt.Rectangle((l, t), r - l, b - t, linewidth=1, edgecolor=color, facecolor='none'))

        save_path = os.path.join(root, 'JPEGImages', '{}_eval.png'.format(file[:-4]))
        plt.savefig(os.path.join(save_path))
        plt.show()


