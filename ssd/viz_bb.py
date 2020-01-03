import pickle
from imp import reload

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

reload(sys)
# sys.setdefaultencoding('utf8')

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'b', 'w']
root = './'
prediction_dir = 'predictions'
prediction_pkl = 'predictions_rfb.pkl'
method = 'rfb'

if __name__ == '__main__':

    image_dir = os.listdir(os.path.join(root, 'JPEGImages'))
    annots = pickle.load(open(os.path.join(root, prediction_dir, prediction_pkl), "rb"))
    print(annots)
    for file in image_dir:
        annot = annots.get(file[:-4], -1)
        if annot == -1:
            continue  # not test image
        img = mpimg.imread(os.path.join(root, 'JPEGImages', file))

        plt.figure()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        for i in range(len(annot)):
            currentAxis = plt.gca()
            obj, color = annot[i], COLORS[i % len(COLORS)]
            l, t, r, b = obj['bbox']
            name = obj['name']
            currentAxis.add_patch(plt.Rectangle((l, t), r - l, b - t, linewidth=2, edgecolor=color, facecolor='none'))
            currentAxis.text(l, t,
                    '{:s} {:.3f}'.format(name, float(obj['confidence'])),
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=10, color='black')


        save_path = os.path.join(root, 'JPEGImages', '{}_{}.png'.format(file[:-4], method))
        plt.savefig(os.path.join(save_path), bbox_inches='tight', pad_inches=0.0)
        plt.show()


