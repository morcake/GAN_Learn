import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

class DataLoader():
    def __init__(self, dataset_name, img_res):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, is_pred=False):
        data_type = "train" if not is_testing else "test"
        if is_pred:
            batch_images = ['Test_imgs/' + x for x in os.listdir('Test_imgs/')]
        else:
            path = glob('%s/*' % (self.dataset_name))
            batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imgread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h/4), int(w/4)
            img_hr = scipy.misc.imresize(img, self.img_res)
            img_ls = scipy.misc.imresize(img, (low_h, low_h))

            if not is_testing and np.random.random() < 0.5 :
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(imgs_lr)
            imgs_hr.append(img_hr)
            imgs_lr.append((img_lr))

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_hr, imgs_lr

    def imgread(self, path) :
        return scipy.misc.imread(path, mode='RGB').astype(np.float)