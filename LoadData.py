import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy

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
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h/4), int(w/4)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_hr, imgs_lr