import os.path as osp
import numpy as np

from PIL import Image
from skimage.io import imread

from queue import Queue
from threading import Thread


class CUB():
    def __init__(self, root, len_limit=None):
        self.root = root
        img_txt_file = open(osp.join(self.root, 'images.txt'))
        label_txt_file = open(osp.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(osp.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.test_img = [osp.join(self.root, 'images', test_file) for test_file in test_file_list]
        self.test_label = [x for i, x in zip(train_test_list, label_list) if not i]
        if len_limit is not None:
            self.test_img = self.test_img[:len_limit]
            self.test_label = self.test_label[:len_limit]

    def __getitem__(self, idx):
        img, target = imread(self.test_img[idx]), self.test_label[idx]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB').resize((600, 600), Image.BILINEAR)
        img = np.array(img)
        img = center_crop(img, 448, 448)
        img = img.astype(np.float32) / 255.0
        img = img - np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        img = img / np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        return img, target

    def __len__(self):
        return len(self.test_label)


def center_crop(x, height, width):
    start_height = int(round((x.shape[0] - height) / 2))
    start_width = int(round((x.shape[1] - width) / 2))
    return x[start_height:start_height + height, start_width:start_width + width, :]
