#!/usr/bin/python3
# coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Data ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()

        with open(os.path.join(cfg.data_root, cfg.test_set,'Test','test.txt'), 'r') as f:
            self.samples = []
            for line in f.readlines():
                self.samples.append(line.strip('\n'))


    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(os.path.join(self.cfg.data_root, self.cfg.test_set, 'Test', 'Image', name + '.jpg'))[:, :, ::-1].astype(np.float32)
        mask  = cv2.imread(os.path.join(self.cfg.data_root, self.cfg.test_set, 'Test', 'GT', name + '.png'), 0).astype(np.float32)        
        shape = mask.shape

        image = self.normalize(image)
        image = self.resize(image)
        image, mask = self.totensor(image, mask)
        return image, mask, shape, name

    def __len__(self):
        return len(self.samples)
