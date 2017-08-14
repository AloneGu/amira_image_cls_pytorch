#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process.py
@time: 2017-06-13 11:13
"""
import torch
from ..utils import getcfg, data_load, get_abspath
from ..model.lenet import LeNet
from ..pytorch_util import train


class ImageClassification(object):
    def __init__(self):
        self.img_w = getcfg('IMG_WIDTH', 224)
        self.img_h = getcfg('IMG_HEIGHT', 224)
        self.epoch = getcfg('EPOCH', 10)
        self.default_shape = (3, self.img_h, self.img_w)  # channel first

        self.model_name = getcfg('MODEL_NAME', 'ALEXNET')
        self.data_path = getcfg('DATA_DIR', '../data/dog_vs_cat')
        self.model_save_path = get_abspath('../models/{}_{}_model.h5'.format(self.model_name, self.epoch))
        print('MODEL NAME', self.model_name, 'EPOCHS', self.epoch, 'DATA PATH', self.data_path)
        print('MODEL SAVE PATH', self.model_save_path)

        # load data
        self.d_loader, self.d_set = data_load(self.data_path, self.img_w)
        self.num_class = len(self.d_set.classes)

        # load model
        self.model = LeNet(self.num_class)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        self.train_nn()

    def train_nn(self):
        # use data augmentation
        for epoch in range(self.epoch):
            train(self.d_loader, self.model, criterion=self.criterion, optimizer=self.optimizer, epoch=epoch)

    def run(self, img_file_path):
        pass
