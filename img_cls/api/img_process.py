#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: img_process.py
@time: 2017-06-13 11:13
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from ..utils import getcfg, data_load, get_abspath
from ..pytorch_util import train


class ImageClassification(object):
    def __init__(self):
        self.img_w = getcfg('IMG_WIDTH', 224)
        self.img_h = getcfg('IMG_HEIGHT', 224)
        self.epoch = getcfg('EPOCH', 10)
        self.default_shape = (3, self.img_h, self.img_w)  # channel first

        self.model_name = getcfg('MODEL_NAME', 'ALEXNET')
        self.data_path = getcfg('DATA_DIR', '../data/dog_vs_cat')
        self.model_save_path = get_abspath('../models/{}_{}_model.pt'.format(self.model_name, self.epoch))
        print('MODEL NAME', self.model_name, 'EPOCHS', self.epoch, 'DATA PATH', self.data_path)
        print('MODEL SAVE PATH', self.model_save_path)

        # load data
        self.d_loader, self.d_set = data_load(self.data_path, self.img_w)
        self.num_class = len(self.d_set.classes)
        print('DATASET', self.d_set.classes, self.d_set.class_to_idx)

        # load model
        if os.path.exists(self.model_save_path):
            self.model = torch.load(self.model_save_path)
        else:
            if self.model_name == "LENET":
                from ..model.lenet import LeNet
                self.model = LeNet(self.num_class)
            elif self.model_name == "SIMPLENET":
                from ..model.simple_cnn import SimpleNet
                self.model = SimpleNet(self.num_class)
            elif self.model_name == 'ALEXNET':
                from torchvision.models import alexnet
                self.model = alexnet(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'VGG11':
                from torchvision.models import vgg11
                self.model = vgg11(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'VGG11_BN':
                from torchvision.models import vgg11_bn
                self.model = vgg11_bn(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'VGG16':
                from torchvision.models import vgg16
                self.model = vgg16(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'VGG16_BN':
                from torchvision.models import vgg16_bn
                self.model = vgg16_bn(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'INCEPTIONV3':
                from torchvision.models import inception_v3
                self.model = inception_v3(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'RESNET18':
                from torchvision.models import resnet18
                self.model = resnet18(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'RESNET34':
                from torchvision.models import resnet34
                self.model = resnet34(pretrained=False, num_classes=self.num_class)
            elif self.model_name == 'RESNET50':
                from torchvision.models import resnet50
                self.model = resnet50(pretrained=False, num_classes=self.num_class)
            else:
                import sys
                print('UNSUPPORTED NET')
                sys.exit()

        # prepare param and train net work
        self.default_lr = 0.01
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.default_lr, momentum=0.9, weight_decay=5e-4)
        self.train_nn()

        # save model
        torch.save(self.model, self.model_save_path)

    def train_nn(self):
        # use data augmentation
        for epoch in range(self.epoch):
            self.adj_lr(self.optimizer, epoch)
            train(self.d_loader, self.model, criterion=self.criterion, optimizer=self.optimizer, epoch=epoch)

    def adj_lr(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.default_lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run(self, img_file_path):
        # http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
        preprocess = transforms.Compose([
            transforms.Scale((self.img_w,self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_pil = Image.open(img_file_path)
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_variable = torch.autograd.Variable(img_tensor)
        model_out = self.model(img_variable)
        print('MODEL OUT')
        print('LABEL', self.d_set.classes[model_out.data.numpy().argmax()])
