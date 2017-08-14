#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: utils.py
@time: 2017-06-13 09:30
"""

import os
import ast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# function to get os
def getcfg(name, default, app_=None):
    try:
        if app_ is None:
            from .apps import app
            app_ = app
        return app_.config[name]
    except:
        return default


def getenv(name, default=''):
    if name in os.environ:
        try:
            value = ast.literal_eval(os.environ[name])
        except (SyntaxError, ValueError):
            value = os.environ[name]
    else:
        value = default
    return value


def get_abspath(filename):
    return os.path.normpath(os.path.join(__file__, os.path.pardir, filename))


def data_load(data_dir_path, img_size):
    """

    :param data_dir_path: data home dir
    :return: data_loader and data_set
    """
    data_dir = get_abspath(data_dir_path)
    data_transform = transforms.Compose(
        [
            transforms.Scale(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    d_set = datasets.ImageFolder(root=data_dir, transform=data_transform)
    d_loader = DataLoader(dataset=d_set, batch_size=32, shuffle=True)
    return d_loader, d_set
