#!/usr/bin/env python
# encoding: utf-8


"""
@author: Jackling Gu
@file: pytorch_util.py
@time: 17-8-14 11:51
"""
import torch
import time
import numpy as np


def train(train_loader, model, criterion, optimizer, epoch):
    # cal time
    start_t = time.time()

    # switch to train mode
    model.train()
    loss_list = []
    correct = 0
    total = 0

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # data
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss_list.append(loss.data[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc
        test_output = model(input_var)
        _, predicted = torch.max(test_output.data, 1)
        total += target_var.size(0)
        correct += predicted.eq(target_var.data).cpu().sum()

        # measure elapsed time
        end = time.time()

    print('epoch', epoch, 'loss', np.mean(loss_list), 'acc', correct / total, 'time cost', end - start_t)
