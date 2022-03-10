#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # set:删除重复值 np.random.choice：从all_idxs中随机选择num_items个样本
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        #set：set(all_idxs)-dict_user[i]:求差集
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    num_shards:分成200块
    num_imgs:一块有300张图片
    dict_users:存储
    idxs:图片索引
    labels：
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.targets
    # typelab = type(labels)
    # 将Tensor数据转换成numpy操作，targets干嘛的？
    labels = dataset.targets.numpy()

    # sort labels
    # np.vatack(a,b):将数据b放在a下边传成一个数组
    idxs_labels = np.vstack((idxs, labels))
    # argsort(x)：将x数组按大小排序，返回排序后原始x数值下标
    # 对idxs_labels中labels进行排序，idxs对应改变
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs中存储排好序的labels下标
    idxs = idxs_labels[0,:]

    # divide and assign 每个client抽样600张
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # np.concatenate(axis=0/1)数组按照行/列拼接
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)