#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import torch.distributed as dist
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import LocalUpdateF
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

from torch.multiprocessing import Process
from deep_gradient_compression import DGC
import json

# __name__是内置的变量，在执行当前文件（main_fed.py）时，默认值为__main__
# 但是如果其他.py文件import当前文件（main_fed.py）时，在其他文件中执行main_fed.py中的__name__,此时main_fed.py中的__name__默认值为文件名main_fed.py
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = 0
    device_id = rank
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=rank, world_size=args.world_size)



    # if torch.cuda.is_available() and args.gpu != -1 else 'cpu'


    # load dataset and split users
    if args.dataset == 'mnist':
        # ToTensor():归一数据到（0,1），Normalize（）：（date-0.1307）/0.3081,将数据分布到（-1， 1）
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if trans_mnist is not None:
            print(1)
            print(trans_mnist)
        # 测试（60000）和训练集（10000）
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        # Noniid数据
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    # print('df ',img_size) [1,28,28]

    # build model
    # print(args.model)
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)

    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            # print('x取值',x)
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        # add
        control_global = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # 设置为训练模型
    net_glob.train()
    print(net_glob)

    control_weights =control_global.state_dict()
    # copy weights
    # 初始化全局权重
    w_glob = net_glob.state_dict()
    c_glob = copy.deepcopy(net_glob.state_dict())

    # print(w_glob)
    # training
    loss_train = []
    accuracy = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    count = 0, 0
    test_acc_list = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        # add



    else:
        # 初始化本地权重
        c_local = [MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device) for i in
                   range(args.num_users)]
        for net in c_local:
            net.load_state_dict(control_weights)
        delta_c = copy.deepcopy(net_glob.state_dict())
        # delta_x = copy.deepcopy(net_glob.state_dict())
        # with open("test.txt", "w") as f:
        #     for i in range(0, len(c_local)):
        #         for k,v in c_local[i].state_dict().items():
        #             f.write(f"{k},{v}\n".format(k,v))
        # with open("test.txt", "a") as f:
        #     for i in range(0, len(c_local)):
        #         for k, v in w_locals[i].items():
        #             f.write(f"{k},{v}\n".format(k, v))
            # add 初始化变化量

        # print("why?")



    for iter in range(args.epochs):
        # 初始换控制变量
        for i in delta_c:
            delta_c[i] = 0.0
        # for i in delta_x:
        #     delta_x[i] = 0.0

        loss_locals = []

        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)

        # 每次随机十位幸运观众
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # momentum法SGD
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, local_delta_c, local_delta, control_local_w= local.train(net=copy.deepcopy(net_glob).to(args.device), control_local
                = c_local[idx], control_global=control_global, rank=rank, device_id=device_id, size=args.world_size)

            # add
            if iter != 0:
                c_local[idx].load_state_dict(control_local_w)


            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            # add
            loss_locals.append(copy.deepcopy(loss))

            # add
            for i in delta_c:
                if iter != 0:
                    delta_c[i] += w[i]
                else:
                    delta_c[i] += local_delta_c[i]
                    # delta_x[i] += local_delta[i]

        # add
        # update the delta C
        for i in delta_c:
            delta_c[i] /= m
            # delta_x[i] /= m

        # update global weights
        w_glob = FedAvg(w_locals)
        # add 更新全局c，w
        # w_glob = net_glob.state_dict()
        control_global_w = control_global.state_dict()
        for i in control_global_w:
            if iter !=0:
            #     w_glob[i] = delta_x[i]
            # else:
            #     w_glob[i] += delta_x[i]
                control_global_w[i] += (m / args.num_users) * delta_c[i]


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # add
        control_global.load_state_dict(control_global_w)


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        accuracy.append(acc_test)


        # add
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            torch.cuda.empty_cache()

        # net_glob.eval()

        # print("Training accuracy: {:.2f}".format(acc_train))
        # print("Testing accuracy: {:.2f}".format(acc_test))

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
    # Fedavg
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_globF = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_globF = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_globF = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_globF)
    net_globF.train()

    # copy weights
    w_globF = net_globF.state_dict()

    # training
    loss_trainF = []
    accuracyF = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_localsF = [w_globF for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_localsF = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            localF = LocalUpdateF(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = localF.train(net=copy.deepcopy(net_globF).to(args.device))
            if args.all_clients:
                w_localsF[idx] = copy.deepcopy(w)
            else:
                w_localsF.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_globF = FedAvg(w_localsF)

        # copy weight to net_globF
        net_globF.load_state_dict(w_globF)

        # print loss
        loss_avgF = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avgF))
        loss_trainF.append(loss_avgF)

        acc_test, loss_test = test_img(net_globF, dataset_test, args)
        accuracyF.append(acc_test)



    # plot loss curve
    plt.figure()
    print(loss_train, loss_trainF)
    plt.plot(range(len(loss_train)), loss_train, label='Scaffold', zorder=2)
    plt.plot(range(len(loss_trainF)), loss_trainF, 'r', label='FedAvg',zorder=1)
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig('./save/fed_{}_{}_{}_{}_iid{}.png'.format(args.dataset, args.model, args.epochs, 'train_loss', args.iid))


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    # plot loss curve
    plt.figure()
    # plt.plot((np.arange(1, len(accuracy)), 1), accuracy, 'r')
    plt.plot(range(len(accuracy)), accuracy, label='Scaffold', zorder=2)
    plt.plot(range(len(accuracyF)), accuracyF, 'r', label='FedAvg', zorder=1)
    plt.ylabel('test_acc')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig('./save/fed_{}_{}_{}_{}_iid{}.png'.format(args.dataset, args.model, args.epochs, 'acc_test', args.iid))

