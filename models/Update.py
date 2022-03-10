#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import torch
import torch.distributed as dist
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from deep_gradient_compression import DGC
from utils.options import args_parser
import numpy as np
from torch.multiprocessing import Process
import random
from sklearn import metrics







class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        # add
    def train(self, net, control_local, control_global, rank, device_id, size):
        global_weights = copy.deepcopy(net.state_dict())
        net.train()
        # world_size = 1

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        dgc_trainer = DGC(model=net, rank=rank, size=size, device_id=device_id,
                          momentum=self.args.momentum, full_update_layers=[3], persentages=self.args.persentages,
                          itreations=self.args.iters)
        # add
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        # for i in control_global_w:
        #     print(control_global_w[i].size())
        count = 0

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                it = iter * len(self.ldr_train) + batch_idx
                # print(labels.size())
                # print(images.size())
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                dgc_trainer.gradient_update(it)
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

                # add
                local_weight = net.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_local_w[w] - control_local_w[w])
                net.load_state_dict(local_weight)
                count +=1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # add
        new_control_local_w =control_local.state_dict()
        # with open("test.txt", "w") as f:
        #         # for i in range(0, len(c_local)):
        #     for k,v in control_local.state_dict().items():
        #         f.write(f"{k},{v}\n".format(k,v))
        control_delta =copy.deepcopy(control_local_w)
        # add
        net_weights = net.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] +(global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]

        # with open("test.txt", "w") as f:
        #     # for i in range(0, len(c_local)):
        #     for k, v in new_control_local_w.items():
        #         f.write(f"{k},{v}\n".format(k, v))
        # with open("test.txt", "a") as f:
        #         # for i in range(0, len(c_local)):
        #     for k,v in control_global_w.items():
        #         f.write(f"{k},{v}\n".format(k,v))
        # with open("test.txt", "a") as f:
        #             # for i in range(0, len(c_local)):
        #     for k, v in global_weights.items():
        #         f.write(f"{k},{v}\n".format(k, v))
        # with open("test.txt", "a") as f:
        #             # for i in range(0, len(c_local)):
        #     for k, v in net_weights.items():
        #         f.write(f"{k},{v}\n".format(k, v))
            # update new control_local model
            # control_local.load_state_dict(new_control_local_w)

        # add
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, new_control_local_w

    def init_processing(rank, size, fn, backend='gloo'):
        """initiale each process by indicate where the master node is located(by ip and port) and run main function
        :parameter
        rank : int , rank of current process
        size : int, overall number of processes
        fn : function, function to run at each node
        backend : string, name of the backend for distributed operations
        """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend=backend, rank=rank, world_size=size)
        fn(rank, size)

class LocalUpdateF(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

