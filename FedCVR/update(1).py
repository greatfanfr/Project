#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random

import torch
import math
from torch import optim
from torch.optim import lr_scheduler, Adam
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import torch.nn.functional as F
from itertools import cycle
import time
from torch.autograd import Variable
from models import MLP, CNNMnist, CNN13, client_model, CNNFashion_Mnist, ResNet18, Net2

from new_optimizer import SGD_Prox, SGD_SCAFFOLD, SGD_CVR, SGD_VRA
# from resnet import ResNet18, ResNet50, ResNet34
from loss import mse_with_softmax


# from convlarge import convLarge
from utils import quantize_model_weights, sparsify_model_weights_by_topK

import GaussianM

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label), item
        return image, label, item


class LocalUpdate_Avg(object):
    def __init__(self, args, dataset, data_idxs, global_weights):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.dataset == 'mnist':
            # global_model = CNNMnist(args=args)
            self.model = client_model(args.dataset).to(self.device)
        elif args.dataset == 'fmnist':
            self.model = CNNFashion_Mnist(args=args).to(self.device)
        elif args.dataset == 'cifar10':
            if args.model == "cnn":
                # self.model = client_model(args.dataset).to(self.device)
                self.model = Net2().to(self.device)
            elif args.model == "resnet":
                self.model = ResNet18().to(self.device)
                # self.model = client_model("Resnet18").to(self.device)
            else:
                exit("error: unsupported model!!!")
            # self.model = client_model(args.dataset).to(self.device)
        else:
            exit('Error: unrecognized dataset!!!')

        # self.model.train(False)
        self.global_weights = global_weights
        self.model.load_state_dict(global_weights)
        self.trainloader = DataLoader(DatasetSplit(dataset, list(data_idxs)),
                                      batch_size=self.args.batch_size, pin_memory=True, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                         momentum=self.args.momentum, weight_decay=5e-4)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, global_round):
        epoch_loss = []
        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep
        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            iter = 0
            for inputs_x, targets_x, indx in self.trainloader:
                iter += 1
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x.float())
                loss = self.ce_loss(x_output, targets_x.long())
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                ###设置clip值
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.2)
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += param.grad
                self.model.zero_grad()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # add Gaussian noise

            for name, param in self.model.named_parameters():
                print(clipped_grads[name])
                print("#################")
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, 0.2, 0.002,
                                                      device=self.device)
                print(clipped_grads[name])
                exit()
                # scale back
            # for name, param in self.model.named_parameters():
            # clipped_grads[name] /= (10)

            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]

            # update local model
            self.optimizer.step()
        self.model.eval()
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        return self.model.state_dict(), bb_loss


class LocalUpdate_PAQ(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, global_weights, cpr_level):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)
        self.cpr_level = cpr_level

    def update_weights(self, global_round):
        epoch_loss = []
        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep
        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.model.eval()
        bb_loss = sum(epoch_loss) / len(epoch_loss)

        # compute the local difference
        local_weights = copy.deepcopy(self.model.state_dict())
        delta_weights = copy.deepcopy(self.global_weights)
        for key in self.global_weights.keys():
            delta_weights[key] = local_weights[key] - self.global_weights[key]

        # compress the local difference
        if self.cpr_level <= 0:
            cpr_delta_weights = copy.deepcopy(delta_weights)
        else:
            cpr_delta_weights = quantize_model_weights(delta_weights, self.cpr_level, self.device)

        return cpr_delta_weights, bb_loss
    """
    def quantize_Q(self, model_weights, QR_accuracy):
        q_weights = copy.deepcopy(model_weights)
        for key in model_weights.keys():
            ef_weight = model_weights[key]
            max_abs = (torch.max(torch.abs(ef_weight))).to(torch.float64)
            min_abs = (torch.min(torch.abs(ef_weight))).to(torch.float64)
            if max_abs - min_abs == 0:
                x = torch.zeros(ef_weight.size()).to(self.device)
            else:
                x = (torch.abs(ef_weight) - min_abs) / ((max_abs - min_abs) / (2 ** int(QR_accuracy) - 1))
            random_flag = torch.rand(x.size()).to(self.device)
            x[random_flag <= x - torch.floor(x)] = torch.floor(x[random_flag <= x - torch.floor(x)])
            x[random_flag > x - torch.floor(x)] = torch.ceil(x[random_flag > x - torch.floor(x)])
            q_weights[key] = torch.sign(ef_weight) * (x * (max_abs - min_abs) / (2 ** int(QR_accuracy) - 1) + min_abs)
        return q_weights
    """


class LocalUpdate_COM(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, global_weights, cpr_level):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)
        self.cpr_level = cpr_level

    def update_weights(self, global_round):
        epoch_loss = []
        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep
        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.model.eval()
        bb_loss = sum(epoch_loss) / len(epoch_loss)

        # compute the local difference
        local_weights = copy.deepcopy(self.model.state_dict())
        delta_weights = copy.deepcopy(self.global_weights)
        for key in self.global_weights.keys():
            delta_weights[key] = local_weights[key] - self.global_weights[key]

        # compress the local difference
        if self.cpr_level <= 0:
            cpr_delta_weights = copy.deepcopy(delta_weights)
        else:
            cpr_delta_weights = quantize_model_weights(delta_weights, self.cpr_level, self.device)

        return cpr_delta_weights, bb_loss
    """
    def quantize_Q(self, model_weights, QR_accuracy):
        q_weights = copy.deepcopy(model_weights)
        for key in model_weights.keys():
            ef_weight = model_weights[key] / self.args.lr
            max_abs = (torch.max(torch.abs(ef_weight))).to(torch.float64)
            min_abs = (torch.min(torch.abs(ef_weight))).to(torch.float64)
            if max_abs - min_abs == 0:
                x = torch.zeros(ef_weight.size()).to(self.device)
            else:
                x = (torch.abs(ef_weight) - min_abs) / ((max_abs - min_abs) / (2 ** int(QR_accuracy) - 1))
            random_flag = torch.rand(x.size()).to(self.device)
            x[random_flag <= x - torch.floor(x)] = torch.floor(x[random_flag <= x - torch.floor(x)])
            x[random_flag > x - torch.floor(x)] = torch.ceil(x[random_flag > x - torch.floor(x)])
            q_weights[key] = torch.sign(ef_weight) * (x * (max_abs - min_abs) / (2 ** int(QR_accuracy) - 1) + min_abs)
        return q_weights
    """


class LocalUpdate_CAMS(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, local_cpr_err, global_weights, spar_level):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)
        self.spar_level = spar_level
        self.local_cpr_err = local_cpr_err

    def update_weights(self, global_round):
        epoch_loss = []
        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep
        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        self.model.eval()
        bb_loss = sum(epoch_loss) / len(epoch_loss)

        # compute the local
        local_weights = copy.deepcopy(self.model.state_dict())
        delta_weights = copy.deepcopy(self.global_weights)
        ef_delta_weights = copy.deepcopy(self.global_weights)
        for key in self.global_weights.keys():
            delta_weights[key] = local_weights[key] - self.global_weights[key]
            ef_delta_weights[key] = delta_weights[key] + self.local_cpr_err[key]

        # compress the local difference
        if self.spar_level <= 0:
            spar_delta_weights = copy.deepcopy(delta_weights)
        else:
            spar_delta_weights = sparsify_model_weights_by_topK(ef_delta_weights, self.spar_level, self.device)

        # update the local compression error
        if self.spar_level > 0:
            for key in local_weights.keys():
                self.local_cpr_err[key] = delta_weights[key] + self.local_cpr_err[key] - spar_delta_weights[key]

        return spar_delta_weights, self.local_cpr_err, bb_loss


class LocalUpdate_Prox(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, global_weights, penalty):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_Prox(params=self.model.parameters(), global_params=global_weights,
                                  penalty=penalty, lr=self.args.lr, momentum=self.args.momentum,
                                  weight_decay=5e-4)


class LocalUpdate_SCAFFOLD(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, global_weights, global_ctl, local_ctl):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_SCAFFOLD(params=self.model.parameters(), global_ctl=global_ctl,
                                      local_ctl=local_ctl, lr=self.args.lr,
                                      momentum=self.args.momentum, weight_decay=5e-4)
        self.local_ctl = local_ctl
        self.global_ctl = global_ctl

    def update_weights(self, global_round):
        # Set mode to train model
        # global_weights = copy.deepcopy(global_model.state_dict())
        # self.model.load_state_dict(global_weights)
        epoch_loss = []
        local_update_num = 0

        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep

        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                local_update_num += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        if self.args.dp!=0:
              for p in self.model.parameters():
                sigma_g = 0.005
                C_norm = np.median([float(d.data.norm(2)) for d in p.data])
                #print(C_norm)
                p.data = torch.stack([d / max(1, float(d.data.norm(2)) / C_norm) for d in p.data])
                p.data = GaussianM.GaussianMechanism(p.data, sigma_g, C_norm, 1,self.args.gpu)
                #print(p.data)
        self.model.eval()
        # update local control
        local_weights = copy.deepcopy(self.model.state_dict())
        local_ctl_delta = copy.deepcopy(self.local_ctl)
        a = 1 / (local_update_num * self.args.lr)
        for key in self.local_ctl.keys():
            self.local_ctl[key] = self.local_ctl[key] - self.global_ctl[key] - (
                    local_weights[key] - self.global_weights[key]) * a
            local_ctl_delta[key] = self.local_ctl[key] - local_ctl_delta[key]

        return local_weights, self.local_ctl, bb_loss, local_ctl_delta


class LocalUpdate_Dyn(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, dual_k, global_weights, penalty):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_VRA(params=self.model.parameters(), dual_params=dual_k,
                                 global_params=global_weights, penalty=penalty, lr=self.args.lr,
                                 momentum=self.args.momentum, weight_decay=5e-4)
        self.client_dual = dual_k
        self.penalty = penalty

    def update_weights(self, global_round):
        # Set mode to train model
        # global_weights = copy.deepcopy(global_model.state_dict())
        # self.model.load_state_dict(global_weights)
        epoch_loss = []

        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep

        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            # correct_l, correct_u = 0, 0
            # total_l, total_u = 0, 0
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        self.model.eval()
        # update local control
        local_weights = copy.deepcopy(self.model.state_dict())
        for key in self.client_dual.keys():
            self.client_dual[key] = self.client_dual[key] \
                                    + (self.global_weights[key] - local_weights[key]) * self.penalty

        return local_weights, self.client_dual, bb_loss


class LocalUpdate_CVR(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, control_k, global_weights, penalty, local_ratio):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_CVR(params=self.model.parameters(), lctr_params=control_k,
                                 global_params=global_weights, penalty=penalty, lr=self.args.lr,
                                 momentum=self.args.momentum, weight_decay=5e-4)
        self.client_control = control_k
        self.penalty = penalty
        self.local_ratio = local_ratio

    def update_weights(self, global_round):
        # Set mode to train model
        # global_weights = copy.deepcopy(global_model.state_dict())
        # self.model.load_state_dict(global_weights)
        epoch_loss = []
        local_update_num = 0

        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep

        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                local_update_num += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        self.model.eval()
        # compute effective number of local update steps
        eff_num = (1.0 - 1.0 / (1.0 + self.penalty * self.args.lr) ** local_update_num) / (self.penalty * self.args.lr)
        print("the effective local control ratio: " + str(self.local_ratio / (self.args.lr * eff_num)))
        # update local control
        local_weights = copy.deepcopy(self.model.state_dict())
        ctr_ratio = self.local_ratio / (self.args.lr * eff_num)
        for key in self.client_control.keys():
            self.client_control[key] = self.client_control[key] \
                                       + (self.global_weights[key] - local_weights[key]) * ctr_ratio
            # self.client_control[key] = self.client_control[key] + (
            #        self.global_weights[key] - local_weights[key]) * self.penalty

        return local_weights, self.client_control, bb_loss, ctr_ratio


class LocalUpdate_QCVR(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, control_k, local_cpr_err, global_weights, penalty, local_ratio,
                 cpr_level):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_CVR(params=self.model.parameters(), lctr_params=control_k,
                                 global_params=global_weights, penalty=penalty, lr=self.args.lr,
                                 momentum=self.args.momentum, weight_decay=5e-4)
        self.client_control = control_k
        self.penalty = penalty
        self.local_ratio = local_ratio
        self.local_cpr_err = local_cpr_err
        self.cpr_level = cpr_level
        # self.delta_weights = torch.zeros.

    def update_weights(self, global_round):
        # Set mode to train model
        # global_weights = copy.deepcopy(global_model.state_dict())
        # self.model.load_state_dict(global_weights)
        epoch_loss = []
        local_update_num = 0

        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep

        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                local_update_num += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        self.model.eval()
        # compute effective number of local update steps
        eff_num = (1.0 - 1.0 / (1.0 + self.penalty * self.args.lr) ** local_update_num) / (self.penalty * self.args.lr)
        print("the effective local control ratio: " + str(self.local_ratio / (self.args.lr * eff_num)))

        # compute the local
        local_weights = copy.deepcopy(self.model.state_dict())
        delta_weights = copy.deepcopy(self.global_weights)
        ef_delta_weights = copy.deepcopy(self.global_weights)
        for key in self.global_weights.keys():
            delta_weights[key] = local_weights[key] - self.global_weights[key]
            ef_delta_weights[key] = delta_weights[key] + self.local_cpr_err[key]

        # compress the local difference
        if self.cpr_level <= 0:
            cpr_delta_weights = copy.deepcopy(delta_weights)
        else:
            cpr_delta_weights = quantize_model_weights(ef_delta_weights, self.cpr_level, self.device)

        # update the local compression error and local control
        ctr_ratio = self.local_ratio / (self.args.lr * eff_num)
        for key in self.client_control.keys():
            if self.cpr_level > 0:
                self.local_cpr_err[key] = delta_weights[key] + self.local_cpr_err[key] - cpr_delta_weights[key]
            self.client_control[key] = self.client_control[key] - cpr_delta_weights[key] * ctr_ratio

        return cpr_delta_weights, self.client_control, self.local_cpr_err, bb_loss, ctr_ratio
    """
    def quantize_Q(self, model_weights, cpr_err, QR_accuracy):
        q_weights = copy.deepcopy(model_weights)
        for key in model_weights.keys():
            ef_weight = model_weights[key] + cpr_err[key]
            max_abs = (torch.max(torch.abs(ef_weight))).to(torch.float64)
            min_abs = (torch.min(torch.abs(ef_weight))).to(torch.float64)
            if max_abs - min_abs == 0:
                x = torch.zeros(ef_weight.size()).to(self.device)
            else:
                x = (torch.abs(ef_weight) - min_abs) / ((max_abs - min_abs) / (2 ** int(QR_accuracy) - 1))
            random_flag = torch.rand(x.size()).to(self.device)
            x[random_flag <= x - torch.floor(x)] = torch.floor(x[random_flag <= x - torch.floor(x)])
            x[random_flag > x - torch.floor(x)] = torch.ceil(x[random_flag > x - torch.floor(x)])
            q_weights[key] = torch.sign(ef_weight) * (x * (max_abs - min_abs) / (2 ** int(QR_accuracy) - 1) + min_abs)
        return q_weights
    """


class LocalUpdate_SCVR(LocalUpdate_Avg):
    def __init__(self, args, dataset, data_idxs, control_k, local_cpr_err, global_weights, penalty, local_ratio,
                 spar_level):
        super().__init__(args=args, dataset=dataset, data_idxs=data_idxs, global_weights=global_weights)

        self.optimizer = SGD_CVR(params=self.model.parameters(), lctr_params=control_k,
                                 global_params=global_weights, penalty=penalty, lr=self.args.lr,
                                 momentum=self.args.momentum, weight_decay=5e-4)
        self.client_control = control_k
        self.penalty = penalty
        self.local_ratio = local_ratio
        self.local_cpr_err = local_cpr_err
        self.spar_level = spar_level
        # self.delta_weights = torch.zeros.

    def update_weights(self, global_round):
        # Set mode to train model
        # global_weights = copy.deepcopy(global_model.state_dict())
        # self.model.load_state_dict(global_weights)
        epoch_loss = []
        local_update_num = 0

        if self.args.local_ep_var:
            num_epochs = random.randint(1, self.args.local_ep)
        else:
            num_epochs = self.args.local_ep

        print("number of local epochs:" + str(num_epochs))
        self.model.train()
        for iter_local in range(num_epochs):
            batch_loss, batch_loss_l, batch_loss_u = [], [], []
            for inputs_x, targets_x, indx in self.trainloader:
                # targets_x = torch.zeros(len(targets_x), 10).scatter_(1, targets_x.view(-1, 1), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                x_output = self.model(inputs_x)
                loss = self.ce_loss(x_output, targets_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                local_update_num += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bb_loss = sum(epoch_loss) / len(epoch_loss)
        self.model.eval()
        # compute effective number of local update steps
        eff_num = (1.0 - 1.0 / (1.0 + self.penalty * self.args.lr) ** local_update_num) / (self.penalty * self.args.lr)
        print("the effective local control ratio: " + str(self.local_ratio / (self.args.lr * eff_num)))

        # compute the local
        local_weights = copy.deepcopy(self.model.state_dict())
        delta_weights = copy.deepcopy(self.global_weights)
        ef_delta_weights = copy.deepcopy(self.global_weights)
        for key in self.global_weights.keys():
            delta_weights[key] = local_weights[key] - self.global_weights[key]
            ef_delta_weights[key] = delta_weights[key] + self.local_cpr_err[key]

        # compress the local difference
        if self.spar_level <= 0:
            spar_delta_weights = copy.deepcopy(delta_weights)
        else:
            # cpr_delta_weights = self.quantize_Q(delta_weights, self.local_cpr_err, self.cpr_level)
            spar_delta_weights = sparsify_model_weights_by_topK(ef_delta_weights, self.spar_level, self.device)

        # update the local compression error and local control
        ctr_ratio = self.local_ratio / (self.args.lr * eff_num)
        for key in self.client_control.keys():
            if self.spar_level > 0:
                self.local_cpr_err[key] = delta_weights[key] + self.local_cpr_err[key] - spar_delta_weights[key]
            self.client_control[key] = self.client_control[key] - spar_delta_weights[key] * ctr_ratio

        return spar_delta_weights, self.client_control, self.local_cpr_err, bb_loss, ctr_ratio

    """
    def sparsify_topk(self, model_weights, cpr_err, spar_level):
        spar_weights = copy.deepcopy(model_weights)
        for key in model_weights.keys():
            spar_weights[key] = model_weights[key] + cpr_err[key]
        # abs_weights = torch.abs(model_weights)
        spar_one = torch.cat([value.flatten() for key, value in spar_weights.items()])
        top_num = int(torch.numel(spar_one) * spar_level)
        print("the number of elements to be kept: " + str(top_num))
        topK_values, _ = torch.topk(torch.abs(spar_one), top_num)
        min_abs_top = topK_values[-1]
        # print(topK_values)
        # print(min_abs_top)
        # print(topK_indices)
        count = 0
        for key in model_weights.keys():
            abs_weight = torch.abs(spar_weights[key])
            mask = torch.zeros(abs_weight.size()).to(self.device)
            mask[abs_weight >= min_abs_top] = 1
            count += torch.count_nonzero(mask)
            spar_weights[key] = spar_weights[key] * mask
            # print(ef_weight)
            # print(ef_weight.size())
            # print(topK_values)
            # print(topK_indices)
            # break
        if abs(count - top_num) > 0.001 * torch.numel(spar_one):
            raise ValueError("The Top K is not " + str(count))
        return spar_weights
    """


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=1)
    ba_loss = []
    for batch_idx, (images, targets) in enumerate(testloader):
        # images, targets = images.to(device), labels.to(device)

        # Inference

        labels = torch.zeros(len(targets), 10).scatter_(1, targets.view(-1, 1), 1)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda(non_blocking=True)
            targets = targets.cuda()
        outputs = model(images)
        batch_loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * labels, dim=1))

        ba_loss.append(batch_loss.item())

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        # print("pred_labels:",pred_labels)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, targets)).item()
        total += len(targets)
    loss = sum(ba_loss) / len(ba_loss)
    accuracy = correct / total

    return accuracy, loss
