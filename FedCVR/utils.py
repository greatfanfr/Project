#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
from pathlib import Path
import random

import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.utils.data as data
import pandas as pd

import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


class CustomDataset(data.Dataset):
    def __init__(self, path, transforms=None):
        self.X, self.Y = self.load_data_to_tensors(path)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.X)

    """
    def data_loader(self, batch_size):
        return data.DataLoader(dataset=self, batch_size=batch_size)
    """

    def load_data_to_tensors(self, data_path, label_path):
        data_df = pd.read_csv(data_path)
        label_df = pd.read_csv(label_path)
        X, Y = list(), list()
        for row_num in range(len(data_df.index)):
            if row_num % 2 == 0:
                data_mat = data_df.iloc[row_num:row_num + 2, :].values
                data_tensor = torch.tensor(data_mat, torch.float32)
                data_tensor.reshape(len(data_df.columns), 1, 2)
                X.append(data_tensor)
                label_mat = data_df.iloc[row_num:row_num + 2, :].values
                label_tensor = torch.tensor(label_mat, torch.float32)
                label_tensor.reshape(len(label_df.columns), 1, 2)
                Y.append(label_tensor)
        X = torch.stack(X)
        Y = torch.LongTensor(Y)  # may change, depends on the model
        return X, Y


def usege_example(path):
    dataset = CustomDataset(path)
    data = dataset.data_loader(4)  # batch size of 4
    for item in enumerate(data):
        print(item)


def get_dataset2(args, seed=0):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    np.random.seed(seed)
    random.seed(seed)
    train_dataset, test_dataset, num_classes = None, None, 10

    if args.dataset == 'cifar10':
        data_dir = './data/cifar10/'
        # the transformation with data augumentation for train dataset
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=transform_test)
        # sample training data amongst users
        num_classes = 10

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        num_classes = 10

    if args.dataset == 'SVHN':
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                      transform=apply_transform)

        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                     transform=transform_test)

    par_mode = 0
    unbalanced_sgm = 0
    if args.iid == 0:
        if args.unequal:
            par_mode = 2
            unbalanced_sgm = 0.6
        else:
            par_mode = 1

    print("mode:" + str(par_mode) + ", unbalanced:" + str(unbalanced_sgm))

    user_groups, p_ratio = partition2(args=args,
                                      dataset=train_dataset,
                                      mode=par_mode,
                                      num_users=args.num_users,
                                      num_classes=num_classes,
                                      dir_alpha=0.2,
                                      unbalanced_sgm=unbalanced_sgm)
    print("data distribution is done!")

    return train_dataset, test_dataset, user_groups, p_ratio


def partition2(args, dataset, mode: int, num_users: int, num_classes: int,
               dir_alpha: float = 0.2, unbalanced_sgm: float = 0.0):
    """
    The function
    """
    # There are three modes to partition the training data among clients.
    #   1. mode = 0:  i.i.d data
    #   2. mode = 1:  balanced non-i.i.d data
    #   3. mode = 2:  unbalanced non-i.i.d data
    num_data = len(dataset)
    num_data_per_user = num_data // num_users
    dict_users = {i: [] for i in range(num_users)}
    p_ratio = np.ones(num_users) / num_users
    if mode == 0:  # IID
        # Shuffle data to make it to be i.i.d data
        # data_ids = torch.randperm(num_data, dtype=torch.int32)
        data_ids = [i for i in range(num_data)]
        np.random.shuffle(data_ids)
        # prepare local data for each client
        for i in range(num_users):
            # TODO: Make this parallel for large number of clients & large datasets (Maybe not required)
            dict_users[i] = data_ids[i * num_data_per_user: (i + 1) * num_data_per_user]
            # print (dict_users[i])
    else:  # Non IID
        if mode not in {1, 2}:
            raise ValueError("Unknown mode. Mode must be {0,1，2}")
        if mode == 2 and unbalanced_sgm != 0:  # if non-iid unbalanced
            # Draw from lognormal distribution
            user_data_list = (np.random.lognormal(mean=np.log(num_data_per_user),
                                                  sigma=unbalanced_sgm, size=num_users))
            user_data_list = (user_data_list / np.sum(user_data_list) * num_data).astype(int)
            diff = np.sum(user_data_list) - num_data

            # Add/Subtract the excess number starting from first client
            if diff != 0:
                for user_i in range(num_users):
                    if user_data_list[user_i] > diff:
                        user_data_list[user_i] -= diff
                        break
        else:
            user_data_list = (np.ones(num_users) * num_data_per_user).astype(int)


        # Label distribution skew 基于分布标签的不平衡
        # 使用狄利克雷分布模拟数据的 非独立同分布
        # 非独立同分布数据孤岛的联邦学习：一项实验研究 - 我爱计算机视觉的文章 - 知乎
        # https://zhuanlan.zhihu.com/p/557553708
        cls_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes, size=num_users)
        prior_cumsum = np.cumsum(cls_priors, axis=1)
        if args.dataset == "cifar10":
            idx_list = [np.where(torch.Tensor(dataset.targets) == i)[0] for i in range(num_classes)]
        else:
            idx_list = [np.where(dataset.targets == i)[0] for i in range(num_classes)]

        cls_size = [len(idx_list[i]) for i in range(num_classes)]

        rest_users = list(range(num_users))
        used_cls = []
        while len(rest_users) > 0:
            # curr_client = np.random.randint(num_clients)
            curr_user = np.random.choice(rest_users)
            # print("current user " + str(curr_user) + ", num_data" + str(user_data_list[curr_user]))
            # If current node is full resample a client
            # print('Remaining Data: %d' % np.sum(client_data_list))
            if user_data_list[curr_user] <= 0:
                rest_users.remove(curr_user)
                continue
            curr_prior = prior_cumsum[curr_user]
            temp = (np.random.uniform() <= curr_prior)
            # print(temp)
            cls_label = np.argmax(temp)
            while cls_size[cls_label] <= 0:
                if cls_label <= 0:
                    temp = (np.random.uniform() <= curr_prior)
                else:
                    temp[cls_label] = False
                cls_label = np.argmax(temp)
            # print (cls_label)
            cls_size[cls_label] -= 1
            dict_users[curr_user].append(idx_list[cls_label][cls_size[cls_label]])
            user_data_list[curr_user] -= 1

    for user_id in range(num_users):
        user_num = len(dict_users[user_id])
        p_ratio[user_id] = user_num / num_data

    return dict_users, p_ratio


def partition_noniid(self, num_clients,
                     num_cls_per_client: int = 2,
                     unbalance_flag: bool = True,
                     client_test_ratio: float = 0.0,
                     client_unlabel_ratio: float = 0.0,
                     show_plots: bool = False):
    """

        :param client_test_ratio:
        :param num_of_clients:
        :param unbalance_flag:
        :param data:
        :param num_cls_per_client:
        :return:
        """
    print(self.root_dir)
    if client_test_ratio < 0 or client_test_ratio > 1 or client_unlabel_ratio < 0 or client_unlabel_ratio > 1:
        raise ValueError("the ratio of local test or unlabel data is invalid!!!")
    # set the data path as "client data" under root
    client_data_path = Path(self.root_dir, "client_data_" + self.dataset_name
                            + "_#" + str(num_clients), "noniid_cls#" + str(num_cls_per_client),
                            "test_ratio_" + str(client_test_ratio)
                            + ",unlabel_ratio_" + str(client_unlabel_ratio))

    # if the dir "client_data" under root already exists, then return
    if os.path.exists(client_data_path):
        # shutil.rmtree(client_data_path)
        return client_data_path

    # client_data_path.mkdir()
    os.makedirs(client_data_path)

    # record the number of unlabel data in each client and return
    # unlabel_sizes = []

    if not isinstance(self.train_data.targets, torch.Tensor):
        self.train_data.targets = torch.tensor(self.train_data.targets)
    total_train_data = [self.train_data[j] for j in range(len(self.train_data))]
    # save the train data
    torch.save(total_train_data, client_data_path / "train_data.pth")

    if not isinstance(self.test_data.targets, torch.Tensor):
        self.test_data.targets = torch.tensor(self.test_data.targets)
    test_data = [self.test_data[j] for j in range(len(self.test_data))]
    # save the test data
    torch.save(test_data, client_data_path / "test_data.pth")

    labels = self.train_data.targets
    labels = torch.tensor(labels)
    classes = labels.unique()
    num_of_data = len(self.train_data)
    if unbalance_flag:
        min_size = int(num_of_data / (len(classes) * num_clients))
        slice_sizes = min_size * np.ones((num_cls_per_client, num_clients), dtype=int)
        for i in range(num_cls_per_client):
            total_remainder = int(num_of_data / num_cls_per_client) - min_size * num_clients
            ind = np.sort(np.random.choice(np.arange(0, total_remainder), num_clients - 1, replace=False))
            ind = np.insert(ind, 0, 0)
            ind = np.insert(ind, len(ind), total_remainder)
            cls_sizes = ind[1:] - ind[:-1]
            slice_sizes[i, :] += cls_sizes
    else:
        slice_size = int(num_of_data / (num_clients * num_cls_per_client))
        slice_sizes = np.zeros((num_cls_per_client, num_clients), dtype=int)
        slice_sizes += slice_size

    label_array = labels.numpy()
    class_array = classes.numpy()
    sorted_ind = np.concatenate([np.squeeze(np.argwhere(label_array == c)) for c in class_array], axis=0)
    indices = list(sorted_ind)

    # partition indices
    from_index = 0
    partitions = []
    for n_class in range(num_cls_per_client):
        for client in range(num_clients):
            to_index = from_index + slice_sizes[n_class, client]
            if n_class == 0:
                partitions.append(indices[from_index:to_index])
            else:
                partitions[client].extend(indices[from_index:to_index])
            from_index = to_index

    for client_id in range(num_clients):
        train_data = [self.train_data[j] for j in partitions[client_id]]
        client_path = Path(client_data_path / str(client_id))
        client_path.mkdir()

        # Shuffle data to make it to be i.i.d data
        ta_ids = torch.randperm(len(train_data), dtype=torch.int32)
        num_of_test = int(len(train_data) * client_test_ratio)
        client_test_data = [train_data[j] for j in ta_ids[:num_of_test]]
        client_train_data = [train_data[j] for j in ta_ids[num_of_test:]]

        # print("len of train _data: " + str(len(train_data)))
        # print("len of client test _data: " + str(len(client_test_data)))
        # print("len of client train _data: " + str(len(client_train_data)))
        # Then, we also use a partition of the rest as unlabelled data
        num_of_ulabel = int(len(client_train_data) * client_unlabel_ratio)
        if num_of_ulabel <= 0:
            raise ValueError("Error: the number of unlabeled data is zero !!!")
        client_unlabel_data = client_train_data[:num_of_ulabel]
        client_label_data = client_train_data[num_of_ulabel:]
        # unlabel_sizes[client_id] = num_of_ulabel
        # print("len of client unlabel _data: " + str(len(client_unlabel_data)))
        # print("len of client label _data: " + str(len(client_label_data)))
        # Split data equally and send to the client
        torch.save(train_data, client_data_path / str(client_id) / "data.pth")
        torch.save(client_train_data, client_data_path / str(client_id) / "train_data.pth")
        torch.save(client_test_data, client_data_path / str(client_id) / "test_data.pth")
        torch.save(client_unlabel_data, client_data_path / str(client_id) / "unlabel_data.pth")
        torch.save(client_label_data, client_data_path / str(client_id) / "label_data.pth")

        if show_plots:
            self._plot(train_data, title=f"Client {client_id + 1} Data Distribution, #" + str(len(train_data)))

    return client_data_path


def average_weights(global_weights, local_weights):
    """
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys():
        wx = 0
        for i in range(len(w)):
             wx+= w_ratio[i]*w[i][key]
        w_avg[key] = wx
    return w_avg
    """
    # N = sum(u["n_samples"] for u in updates)
    for key, value in global_weights.items():
        weight_sum = [
            w[key] for w in local_weights
        ]
        if len(sum(weight_sum).size()) != 0:
            value[:] = sum(weight_sum) / 10.0

    return global_weights


def update_global_weights(global_weights, local_weights, idxs_users, p_ratio, tau, lr):
    aa = np.sum(p_ratio[idxs_users])
    tau_w = np.dot(p_ratio[idxs_users], tau[idxs_users]) / aa
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] / aa * (global_weights[key] - local_weights[idx][key]) / (lr * tau[idx])

        global_weights[key] = global_weights[key] - wx * tau_w * lr

    return global_weights, tau_w


def update_global_weights_Avg(global_weights, local_weights, idxs_users, num_users, p_ratio):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] * (local_weights[idx][key] - global_weights[key])

        global_weights[key] = global_weights[key] + wx * num_users / len(idxs_users)

    return global_weights


def update_global_weights_PAQ(global_weights, local_diffs, idxs_users, num_users, p_ratio):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            # wx += p_ratio[idx] * (local_weights[idx][key] - global_weights[key])
            wx += p_ratio[idx] * local_diffs[idx][key]

        global_weights[key] = global_weights[key] + wx * num_users / len(idxs_users)

    return global_weights


def update_global_weights_COM(global_weights, local_diffs, idxs_users, num_users, p_ratio, lr, glr):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            # wx += p_ratio[idx] * (local_weights[idx][key] - global_weights[key])
            wx += p_ratio[idx] * local_diffs[idx][key]

        global_weights[key] = global_weights[key] + lr * glr * wx * num_users / len(idxs_users)

    return global_weights


"""
def update_global_weights_scaffold(global_weights, global_ctl, local_weights, local_ctls,
                                   idxs_users, num_users, p_ratio):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] * local_weights[idx][key]

        global_weights[key] = wx * num_users / len(idxs_users)

    for key in global_ctl.keys():
        ctl = 0
        for idx in range(num_users):
            ctl += p_ratio[idx] * local_ctls[idx][key]
        global_ctl[key] = ctl

    return global_weights, global_ctl
"""


def update_global_weights_scaffold(global_weights, global_ctl, local_weights, local_ctl_deltas,
                                   idxs_users, num_users, p_ratio):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] * local_weights[idx][key]

        global_weights[key] = wx * num_users / len(idxs_users)

    for key in global_ctl.keys():
        ctl = 0
        for idx in idxs_users:
            ctl += p_ratio[idx] * local_ctl_deltas[idx][key]
        global_ctl[key] += ctl

    return global_weights, global_ctl


def update_global_weights_Dyn(global_weights, local_weights, duals, idxs_users, num_users, p_ratio, penalty):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] * (local_weights[idx][key] - global_weights[key])

        global_weights[key] = global_weights[key] + wx * num_users / len(idxs_users)

    for key in global_weights.keys():
        dual_sum = 0
        for idx in range(num_users):
            dual_sum += p_ratio[idx] * duals[idx][key]
        global_weights[key] = global_weights[key] - dual_sum / penalty

    return global_weights


def update_global_weights_CVR(global_init, global_weights, global_ctl, local_weights,
                              idxs_users, num_users, p_ratio, penalty, local_ctr_ratios):
    for key in global_init.keys():
        wx = 0
        for idx in idxs_users:
            wx += p_ratio[idx] * (local_weights[idx][key] - global_init[key])

        global_weights[key] = global_init[key] + wx * num_users / len(idxs_users)

    for key in global_init.keys():
        ctl = 0
        for idx in idxs_users:
            ctl += p_ratio[idx] * (global_init[key] - local_weights[idx][key]) * local_ctr_ratios[idx]
        global_ctl[key] += ctl
        global_init[key] = global_weights[key] - global_ctl[key] / penalty

    # return global_weights, global_ctl, global_init
    return global_weights, global_ctl, global_init


def update_global_weights_QCVR(global_init, global_weights, global_ctl, local_diffs,
                               idxs_users, num_users, p_ratio, penalty, local_ctr_ratios):
    for key in global_init.keys():
        wx = 0
        for idx in idxs_users:
            # wx += p_ratio[idx] * (local_weights[idx][key] - global_init[key])
            wx += p_ratio[idx] * local_diffs[idx][key]

        global_weights[key] = global_init[key] + wx * num_users / len(idxs_users)

    for key in global_init.keys():
        ctl = 0
        for idx in idxs_users:
            # ctl += p_ratio[idx] * (global_init[key] - local_weights[idx][key]) * local_ctr_ratios[idx]
            ctl += p_ratio[idx] * local_diffs[idx][key] * local_ctr_ratios[idx]
        global_ctl[key] -= ctl
        global_init[key] = global_weights[key] - global_ctl[key] / penalty

    # return global_weights, global_ctl, global_init
    return global_weights, global_ctl, global_init


def update_global_weights_Adam(global_weights, local_weights, idxs_users, num_users, p_ratio, moment1, moment2,
                               step_size=0.1):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += (local_weights[idx][key] - global_weights[key]) / len(idxs_users)
        moment1[key] = 0.9 * moment1[key] + 0.1 * wx
        moment2[key] = 0.999 * moment2[key] + 0.001 * (wx ** 2)

        global_weights[key] = global_weights[key] + step_size * moment1[key] / (torch.sqrt(moment2[key]) - 3)

    return global_weights, moment1, moment2


def update_global_weights_AMS(global_weights, local_weights, idxs_users, num_users, p_ratio, moment1, moment2,
                              moment3, step_size=0.1):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += (local_weights[idx][key] - global_weights[key]) / len(idxs_users)
        moment1[key] = 0.9 * moment1[key] + 0.1 * wx
        moment2[key] = 0.999 * moment2[key] + 0.001 * (wx ** 2)
        tmp = torch.maximum(moment3[key], moment2[key])
        moment3[key] = torch.clamp(tmp, min=0.001)
        global_weights[key] = global_weights[key] + step_size * moment1[key] / torch.sqrt(moment3[key])

    return global_weights, moment1, moment2, moment3


def update_global_weights_CAMS(global_weights, local_diffs, idxs_users, num_users, p_ratio, moment1, moment2,
                               moment3, step_size=0.1):
    for key in global_weights.keys():
        wx = 0
        for idx in idxs_users:
            wx += local_diffs[idx][key] / len(idxs_users)
        moment1[key] = 0.9 * moment1[key] + 0.1 * wx
        moment2[key] = 0.999 * moment2[key] + 0.001 * (wx ** 2)
        tmp = torch.maximum(moment3[key], moment2[key])
        moment3[key] = torch.clamp(tmp, min=0.001)
        global_weights[key] = global_weights[key] + step_size * moment1[key] / torch.sqrt(moment3[key])

    return global_weights, moment1, moment2, moment3


def compute_accuracy_and_loss(model, dataset, args):
    """
    The function computes the loss value and the associated accuracy based on
    the data

    Args:
        model       ----    the neural network architecture
        data_loader ----    the loader to assess the data
        criterion   ----    the loss function
        device      ----    which device to perform the computation\

    Returns:
        the loss value and accuracy
    """
    # a flag indicating whether the model was being used for training or not before this function
    was_training = False
    # if the model is being used for training, then set the mode to evaluation and was_training = True
    if model.training:
        model.eval()
        was_training = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=1000,
                             shuffle=False, num_workers=1)

    correct, total, loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        p_bar = tqdm(data_loader, desc="Evaluating")
        for data, labels in p_bar:
            data, labels = data.to(device), labels.to(device)
            y = model(data)
            loss += criterion(y, labels).item()  # sum up batch loss
            _, predicted = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(data_loader)
    if was_training:
        model.train()
    return correct / float(total), loss


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def plot_results(exp_name: str, train_loss: [], train_accuracy: [],
                 test_loss: [], test_accuracy: []):
    value = exp_name.split("/")
    result_dir = Path("results", value[0], value[1])
    exp_name = value[2]
    if not os.path.exists(result_dir):
        result_dir.mkdir(parents=True, exist_ok=True)

    # save data to npy file
    np.savetxt(Path(result_dir, "trainloss_" + exp_name + ".csv"), train_loss, delimiter=',')
    # np.savetxt(Path(result_dir, "trainacc_" + exp_name + ".csv"), self.train_accuracy, delimiter=',')
    # np.savetxt(Path(result_dir, "testloss_" + exp_name + ".csv"), self.test_loss, delimiter=',')
    np.savetxt(Path(result_dir, "testacc_" + exp_name + ".csv"), test_accuracy, delimiter=',')

    matplotlib.rcdefaults()
    markers = ['o', '+', '*', '>', 's', 'd', 'v', '^']
    colors = ['r', 'b', 'y', 'g', 'o', 'p', 'k']
    marker_style = dict(markersize=25, markerfacecoloralt='tab:red', markeredgewidth=2,
                        markevery=max(1, int(len(train_loss) / 6)), fillstyle='none')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15.5, 8))
    ax[0].plot(train_loss, lw=1, marker=markers[0], color=colors[0], label='Train Loss', **marker_style)
    ax[0].plot(test_loss, lw=1, marker=markers[1], color=colors[1], label='Test Loss', **marker_style)
    ax[0].set_ylabel('Loss', fontsize=25, color='k')
    ax[0].set_xlabel('Round $r$', fontsize=25)
    ax[0].set_xlim([0, len(test_loss)])
    ax[0].grid(linestyle=':', alpha=0.2, lw=2)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].legend(fontsize="25", edgecolor='k')

    ax[1].plot(train_accuracy, lw=1, marker=markers[0], color=colors[0], label='Train Accuracy',
               **marker_style)
    ax[1].plot(test_accuracy, lw=1, marker=markers[1], color=colors[1], label='Test Accuracy', **marker_style)
    ax[1].set_ylabel('Accuracy', fontsize=25, color='k')
    ax[1].set_xlabel('Round $r$', fontsize=25)
    ax[1].set_xlim([0, len(test_accuracy)])
    ax[1].set_ylim([0.1, 1])
    ax[1].grid(linestyle=':', alpha=0.2, lw=2)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[1].legend(fontsize="25", edgecolor='k')

    fig.subplots_adjust(
        top=0.9,
        bottom=0.080,
        left=0.06,
        right=0.9,
        hspace=0.1,
        wspace=0.15
    )
    plt.savefig(Path(result_dir, f"{exp_name}_results.pdf"), dpi=200)
    # plt.show()


def init_seed(seed: int):
    """
    Set all random generators with the same seed to reproduce the results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#############################################################################
# Compression methods
#############################################################################
def quantize_model_weights(model_weights, QR_accuracy, device):
    """
    Compress the model weights by stochastic (random) quantization
    @args
        model_weights ------- a dict of model parameters including keys and weights
        QR_accuracy   ------- an integer in [1, 64] denoting the number of bits to represent a floating number
        device        ------- the device to store and compute
    """
    q_weights = copy.deepcopy(model_weights)
    spar_one = torch.cat([value.flatten() for key, value in model_weights.items()])
    max_abs = (torch.max(torch.abs(spar_one))).to(torch.float64)
    min_abs = (torch.min(torch.abs(spar_one))).to(torch.float64)
    for key in model_weights.keys():
        ef_weight = model_weights[key]
        # max_abs = (torch.max(torch.abs(ef_weight))).to(torch.float64)
        # min_abs = (torch.min(torch.abs(ef_weight))).to(torch.float64)
        if max_abs - min_abs == 0:
            x = torch.zeros(ef_weight.size()).to(device)
        else:
            x = (torch.abs(ef_weight) - min_abs) / ((max_abs - min_abs) / (2 ** int(QR_accuracy) - 1))
        random_flag = torch.rand(x.size()).to(device)
        x[random_flag <= x - torch.floor(x)] = torch.floor(x[random_flag <= x - torch.floor(x)])
        x[random_flag > x - torch.floor(x)] = torch.ceil(x[random_flag > x - torch.floor(x)])
        q_weights[key] = torch.sign(ef_weight) * (x * (max_abs - min_abs) / (2 ** int(QR_accuracy) - 1) + min_abs)
    return q_weights


def sparsify_model_weights_by_topK(model_weights, spar_level, device):
    """
    @args
        model_weights ------- a dict of model parameters including keys and weights
        QR_accuracy   ------- an integer in [1, 64] denoting the number of bits to represent a floating number
        device        ------- the device to store and compute
    """
    spar_weights = copy.deepcopy(model_weights)
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
        mask = torch.zeros(abs_weight.size()).to(device)
        mask[abs_weight >= min_abs_top] = 1
        count += torch.count_nonzero(mask)
        spar_weights[key] = spar_weights[key] * mask
        if count > top_num:
            break
        # print(ef_weight)
        # print(ef_weight.size())
        # print(topK_values)
        # print(topK_indices)
        # break
    # if abs(count - top_num) > 0.001 * torch.numel(spar_one):
    #    raise ValueError("The Top K is not " + str(count))
    return spar_weights
