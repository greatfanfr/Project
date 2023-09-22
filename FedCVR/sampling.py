#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
from torchvision import datasets, transforms
import torch


def split_relabel_data(np_labs, labels, label_per_class,
                       num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs == id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    # np.random.shuffle(labeled_idxs)
    # np.random.shuffle(unlabed_idxs)

    # relabel dataset
    # for idx in unlabed_idxs:
    # labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs


def mnist_iid(dataset, labeled_idxs, unlabed_idxs, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    l_num_items = int(len(labeled_idxs) / num_users)
    dict_users_l, l_all_idxs = {}, [i for i in range(len(labeled_idxs))]
    u_num_items = int(len(unlabed_idxs) / num_users)
    dict_users_u, u_all_idxs = {}, [i for i in range(len(unlabed_idxs))]

    p_ratio = np.ones(num_users) / num_users
    ni_l = np.zeros(num_users) / num_users
    ni_u = np.zeros(num_users) / num_users
    for i in range(num_users):
        l_local = np.random.choice(l_all_idxs, l_num_items, replace=False)

        dict_users_l[i] = np.array(labeled_idxs)[l_local]
        ni_l[i] = len(dict_users_l[i])
        u_local = np.random.choice(u_all_idxs, u_num_items, replace=False)
        dict_users_u[i] = np.array(unlabed_idxs)[u_local]
        l_all_idxs = list(set(l_all_idxs) - set(l_local))
        u_all_idxs = list(set(u_all_idxs) - set(u_local))
        ni_u[i] = len(dict_users_u[i])
        # y_rand = torch.rand([num_unlabel,10])

    ni = ni_l + ni_u
    p_ratio = ni / np.sum(ni)

    return dict_users_l, dict_users_u, p_ratio, ni_u


def mnist_noniid(dataset, labeled_idxs, unlabed_idxs, num_users):
    l_num_items = int(len(labeled_idxs) / num_users)  # 60
    # u_num_items = int(len(unlabed_idxs)/num_users)
    proportions = np.random.dirichlet(np.repeat(num_users, num_users))
    u_num_items = len(unlabed_idxs) * proportions
    u_num_items = u_num_items.astype(int)
    num_shards = 20  # 20, 50, 100
    num_imgs_l = int(len(labeled_idxs) / num_shards)  # 30
    num_imgs_u = int(len(unlabed_idxs) / num_shards)
    # p_ratio = np.ones(num_users)/num_users
    idx_shard = [i for i in range(num_shards)]
    idx_shard_un = [i for i in range(num_shards)]
    dict_users_l = {i: np.array([]) for i in range(num_users)}
    dict_users_u = {i: np.array([]) for i in range(num_users)}
    u_all_idxs = [i for i in range(len(unlabed_idxs))]
    idxs_l = np.arange(num_shards * num_imgs_l * num_users)
    # labels = np.array(dataset.targets)

    # sort labels
    # idxs_labels = np.vstack((idxs_l, labeled_idxs))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs_l = idxs_labels[0, :]

    # sort labels
    # idxs_u = np.arange(num_shards*num_imgs_u*num_users)
    # idxs_unlabels = np.vstack((idxs_u, unlabed_idxs))
    # idxs_unlabels = idxs_unlabels[:, idxs_labels[1, :].argsort()]
    # idxs_u = idxs_labels[0, :]
    # divide and assign 2 shards/client

    ni_l = np.zeros(num_users) / num_users
    ni_u = np.zeros(num_users) / num_users
    for i in range(num_users):
        rand_set = np.random.choice(idx_shard, int(l_num_items / num_imgs_l), replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users_l[i] = np.concatenate(
                (dict_users_l[i], labeled_idxs[rand * num_imgs_l:(rand + 1) * num_imgs_l]), axis=0)
        ni_l[i] = len(dict_users_l[i])

    np.random.shuffle(unlabed_idxs)
    for i in range(num_users):
        u_local = np.random.choice(u_all_idxs, u_num_items[i], replace=False)
        dict_users_u[i] = np.array(unlabed_idxs)[u_local]
        u_all_idxs = list(set(u_all_idxs) - set(u_local))
        #  rand_set = np.random.choice(idx_shard_un, int(u_num_items/num_imgs_u), replace=False)
        # idx_shard_un = list(set(idx_shard_un) - set(rand_set))
        # for rand in rand_set:
        #  dict_users_u[i] = np.concatenate(
        # (dict_users_u[i], unlabed_idxs[rand*num_imgs_u:(rand+1)*num_imgs_u]), axis=0)
        ni_u[i] = len(dict_users_u[i])

    ni = ni_l + ni_u
    p_ratio = ni / np.sum(ni)

    """
    l_num_items = int(len(labeled_idxs)/num_users) #600
    dict_users_l, l_all_idxs = {}, [i for i in range(len(labeled_idxs))]
    u_num_items = int(len(unlabed_idxs)/num_users) #5400
    dict_users_u, u_all_idxs = {},[i for i in range(len(unlabed_idxs))]
    num_classes=10
    
    class_label=np.array([dataset.targets[i] for i in labeled_idxs])
    
    labeled_idxs=np.array(labeled_idxs)
    #print(labeled_idxs)
    class_label=np.resize(class_label,(1,6000))
    labeled_idxs=np.resize(labeled_idxs,(1,6000))

   

    # divide and assign、
    unlabel_soft = {}
    p_ratio = np.ones(num_users)/num_users

    labeled_i = np.zeros((num_users,l_num_items))
    for i in range(num_classes):
        indexes = labeled_idxs[class_label==i]
        
        np.random.shuffle(indexes)
        labeled_i[i]=indexes[:]
        
    
    for user in range(num_users):
      
      if user!=9:
        rand1=np.random.choice(600,300,replace=False) 
        rand2=np.random.choice(600,300,replace=False) 
        con=np.concatenate((labeled_i[user][rand1],labeled_i[user+1][rand2]))
        dict_users_l[user]=con
      if user==9:
        rand1=np.random.choice(600,300,replace=False) 
        rand2=np.random.choice(600,300,replace=False) 
        con=np.concatenate((labeled_i[user][rand1],labeled_i[0][rand2]))
        dict_users_l[user]=con


    
    for i in range(num_users):
        u_local = np.random.choice(u_all_idxs, u_num_items,replace=False) 
        dict_users_u[i] = np.array(unlabed_idxs)[u_local]
        u_all_idxs = list(set(u_all_idxs) - set(u_local))
        num_unlabel= len(u_local)   
        y_rand = torch.ones(num_unlabel,10)
        target_p = y_rand / y_rand.sum(dim=1, keepdim=True)
        target_p = target_p.type(torch.cuda.FloatTensor)
        with torch.no_grad(): 
             unlabel_soft[i] = target_p.detach()
    """

    return dict_users_l, dict_users_u, p_ratio, ni_u


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, labeled_idxs, unlabed_idxs, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    l_num_items = int(len(labeled_idxs) / num_users)
    dict_users_l, l_all_idxs = {}, [i for i in range(len(labeled_idxs))]
    u_num_items = int(len(unlabed_idxs) / num_users)
    dict_users_u, u_all_idxs = {}, [i for i in range(len(unlabed_idxs))]

    p_ratio = np.ones(num_users) / num_users
    ni_l = np.zeros(num_users) / num_users
    ni_u = np.zeros(num_users) / num_users
    for i in range(num_users):
        l_local = np.random.choice(l_all_idxs, l_num_items, replace=False)

        dict_users_l[i] = np.array(labeled_idxs)[l_local]
        ni_l[i] = len(dict_users_l[i])
        u_local = np.random.choice(u_all_idxs, u_num_items, replace=False)
        dict_users_u[i] = np.array(unlabed_idxs)[u_local]
        l_all_idxs = list(set(l_all_idxs) - set(l_local))
        u_all_idxs = list(set(u_all_idxs) - set(u_local))
        ni_u[i] = len(dict_users_u[i])
        # y_rand = torch.rand([num_unlabel,10])

    ni = ni_l + ni_u
    p_ratio = ni / np.sum(ni)

    return dict_users_l, dict_users_u, p_ratio, ni_u


def cifar_noniid(dataset, labeled_idxs, unlabed_idxs, num_users):
    l_num_items = int(len(labeled_idxs) / num_users)  # 400
    # u_num_items = int(len(unlabed_idxs)/num_users)
    proportions = np.random.dirichlet(np.repeat(num_users, num_users))
    u_num_items = len(unlabed_idxs) * proportions
    u_num_items = u_num_items.astype(int)
    num_shards = 100
    num_imgs_l = int(len(labeled_idxs) / num_shards)

    # p_ratio = np.ones(num_users)/num_users
    idx_shard = [i for i in range(num_shards)]
    idx_shard_un = [i for i in range(num_shards)]
    dict_users_l = {i: np.array([]) for i in range(num_users)}
    dict_users_u = {i: np.array([]) for i in range(num_users)}
    u_all_idxs = [i for i in range(len(unlabed_idxs))]
    idxs_l = np.arange(num_shards * num_imgs_l * num_users)
    # labels = np.array(dataset.targets)

    # sort labels
    # idxs_labels = np.vstack((idxs_l, labeled_idxs))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs_l = idxs_labels[0, :]

    # sort labels
    # idxs_u = np.arange(num_shards*num_imgs_u*num_users)
    # idxs_unlabels = np.vstack((idxs_u, unlabed_idxs))
    # idxs_unlabels = idxs_unlabels[:, idxs_labels[1, :].argsort()]
    # idxs_u = idxs_labels[0, :]
    # divide and assign 2 shards/client

    ni_l = np.zeros(num_users) / num_users
    ni_u = np.zeros(num_users) / num_users
    for i in range(num_users):
        rand_set = np.random.choice(idx_shard, int(l_num_items / num_imgs_l), replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users_l[i] = np.concatenate(
                (dict_users_l[i], labeled_idxs[rand * num_imgs_l:(rand + 1) * num_imgs_l]), axis=0)
        ni_l[i] = len(dict_users_l[i])

    np.random.shuffle(unlabed_idxs)
    for i in range(num_users):
        u_local = np.random.choice(u_all_idxs, u_num_items[i], replace=False)
        dict_users_u[i] = np.array(unlabed_idxs)[u_local]
        u_all_idxs = list(set(u_all_idxs) - set(u_local))
        #  rand_set = np.random.choice(idx_shard_un, int(u_num_items/num_imgs_u), replace=False)
        # idx_shard_un = list(set(idx_shard_un) - set(rand_set))
        # for rand in rand_set:
        #  dict_users_u[i] = np.concatenate(
        # (dict_users_u[i], unlabed_idxs[rand*num_imgs_u:(rand+1)*num_imgs_u]), axis=0)
        ni_u[i] = len(dict_users_u[i])

    ni = ni_l + ni_u
    p_ratio = ni / np.sum(ni)

    # l_num_items = int(len(labeled_idxs)/num_users) #400
    # dict_users_l, l_all_idxs = {}, [i for i in range(len(labeled_idxs))]
    # u_num_items = int(len(unlabed_idxs)/num_users) #4600
    # dict_users_u, u_all_idxs = {},[i for i in range(len(unlabed_idxs))]
    # num_classes=10

    # class_label=np.array([dataset.targets[i] for i in labeled_idxs])

    # labeled_idxs=np.array(labeled_idxs)
    ##print(labeled_idxs)
    # class_label=np.resize(class_label,(1,4000))
    # labeled_idxs=np.resize(labeled_idxs,(1,4000))

    ## divide and assign、
    # unlabel_soft = {}
    # p_ratio = np.ones(num_users)/num_users

    # labeled_i = np.zeros((num_users,l_num_items))
    # for i in range(num_classes):
    # indexes = labeled_idxs[class_label==i]

    # np.random.shuffle(indexes)
    # labeled_i[i]=indexes[:]

    # for user in range(num_users):

    # if user!=9:
    # rand1=np.random.choice(400,200,replace=False)
    # rand2=np.random.choice(400,200,replace=False)
    # con=np.concatenate((labeled_i[user][rand1],labeled_i[user+1][rand2]))
    # dict_users_l[user]=con
    # if user==9:
    # rand1=np.random.choice(400,200,replace=False)
    # rand2=np.random.choice(400,200,replace=False)
    # con=np.concatenate((labeled_i[user][rand1],labeled_i[0][rand2]))
    # dict_users_l[user]=con

    # for i in range(num_users):
    # u_local = np.random.choice(u_all_idxs, u_num_items,replace=False)
    # dict_users_u[i] = np.array(unlabed_idxs)[u_local]
    #  u_all_idxs = list(set(u_all_idxs) - set(u_local))
    # num_unlabel= len(u_local)
    # y_rand = torch.ones(num_unlabel,10)
    # target_p = y_rand / y_rand.sum(dim=1, keepdim=True)
    # target_p = target_p.type(torch.cuda.FloatTensor)
    # with torch.no_grad():
    # unlabel_soft[i] = target_p.detach()

    """  
    l_num_items = int(len(labeled_idxs)/num_users)
    u_num_items = int(len(unlabed_idxs)/num_users)
    num_shards, num_imgs = 1250, 40
    p_ratio = np.ones(num_users)/num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users_l = {i: np.array([]) for i in range(num_users)}
    dict_users_u = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign、
    unlabel_soft = {}
    for i in range(num_users):
        rand_set = np.random.choice(idx_shard, int(l_num_items/num_imgs), replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users_l[i] = np.concatenate(
                (dict_users_l[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        print(len(dict_users_l[i]))
    for i in range(num_users):
        rand_set = np.random.choice(idx_shard, int(u_num_items/num_imgs), replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users_u[i] = np.concatenate(
                (dict_users_u[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        print(len(dict_users_u[i]))
   
        y_rand = torch.ones(u_num_items,10)
        target_p = y_rand / y_rand.sum(dim=1, keepdim=True)
        target_p = target_p.type(torch.cuda.FloatTensor)
        with torch.no_grad(): 
             unlabel_soft[i] = target_p.detach()
    """

    return dict_users_l, dict_users_u, p_ratio, ni_u
