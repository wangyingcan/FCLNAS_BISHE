#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_uneuqal, cifar_noniid_uneuqal_test

import os
import logging
import shutil
import torchvision.datasets as dset
import numpy as np
import preproc


def save_checkpoint(state, ckpt_dir, hardware_class, is_best=False):
    filename = os.path.join(ckpt_dir,
                            hardware_class + 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir,
                                     hardware_class + 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def all_img_idx_for_debug(dataset):
    """
    return train_dataset, test_dataset, user_groups
    Get torchvision dataset
    """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
    else:
        raise ValueError(dataset)

    # trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)./dataset/cifar10
    trn_data = dset_cls(root='/home/grouplcheng/data/zch/dataset/cifar10/train',
                        # /home/grouplcheng/leo/zch/fedPlNAS/search/dataset /home/grouplcheng/data/zch/dataset
                        train=True,
                        download=True)
    # sample training data amongst users
    ret = []
    for i in range(len(trn_data)):
        ret.append(i)
    return ret


def set_target_hardware(idx=0):
    if idx in [0, 1, 2]:
        return 'gpu8'
    elif idx in [3, 4, 5]:
        return 'cpu'
    elif idx in [6, 7, 8, 9]:
        return 'flops'


def get_train_user_image_id(dataset, num_users, unequal=0, iid=True):
    """
    return train_dataset, test_dataset, user_groups
    Get torchvision dataset
    """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
    else:
        raise ValueError(dataset)

    # trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)./dataset/cifar10 /home/grouplcheng/data/zch/dataset
    trn_data = dset_cls(root='/home/grouplcheng/data/zch/dataset/cifar10/train',
                        train=True,
                        download=True)
    # sample training data amongst users
    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(trn_data, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unequal:
            # Chose uneuqal splits for every user
            user_groups = cifar_noniid_uneuqal(trn_data, num_users)
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(trn_data, num_users)

    ret = user_groups
    return ret


def get_test_img_id(dataset, num_users, unequal=0, iid=True):
    """
    return train_dataset, test_dataset, user_groups
    Get torchvision dataset
    """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
    else:
        raise ValueError(dataset)

    # trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    tst_data = dset_cls(root='/home/grouplcheng/data/zch/dataset/cifar10/val',
                        train=False,
                        download=True)
    # sample training data amongst users
    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(tst_data, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unequal:
            # Chose uneuqal splits for every user
            user_groups = cifar_noniid_uneuqal_test(tst_data, num_users)
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid_uneuqal_test(tst_data, num_users)  # cifar_noniid(tst_data, num_users)

    ret = user_groups
    return ret


def get_trn_data(dataset, data_path, num_users, cutout_length, unequal=0, iid=True):
    """
    return train_dataset, test_dataset, user_groups
    Get torchvision dataset
    """
    dataset = dataset.lower()
    dset_cls = dset.CIFAR10
    n_classes = 10

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # sample training data amongst users
    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(trn_data, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(trn_data, num_users)

    # assuming shape is NHW or NHWC
    shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data, user_groups]

    return ret


def get_test_data(dataset, data_path, num_users, cutout_length, unequal=0, iid=True):
    """
    return train_dataset, test_dataset, user_groups
    Get torchvision dataset
    """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    tst_data = dset_cls(root=data_path, train=False, download=True, transform=trn_transform)

    # sample training data amongst users
    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(tst_data, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(tst_data, num_users)

    # assuming shape is NHW or NHWC
    shape = tst_data.test_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, tst_data, user_groups]

    return ret


def get_dataset(args):
    """ return train_dataset, test_dataset, user_groups
    train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def _error_at_init_w_avg_0(w, list_local_sample_weight):
    """
    Returns the average of the weights.
    """
    list_local_sample_weight = torch.tensor(list_local_sample_weight)
    total_weight = torch.sum(list_local_sample_weight)

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * list_local_sample_weight[i]
        w_avg[key] = torch.div(w_avg[key], total_weight)
    return w_avg


def average_weights(w, list_local_sample_weight):
    """
    Returns the average of the weights.
    """
    # 过滤掉本地数据量为 0 的客户端，避免 total_weight=0 导致 NaN
    filtered_items = [
        (state_dict, weight)
        for state_dict, weight in zip(w, list_local_sample_weight)
        if weight > 0
    ]
    if len(filtered_items) == 0:
        raise ValueError("average_weights: no client has positive sample weight.")

    w_filtered, weight_filtered = zip(*filtered_items)
    weight_tensor = torch.tensor(weight_filtered)
    total_weight = torch.sum(weight_tensor)
    # 用 0 初始化，再按权重求和，避免初始时放大第一个客户端的参数
    w_avg = copy.deepcopy(w_filtered[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * 0
    for key in w_avg.keys():
        for i in range(len(w_filtered)):
            w_avg[key] += w_filtered[i][key] * weight_tensor[i]
        w_avg[key] = torch.div(w_avg[key], total_weight)
    return w_avg


def average_alphas(alphas, list_local_sample_weight):
    """
    input: alphas_list_of_local_user sample_weight
    Return: the average of the weights.

    def alphas(self):
        for p in self._alphas:
            yield p
    """
    list_local_sample_weight = torch.tensor(list_local_sample_weight)
    # 类型转换
    total_weight = torch.sum(list_local_sample_weight)
    alphas_avg = copy.deepcopy(alphas[0])

    # initial--------------------------------------------
    for key in alphas_avg.keys():
        alphas_avg[key] += alphas_avg[key] * list_local_sample_weight[0]
    # initial--------------------------------------------

    for key in alphas_avg.keys():
        for i in range(1, len(alphas)):
            alphas_avg[key] += alphas[i][key] * list_local_sample_weight[i]
        alphas_avg[key] = torch.div(alphas_avg[key], total_weight)
    return alphas_avg


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
