"""
======================
@author:Mr.li
@time:2022/2/24:16:56
@email:1035626671@qq.com
======================
"""
import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from load_bank import *


def partition_data(conf,Y_train,train_datasets_all, n_parties, beta):

    n_train = len(train_datasets_all)
    print(n_train)





    min_size = 0
    min_require_size = 10
    K = 10

    if conf["dataset"] == "bank":
        min_require_size = 600


    if conf["dataset"] == "adult" or conf["dataset"] == "bank":
        K = 2
    if conf["dataset"] == "gtrsb":
        K = 43

    N = len(train_datasets_all)
    #net_dataidx_map = {}
    data_indices = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            if conf["dataset"] == "adult":
                idx_k = np.where(train_datasets_all.targets.cpu().numpy() == k)[0]
            elif conf["dataset"] == "cifar10":
                idx_k = np.where(torch.Tensor(train_datasets_all.targets) == k)[0]  # 将每类的索引拿出来
            elif conf["dataset"] == "bank":
                idx_k = np.where(train_datasets_all.y == k)[0]  # 将每类的索引拿出来
            elif conf["dataset"] == "gtrsb":
                idx_k = np.where(Y_train == k)[0]
            else:
                idx_k = np.where(train_datasets_all.targets == k)[0]  # 将每类的索引拿出来
            np.random.shuffle(idx_k)  # 随机打乱
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))  # 从狄利克雷分布中抽取样本
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        #net_dataidx_map[j] = idx_batch[j]
        data_indices.append(idx_batch[j])

    return data_indices
