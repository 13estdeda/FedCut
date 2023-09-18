from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def get_dataset(dir, name,conf):
    if name == 'mnist':
        train_datasets = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())

        train_dataset, _ = train_test_split(train_datasets, test_size=0.2, random_state=42)  # 4万个样本 1万个样本

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_datasets = datasets.CIFAR10(dir, train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
        train_dataset, _ = train_test_split(train_datasets, test_size=0.2, random_state=42)  # 4万个样本 1万个样本


    if conf["no_iid"] == "yes":
        Y_train = 0
        data_indices = partition_data(conf, Y_train, train_datasets, 10, conf["beta"])
    else:
        data_indices = 0

   


    return train_datasets, test_dataset,data_indices



def partition_data(conf,Y_train,train_datasets_all, n_parties, beta):

    n_train = len(train_datasets_all)
    print(n_train)


    min_size = 0
    min_require_size =10
    K = 10



    if conf["dataset"] == "adult" or conf["dataset"] == "bank":
        K = 2
        min_require_size = 600





    if conf["dataset"] == "gtrsb":
        K = 43

    N = len(train_datasets_all)
    #net_dataidx_map = {}
    data_indices = []

    while min_size < min_require_size :
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




from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
setup = dict(device=device, dtype=torch.float)
To_image = transforms.ToPILImage()

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
import matplotlib.pyplot as plt
def  data_one_inver(data,img_index,conf):
    To_tensor = transforms.ToTensor()
    To_image = transforms.ToPILImage()
    device = "cuda" if torch.cuda.is_available() else "cpu"



    dm = torch.as_tensor(cifar100_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar100_std, **setup)[:, None, None]

    if conf["dataset"] == 'cifar10':
        # gt_data = torch.as_tensor(np.array(data[img_index][0].resize((32, 32), Image.BICUBIC)) / 255,
        #                                **setup)

        gt_data = torch.as_tensor(np.array(Image.open("./util/cifar10-cat-3.jpg").resize((32, 32), Image.BICUBIC)) / 255,
                                  **setup)
        gt_data = gt_data.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        num_channel = 3
        dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
    if conf["dataset"] == 'mnist':
        gt_data = torch.as_tensor(np.array(data[img_index][0].resize((28, 28), Image.BICUBIC)) / 255,
                                       **setup)
        gt_data = gt_data.sub(dm).div(ds).unsqueeze(0).contiguous()
        num_channel = 1
        dm = torch.as_tensor(mnist_mean, **setup)[:, None, None]
        ds = torch.as_tensor(mnist_std, **setup)[:, None, None]

    # gt_label = torch.Tensor([data[img_index][1]]).long().to(device)
    # gt_onehot_label = label_to_onehot(gt_label, 10)
    gt_onehot_label = torch.as_tensor((3,), device=setup['device'])  # 车是1 猫是3

    img_shape = (num_channel, gt_data.shape[2], gt_data.shape[3])

    return gt_data, gt_onehot_label,img_shape,dm,ds



def data_one_dig(data,img_index):

    gt_data = (data[img_index][0]).to(device) # image_index[i][0]表示的是第I张图片的data，image_index[i][1]表示的是第i张图片的lable
    gt_data = gt_data.view(1, *gt_data.size())

    gt_label = torch.Tensor([data[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label,10)

    plt.imshow(To_image(gt_data[0].cpu())) #imshow方法首先将二维数组的值标准化为0到1之间的值，然后根据指定的渐变色依次赋予每个单元格对应的颜色，就形成了热图
    plt.show()
    
    return gt_data,gt_onehot_label


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target