"""
======================
@author:Mr.li
@time:2022/6/20:13:20
@email:1035626671@qq.com
======================
"""
import numpy as np
import torch
from PIL import Image
import sys
import os
# print(os.path)
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
sys.path.append("../inver")
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
from inver.reconstruction_algorithms import *
from inver.inver_model import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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







def get_ground_truth(model,ground_truth,labels,img_shape,dm, ds):
    # dm = torch.as_tensor(cifar100_mean, **setup)[:, None, None]
    # ds = torch.as_tensor(cifar100_std, **setup)[:, None, None]
    #
    # if conf["dataset"] == 'cifar10':
    #     ground_truth = torch.as_tensor(np.array(Image.open(data).resize((32, 32), Image.BICUBIC)) / 255,
    #                                    **setup)
    #     ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
    #     num_channel = 3
    #     dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
    #     ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
    # if conf["dataset"] == 'mnist':
    #     ground_truth = torch.as_tensor(np.array(Image.open(data).resize((28, 28), Image.BICUBIC)) / 255,
    #                                    **setup)
    #     ground_truth = ground_truth.sub(dm).div(ds).unsqueeze(0).contiguous()
    #     num_channel = 1
    #     dm = torch.as_tensor(mnist_mean, **setup)[:, None, None]
    #     ds = torch.as_tensor(mnist_std, **setup)[:, None, None]
    #
    # model, model_seed = construct_model("ResNet20-4", num_classes=10, num_channels=num_channel)
    # model.to(**setup)
    # model.eval()
    #
    # label = 1
    # labels = torch.as_tensor((label,), device=setup['device'])  # 车是1 猫是3
    #
    # img_shape = (num_channel, ground_truth.shape[2], ground_truth.shape[3])
    #
    # plt.imshow(To_image((torch.clamp(ground_truth * ds + dm, 0, 1))[0].cpu()))
    # plt.show()
    # plt.clf()



    loss_fn = cross_entropy_for_onehot
    model.zero_grad()
    target_loss= loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())  # 输出梯度


    config = dict(signed=True,
                  boxed=True,
                  cost_fn="sim",
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim="adam",
                  restarts=10,
                  max_iterations=500,  # 24_000,
                  total_variation=0.00001,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss'

                  )
    num_images = 1
    rec_machine = GradientReconstructor(model, (dm, ds,ground_truth), config, num_images=num_images)
    output, stats,min_loss = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=False)


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
