"""
======================
@author:Mr.li
@time:2022/6/19:10:22
@email:1035626671@qq.com
======================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def origin(model,gt_data,gt_onehot_label,original_dy_dx,method):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    torch.manual_seed(1234)
    To_tensor = transforms.ToTensor()
    To_image = transforms.ToPILImage()


    criterion = cross_entropy_for_onehot
    optimizer = torch.optim.LBFGS([gt_data, gt_onehot_label])
    # model.train()
    dy_dx_all = []
    # for e in range(3):
    #     def closure():
    #         optimizer.zero_grad()
    #         pred = model(gt_data)
    #         y = criterion(pred, gt_onehot_label)
    #         dy_dx = torch.autograd.grad(y, model.parameters(),create_graph=True)
    #         if e==0 or e==5:
    #             dy_dx_all.append(list(dy_dx))
    #         y.backward()
    #
    #         return y
    #     optimizer.step(closure)

    # last_grad_1 =  torch.zeros_like(dy_dx_all[0][-1])
    # last_grad_2 =  torch.zeros_like(dy_dx_all[0][-2])
    # dy_dx_new = []
    # for i in range(len(dy_dx_all[0])):
    #     if i%2==0:
    #         dy_dx_one = dy_dx_all[0][i]
    #     else:
    #         dy_dx_one = dy_dx_all[1][i]
    #     dy_dx_new.append(dy_dx_one)

    # dy_dx_new[-1] = last_grad_1
    # dy_dx_new[-2] = last_grad_2
    #
    # print(dy_dx_all[0][1])
    # print("======================")
    # print(dy_dx_all[1][1])
    #
    #
    # dy_dx = tuple(dy_dx_new)

    # dy_dx = tuple(dy_dx_all[0])


    #
    # original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)  # 需要保存相关梯度信息
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(To_image(dummy_data[0].cpu()))

    # optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    lr = 0.1
    if method == "DLG":
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    elif method == "iDLG":
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(
            False)

    # net = copy.deepcopy(net)
    history = []
    for iters in range(3000):
        def closure():
            optimizer.zero_grad()  # 梯度清零

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)  # 是对某一维度的行进行softmax运算

            if method == "DLG":
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            if method == "iDLG":
                criterion_IDLG = nn.CrossEntropyLoss().to(device)
                dummy_loss = criterion_IDLG(dummy_pred, label_pred)

            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)  # faked数据得到的梯度

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # # 打包为元组的列表
                grad_diff += ((gx - gy) ** 2).sum()  # 计算fake梯度与真实梯度的均方损失

            #     grad_diff += 1 - torch.nn.functional.cosine_similarity(gx.flatten(),gy.flatten(),0, 1e-10)
            # grad_diff /= len(dummy_dy_dx)
            grad_diff.backward()  # 对损失进行反向传播    优化器的目标是fake_data, fake_label

            return grad_diff

        optimizer.step(closure)
        if iters % 100 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(To_image(dummy_data[0].cpu()))

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    plt.show()

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
