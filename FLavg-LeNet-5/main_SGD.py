"""
======================
@author:Mr.li
@time:2023/1/17:19:22
@email:1035626671@qq.com
======================
"""
import argparse, json
import datetime
import os
import logging
import torch, random
import time
from server import *

import client
from dirichlet import *

from inver.invert_grad import *
from util.plot_loss import *

from util.cos import *
import torch

from datasets import *
from grad_leage import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default="./util/conf.json", dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, test_datasets, data_indices = get_dataset("./data/", conf["dataset"], conf)

    server = Server(conf, test_datasets)

    if conf["no_iid"] == "yes":
        data_indices = partition_data(conf, 0, train_datasets, 10, conf["beta"])

    method = ""
    # method = "inver"
    # method = "iDLG"
    # method = "DLG"
    #####################################################################################################
    if method == "inver":
        gt_data, gt_onehot_label, img_shape, dm, ds = data_one_inver(train_datasets, 0, conf)

        if conf["dataset"] == 'cifar10':
            num_channel = 3

        if conf["dataset"] == 'mnist':
            num_channel = 1

        model, model_seed = construct_model("ResNet20-4", num_classes=10, num_channels=num_channel)
        server.global_model = model

    if method == "iDLG" or method == "DLG":
        gt_data, gt_onehot_label = data_one_dig(train_datasets, 0)
    #####################################################################################################

    clients = []

    models_client = []

    for c in range(conf["no_models"]):  # 本地客户端
        clients.append(
            client.Client(conf, deepcopy(server.global_model), train_datasets, test_datasets, data_indices, c))
        models_client.append(deepcopy(server.global_model))

    loss_all = []

    for e in range(conf["global_epochs"]):
        updated_models = []

        grads_all = []
        for i, c in enumerate(clients):
            local_grads, new_model, dy_dx, total_l = c.local_train_sgd(deepcopy(models_client[i]), i, e)


            print("+")
            if i == 0:
                loss_all.append(total_l)  #loss

            # grads_all.append(local_grads)  #收集客户端上传的梯度信息

            ###############梯度泄露
            if method == "inver" and i == 0:
                get_ground_truth(deepcopy(new_model), gt_data, gt_onehot_label, img_shape, dm, ds)

            if (method == "iDLG" or method == "DLG") and i == 0:
                original_dy_dx = dy_dx
                origin(deepcopy(new_model), gt_data, gt_onehot_label, original_dy_dx, method)
            ###############

            models_client[i] = deepcopy(new_model)
            acc = c.verification(models_client[i])
            print("client %d, acc: %f\n" % (i, acc))



        #####################全部上传
        grads_all =server.load_grads()
        global_model = server.average_grads(grads_all)
        for i,model in enumerate(models_client):
            models_client[i] = deepcopy(global_model)




        #############梯度泄露 本地客户端接受全局模型
        # if i == 0:
        # 	origin(deepcopy(models_client[i]), gt_data, gt_onehot_label,original_dy_dx,method)

        acc, loss_sever = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss_sever))
        # if (e+1)%10==0:
        #     print("Epoch %d" % (e))
        # print("loss_all", loss_all)




