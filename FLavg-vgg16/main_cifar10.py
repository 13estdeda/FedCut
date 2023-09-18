import os
import torch,random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
import argparse, json
import datetime

import logging

import time
from server import *
# from client_resnet import *
from client import *

from dirichlet import *

from inver.invert_grad import *
from util.plot_loss import *
from util.k_means import *
from util.cos import *
import torch

from datasets import *
from grad_leage import *




if __name__ == '__main__':






	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf',default="./util/conf_cifar10.json" ,dest='conf')
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
	#method = "DLG"
#####################################################################################################
	if method == "inver":
		gt_data, gt_onehot_label,img_shape,dm,ds = data_one_inver(train_datasets, 0,conf)

		if conf["dataset"] == 'cifar10':
			num_channel = 3

		if conf["dataset"] == 'mnist':
			num_channel = 1

		model, model_seed = construct_model("ResNet20-4", num_classes=10, num_channels=num_channel)
		server.global_model = model

	if  method == "iDLG" or method == "DLG":
		gt_data, gt_onehot_label = data_one_dig(train_datasets, 0)
#####################################################################################################




	clients = []

	models_client = []

	for c in range(conf["no_models"]):  # 本地客户端
		clients.append(Client(conf, deepcopy(server.global_model), train_datasets,test_datasets,data_indices, c))
		models_client.append(deepcopy(server.global_model))

		


	for e in range(conf["global_epochs"]):
		updated_models = []
		diff_all = []

		for i,c in enumerate(clients):
			# diffs,new_model,dy_dx = c.local_train_subsection(deepcopy(models_client[i]),i,e)
			diffs,new_model,dy_dx = c.local_train_sub(deepcopy(models_client[i]),i,e)
			# diffs,new_model,dy_dx = c.local_train(deepcopy(models_client[i]),i,e)
			#
			print("+")


			###############梯度泄露
			if method == "inver" and i==0:
				get_ground_truth(deepcopy(new_model), gt_data, gt_onehot_label,img_shape, dm, ds)

			if (method == "iDLG" or method == "DLG") and i == 0:
				original_dy_dx = dy_dx
				origin(deepcopy(new_model), gt_data, gt_onehot_label, original_dy_dx, method)
			###############



			models_client[i] = deepcopy(new_model)
			acc = c.verification(models_client[i])
			print("client %d, acc: %f\n" % (i, acc))



			diff_all.append(diffs)

		#####################全部上传
		# weights = server.global_param_update(models_client,server.global_model,e)
		# for i,model in enumerate(models_client):
		# 	models_client[i] = deepcopy(server.global_model)



		#####################分层上传
		weight = server.global_param_update_sub(diff_all, server.global_model, e)
		for i,model in enumerate(models_client):
			client_model = server.issue_client_sub(model,weight,conf["no_models"])
			models_client[i] = client_model

			#############梯度泄露 本地客户端接受全局模型
			# if i == 0:
			# 	origin(deepcopy(models_client[i]), gt_data, gt_onehot_label,original_dy_dx,method)






		acc, loss_sever = server.model_eval()

		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss_sever))



