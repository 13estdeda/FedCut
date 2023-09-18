from copy import deepcopy
import models, torch, copy
from random import choices
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def add_noise(parameters, dp, dev,sigma=0.1):
    noise = None
    # sigma = 0.01
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape)).cuda()
    # 高斯噪声
    else:
        noise = torch.FloatTensor(parameters.shape).normal_(0, sigma).cuda()

    return noise

def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0  # np.inf
    device = input_gradient[0].device

    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in input_gradient]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None] * len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient

def compute_grad_update(old_model, new_model, device=None):
	# maybe later to implement on selected layers/parameters
	if device:
		old_model, new_model = old_model.to(device), new_model.to(device)
	return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]



class Client(object):

	def __init__(self, conf, model, train_dataset,test_datasets,data_indices, id = -1):

		self.conf = conf

		self.local_model = model

		self.client_id = id

		self.train_dataset = train_dataset

		self.test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=self.conf["batch_size"], shuffle=True)

		all_range = list(range(len(self.train_dataset)))
		# data_len = int(len(self.train_dataset) / 15)
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		if conf["no_iid"] == "yes":
			train_indices = data_indices[self.client_id]
		else:
			train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
								sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))

		print("客户端", self.client_id, "数据量：", len(train_indices))






	def local_train(self,model,c,epoch):

		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)

		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量
		for name, data in model.state_dict().items():  # 更新后的本地模型
			# print(data)
			diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		return diff,deepcopy(model),original_dy_dx

	def local_train_prox(self,model,c,epoch):

		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)

		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)

				proximal_term = 0.0
				for w, w_t in zip(model.parameters(), old_model.parameters()):
					proximal_term += (w - w_t).norm(2)

				loss = torch.nn.functional.cross_entropy(output, target)+ (0.01 / 2) * proximal_term
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量
		for name, data in model.state_dict().items():  # 更新后的本地模型
			# print(data)
			diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		return diff,deepcopy(model),original_dy_dx



	def local_train_scaffold(self,model,c,epoch):

		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		optimizer = ScaffoldOptimizer(model.parameters(), lr=self.conf['lr'], weight_decay=1e-4)
		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step(old_model.control,model.control)

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量
		for name, data in model.state_dict().items():  # 更新后的本地模型
			# print(data)
			diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		return diff,deepcopy(model),original_dy_dx



	def local_train_subsection(self,model,c,epoch):




		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)

		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model_1 = deepcopy(model)

		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model_2 = deepcopy(model)


		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量

		i = 0

		for _,(name, data) in enumerate(new_model_1.state_dict().items()):  # 更新后的本地模型
			# print(data)
			if _==i or _==i+1:
				diff[name] = (data - old_model.state_dict()[name])  # 更新的参数
				i=i+4


		j = 2
		for _, (name, data) in enumerate(new_model_2.state_dict().items()):  # 更新后的本地模型
			if _==j or _==j+1:
				diff[name] = (data - new_model_1.state_dict()[name])  # 更新的参数
				j=j+4




		return diff,deepcopy(model),original_dy_dx

	def local_train_sub(self,model,c,epoch):




		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		total_loss = 0.0
		dataset_size = 0
		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				dataset_size += data.size()[0]
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				total_loss += loss
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model = deepcopy(model)
		total_l = total_loss / dataset_size

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量


		if epoch%2==0:
			for _,(name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型

				if _<=3:   #features.features.28.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

					# noise=add_noise(data,1,dev=0,sigma=0.001)
					# diff[name] = (data - old_model.state_dict()[name])+noise#DP
					# a=gradient_compression (data - old_model.state_dict()[name],50)
					# a=torch.cat([torch.unsqueeze(a[i],0) for i in range (len(a))],0)
					#
					# diff[name] =a#GC
					#

					# diff[name] = gradient_clipping(data - old_model.state_dict()[name], 4)  # GS

		if epoch % 2 == 1:
			for _,(name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				#全连接层
				if 4<=_<=7:  #classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

					# noise=add_noise(data,1,dev=0,sigma=0.001)
					# diff[name] = (data - old_model.state_dict()[name])+noise#DP

					# diff[name] =gradient_compression (data - old_model.state_dict()[name],50)#GC
					#
					# diff[name] = gradient_clipping(data - old_model.state_dict()[name], 4)  # GS






		return diff,deepcopy(model),original_dy_dx,total_l

	def local_train_sub_3(self,model,c,epoch):




		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		total_loss = 0.0
		dataset_size = 0
		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				dataset_size += data.size()[0]
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)


				loss = torch.nn.functional.cross_entropy(output, target)
				total_loss += loss
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model = deepcopy(model)
		total_l = total_loss / dataset_size

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict() #收集梯度量


		if epoch%3==0:
			for _,(name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型

				if _<=2:   #features.features.28.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 3 == 1:
			for _,(name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				#全连接层
				if 2<=_<=3:  #classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 3 == 2:
			for _,(name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				#全连接层
				if 4<=_<=7:  #classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数



		return diff,deepcopy(model),original_dy_dx,total_l

	def local_train_sub_4(self, model, c, epoch):

		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		total_loss = 0.0
		dataset_size = 0
		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				dataset_size += data.size()[0]
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)

				loss = torch.nn.functional.cross_entropy(output, target)
				total_loss += loss
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model = deepcopy(model)
		total_l = total_loss / dataset_size

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict()  # 收集梯度量

		if epoch % 4 == 0:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型

				if _ <= 2:  # features.features.28.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 4 == 1:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 2 <= _ <= 3:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 4 == 2:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 4 <= _ <= 5:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 4 == 3:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 6 <= _ <= 7:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		return diff, deepcopy(model), original_dy_dx, total_l

	def local_train_sub_5(self, model, c, epoch):

		optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		total_loss = 0.0
		dataset_size = 0
		model.train()
		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				dataset_size += data.size()[0]
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = model(data)

				loss = torch.nn.functional.cross_entropy(output, target)
				total_loss += loss
				dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
				loss.backward()
				optimizer.step()

		new_model = deepcopy(model)
		total_l = total_loss / dataset_size

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		diff = dict()  # 收集梯度量

		if epoch % 5 == 0:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型

				if _ <= 1:  # features.features.28.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 5 == 1:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 1 <= _ <= 2:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 5 == 2:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 3 <= _ <= 4:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 5 == 3:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 4 <= _ <= 5:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		if epoch % 5 == 4:
			for _, (name, data) in enumerate(new_model.state_dict().items()):  # 更新后的本地模型
				# 全连接层
				if 6 <= _ <= 7:  # classifier.3.bias
					diff[name] = (data - old_model.state_dict()[name])  # 更新的参数

		return diff, deepcopy(model), original_dy_dx, total_l


	def verification(self, local_model):

		local_model.eval()
		correct = 0
		dataset_size = 0
		with torch.no_grad():
			for e in range(self.conf["local_epochs"]):

				for batch_id, batch in enumerate(self.test_loader):
					data, target = batch
					dataset_size += data.size()[0]

					if torch.cuda.is_available():
						data = data.cuda()
						target = target.cuda()

					# optimizer.zero_grad()
					output = local_model(data)
					# loss = torch.nn.functional.cross_entropy(output, target)
					# loss.backward()

					# optimizer.step()

					pred = output.data.max(1)[1]  # get the index of the max log-probability
					correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		return acc


	def local_train_sgd(self,model,c,epoch):




		# optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'],
		# 							momentum=self.conf['momentum'])

		old_model = deepcopy(model)
		total_loss = 0.0
		dataset_size = 0
		model.train()

		for batch_id, batch in enumerate(self.train_loader):
			data, target = batch
			dataset_size += data.size()[0]
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			# optimizer.zero_grad()
			output = model(data)
			loss = torch.nn.functional.cross_entropy(output, target)
			total_loss += loss
			dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
			loss.backward()
			#optimizer.step()

		new_model = deepcopy(model)
		total_l = total_loss / dataset_size



		local_grads = {'n_samples': len(self.train_loader.dataset), 'named_grads': {}}
		for name, param in model.named_parameters():
			local_grads['named_grads'][name] = param.grad

		torch.save(local_grads, './cache/grads_{}.pkl'.format(self.client_id))

		original_dy_dx = list((_.detach().clone() for _ in dy_dx))

		return local_grads,deepcopy(model),original_dy_dx,total_l

	def model_eval(self,model):
		model.eval()

		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.test_loader):
			data, target = batch
			dataset_size += data.size()[0]

			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()

			output = model(data)

			total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l



from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']
