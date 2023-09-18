import models, torch
import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
class Server(object):

    def __init__(self, conf, test_datasets):

        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])

        self.test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=self.conf["batch_size"], shuffle=True)



    def global_param_update(self, local_models, global_model,e):

        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data


        for _,(name, param) in enumerate(global_model.named_parameters()):
                weights[name] /= len(local_models)
                param.data = weights[name]


        return weights

    def issue_client(self,client_model,weights):
        for _, (name_client, param_client) in enumerate(client_model.named_parameters()):
                param_client.data = weights[name_client]
        return client_model




    #
    # def global_param_update_2(self, diff_all, global_model,e):
    #
    #     grad = {}
    #
    #     weight = {}
    #
    #     for diff in diff_all:
    #         for name, params in global_model.state_dict().items():
    #             if name not in grad:
    #                 grad[name] = torch.zeros_like(params.data)
    #                 weight[name] = torch.zeros_like(params.data)
    #             grad[name].add_(diff[name])
    #
    #     # 单双层上传
    #     if e % 2 == 0:
    #         i=0
    #     if e % 2 == 1:
    #         i=2
    #     base = 2
    #     for _, (name, param) in enumerate(global_model.named_parameters()):
    #         print(name)
    #         if _ == i or _ == i+1:
    #
    #             if _ == (len(diff_all[0]) - 1) or _ == (len(diff_all[0]) - 1) - 1:  # 最后层传全0
    #                 grad[name] = torch.zeros_like(grad[name])
    #                 param.data += grad[name]
    #                 weight[name] = param.data
    #
    #             # 单双层上传
    #             param.data += grad[name]
    #             weight[name] = param.data
    #
    #
    #             base -= 1
    #             if base==0:
    #                 i = i + 4
    #                 base=2
    #
    #
    #
    #     return weight
    def global_param_update_2(self, diff_all):

        weight_accumulator = {}

        for diff in diff_all:
            for name, params in self.global_model.state_dict().items():
                if name not in weight_accumulator:
                    weight_accumulator[name] = torch.zeros_like(params)

                weight_accumulator[name].add_(diff[name])


        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] / float(len(diff_all))
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)



    def global_param_update_sub(self, diff_all, global_model,e):

        donwload = {}

        weight = {}

        for diff in diff_all:
            for name, params in global_model.state_dict().items():
                if name in diff_all[0]:
                    if name not in weight:
                        donwload[name] = torch.zeros_like(params.data)
                        weight[name] = torch.zeros_like(params.data)

                    weight[name].add_(diff[name])


        for name, data in self.global_model.state_dict().items():
            if name in diff_all[0]:
                update_per_layer = weight[name] / float(len(diff_all))
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)


        return weight


    def issue_client_sub(self, client_model, weight,n):
        # for name_client, param_client in client_model.named_parameters():
        #     if name_client in weight:
        #         param_client.data = weight[name_client]

        for name, data in client_model.state_dict().items():
            if name in weight:
                update_per_layer = weight[name] / float(n)
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)


        return client_model

    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                # data = data.cuda()
                # target = target.cuda()

            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
