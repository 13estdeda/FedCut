"""
======================
@author:Mr.li
@time:2022/6/12:19:39
@email:1035626671@qq.com
======================
"""
import torch

def Cosine_similarity(updated_models,server_global_model,device=None):
    grads = compute_grad_update(updated_models,server_global_model,device)

    cos_all = []
    for __, grads_y in enumerate(grads):
        y = (grads_y)
        cos_sum = 0
        for __, grads_x in enumerate(grads):
            x = (grads_x)
            cos = torch.nn.functional.cosine_similarity(x, y, 0)
            cos_sum += cos.item()

        cos_all.append((cos_sum/len(grads)))

    cos_all = (torch.div(torch.tensor(cos_all).float(),torch.sum(torch.tensor(cos_all).float()))).tolist()

    print("cos:",cos_all)


    return cos_all





def compute_grad_update(updated_models, server_global_model, device=None):

    grads = []

    for i in range(len(updated_models)):
        if device:
            local_model, sever_model = updated_models[i].to(device), server_global_model.to(device)

        grad = flatten([(local_param.data - sever_param.data) for local_param, sever_param in zip(local_model.parameters(), sever_model.parameters())])
        grads.append(grad)
    return grads



def flatten(grad_update):
	return torch.cat([update.data.view(-1) for update in grad_update])