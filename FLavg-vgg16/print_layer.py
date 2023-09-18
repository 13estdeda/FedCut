"""
======================
@author:Mr.li
@time:2023/7/1:15:11
@email:1035626671@qq.com
======================
"""

import argparse, json
import models, torch
import re

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('-c', '--conf', default="./util/conf.json", dest='conf')
args = parser.parse_args()

with open(args.conf, 'r') as f:
    conf = json.load(f)

global_model = models.get_model(conf["model_name"])



for _, (name, data) in enumerate(global_model.state_dict().items()):  # 更新后的本地模型
    print(name)


print("==========================================================================")



for _, (name, data) in enumerate(global_model.state_dict().items()):  # 更新后的本地模型
    # print(name)
    if _ <= 25:  # vgg16                  #   if _<=9: alenet
        print(name)

print("=====================================")

for _, (name, data) in enumerate(global_model.state_dict().items()):  # 更新后的本地模型
    # 全连接层
    if 26 <= _ <= 29:  # vgg16          if 10<=_<=13: alenet
        print(name)
