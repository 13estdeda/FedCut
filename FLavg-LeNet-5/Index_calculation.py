"""
======================
@author:Mr.li
@time:2022/6/1:14:05
@email:1035626671@qq.com
======================
"""
from collections import defaultdict
import numpy as np


import matplotlib.pyplot as plt


def _WFD_create_distances(last_weight_all):
    distances = defaultdict(dict)
    for i in range(len(last_weight_all)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(last_weight_all[i] - last_weight_all[j], ord=None)
    return distances


def Index_calculation(conf,last_weight_all):

    distance_sum_all = [0.0 for _ in range(conf["no_models"] )]
    distances = _WFD_create_distances(last_weight_all)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors)

        distance_sum_all[user] = current_error

    print(distance_sum_all)



