"""
======================
@author:Mr.li
@time:2022/6/12:16:34
@email:1035626671@qq.com
======================
"""
# -*- coding: utf-8 -*-
# python实现聚类
# import numpy as np
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# 加载数据集
# def loadDataSet(fileName):
#     dataMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         """
#         strip()将line行开头和结尾的空白符号删掉。split('\t') 函数以‘\t’为切分符号对每行切分
#         split函数返回的结果是一个列表，赋值给curLine
#         """
#         curLine = line.strip().split('\t')
#         """#map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
#         并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。"""
#         fltLine = map(float, curLine)
#         dataMat.append(fltLine)
#     return dataMat
#
#
# # 计算欧式距离的函数
# def countDist(vecA, vecB):
#     return sqrt(sum(power(vecA - vecB, 2)))
#
#
# # 测试countDist
# # vecA,vecB = np.array([1,2]),np.array([3,4])
# # x = countDist(vecA,vecB)
# # print(x)
#
# # 随机设置k个质心
# def randCent(dataSet, k):
#     n = shape(dataSet)[1]
#     # 初始化一个k行n列的二维数组，数组初始值全部为0，然后用mat函数将其转化为矩阵
#     centroids = mat(zeros([k, n]))
#     for j in range(n):
#         minj = min(dataSet[:, j])
#         rangej = float(max(dataSet[:, j]) - minj)
#         centroids[:, j] = mat(minj + rangej * random.rand(k, 1))
#     return centroids
#
#
# # k-Means算法核心部分
# def kMeans(dataSet, k, distMeas=countDist, createCent=randCent):
#     # 将函数distEclud和randCent作为参数传进来，可以更好的封装kMeans函数
#     m = shape(dataSet)[0]
#
#     """ clusterAssment是一个m行2列的矩阵，第一列存放每个样本所属聚类中心的下标，
#         第二列存放该样本与该聚类中心的距离
#     """
#     clusterAssment = mat(zeros((m, 2)))
#     centroids = createCent(dataSet, k)
#     clusterChanged = True
#     while clusterChanged:
#         clusterChanged = False
#         # 下面两个for循环计算每一个样本到每一个聚类中心的距离
#         for i in range(m):  # 遍历样本
#             minDist = inf  # inf表示无穷大，-inf表示负无穷
#             minIndex = -1
#             for j in range(k):  # 遍历聚类中心
#                 distJI = distMeas(centroids[j, :], dataSet[i, :])
#                 if distJI < minDist:
#                     minDist = distJI
#                     minIndex = j
#             if clusterAssment[i, 0] != minIndex:
#                 clusterChanged = True
#             clusterAssment[i, :] = minIndex, minDist ** 2
#             # print(centroids)
#         for cent in range(k):  # 重新计算聚类中心，cent从0遍历到k
#             ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # nonzero函数：返回不为0元素的行列下标组成的元组
#             """  #clusterAssment[:,0]:取得clusterAssment中第0列元素。
#                  #clusterAssment[:,0].A：转置成一个1行m列的行矩阵
#                  #clusterAssment[:,0].A==cent ：将行矩阵中每一个元素与
#                  cent进行比较，得到一个1行m列、元素取值为True和false
#                  的矩阵，元素为true表示该样本是第cent个聚类中心的点
#                  #nonzero(clusterAssment[:,0].A==cent)[0]：获得元素值为
#                  True的元素下标，这些下标集合即为所有属于cent类的样 本下标
#                  #整句含义：取得数据集中属于第cent个簇的样本集
#             """
#             centroids[cent, :] = mean(ptsInClust, axis=0)
#     return centroids, clusterAssment
#
#
# def main():
#     # dataMat = mat(loadDataSet('testSet.txt'))  #加载数据集。若没有源数据，可用下面两句随机生成数据
#     dataMat = random.rand(100, 2)  # 随机生成100行2列的数据
#     dataMat = np.array([1,2,1,2,1,2,4,6,7,45,8,4,8]).reshape(-1, 1)
#
#     dataMat = mat(dataMat)
#     myCentroids, clustAssing = kMeans(dataMat, 3)
#     print(clustAssing)
#     print("==============================")
#     print(myCentroids)
#
#
#
#
#
#
# if __name__ == '__main__':
#
#     main()
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算欧式距离
def Distance(dataSet, centroids, k) -> np.array:
    dis = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  # 行数上复制k份，方便作差
        temp1 = diff ** 2
        temp2 = np.sum(temp1, axis=1)  # 按行相加
        dis_temp = temp2 ** 0.5
        dis.append(dis_temp)
    dis = np.array(dis)  # 转换为一个array类型
    return dis


# 更新质心
def Update_cen(dataSet, centroids, k):
    # 计算每个样本到质心的距离，返回值是array数组
    distance = Distance(dataSet, centroids, k)
    # print("输出所有样本到质心的距离：", distance)
    # 分组并计算新的质心
    minIndex = np.argmin(distance, axis=1)  # axis=1 返回每行最小值的索引
    # print("输出最小值索引", minIndex)
    newCentroids = pd.DataFrame(dataSet).groupby(minIndex).mean()
    # print("新的质心(dataframe)：", newCentroids)
    newCentroids = newCentroids.values
    # print("新的质心(值）：", newCentroids)

    # 计算变化量
    changed = newCentroids - centroids
    return changed, newCentroids


# k-means 算法实现
def kmeans(dataSet, k):
    # (1) 随机选定k个质心
    centroids = random.sample(dataSet, k)
    print("随机选定两个质心：", centroids)

    # (2) 计算样本值到质心之间的距离，直到质心的位置不再改变
    changed, newCentroids = Update_cen(dataSet, centroids, k)
    while np.any(changed):
        changed, newCentroids = Update_cen(dataSet, newCentroids, k)
    centroids = sorted(newCentroids.tolist())

    # (3) 根据最终的质心，计算每个集群的样本
    cluster = []
    dis = Distance(dataSet, centroids, k)  # 调用欧拉距离
    minIndex = np.argmin(dis, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minIndex):  # enumerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


# 创建数据集
def createDataSet():
    #return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
    return [1,2,3,1,24,5,7,9,4,2]



def kmeans_start(loss,cos_all):
    loss = (torch.div(torch.tensor(loss).float(), torch.sum(torch.tensor(loss).float()))).tolist()
    print("loss_normalization :",loss)
    dataset = []
    for i in range(len(loss)):
        dataset.append([])
        dataset[i].append(loss[i])
        dataset[i].append(cos_all[i])

    print("dataset:",dataset)

    centroids, cluster = kmeans(dataset, 2)  # 2 代表的是分为2类=2个质心
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)

    k_1 = []
    for i in cluster[0]:
        for j in range(len(loss)):
            if i[0]==loss[j]:
                k_1.append(j)

    k_2 = []
    for i in cluster[1]:
        for j in range(len(loss)):
            if i[0]==loss[j]:
                k_2.append(j)

    print('集群1索引为：%s' % k_1)
    print('集群2索引为：%s' % k_2)


    if len(k_1)>len(k_2):
        return k_1

    return k_2
