"""
======================
@author:Mr.li
@time:2021/12/21:19:39
@email:1035626671@qq.com
======================
"""
# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']





def plotloss(loss,conf):

    # loss_client0 = []
    # loss_client1 = []
    # loss_client2 = []
    #
    # for i in range(0,len(loss),3):
    #     loss_client0.append(loss[i])
    #     loss_client1.append(loss[i+1])
    #     loss_client2.append(loss[i+2])
    #

    x = range(conf["global_epochs"])


    #plt.plot(x, y, 'ro-')
    #plt.plot(x, y1, 'bo-')
    #pl.xlim(-1, 11)  # 限定横轴的范围
    #pl.ylim(-1, 110)  # 限定纵轴的范围
    plt.plot(x, loss[0], mec='r', mfc='w',label=u'client0')
    plt.plot(x, loss[1], mec='r', mfc='w',label=u'client1')
    plt.plot(x, loss[2], mec='r', mfc='w',label=u'client2')


    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"epoch") #X轴标签
    plt.ylabel("loss") #Y轴标签
    # plt.title("Accuracy gained by free-riding") #标题

    plt.show()