"""
======================
@author:Mr.li
@time:2021/10/7:17:07
@email:1035626671@qq.com
======================
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.neighbors import KernelDensity
from pylab import *  # 支持中文
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

mpl.rcParams['font.sans-serif'] = ['SimHei']



def picture_1(local_acc_final,sever_acc_final):
    free_rider_acc = []
    fair_acc1 = []
    fair_acc2 = []
    fair_acc3 = []
    fair_acc4 = []
    fair_acc5 = []
    sever_acc = []

    epoch = 50

    for _,i in enumerate(local_acc_final):
        free_rider_acc.append(i[-1])
        fair_acc1.append(i[0])
        fair_acc2.append(i[1])
        fair_acc3.append(i[2])
        fair_acc4.append(i[3])
        fair_acc5.append(i[4])



    x = range(epoch)
   # y1 = free_rider_acc[1:12]

    y1 = fair_acc1[:]
    y2 = fair_acc2[:]
    y3 = fair_acc3[:]
    y4 = fair_acc4[:]
    y5 = fair_acc5[:]

    y = []
    y.append(y1)
    y.append(y2)
    y.append(y3)
    y.append(y4)
    y.append(y5)

    max_y = y1
    min_y = y1
    for i in y:
        if i != y1:

            max_y = np.maximum(max_y,i)
            min_y = np.minimum(min_y,i)


    #y = sever_acc_final[:11]


    epoch = [ i for i in range(epoch)]

    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    plt.ylim(0, 100)
    # pl.ylim(-1, 110)  # 限定纵轴的范围
    plt.plot(x, free_rider_acc[1:12], marker='o', mec='r', mfc='w', label=u'free_rider_acc')
    #plt.plot(x, y1, marker='v',mec='b', ms=10, label=u'fair_acc')
   # plt.plot(x, y2, marker='v', mec='b', ms=10, label=u'fair_acc')
    plt.plot(x, sever_acc_final[:11], marker='*', mec='k',ms=10, label=u'sever_acc')
    plt.fill_between(x,
                     min_y,
                     max_y,
                     color='b',
                     alpha=0.2)

    plt.legend()  # 让图例生效
    plt.xticks(x, epoch, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"轮次")  # X轴标签
    plt.ylabel("ACC")  # Y轴标签
    plt.title("acc")  # 标题

    plt.savefig("./util/acc_1.jpg")

    plt.show()

    plt.clf()


def acc(local_acc_final,sever_acc_final,server_acc_2_final,epoch):
    free_rider_acc = []
    fair_acc1 = []
    fair_acc2 = []
    fair_acc3 = []
    fair_acc4 = []
    fair_acc5 = []
    sever_acc = []



    for _,i in enumerate(local_acc_final):
        free_rider_acc.append(i[-1])
        fair_acc1.append(i[0])
        # fair_acc2.append(i[1])
        # fair_acc3.append(i[2])
        # fair_acc4.append(i[3])
        # fair_acc5.append(i[4])



    x = range(epoch)


    plt.ylim(0, 100)
    # pl.ylim(-1, 110)  # 限定纵轴的范围
    # plt.plot(x, free_rider_acc[:], marker='o', mec='r', mfc='w', label=u'free_rider')
    # plt.plot(x, fair_acc1[:], marker='*', mec='k', ms=10, label=u'normal_client')

    # plt.plot(x, free_rider_acc[:], label='free_rider', linewidth=2, color='black', marker='o', markerfacecolor='red', markersize=4)
    # plt.plot(x, fair_acc1[:], label='normal_client', linewidth=2, color='red', marker='o', markerfacecolor='black', markersize=4)

    if sever_acc_final[-1] > server_acc_2_final[-1]:
        plt.plot(x, sever_acc_final[:], label='normal_client', linewidth=2, color='limegreen', marker='o',markerfacecolor='limegreen', markersize=4)
        plt.plot(x, server_acc_2_final[:], label='free_rider', linewidth=2, color='orange', marker='^',markerfacecolor='orange', markersize=4)
    else:
        plt.plot(x, sever_acc_final[:], label='free_rider', linewidth=2, color='orange', marker='^', markerfacecolor='orange',markersize=4)
        plt.plot(x, server_acc_2_final[:], label='normal_client', linewidth=2, color='limegreen', marker='o', markerfacecolor='limegreen',markersize=4)

    plt.legend()  # 让图例生效

    x_major_locator = MultipleLocator(5)

    y_major_locator = MultipleLocator(20)

    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #plt.xticks(x, epoch, rotation=45)


    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Communication rounds",size=15)  # X轴标签
    plt.ylabel("Accuracy (%)",size=15)  # Y轴标签
    # plt.title("acc")  # 标题

    plt.savefig("./util/acc.jpg")


    plt.show()

    plt.clf()



def picture_2(local_acc_final,sever_acc_final):
    size = 6
    x = np.arange(size)


    free_rider_acc = []
    fair_acc_1 = []
    fair_acc_2 = []
    fair_acc_3 = []
    fair_acc_4 = []
    fair_acc_5 = []

    for _, i in enumerate(local_acc_final):
        if( 0<_<7):
            free_rider_acc.append(i[-1])
            fair_acc_1.append(i[0])
            fair_acc_2.append(i[1])
            fair_acc_3.append(i[2])
            fair_acc_4.append(i[3])
            fair_acc_5.append(i[4])

    for i in range(len(sever_acc_final)):   #6轮
        if i < 6:
            fair_acc_1[i] = fair_acc_1[i] - sever_acc_final[i]   #当前本地模型的acc-上一轮服务器全局模型acc  即相对于上一轮，本轮的acc提升度
            fair_acc_2[i] = fair_acc_2[i] - sever_acc_final[i]
            fair_acc_3[i] = fair_acc_3[i] - sever_acc_final[i]
            fair_acc_4[i] = fair_acc_4[i] - sever_acc_final[i]
            fair_acc_5[i] = fair_acc_5[i] - sever_acc_final[i]
            free_rider_acc[i] = free_rider_acc[i] - sever_acc_final[i]


    total_width, n = 0.5, 6
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, free_rider_acc, width=width, label='free_rider')
    plt.bar(x + width, fair_acc_1, width=width, label='fair_1')
    plt.bar(x + 2 * width, fair_acc_2, width=width, label='fair_2')
    plt.bar(x + 3 * width, fair_acc_3, width=width, label='fair_3')
    plt.bar(x + 4 * width, fair_acc_4, width=width, label='fair_4')
    plt.bar(x + 5 * width, fair_acc_5, width=width, label='fair_5')
    plt.legend()

    plt.savefig("./util/acc_2.jpg")

    plt.show()

    plt.clf()




def picture_3(weight_all):
    # y = torch.from_numpy(weight_sever)
    for _, i in enumerate(weight_all):
        [f, xi] = KernelDensity(i.any())
        plot(xi, f)












import seaborn as sns
def relitu(Frequency_sum_all,data):

    for i in range(len(Frequency_sum_all)):
        total_ac = Frequency_sum_all[i]

        if data == "mnist" or data == "fmnist":
            total_ac = total_ac.reshape(84,120)
        elif data == "cifar10":
            # total_ac = total_ac.reshape(16, 32)
            # total_ac = total_ac.reshape(32, 86)
            # total_ac = total_ac.reshape(43, 64)
            total_ac = total_ac.reshape(32, 40)
        elif data == "adult":
        #     total_ac = total_ac.reshape(32, 86)
            total_ac = total_ac.reshape(43, 64)
        # # print(total_ac)

       # total_ac = total_ac.cpu().numpy()

        f, ax1 = plt.subplots(figsize=(6, 4), nrows=1)


        # cmap用cubehelix map颜色
        sns_plot = sns.heatmap(total_ac, linewidths=0.05, ax=ax1, cmap='YlGnBu',vmin=0, vmax=150)  #rainbow  YlGnBu

        # 热力图刻度大小
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)

        # cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
        # sns.heatmap(total_ac, linewidths=0.05, ax=ax1, cmap=cmap)
        ax1.set_title('',fontsize=20)

        # ax1.set_xlabel('row')
        ax1.set_xticklabels([])  # 设置x轴图例为空值
        ax1.set_yticklabels([])

        # if data == "mnist" or data == "fmnist":
        #     ax1.set_xlabel('120')
        #     ax1.set_ylabel('84')
        # elif data == "cifar10":
        #     ax1.set_xlabel('40')
        #     ax1.set_ylabel('32')
        # elif data == "adult":
        #     ax1.set_xlabel('64')
        #     ax1.set_ylabel('43')

        if data == "mnist" or data == "fmnist":
            ax1.set_xlabel('X-axis size: 120',size=20)
            ax1.set_ylabel('Y-axis size: 84',size=20)
        elif data == "cifar10":
            ax1.set_xlabel('X-axis size: 40',size=20)
            ax1.set_ylabel('Y-axis size: 32',size=20)
        elif data == "adult":
            ax1.set_xlabel('X-axis size: 64',size=20)
            ax1.set_ylabel('Y-axis size: 43',size=20)


        plt.show()

        plt.savefig("./util/" + str(i) + "relitu.jpg")
        plt.clf()



def relitu_distance(distance_list_all,e):
    total_ac = np.array(distance_list_all).reshape(6, 6)
    f, ax1 = plt.subplots(figsize=(6, 4), nrows=1)
    ax1.set_yscale('log')
    # cmap用cubehelix map颜色
    sns.heatmap(total_ac, linewidths=0.05, ax=ax1, cmap='rainbow')
    # cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    # sns.heatmap(total_ac, linewidths=0.05, ax=ax1, cmap=cmap)
    ax1.set_title('Activation Value Matrix',fontsize=20)
    # ax1.set_xlabel('row')
    ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_yticklabels([])  # 设置x轴图例为空值
    # ax1.set_ylabel('column')


    plt.savefig("./util/" + str(e)+ "relitu_distance.jpg")

    plt.clf()
    # plt.show()




def picture_avg(av_count_all):
    free_rider_avg = []
    fair_avg1 = []
    fair_avg2 = []
    fair_avg3 = []
    fair_avg4 = []
    fair_avg5 = []


    epoch = 20

    for _, i in enumerate(av_count_all):
        free_rider_avg.append(i[-1])
        fair_avg1.append(i[0])
        fair_avg2.append(i[1])
        fair_avg3.append(i[2])
        fair_avg4.append(i[3])
        fair_avg5.append(i[4])

    x = range(20)
    # y1 = free_rider_acc[1:12]

    # y1 = fair_acc1[1:12]
    # y2 = fair_acc2[1:12]
    # y3 = fair_acc3[1:12]
    # y4 = fair_acc4[1:12]
    # y5 = fair_acc5[1:12]


    epoch = [i for i in range(epoch)]

    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    plt.ylim(0, 30)
    # pl.ylim(-1, 110)  # 限定纵轴的范围
    plt.plot(x, free_rider_avg, mec='r', label=u'free_rider')

    plt.plot(x, fair_avg1, mec='b',  label=u'fair1')
    plt.plot(x, fair_avg2, mec='g',  label=u'fair2')
    plt.plot(x, fair_avg3, mec='c',  label=u'fair3')
    plt.plot(x, fair_avg4, mec='y',  label=u'fair4')
    plt.plot(x, fair_avg5, mec='k', label=u'fair5')

    # plt.fill_between(x,
    #                  min_y,
    #                  max_y,
    #                  color='b',
    #                  alpha=0.2)

    plt.legend()  # 让图例生效
    plt.xticks(x, epoch, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"轮次")  # X轴标签
    plt.ylabel("avg")  # Y轴标签
    plt.title("avg")  # 标题



    plt.show()

    plt.clf()


def picture_threshold_value_all(threshold_value_all):


    # 通过切片获取横坐标x1
    x1 = [ i for i in range(150)]
    y1 = []
    # 通过切片获取纵坐标R
    for threshold_value in threshold_value_all:
        y1.append(threshold_value[0])
        y1.append(threshold_value[1])
        y1.append(threshold_value[2])

    print(x1)
    print(y1)
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    # ax1.set_title('Result Analysis')

    # style =‘sci’ 指明用科学记数法；
    # scilimits = (-1, 2)表示对(0.1, 100)范围之外的值换科学记数法，范围内的数不换；
    # axis =‘y’ 指明对y轴用，亦可以是x，或者两者同时
    ax1.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(16)  # 设置1e6的大小与位置


    # 设置横坐标名称
    ax1.set_xlabel('iteration round',size=20)
    # 设置纵坐标名称
    ax1.set_ylabel('threshold value',size=20)
    # 画散点图
    ax1.scatter(x1, y1, c='r', marker='.')

    # 调整横坐标的上下界
    # plt.xlim(xmax=5, xmin=0)
    # plt.ylim(xmax=0., xmin=0)
    # 显示
    plt.show()

