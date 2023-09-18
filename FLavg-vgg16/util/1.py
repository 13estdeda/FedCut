"""
======================
@author:Mr.li
@time:2021/11/17:9:05
@email:1035626671@qq.com
======================
"""
import numpy as np
import matplotlib.pyplot as plt


#
# plt.style.use("ggplot")
# from sklearn.model_selection import train_test_split
#
# #mnist
# x = ["2", "3", "4", "5"]
# y1 = [11, 11, 11, 11] #noise
# y2 = [22.5,12.53,10.83,9.16]
#
# # 柱状图和散点图不同，散点图的(x,y)均为数值变量
# # 柱状图的x轴显示分类变量，有两种处理方式
# # 方式1：自己创建x轴坐标，并提供对应的标签
# # 方式2：让Matplotlib自动完成映射
#
# # 方式1
# # xticks = np.arange(len(x))  # 每根柱子的x轴坐标
# # xlabels = x  # 每根柱子的标签
# # fig, ax = plt.subplots(figsize=(10, 7))
# # ax.bar(x=xticks, height=y, tick_label=xlabels)
#
# # 方式2（推荐）
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(
#     x=x,  # Matplotlib自动将非数值变量转化为x轴坐标
#     height=y1,  # 柱子高度，y轴坐标
#     width=0.4,  # 柱子宽度，默认0.8，两根柱子中心的距离默认为1.0
#     align="center",  # 柱子的对齐方式，'center' or 'edge'
#     color="green",  # 柱子颜色
#     # edgecolor="red",  # 柱子边框的颜色
#     linewidth=1.0  # 柱子边框线的大小
# )
#
# ax.bar(
#     x=x,  # Matplotlib自动将非数值变量转化为x轴坐标
#     height=y2,  # 柱子高度，y轴坐标
#     width=0.4,  # 柱子宽度，默认0.8，两根柱子中心的距离默认为1.0
#     align="center",  # 柱子的对齐方式，'center' or 'edge'
#     color="blue",  # 柱子颜色
#     # edgecolor="red",  # 柱子边框的颜色
#     linewidth=1.0  # 柱子边框线的大小
# )
#
#
# ax.set_title("Accuracy gained by free-riding", fontsize=15)
#
# plt.show()


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

labels = ['2', '3', '4', '5']
# y1 = [11, 11, 11, 11] #mnist
# y2 = [22 , 12, 10, 9]

# y1 = [21, 10, 11, 11] #cifar
# y2 = [23 , 20, 19, 25]

y1 = [51.38, 50.28, 50.59, 51.24] #adult
y2 = [57 , 57, 53, 59]

matplotlib.rcParams['xtick.labelsize'] = 26  #x轴 y轴字体大小
matplotlib.rcParams['ytick.labelsize'] = 26

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y1, width, label='noise')
rects2 = ax.bar(x + width/2, y2, width, label='ZL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('acc',fontsize=25)
ax.set_title('Accuracy gained by free-riding',fontsize=25)
ax.set_xticks(x)



ax.legend(fontsize=20)  ##显示图例
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

#
# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()
