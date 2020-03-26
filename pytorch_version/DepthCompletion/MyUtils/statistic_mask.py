import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = 'E:\Projects\Datasets\LevelMaskDataset'

scenes = [len(os.listdir('%s/%s' %(DATA_DIR, i))) for i in os.listdir(DATA_DIR)]
xx = [i for i in range(10, 110, 10)]
print(scenes)


plt.figure(figsize=(10, 10), dpi=80)
# 再创建一个规格为 1 x 1 的子图
# plt.subplot(1, 1, 1)
# 柱子总数
N = 10
# 包含每个柱子对应值的序列
values = scenes
# 包含每个柱子下标的序列
index = xx
# 柱子的宽度
width = 2.0
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
# 设置横轴标签
plt.xlabel('Miss ratio')
# 设置纵轴标签
plt.ylabel('Mask Num')
# 添加标题
# plt.title('')
# 添加纵横轴的刻度
plt.xticks(index, ('0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'))
# plt.yticks(np.arange(0, 10000, 10))
# 添加图例
plt.legend(loc="upper right")
plt.show()

print(xx)