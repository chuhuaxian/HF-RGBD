import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
depth = cv2.imread('E:\Projects\Datasets\IRS\Home_ArchvizKitchen\\0-depth.png', -1).astype(np.float32)
color = cv2.imread('E:\Projects\Datasets\IRS\Home_ArchvizKitchen\\0-color.jpg', -1)
color = color[:,:,(2, 1, 0)]
# color = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\0001-color.jpg')[:, :, (2,1,0)]
# depth = np.where(depth>10000, 10000, depth)
#
# d1 = depth[::-1, ::-1]
# c1 = color[:, :-1, :]
#
# res_wid, res_hei = 640, 480
# img_wid, img_hei = depth.shape[::-1]
# wid_range = img_wid - res_wid
# hei_range = img_hei - res_hei
#
# # print()
# start_wid, start_hei = random.randint(0, wid_range - 1), random.randint(0, hei_range - 1)
# box = (start_wid, start_hei, start_wid + res_wid, start_hei + res_hei)  # 设置要裁剪的区域
# d2 = depth[box[1]:box[3], box[0]:box[2]]
# c2 = color[box[1]:box[3], box[0]:box[2], :]

# print(d2.shape)
plt.subplot(231), plt.imshow(color[:, :, (0,1,2)]), plt.axis('off')
plt.subplot(232), plt.imshow(color[:, :, (0,2,1)]), plt.axis('off')
plt.subplot(233), plt.imshow(color[:, :, (1,0,2)]), plt.axis('off')
plt.subplot(234), plt.imshow(color[:, :, (1,2,0)]), plt.axis('off')
plt.subplot(235), plt.imshow(color[:, :, (2,1,0)]), plt.axis('off')
plt.subplot(236), plt.imshow(color[:, :, (2,0,1)]), plt.axis('off')

# plt.subplot(244), plt.imshow(color[: ,:, (1,2, 0)])
# plt.subplot(133), plt.imshow(grad_y, cmap='gray')
plt.show()