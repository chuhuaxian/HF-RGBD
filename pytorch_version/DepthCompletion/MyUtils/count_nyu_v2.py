import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
BASE_DIR = 'E:\Projects\Datasets\\NyuV2Dataset'
RES_DIR = 'E:\\Projects\\Datasets\\TestDataset'
depth_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'raw' in i]
color_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'color' in i]
full_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'depth' in i]
count = 0
len_ = len(depth_lst)
for i in range(len_):
    raw = cv2.imread(depth_lst[i], -1)[45:-40, 45:-40]
    depth = cv2.imread(full_lst[i], -1)[45:-40, 45:-40]
    color = cv2.imread(color_lst[i])[45:-40, 45:-40, :]

    mask = np.where(raw!=0, 1, 0)


    if np.sum(mask) > raw.shape[0]*raw.shape[1]*0.95:
        # res_cname = '%s/%s-color.jpg' % (RES_DIR, count)
        res_dname = '%s/%s-depth.png' % (RES_DIR, count)

        # res_mname = '%s/%s-mask.png' % (RES_DIR, count)

        # plt.subplot(131), plt.imshow(raw)
        # cv2.imwrite(res_cname, color)
        cv2.imwrite(res_dname, depth)
        # mask = mask * 255.
        # cv2.imwrite(res_mname, mask.astype(np.uint8))
        print()

        count+=1
    # print(depth.shape)
    # plt.imshow(depth)
    # plt.show()
print(count)