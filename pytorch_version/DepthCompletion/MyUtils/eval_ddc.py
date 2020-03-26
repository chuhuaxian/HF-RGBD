import os
import numpy as np
import cv2

BASE_DIR = 'E:\Projects\Datasets\\NyuV2Dataset'


file_list = [i for i in os.listdir(BASE_DIR)]

raw_list = ['%s/%s'%(BASE_DIR,i) for i in file_list if 'raw' in i]
depth_list = ['%s/%s'%(BASE_DIR,i) for i in file_list if 'depth' in i]

loss_count = 0
cou_ = 0
for i in range(len(raw_list)):
    raw = cv2.imread(raw_list[i], -1)/1000.
    depth = cv2.imread(depth_list[i], -1)/1000.

    valid_mask = raw > 0
    depth = depth[valid_mask]
    raw = raw[valid_mask]

    loss = np.mean(np.abs(raw-depth))
    print(loss)
    loss_count += loss
    cou_ += 1


print('loss= ' ,np.sqrt(loss_count/cou_))


