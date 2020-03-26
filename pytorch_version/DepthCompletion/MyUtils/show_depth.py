import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil

BASE_DIR = 'E:\Projects\Datasets\\realsense1'
SAVE_DIR = 'E:\Projects\Datasets\\NyuV2Dataset'
file_lst = os.listdir(BASE_DIR)

for i in file_lst:
    order = i.split('-')[0]
    file_name = os.path.join(BASE_DIR, i)
    save_name = os.path.join(BASE_DIR, '%s-depth.png' % order)
    save_name1 = os.path.join(SAVE_DIR, '%04d-depth.png' % int(order))
    # print(save_name1)

    print(save_name, save_name1)
    # os.remove(save_name1)
    shutil.move(save_name, save_name1)
    # print(order)
    # f = h5py.File(file_name, 'r')
    #
    # normal = f['result']
    # normal = np.transpose(normal, (1, 2, 0))
    #
    # nx, ny, nz = np.split(normal, 3,  axis=2)
    # ny = -ny
    # normal = np.concatenate([nx, ny, nz], axis=-1)
    # normal = (normal+1) / 2
    # normal = normal[:, :, (0, 2, 1)]
    # normal = np.clip(normal*255., 0, 255)
    # normal = normal.astype(np.uint8)
    #
    # cv2.imwrite(save_name, normal[:, :, (2, 1, 0)])
    # break
    # plt.imshow(normal)
    # plt.show()
# print(list(f.keys()))