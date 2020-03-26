import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil
BASE_DIR = 'E:\Projects\Datasets\DDC\\realsense\\normal_scannet_realsense_test'
SAVE_DIR = 'E:\Projects\Datasets\SunRGBD\\realsense'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
file_lst = os.listdir(BASE_DIR)

for i in file_lst:

    # order = i.split('_')[1]
    # file_name = os.path.join(BASE_DIR, i)
    # save_name = os.path.join(SAVE_DIR, '%s-bound.png' % order)
    # save_name1 = os.path.join(SAVE_DIR, '%04d-bound.png' % int(order))
    # print(save_name1)


    # shutil.move(save_name, save_name1)


    order = i.split('_')[1]
    file_name = os.path.join(BASE_DIR, i)
    save_name = os.path.join(SAVE_DIR, '%s-normal.png' % order)
    # print(order)
    f = h5py.File(file_name, 'r')

    normal = f['result']
    normal = np.transpose(normal, (1, 2, 0))

    nx, ny, nz = np.split(normal, 3,  axis=2)
    ny = -ny
    # nx = -nx
    normal = np.concatenate([nx, ny, nz], axis=-1)
    normal = (normal+1) / 2
    normal = normal[:, :, (0, 2, 1)]
    normal = np.clip(normal*255., 0, 255)
    normal = normal.astype(np.uint8)

    cv2.imwrite(save_name, normal[:, :, (2, 1, 0)])
    # break
    # plt.imshow(normal)
    # plt.show()
# print(list(f.keys()))