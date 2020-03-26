import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil

BASE_DIR = 'E:\Projects\Datasets\DDC\\realsense\\results\\realsense'
SAVE_DIR = 'E:\Projects\Datasets\SunRGBD\\realsense'
file_lst = os.listdir(BASE_DIR)
# color_lst = [i for i in file_lst if 'color' in i]
raw_lst = [i for i in file_lst if 'ddc' in i]

for i in range(len(raw_lst)):
    # order = i.split('-')[0]
    # cname = '%s/%04d-color.jpg' % (BASE_DIR, i)
    dname = '%s/%04d-ddc.png' % (BASE_DIR, i)

    # res_cname = '%s/%04d-color.jpg' % (SAVE_DIR, i)
    res_dname = '%s/%04d-depth.png' % (SAVE_DIR, i)

    # file_name = os.path.join(BASE_DIR, i)
    # save_name = os.path.join(SAVE_DIR, '%s-bound.png' % order)
    # save_name1 = os.path.join(SAVE_DIR, '%04d-bound.png' % int(order))
    print()
    # print(save_name1)


    # shutil.move(cname, res_cname)
    shutil.move(dname, res_dname)
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