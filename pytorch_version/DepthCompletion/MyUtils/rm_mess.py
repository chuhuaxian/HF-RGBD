import os

import os
import cv2
import matplotlib.pyplot as plt
import shutil


base_dir = 'E:\\Projects\\Datasets\\ScanNet'

scene_lst = os.listdir(base_dir)

count = 0
for i in scene_lst:
    scenes = os.listdir(os.path.join(base_dir, i))
    count += len(scenes)


print(count)
    # color_dir = '%s/%s/color' % (base_dir, i)
    # depth_dir = '%s/%s/depth' % (base_dir, i)

    # os.rmdir(color_dir)
    # os.rmdir(depth_dir)
    # color_lst = os.listdir(color_dir)
    # depth_lst = os.listdir(depth_dir)
    # len_ = len(color_lst)
    #
    # for j in range(len_):
    #     c_path = '%s/%s/color/%s.jpg' % (base_dir, i, j * 10)
    #     d_path = '%s/%s/depth/%s.png' % (base_dir, i, j * 10)
    #
    #
    #     res_c_path = '%s/%s/%s.jpg' % (base_dir, i, j)
    #     res_d_path = '%s/%s/%s.png' % (base_dir, i, j)
    #
    #     shutil.move(c_path, res_c_path)
    #     shutil.move(d_path, res_d_path)

    # print()
