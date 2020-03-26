import os
import pyexr
import matplotlib.pyplot as plt
import numpy as np
import math
# normal_path = 'C:\\Users\\39796\Desktop\\0000000061_normal.exr'

camera_cx = 528/2
camera_cy = 384/2
camera_fx = 548.9937
camera_fy = 548.9937


# row_map = np.ones(shape=(528, 384), dtype=np.float)*np.arange(0, 512)
# col_map = np.transpose(row_map)
# row_map = np.expand_dims(row_map, axis=-1)-camera_cx
# col_map = np.expand_dims(col_map, axis=-1) - camera_cy
# import cv2
#
# def depth2position(depth_map):
#     depth_map = 1 - depth_map
#     xx = 0.5+np.multiply(row_map, depth_map) / camera_fx
#     yy = 0.35-np.multiply(col_map, depth_map) / camera_fy
#     return np.concatenate([yy, xx, depth_map], axis=-1)
#
#
import cv2
# position = cv2.imread('C:\\Users\\39796\Desktop\\realsense\\46-ours.png', -1)

order = 162
position = cv2.imread('C:\\Users\\39796\Desktop\\align_kv2-i1\\%s-ours.png' % order, -1)

# res = depth2position(np.expand_dims(position, axis=-1))
# print()
f = open('C:\\Users\\39796\Desktop\\%s.obj' % order, mode='w')
for n in range(position.shape[0]):
    for m in range(position.shape[1]):
        d = position[n, m]
        z = 1-d
        x = (n - camera_cx) * z / camera_fx
        y = (m - camera_cy) * z / camera_fy

        f.write('v %s %s %s\n' % (x, y, z))
        # f.write('v %s %s %s\n' % (res[n, m, 0], res[n, m, 1], res[n, m, 2]))

f.close()
