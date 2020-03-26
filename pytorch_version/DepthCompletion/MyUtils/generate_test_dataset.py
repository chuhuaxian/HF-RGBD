import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import h5py
                  #需要读取的mat文件路径



nyu_v2_dir = 'E:\Document\\nyu_depth_v2_labeled.mat'
res_dir = 'E:\Projects\Datasets\\NyuV2Dataset'
feature = h5py.File(nyu_v2_dir)               #读取mat文件
depth = feature['depths']
raw = feature['rawDepths']
color = feature['images']
# data = loadmat(nyu_v2_dir)

print(color.shape)
print()
for i in range(1449):
    d = depth[i] * 1000.
    c = color[i]
    c = np.transpose(c, (2, 1, 0))
    c = c[:, :, (2, 1, 0)]
    r = raw[i] * 1000.

    d = d.astype(np.uint16)
    r = r.astype(np.uint16)
    mask = r != 0
    mask = mask * 255
    mask = mask.astype(np.uint8)
    cv2.imwrite('%s/%04d-raw.png' % (res_dir, i), r.transpose())
    cv2.imwrite('%s/%04d-depth.png' % (res_dir, i), d.transpose())
    cv2.imwrite('%s/%04d-color.jpg' % (res_dir, i), c)
    cv2.imwrite('%s/%04d-mask.png' % (res_dir, i), mask.transpose())
    # break

    print()
print()
# for mask in mask_lst:
#     # print('%s/%s' % (nyu_v2_mask_dir, mask))
#     order = mask.split('-')[0]
#     cp = '%s/%s_colors.png' % (nyu_v2_dir, order)
#     dp = '%s/%s_depth.png' % (nyu_v2_dir, order)
#     mp = '%s/%s' % (nyu_v2_mask_dir, mask)
#
#     depth = cv2.imread(dp, -1)
#     mask = cv2.imread(mp)[:, :, 0]
#     mask = mask.astype(np.uint8)
#     color = cv2.imread(cp)
#
#     plt.subplot(121), plt.imshow(depth)
#     plt.subplot(122), plt.imshow(mask)
#     print(cp)
