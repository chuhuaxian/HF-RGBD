import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# depth = cv2.imread('E:\Projects\Datasets\\Nyu_sub\\basement_0001a\\280.png', -1)
# color = cv2.imread('D:\Projects\Datasets\RGBD_DATA\kv1\\xtion_sun3ddata\\00000-depth.png', -1)
# # #
# # #
# # # #39.1
# # # # depth = cv2.imread('D:\\Projects\Datasets\\RGBD_DATA\kv1\\realsense\\00722-depth.png', -1)
# # #
# # # # mask = depth!=0
# # # #
# # # # depth = depth + 1e-10
# # # # depth = 100*1000* 0.1*480/depth
# # # #
# # # # depth = depth * mask
# # # #
# # # # depth = depth.astype(np.uint16)
# # #
# # # plt.imshow(color)
# # # # plt.subplot(121), plt.imshow(color)
# # # # plt.subplot(122), plt.imshow(255-color)
# # # # plt.subplot(133), plt.imshow(depth2)
# # # plt.show()
import h5py
path = 'C:\\Users\\39796\Desktop\\00002.h5'

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    rgb = f['rgb'][:].transpose(1,2,0)
    depth = f['depth'][:]
    return (rgb, depth)

depth = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\0001-raw.png', -1) / 1000.
color = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\0001-color.jpg')[:, :, (2,1,0)]


mask = depth > 0

valid = depth[mask]
c, d = load_h5(path)

plt.subplot(121), plt.imshow(c)
plt.subplot(122), plt.imshow(d)

# plt.subplot(232), plt.imshow(color)
# plt.subplot(235), plt.imshow(depth)
#
# plt.subplot(233), plt.imshow(c/color)
# plt.subplot(236), plt.imshow(d/depth)


plt.show()