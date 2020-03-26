import numpy as np
import matplotlib.pyplot as plt
import cv2

# focal_length = 480
# baseline = 0.1
# left = np.ones(shape=(480, 320, 1), dtype=np.float32)
# right = np.zeros(shape=(480, 320, 1), dtype=np.float32)
# cat = np.concatenate([left, right], axis=1)
#
# img = cv2.imread('C:\\Users\\39796\Desktop\\00000_colors.png').astype(np.float32)
# # xx = np.ra(shape=(480, 640, 3), dtype=np.float32)
# xxx = img*cat
# print(cat.shape)
# plt.imshow(xxx.astype(np.uint8))
# plt.show()

# aa = np.array([1,1,2,3,4,5,6])
# #
# # aa = aa[aa>3]
# # print(aa)
state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict()}
f = open('C:\\Users\\39796\Desktop\\realsense_list.txt', mode='w')

import os

path = 'C:\\Users\\39796\Desktop\SunRGBD\\realsense'
c_files = ['%s/%s' %(path, i) for i in os.listdir(path) if 'color' in i]
d_files = ['%s/%s' %(path, i) for i in os.listdir(path) if 'depth' in i]
for i,j in zip(c_files, d_files):
    f.write('%s,%s\n' % (i,j))
    print(i)
f.close()
