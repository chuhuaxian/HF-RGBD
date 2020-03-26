import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil
BASE_DIR = 'E:\Projects\Matlab\Bounds'
SAVE_DIR = 'E:\Projects\Matlab\\Normal'
file_lst = os.listdir(BASE_DIR)

count = 0
for i in file_lst:

    # if count % 1000 != 0:
    #     count += 1
    #     continue
    # else:

    order = i.split('_')[1]
    file_name = os.path.join(BASE_DIR, i)
    save_name = os.path.join(BASE_DIR, '%s-bound.png' % order)
    # print(file_name, save_name)
    shutil.move(file_name, save_name)
    # bound = cv2.imread(file_name)

    # print(order)
    # f = h5py.File(file_name, 'r')

    # normal = f['result']
    # normal = np.transpose(normal, (1, 2, 0))

    # nx, ny, nz = np.split(normal, 3,  axis=2)
    # ny = -ny
    # normal = np.concatenate([nx, ny, nz], axis=-1)
    # normal = (normal+1) / 2
    # normal = normal[:, :, (0, 2, 1)]
    # normal = np.clip(normal*255., 0, 255)
    # normal = normal.astype(np.uint8)
    # bound = bound / 255.
    # weight = np.power(1-bound[:,:,0], 3)
    # weight = np.where(weight>0.05, 0, weight)
    # count += 1
    # weight  = weight*1000
    # plt.imshow(weight)
    # plt.show()

# cv2.imwrite(save_name, normal[:, :, (2, 1, 0)])
#     break

# print(list(f.keys()))