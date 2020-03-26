import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RES_DIR = 'E:\Projects\Datasets\\realsense4'

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


BASE_DIR = 'E:\\Projects\\Datasets\\TestDataset'


def dense2sparse(depth, test_size=1000):
    nonzeros = np.nonzero(depth)
    nonzeros = np.concatenate([np.expand_dims(nonzeros[0], axis=-1), np.expand_dims(nonzeros[1], axis=-1)], axis=-1)
    samples, _ = train_test_split(nonzeros, test_size=test_size, random_state=42)
    samples = (samples[:, 0], samples[:, 1])

    input = depth.copy()
    input[samples] = 0
    return input


depth_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'depth' in i]
color_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'color' in i]

count = 0

sum_ = 0
len_ = len(depth_lst)
f = open('E:\\Projects\\Datasets\\realsense_list_sparse.txt', mode='w')
for i in range(len_):
    raw = cv2.imread(depth_lst[i], -1)
    color = cv2.imread(color_lst[i])

    res_dname = '%s/%05d-gt.png' % (RES_DIR, sum_)
    cv2.imwrite(res_dname, (raw).astype(np.uint16))
    # plt.subplot(121)
    # plt.imshow(raw)

    raw = dense2sparse(raw)
    # plt.subplot(122)
    # plt.imshow(raw)
    # plt.show()

    # for j in range(10):

    res_cname = '%s/%05d-color.jpg' % (RES_DIR, sum_)
    res_dname = '%s/%05d-sparse.png' % (RES_DIR, sum_)

    # res_gtname = '%s/%05d-gt.png' % (RES_DIR, sum_)
    # res_mname = '%s/mask/%05d-mask.png' % (RES_DIR, sum_)
    #
    cv2.imwrite(res_cname, color)
    cv2.imwrite(res_dname, (raw).astype(np.uint16))
    f.write('%05d\n' % sum_)
    sum_ += 1

f.close()

