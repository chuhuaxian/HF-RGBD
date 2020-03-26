import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

nyu_lst = open('C:\\Users\\39796\PycharmProjects\DepthCompletion\\nyu_list.txt').readlines()
scan_lst = open('C:\\Users\\39796\PycharmProjects\DepthCompletion\\scannet_list.txt').readlines()

res_dir = 'E:\Projects\Datasets\MaskDataset'
nyu_mask_lst = [i.split(',')[2] for i in nyu_lst]
scan_mask_lst = [i.split(',')[2] for i in scan_lst]

total_mask_lst = nyu_mask_lst + scan_mask_lst

for idx, i in enumerate(total_mask_lst):
    raw = cv2.imread(i, -1)
    mask = raw != 0
    mask = mask * 255
    mask = mask.astype(np.uint8)
    print(idx)
    cv2.imwrite('%s/%s-mask.png' % (res_dir, idx), mask)

print()


