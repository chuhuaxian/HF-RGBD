import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

res_dir = 'E:\Projects\Datasets\LevelMaskDataset'
mask_dir = 'E:\Projects\Datasets\MaskDataset'
file_lst = os.listdir(mask_dir)[::-1]

# index = len(os.listdir(mask_dir))
# print(index)
level = ['010', '020','030','040','050','060','070','080','090','100']

cc = 0
for i in level:
    # os.makedirs('%s/%s' % (res_dir, i))
    cc += len(os.listdir('%s/%s' % (res_dir, i)))
print(cc)

# idex = len(file_lst)
# for file in file_lst:
#     raw = 255-cv2.imread('%s/%s' % (mask_dir, file), -1)
#     mask = raw != 0
#     miss_rate = 1-np.sum(mask)/640/480
#     # print(miss_rate)
#     path = '%03d' % (10+miss_rate*100 // 10 * 10)
#     # print(path)
#     cv2.imwrite('%s/%s/%s-mask.png' % (res_dir, path, idex), raw)
#     idex += 1
#     # print(path)
#     # plt.subplot(121)
#     # plt.imshow(raw)
#     # plt.subplot(122)
#     # plt.imshow(raw)
#     # plt.show()
#
#     print(idex)
#
#     # cv2.imwrite('%s/%s-mask.png' % (res_dir, index), mask)
#     # index += 1
#     # break
#
#     print()


# res_dir = 'E:\Projects\Datasets\MaskDataset'
# nyu_mask_lst = [i.split(',')[2] for i in nyu_lst]
# scan_mask_lst = [i.split(',')[2] for i in scan_lst]
#
# total_mask_lst = nyu_mask_lst + scan_mask_lst
#
# for idx, i in enumerate(total_mask_lst):
#     raw = cv2.imread(i, -1)
#     mask = raw != 0
#     mask = mask * 255
#     mask = mask.astype(np.uint8)
#     print(idx)
#     cv2.imwrite('%s/%s-mask.png' % (res_dir, idx), mask)
#
# print()


