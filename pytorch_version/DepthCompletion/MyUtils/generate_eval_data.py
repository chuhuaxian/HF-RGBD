import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
RES_DIR = 'E:\Projects\Datasets\\realsense4'

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


BASE_DIR = 'E:\\Projects\\Datasets\\TestDataset'

MASK_30_DIR = 'E:\\Projects\\Datasets\\LevelMaskDataset\\030'
MASK_50_DIR = 'E:\\Projects\\Datasets\\LevelMaskDataset\\050'
MASK_70_DIR = 'E:\\Projects\\Datasets\\LevelMaskDataset\\070'
MASK_90_DIR = 'E:\\Projects\\Datasets\\LevelMaskDataset\\090'


mask_30_lst = ['%s/%s' %(MASK_30_DIR, i) for i in os.listdir(MASK_30_DIR)]
mask_50_lst = ['%s/%s' %(MASK_50_DIR, i) for i in os.listdir(MASK_50_DIR)]
mask_70_lst = ['%s/%s' %(MASK_70_DIR, i) for i in os.listdir(MASK_70_DIR)]
mask_90_lst = ['%s/%s' %(MASK_90_DIR, i) for i in os.listdir(MASK_90_DIR)]

idx_30 = [np.random.randint(0, len(os.listdir(MASK_30_DIR))) for i in range(10)]
idx_50 = [np.random.randint(0, len(os.listdir(MASK_50_DIR))) for i in range(10)]
idx_70 = [np.random.randint(0, len(os.listdir(MASK_70_DIR))) for i in range(10)]
idx_90 = [np.random.randint(0, len(os.listdir(MASK_90_DIR))) for i in range(10)]

print(idx_30)
print(idx_50)
print(idx_70)
print(idx_90)

depth_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'depth' in i]
color_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'color' in i]
mask_lst = ['%s/%s' %(BASE_DIR, i) for i in os.listdir(BASE_DIR) if 'mask' in i]
count = 0

sum_ = 0
len_ = len(depth_lst)
# f = open('E:\\Projects\\Datasets\\realsense_list_90.txt', mode='w')
for i in range(len_):
    raw = cv2.imread(depth_lst[i], -1)
    # print(raw.shape)
    # depth = cv2.imread(full_lst[i], -1)[45:-40, 45:-40]
    # color = cv2.imread(color_lst[i])

    for j in range(10):
        # mask_30 = cv2.resize(cv2.imread(mask_30_lst[i]), (555, 395), interpolation=cv2.INTER_NEAREST)[:, :, 0]/255.
        # mask_50 = cv2.resize(cv2.imread(mask_50_lst[i]), (555, 395), interpolation=cv2.INTER_NEAREST)[:, :, 0] / 255.
        # mask_70 = cv2.resize(cv2.imread(mask_70_lst[i]), (555, 395), interpolation=cv2.INTER_NEAREST)[:, :, 0] / 255.
        # mask_90 = cv2.resize(cv2.imread(mask_90_lst[i+j]), (555, 395), interpolation=cv2.INTER_NEAREST)[:, :, 0] / 255.
        #
        res_cname = '%s/%05d-color.jpg' % (RES_DIR, sum_)
        res_dname = '%s/%05d-gt.png' % (RES_DIR, sum_)
        # res_gtname = '%s/%05d-gt.png' % (RES_DIR, sum_)
        # res_mname = '%s/mask/%05d-mask.png' % (RES_DIR, sum_)
        #
        # cv2.imwrite(res_cname, color)
        cv2.imwrite(res_dname, (raw).astype(np.uint16))
        # mask = mask_30 * 255.
        # cv2.imwrite(res_mname, mask.astype(np.uint8))
        # f.write('%05d\n' % sum_)
        sum_ += 1
        # res_cname = '%s/%05d-color.jpg' % (RES_DIR, sum_)
        # res_dname = '%s/%05d-depth.png' % (RES_DIR, sum_)
        # res_gtname = '%s/%05d-gt.png' % (RES_DIR, sum_)
        # res_mname = '%s/mask/%05d-mask.png' % (RES_DIR, sum_)
        #
        # cv2.imwrite(res_cname, color)
        # cv2.imwrite(res_dname, (raw*mask_90).astype(np.uint16))
        # mask = mask_50 * 255.
        # cv2.imwrite(res_mname, mask.astype(np.uint8))
        # f.write('%05d\n' % sum_)
        # sum_ += 1
        # res_cname = '%s/%05d-color.jpg' % (RES_DIR, sum_)
        # res_dname = '%s/%05d-gt.png' % (RES_DIR, sum_)
        # res_mname = '%s/mask/%05d-mask.png' % (RES_DIR, sum_)
        #
        # cv2.imwrite(res_cname, color)
        # cv2.imwrite(res_dname, (raw).astype(np.uint16))
        # mask = mask_70 * 255.
        # cv2.imwrite(res_mname, mask.astype(np.uint8))
        # f.write('%05d\n' % sum_)
        # sum_ += 1
        print()
# f.close()



        # print(mask_30.shape)
        # plt.subplot(131), plt.imshow(mask_30*raw)
        # plt.subplot(132), plt.imshow(mask_50*raw)
        # plt.subplot(133), plt.imshow(mask_70*raw)
        # plt.show()