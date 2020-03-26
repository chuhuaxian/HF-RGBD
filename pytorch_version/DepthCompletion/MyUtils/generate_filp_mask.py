import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


Sun_dir = 'D:\Projects\Datasets\RGBD_DATA\kv1'
res_dir = 'E:\Projects\Datasets\MaskDataset'
file_lst = os.listdir(res_dir)

index = len(os.listdir(res_dir))
print(index)

for file in file_lst:
    raw = cv2.imread('%s/%s' % (res_dir, file), -1)
    # plt.subplot(121)
    # plt.imshow(raw)
    raw = cv2.resize(raw.transpose(), (640, 480), interpolation=cv2.INTER_NEAREST)
    # plt.subplot(122)
    # plt.imshow(raw)
    # plt.show()
    mask = raw != 0
    mask = mask * 255
    mask = mask.astype(np.uint8)
    print(index)

    cv2.imwrite('%s/%s-mask.png' % (res_dir, index), mask)
    index += 1
    # break

    print()


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


