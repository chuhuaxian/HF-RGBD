import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


Matterport_dir = 'E:\Projects\Datasets\Matterport3D'
res_dir = 'E:\Projects\Datasets\MaskDataset'
scene_lst = os.listdir(Matterport_dir)

index = len(os.listdir(res_dir))
print(index)
for scene in scene_lst:
    dp = [i for i in os.listdir('%s/%s' % (Matterport_dir, scene)) if 'depth' in i][0]
    raw_lst = os.listdir('%s/%s/%s' % (Matterport_dir, scene, dp))
    for file in raw_lst:
        print('%s/%s/undistorted_depth_images/%s' % (Matterport_dir, scene, file))
        raw = cv2.imread('%s/%s/%s/%s' % (Matterport_dir, scene, dp, file), -1)
        raw = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_NEAREST)
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


