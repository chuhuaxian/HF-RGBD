import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
BASE_DIR = 'E:\Experimrnt\Completion\RandRemove'
BASE_DIR_121 = 'E:\Experimrnt\Completion\Rand'
file_list = ['%s/%s' % (BASE_DIR, i) for i in os.listdir(BASE_DIR)]
col_list = [i for i in file_list if 'col' in i]
raw_list = [i for i in file_list if 'depth' in i]
gt_list = [i for i in file_list if 'gt' in i]
our_169_list = [i for i in file_list if 'our' in i]

our_121_list = ['%s/%s' % (BASE_DIR_121, i) for i in os.listdir(BASE_DIR_121)]
# ddc_list = [i for i in file_list if 'ddc' in i]
dd_list = [i for i in file_list if 'ddc' in i]
len_ = len(col_list)

def depth2color(depth, cmap='jet'):
    depth = np.squeeze(depth)

    min_ = np.min(depth)
    max_ = np.max(depth)

    res = (depth - min_) / (max_ - min_)

    import matplotlib
    cmapper = matplotlib.cm.get_cmap(cmap)
    res = cmapper(res, bytes=True)  # (nxmx4)
    # res = np.concatenate([c, res[:, :, :3]], axis=1)
    return res[:, :, :3]

def make_grid(arr_lst, pad_value=1, pad_size=10):
    shape_ = arr_lst[0].shape
    pad_ = np.ones((shape_[0], pad_size, 3), dtype=np.uint8)*pad_value*255
    out = []
    for i in range(len(arr_lst)):
        out.append(arr_lst[i])
        if i < len(arr_lst)-1:
            out.append(pad_)
    return np.concatenate(out, axis=1)

def crop(img):
    img  =Image.fromarray(img)
    res_wid, res_hei = img.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = img.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    return np.array(img.crop(box))


RES_DIR = 'E:\Experimrnt\Completion\Rand-pseudo'
for i in range(len_):
    rgb = cv2.imread(col_list[i])[:, :, (2,1,0)]
    raw = cv2.imread(raw_list[i], -1)
    gt = cv2.imread(gt_list[i], -1)
    # our_169 = cv2.imread(our_169_list[i], -1)
    our_121 = cv2.imread(our_121_list[i], -1)
    dd = cv2.imread(dd_list[i], -1)

    rgb = crop(rgb)
    raw = crop(raw)
    gt = crop(gt)
    dd = crop(dd)

    res = np.concatenate([dd, our_121, gt], axis=1)

    res = depth2color(res)

    res = np.split(res, 3, axis=1)

    mask = raw!=0
    mask = np.expand_dims(mask, -1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    raw = res[2]*mask

    res = make_grid([rgb, raw, res[0], res[1], res[2]])

    cv2.imwrite('%s/%04d-pseudo.png' % (RES_DIR, i), res[:,:,(2,1,0)])

    print()

    # plt.imshow(res)
    # plt.show()


