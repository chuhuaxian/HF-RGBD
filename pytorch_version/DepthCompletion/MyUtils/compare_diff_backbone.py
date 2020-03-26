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


def static_data_(gt, pred):
    abs_diff = np.abs(gt - pred)

    rel_loss = np.sum(abs_diff/gt)
    rmse_loss = np.sum(np.square(abs_diff))
    log10_loss = np.sum(np.abs(np.log10(gt) - np.log10(pred)))

    maxRatio = np.concatenate([np.expand_dims(gt / pred, axis=-1), np.expand_dims(pred / gt, axis=-1)], axis=-1)
    maxRatio = np.max(maxRatio, axis=-1)

    sigma1_loss = np.sum(maxRatio < 1.25)
    sigma2_loss = np.sum(maxRatio < 1.25 ** 2)
    sigma3_loss = np.sum(maxRatio < 1.25 ** 3)
    return rel_loss, rmse_loss, log10_loss, sigma1_loss, sigma2_loss, sigma3_loss

def static_data(gt, pred):
    abs_diff = np.abs(gt - pred)

    rel_loss = np.mean(abs_diff/gt)
    rmse_loss = np.mean(np.square(abs_diff))
    log10_loss = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    maxRatio = np.concatenate([np.expand_dims(gt / pred, axis=-1), np.expand_dims(pred / gt, axis=-1)], axis=-1)
    maxRatio = np.max(maxRatio, axis=-1)

    sigma1_loss = np.mean(maxRatio < 1.25)
    sigma2_loss = np.mean(maxRatio < 1.25 ** 2)
    sigma3_loss = np.mean(maxRatio < 1.25 ** 3)
    return rel_loss, rmse_loss, log10_loss, sigma1_loss, sigma2_loss, sigma3_loss

our_total_rel_loss_121 = 0
our_total_rmse_loss_121 = 0
our_total_log_loss_121 = 0
our_total_sigma1_loss_121 = 0
our_total_sigma2_loss_121 = 0
our_total_sigma3_loss_121 = 0

our_total_rel_loss_169 = 0
our_total_rmse_loss_169 = 0
our_total_log_loss_169 = 0
our_total_sigma1_loss_169 = 0
our_total_sigma2_loss_169 = 0
our_total_sigma3_loss_169 = 0


RES_DIR = 'E:\Experimrnt\Completion\RandRemove-pseudo'
count = 0
for i in range(len_):
    rgb = cv2.imread(col_list[i])[:, :, (2,1,0)]
    raw = cv2.imread(raw_list[i], -1)
    gt = cv2.imread(gt_list[i], -1) / 1000.
    our_169 = cv2.imread(our_169_list[i], -1) / 1000.
    our_121 = cv2.imread(our_121_list[i], -1) / 1000.
    dd = cv2.imread(dd_list[i], -1)

    rgb = crop(rgb)
    raw = crop(raw)
    gt = crop(gt)
    dd = crop(dd)
    # print(np.mean(our_121-our_169))

    rel_loss, rmse_loss, log10_loss, sigma1_loss, sigma2_loss, sigma3_loss = static_data(gt, our_121)
    # print(rel_loss)
    our_total_rel_loss_121 += rel_loss
    our_total_rmse_loss_121 += rmse_loss
    our_total_log_loss_121 += log10_loss
    our_total_sigma1_loss_121 += sigma1_loss
    our_total_sigma2_loss_121 += sigma2_loss
    our_total_sigma3_loss_121 += sigma3_loss

    rel_loss, rmse_loss, log10_loss, sigma1_loss, sigma2_loss, sigma3_loss = static_data(gt, our_169)
    # print(rel_loss)
    our_total_rel_loss_169 += rel_loss
    our_total_rmse_loss_169 += rmse_loss
    our_total_log_loss_169 += log10_loss
    our_total_sigma1_loss_169 += sigma1_loss
    our_total_sigma2_loss_169 += sigma2_loss
    our_total_sigma3_loss_169 += sigma3_loss
    count+=1

    print(count)
    if count > 1000:
        break

print(
    'rel:%.3f,   rms:%.3f,    log10:%.3f,    delta1:%.3f,  delta2:%.3f,  delta3:%.3f' % (our_total_rel_loss_121 / count,
                                                                                         np.sqrt(
                                                                                             our_total_rmse_loss_121 / count),
                                                                                         our_total_log_loss_121 / count,
                                                                                         our_total_sigma1_loss_121 / count,
                                                                                         our_total_sigma2_loss_121 / count,
                                                                                         our_total_sigma3_loss_121 / count,))

print(
    'rel:%.3f,   rms:%.3f,    log10:%.3f,    delta1:%.3f,  delta2:%.3f,  delta3:%.3f' % (our_total_rel_loss_169 / count,
                                                                                         np.sqrt(
                                                                                             our_total_rmse_loss_169 / count),
                                                                                         our_total_log_loss_169 / count,
                                                                                         our_total_sigma1_loss_169 / count,
                                                                                         our_total_sigma2_loss_169 / count,
                                                                                         our_total_sigma3_loss_169 / count,))

    # res = np.concatenate([dd, our_169, gt], axis=1)
    #
    # res = depth2color(res)
    #
    # res = np.split(res, 3, axis=1)
    #
    # mask = raw!=0
    # mask = np.expand_dims(mask, -1)
    # mask = np.concatenate([mask, mask, mask], axis=-1)
    # raw = res[2]*mask
    #
    # res = make_grid([rgb, raw, res[0], res[1], res[2]])
    #
    # cv2.imwrite('%s/%04d-pseudo.png' % (RES_DIR, i), res[:,:,(2,1,0)])
    #
    # print()

    # plt.imshow(res)
    # plt.show()


