import os
import cv2
import numpy as np
from PIL import Image

def generate_sun_test_list():
    sun_dir = 'D:\Projects\Datasets\RGBD_DATA\kv1'
    res_dir = 'C:\\Users\\39796\Desktop\SunRGBD'
    scene_lst = os.listdir(sun_dir)
    # f = open('C:\\Users\\39796\Desktop\sun_list.txt', mode='w')

    for scene in scene_lst:
        if not  os.path.exists('%s/%s' % (res_dir, scene)):
            os.makedirs('%s/%s' % (res_dir, scene))
        filelist = ['%s/%s/%s' % (sun_dir, scene, file) for file in os.listdir('%s/%s' % (sun_dir, scene))]
        colors = [i for i in filelist if 'color' in i]
        raws = [i for i in filelist if 'depth' in i]
        len_ = len(colors)
        print()
        for idx in range(len_):
            res_cp = '%s/%s/%04d-color.jpg' % (res_dir, scene, idx)
            res_dp = '%s/%s/%04d-depth.png' % (res_dir, scene, idx)

            # color = cv2.imread(colors[idx])
            # depth = cv2.imread(raws[idx], -1)/10.
            # depth = np.clip(depth, 500, 20000)
            # depth = depth.astype(np.uint16)
            #
            # cv2.imwrite(res_cp, color)
            # cv2.imwrite(res_dp, depth)
            out = '%s,%s\n' % (res_cp, res_dp)


            color = Image.open(colors[idx])
            raw = Image.open(raws[idx])
            xx = color.size

            res_wid, res_hei = color.size
            res_wid, res_hei = res_wid-res_wid%16, res_hei-res_hei%16
            print(xx)
            print(res_wid, res_hei)

            img_wid, img_hei = color.size

            start_wid, start_hei = (img_wid - res_wid) // 2, (img_hei - res_hei) // 2
            box = (start_wid, start_hei, start_wid + res_wid, start_hei + res_hei)  # 设置要裁剪的区域
            # print()
            color = color.crop(box)
            raw = raw.crop(box)
            color = np.array(color)
            raw = np.array(raw)/10.

            raw = np.clip(raw, 0, 20000)
            raw = raw.astype(np.uint16)

            cv2.imwrite(res_cp, color[:, :, (2,1,0)])
            cv2.imwrite(res_dp, raw)
            print()


            # f.write(out)
            # print(out)
    # f.close()

generate_sun_test_list()