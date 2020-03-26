import os
import cv2
import matplotlib.pyplot as plt
import shutil

base_scan_dir = 'E:/Projects/Datasets/Nyu_sub/'
base_nyu_dir = 'E:/Projects/Datasets/ScanNet/'

res_dir = 'E:/Projects/Datasets/RGBD'

scan_scene_lst = [os.path.join(base_scan_dir, i) for i in os.listdir(base_scan_dir)]
nyu_scene_lst = [os.path.join(base_nyu_dir, i) for i in os.listdir(base_nyu_dir)]

total_lst = scan_scene_lst + nyu_scene_lst
count = 0
# f = open('E:\Projects\Datasets\\total_file_lst.txt', mode='w')
f1 = open('E:\Projects\Datasets\\file_lst.txt', mode='w')
for scene in total_lst:
    color_lst = [os.path.join(scene, i) for i in os.listdir(scene) if i.endswith('jpg')]
    depth_lst = [os.path.join(scene, i) for i in os.listdir(scene) if i.endswith('png')]

    len_  = len(color_lst)
    for index in range(len_):
        cname, dname = color_lst[index], depth_lst[index]
        res_cname, res_dname = '%s/%s.jpg' % (res_dir, count), '%s/%s.png' % (res_dir, count)
        # shutil.copy(cname, res_cname)
        # shutil.copy(dname, res_dname)
        out = '%s,%s,%s,%s\n' % (color_lst[index], depth_lst[index], res_cname, res_dname)
        # f.write(out)
        f1.write('%s, %s, %s\n' % (count, cname, dname))
        print(out)

        count += 1
f1.close()
# f.close()
        # if not os.path.exists(color_lst[index]) or not os.path.exists(depth_lst[index]):
        #     print(color_lst[index], ',  ', depth_lst[index])
        #     break
