import os
import cv2
import matplotlib.pyplot as plt
import shutil

base_scan_dir = 'E:/Projects/Datasets/Nyu_sub/'
base_nyu_dir = 'E:/Projects/Datasets/ScanNet/'
rgbd_dir = 'E:/Projects/Datasets/RGBD'
res_dir = 'E:/Projects/Datasets/RGBD'

scan_scene_lst = [os.path.join(base_scan_dir, i) for i in os.listdir(base_scan_dir)]
nyu_scene_lst = [os.path.join(base_nyu_dir, i) for i in os.listdir(base_nyu_dir)]

total_lst = scan_scene_lst + nyu_scene_lst
count = 0
# f = open('E:\Projects\Datasets\\total_file_lst.txt', mode='w')
f1 = open('E:\Projects\Datasets\\file_lst.txt', mode='w')
for scene in total_lst:
    color_lst = ['%s/%s' % (scene, i) for i in os.listdir(scene) if i.endswith('jpg')]
    depth_lst = ['%s/%s' % (scene, i) for i in os.listdir(scene) if i.endswith('png')]

    len_ = len(color_lst)
    for index in range(len_):
        cname, dname = color_lst[index], depth_lst[index]
        if 'Nyu_sub' in cname:
            res_rname = '/'.join(cname.replace('Nyu_sub', 'Nyu').split('/')[:-1])
            if not os.path.exists(res_rname):
                os.makedirs(res_rname)
            order = cname.replace('Nyu_sub', 'Nyu').split('/')[-1].split('.')[0]

            rawname = '%s/%s.png' % (rgbd_dir, count)
            depthname = '%s/%s-depth.png' % (rgbd_dir, count)
            rgbname = '%s/%s.jpg' % (rgbd_dir, count)
            normalname = '%s/%s-normal.png' % (rgbd_dir, count)
            boundname = '%s/%s-bound.png' % (rgbd_dir, count)

            res_rawname = '%s/%s-raw.png' % (res_rname, order)
            res_depthname = '%s/%s-depth.png' % (res_rname, order)
            res_rgbname = '%s/%s-color.jpg' % (res_rname, order)
            res_normalname = '%s/%s-normal.png' % (res_rname, order)
            res_boundname = '%s/%s-bound.png' % (res_rname, order)

            print()

            shutil.copy(rawname, res_rawname)
            shutil.copy(depthname, res_depthname)
            shutil.copy(rgbname, res_rgbname)
            shutil.copy(normalname, res_normalname)
            shutil.copy(boundname, res_boundname)

            # out = '%s,%s,%s,%s\n' % (color_lst[index], depth_lst[index], res_cname, res_dname)
            # f.write(out)
            # f1.write('%s, %s, %s\n' % (count, cname, dname))
            # print(out)

            count += 1
        else:
            res_rname = '/'.join(cname.replace('ScanNet', 'Scan').split('/')[:-1])
            order = cname.replace('Nyu_sub', 'Nyu').split('/')[-1].split('.')[0]
            if not os.path.exists(res_rname):
                os.makedirs(res_rname)

            rawname = '%s/%s.png' % (rgbd_dir, count)
            depthname = '%s/%s-depth.png' % (rgbd_dir, count)
            rgbname = '%s/%s.jpg' % (rgbd_dir, count)
            normalname = '%s/%s-normal.png' % (rgbd_dir, count)
            boundname = '%s/%s-bound.png' % (rgbd_dir, count)

            res_rawname = '%s/%s-raw.png' % (res_rname, order)
            res_depthname = '%s/%s-depth.png' % (res_rname, order)
            res_rgbname = '%s/%s-color.jpg' % (res_rname, order)
            res_normalname = '%s/%s-normal.png' % (res_rname, order)
            res_boundname = '%s/%s-bound.png' % (res_rname, order)

            print()

            shutil.copy(rawname, res_rawname)
            shutil.copy(depthname, res_depthname)
            shutil.copy(rgbname, res_rgbname)
            shutil.copy(normalname, res_normalname)
            shutil.copy(boundname, res_boundname)

            print()
            # res_cname, res_dname = cname.replace('ScanNet', 'Scan'), dname.replace('ScanNet', 'Scan')
            # shutil.copy(cname, res_cname)
            # shutil.copy(dname, res_dname)
            # out = '%s,%s,%s,%s\n' % (color_lst[index], depth_lst[index], res_cname, res_dname)
            # f.write(out)
            # f1.write('%s, %s, %s\n' % (count, cname, dname))
            # print(out)

            count += 1

f1.close()
# f.close()
        # if not os.path.exists(color_lst[index]) or not os.path.exists(depth_lst[index]):
        #     print(color_lst[index], ',  ', depth_lst[index])
        #     break
