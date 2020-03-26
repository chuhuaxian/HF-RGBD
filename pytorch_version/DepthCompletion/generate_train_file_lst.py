import os
from sklearn.model_selection import train_test_split
import pyexr
import matplotlib.pyplot as plt
import numpy as np


def generate_nyu_list():
    nyu_dir = 'E:/Projects/Datasets/NYU'
    scene_lst = os.listdir(nyu_dir)
    f = open('nyu_list.txt', mode='w')
    for scene in scene_lst:
        raw = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'raw' in file]
        depth = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'depth' in file]
        color = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'color' in file]
        normal = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'normal' in file]
        bound = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'bound' in file]
        len_ = len(raw)
        for idx in range(len_):
            out = '%s,%s,%s,%s,%s\n' % (color[idx],  depth[idx], raw[idx], normal[idx], bound[idx])
            f.write(out)
            # print(out)
    f.close()
        # print()


def generate_scanet_list():
    nyu_dir = 'E:/Projects/Datasets/Scannet'
    scene_lst = os.listdir(nyu_dir)
    f = open('scannet_list.txt', mode='w')
    for scene in scene_lst:
        raw = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'raw' in file]
        depth = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'depth' in file]
        color = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'color' in file]
        normal = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'normal' in file]
        bound = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'bound' in file]
        len_ = len(raw)
        for idx in range(len_):
            out = '%s,%s,%s,%s,%s\n' % (color[idx],  depth[idx], raw[idx], normal[idx], bound[idx])
            f.write(out)
            # print(out)
    f.close()


def generate_irs_list():
    nyu_dir = 'E:/Projects/Datasets/IRS'
    scene_lst = os.listdir(nyu_dir)
    f = open('irs_list.txt', mode='w')
    for scene in scene_lst:
        depth = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'depth' in file]
        color = ['%s/%s/%s' % (nyu_dir, scene, file) for file in os.listdir('%s/%s' % (nyu_dir, scene)) if 'color' in file]
        len_ = len(color)
        for idx in range(len_):
            out = '%s,%s\n' % (color[idx], depth[idx])
            f.write(out)
            # print(out)
    f.close()


def generate_mask_list():
    mask_dir = 'E:/Projects/Datasets/LevelMaskDataset'
    scene_lst = os.listdir(mask_dir)
    f = open('mask_list.txt', mode='w')
    for scene in scene_lst:
        mask = ['%s/%s/%s' % (mask_dir, scene, file) for file in os.listdir('%s/%s' % (mask_dir, scene))]
        len_ = len(mask)
        for idx in range(len_):
            out = '%s\n' % (mask[idx])
            f.write(out)
            # print(out)
    f.close()


def generate_test_list():
    test_dir = 'E:\Projects\Datasets\\NyuV2Dataset'
    file_lst = os.listdir(test_dir)

    raw_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'raw' in i]
    depth_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'depth' in i]
    color_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'color' in i]

    f = open('test_list.txt', mode='w')
    for idx in range(len(raw_lst)):

        out = '%s,%s,%s\n' % (color_lst[idx], depth_lst[idx], raw_lst[idx])
        f.write(out)
            # print(out)
    f.close()


def generate_sun_test_list():
    sun_dir = 'D:\Projects\Datasets\RGBD_DATA\kv1'
    res_dir = 'E:\Projects\Datasets\SunRGBD'
    scene_lst = os.listdir(sun_dir)
    f = open('sun_list.txt', mode='w')
    for scene in scene_lst:
        filelist = ['%s/%s/%s' % (sun_dir, scene, file) for file in os.listdir('%s/%s' % (sun_dir, scene))]
        colors = [i for i in filelist if 'color' in i]
        raws = [i for i in filelist if 'depth' in i]
        len_ = len(colors)
        print()
        for idx in range(len_):
            res_cp = '%s/%s/%04d-color.jpg' % (res_dir, scene, idx)
            res_dp = '%s/%s/%04d-depth.png' % (res_dir, scene, idx)
            out = '%s,%s\n' % (colors[idx], raws[idx])
            f.write(out)
            # print(out)
    f.close()

generate_sun_test_list()
# generate_test_list()
# generate_nyu_list()
# generate_scanet_list()
# generate_irs_list()
# generate_mask_list()


def generate_file_list(data_dir):
    lst = [[os.path.join(os.path.join(data_dir, i), j) for j in os.listdir(os.path.join(data_dir, i))] for i in os.listdir(data_dir)]

    file_lst = []
    # remove_lst = []
    # lst = lst[6:]
    # print()
    for i in range(len(lst)):

        ao_lst = [i for i in lst[i] if 'normal' not in i and 'Z D' not in i]
        normal_lst = [i for i in lst[i] if 'normal' in i]
        position_lst = [i for i in lst[i] if 'Z D' in i]
        print(len(ao_lst), len(normal_lst), len(position_lst))
        # 2400 1800 3500 3600

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]

            # depth = pyexr.open(position_fn).get()
            # if 'scene09' not in ao_fn:
            #     continue
            # ao = pyexr.open(ao_fn).get()

            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)

            # if np.mean(ao[:, :, 0]) <=1e-5:
            #     print(our_str)

            # if np.min(depth) < 1e-5:
            #     plt.subplot(121)
            #     plt.imshow(depth)
            #     plt.subplot(122)
            #     plt.imshow(ao)
            #     plt.show()
            #     print(our_str)
            file_lst.append(our_str)
# generate_file_list('D:\\Projects\Datasets\\3dsmax\scene')

# print()
        # for j in range(len(ao_lst)):
        #     our_str = ''
        #     for k in range(3, -1, -1):
        #         index = j - k if j // 100 * 100 == (j - k) // 100 * 100 else j // 100 * 100
        #         ao_fn = ao_lst[index]
        #         normal_fn = normal_lst[index]
        #         position_fn = position_lst[index]
        #         our_str += '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #     file_lst.append(our_str)

    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    train, test = file_lst[:-7176], file_lst[-7176:]
    f = open('train_lst.txt', mode='w')
    for i in train:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test:
        f.write(i)
    f.close()


def generate_deepshading_file_list(data_dir):

    train_dir, test_dir = os.path.join(data_dir, 'Train'), os.path.join(data_dir, 'Test')

    train_lst = [[os.path.join(os.path.join(train_dir, i), j) for j in os.listdir(os.path.join(train_dir, i))] for i in
                 os.listdir(train_dir)]

    test_lst = [[os.path.join(os.path.join(test_dir, i), j) for j in os.listdir(os.path.join(test_dir, i))] for i in
                 os.listdir(test_dir)]

    file_lst = []
    for i in train_lst:

        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            file_lst.append(our_str)

    test_file_lst = []
    for i in test_lst:
        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            test_file_lst.append(our_str)
    # print()
    #
    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in file_lst:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test_file_lst:
        f.write(i)
    f.close()


def generate_nnao_file_list(data_dir):

    train_dir, test_dir = os.path.join(data_dir, 'Train'), os.path.join(data_dir, 'Test')

    train_lst = [[os.path.join(os.path.join(train_dir, i), j) for j in os.listdir(os.path.join(train_dir, i))] for i in
                 os.listdir(train_dir)]

    test_lst = [[os.path.join(os.path.join(test_dir, i), j) for j in os.listdir(os.path.join(test_dir, i))] for i in
                 os.listdir(test_dir)]

    file_lst = []
    for i in train_lst:

        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            file_lst.append(our_str)

    test_file_lst = []
    for i in test_lst:
        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]
        position_lst = [os.path.join(i[3], j) for j in os.listdir(i[3])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            test_file_lst.append(our_str)
    # print()
    #
    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in file_lst:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test_file_lst:
        f.write(i)
    f.close()


def generate_lstm_file_list(data_dir):
    lst = [[os.path.join(os.path.join(data_dir, i), j) for j in os.listdir(os.path.join(data_dir, i))] for i in os.listdir(data_dir)]

    file_lst = []
    for i in range(len(lst)):

        ao_lst = [i for i in lst[i] if 'normal' not in i and 'Z D' not in i]
        normal_lst = [i for i in lst[i] if 'normal' in i]
        position_lst = [i for i in lst[i] if 'Z D' in i]
        print(len(ao_lst), len(normal_lst), len(position_lst))
        # 2400 1800 3500 3600

        # if len(ao_lst) == 1800:
        for j in range(len(ao_lst)):
            our_str = ''
            for k in range(3, -1, -1):
                index = j-k if  j//100*100 == (j-k)//100*100 else j//100*100
                ao_fn = ao_lst[index]
                normal_fn = normal_lst[index]
                position_fn = position_lst[index]
                our_str += '%s,%s,%s,' % (ao_fn, normal_fn, position_fn)
            our_str = our_str[:-1]
            our_str += '\n'
            file_lst.append(our_str)

        # elif len(ao_lst) == 2400:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)
        # elif len(ao_lst) == 4800:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)
        # else:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)

    train, test = train_test_split(file_lst, test_size=1000, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in train:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test:
        f.write(i)
    f.close()

