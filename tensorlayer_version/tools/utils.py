from tensorlayer.prepro import *
import tensorflow as tf
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import imageio
import numpy as np
import tqdm
import pandas as pd
from numpy import *
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries/' + name[:-2]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)


def get_imgs_fn(file_name):
    """ Input an image path and name, return an image array """
    # print(file_name.strip()+'color.png')

    return scipy.misc.imread(file_name.strip())


def color2gray(imgRgb):
    return cv2.cvtColor(imgRgb, cv2.COLOR_BGR2GRAY)


def get_depth_fn(file_name):
    """ Input an image path and name, return an image array """
    if file_name.strip().split('.')[-1] =='png':
        return cv2.imread(file_name.strip(), flags=-1)
        # return scipy.misc.imread(file_name.strip())

    else:
        return cv2.imread(file_name.strip() + 'depth.png', flags=-1)
        # return scipy.misc.imread(file_name.strip() + 'depth.png')


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


# def transform_image_color(image):
#
#     # Normalize
#     image_color = image / (255./2.)
#     image_color = image_color - 1.
#     return image_color


def transform_image_depth(image, scale):

    image_depth = image / scale
    # Convert to 4D
    image_depth = np.expand_dims(image_depth, axis=-1)

    return image_depth


def get_image_depth_mask(image, times=4):

    mask = image != -1.
    mask = mask.astype(np.float32)
    mask = imresize(mask, size=[mask.shape[0]*times, mask.shape[1]*times], interp='nearest', mode='F')
    return mask


# def subsample_color(image, times=4):
#
#     # Downsample
#     image_ = imresize(image, size=[image.shape[0]//times, image.shape[1]//times, 3], interp='bicubic', mode=None)
#
#     image_color = image_ / (255./2.)
#     image_color = image_color - 1.
#     return image_color



# def downsample_depth(image, times=4):
#
#     # Downsample
#     image_ = imresize(image, size=[image.shape[0]*times, image.shape[1]*times], interp='nearest', mode=None)
#     image_depth = (image_ - image.min()) / (image.max() - image.min())
# print(tl.files.load_file_list('D:\Projects\\rgbd-scenes'))


def dilate(img, ks=3, it=1):

    kernel = np.uint8(np.zeros((ks, ks)))
    # # kernel = np.uint8(np.ones((ks, ks)))
    #
    for x in range(ks):
        kernel[x, ks//2] = 1
        kernel[ks//2, x] = 1
    # kernel[ks // 2, :] = 1
    # kernel = np.uint8(np.array([1, 1]))
    # img = cv2.imread(path, flags=mode)

    dilated = cv2.dilate(img, kernel, iterations=it)
    # dilated = cv2.dilate(dilated, kernel)
    # dilated = cv2.dilate(dilated, kernel)
    # imageio.imwrite('C:\\Users\Administrator\Desktop\\00007.png', dilated.astype(np.uint16))



    # cv2.imshow('dilate', dilated)
    # cv2.waitKey(0)

    return dilated


def erode(img, ks=3, it=1):

    kernel = np.uint8(np.zeros((ks, ks)))
    for x in range(ks):
        kernel[x, ks//2] = 1
        kernel[ks//2, x] = 1
    # kernel[ks//2, :] = 1
    # kernel = np.uint8(np.eye(ks, ks))
    # img = cv2.imread(path, flags=mode)
    # kernel = np.uint8(np.array([1, 1]))

    eroded = cv2.erode(img, kernel, iterations=it)
    # cv2.imshow('dilate', dilated)
    # cv2.waitKey(0)

    return eroded


def subsample_depth(image_, times=4):

    # Downsample
    # image = cv2.imread(image, flags=-1)
    # cv2.imshow('dilate', image)
    # cv2.waitKey(0)
    # image_ = np.expand_dims(image, axis=-1)
    image_ = imresize(image_, size=[image_.shape[0]//times, image_.shape[1]//times], interp='nearest', mode='F')
    # mask = np.abs(cv2.absdiff(dilate(image_), erode(image_))) <500
    # mask = mask.astype(np.float32)

    # image_ = np.squeeze(image_) * mask
    # image_ = np.expand_dims(image_, -1)
    # cv2.imshow('image*mask', image_.astype(np.uint16))
    # cv2.waitKey(0)

    # image_ = dilate(image_, it=2)
    # cv2.imshow('dilate', image_)
    # cv2.waitKey(0)

    # image_depth = image_ / 127.5
    # image_depth = image_ - 1.

    return image_
# cp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_14\\00640-depth.png'
# subsample_depth(cp1)


# p = 'D:\Projects\RGB-D_SR\../rgbd-scenes-v2\imgs\scene_14\\00007.png'
p = 'C:\\Users\Administrator\Desktop\\table_1_98_depth.png'
p1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_06\\00101-depth.png'
fp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_10\\00004-depth.png'


def inpaint(path):
    image = cv2.imread(path, flags=-1)
    height, width = image.shape[0], image.shape[1]
    gap_h, gap_w = 1, 1
    for h in range(0, height, gap_h):
        if np.sum(image[h, :]) == 0:
            continue
        begin, end = 0, 0
        while begin < width-1:

            if image[h, 0] == 0:
                while image[h, end] == 0:
                    end += 1
                image[h, begin:end] = image[h, end]
                begin = end
            while image[h, begin] != 0:
                begin += 1
            end = begin
            while image[h, end] == 0 and end <width-1:
                end += 1
            # print(h, begin, end)
            image[h, begin:end] = np.max([image[h, begin-1], image[h, end]])
            begin = end
    return image


def getborder(path, mode):
    ks = 3
    kernel = np.uint8(np.zeros((ks, ks)))
    for x in range(ks):
        kernel[x, ks//2] = 1
        kernel[ks//2, x] = 1
    img = cv2.imread(path, flags=mode)
    mask = img != 0

    mask = imresize(np.expand_dims(mask.astype(np.float32),-1), size=[mask.shape[0] * 4, mask.shape[1] * 4], interp='nearest', mode='F')
    mask = np.squeeze(mask)
    mask = mask >= 1.
    mask = mask.astype(np.float32)
    # cv2.imshow('imask', mask)
    # cv2.waitKey(0)
    # 腐蚀图像
    eroded = cv2.erode(img, kernel)
    # eroded = imresize(eroded, size=[eroded.shape[0]*4, eroded.shape[1]*4], interp='nearest', mode='F')
    # eroded = eroded*mask
    # cv2.imshow('eroded', eroded)
    # cv2.waitKey(0)
    # 膨胀图像
    # dilated = cv2.dilate(img, kernel)
    dilated = img
    dilated = imresize(np.expand_dims(dilated, -1), size=[dilated.shape[0] * 4, dilated.shape[1] * 4], interp='nearest', mode='F')

    # dilated = np.squeeze(dilated)*mask
    dilated = np.squeeze(dilated)
    # dilated = erode(dilated, ks=3)
    imageio.imwrite('C:\\Users\Administrator\Desktop\\0006.png', dilated.astype(np.uint16))

    cv2.imshow('dilate', dilated.astype(np.uint16))
    cv2.waitKey(0)

    result = cv2.absdiff(dilated, eroded)
    # plt.imshow(result)
    # plt.show()
    # cv2.imshow('result', result)
    # cv2.waitKey(0)

    result = result.astype(np.int32)
    result -= 700
    BM = result > 0

    # for h in range(height):
    #     for w in range(width):
    #         # result[h][w] = max_ - result[h][w]
    #         result[h][w] = 65535 if result[h][w] < 700 else 0
    return BM


def clip_by_border(path):
    f = open(path, mode='r')
    lines = f.readlines()
    lines = lines[11408:]
    print(lines[0])
    for line in lines:
        # print('%s-%s'%(line.strip().split('\\')[-3], line.strip().split('\\')[-1]))
        border = getborder(line.strip()+'depth.png', mode=-1)

        depth = cv2.imread(line.strip()+'depth.png', flags=-1)
        border = depth == 0
        # border = border.astype(np.uint8)*255
        # cv2.imshow('result', border)
        # cv2.waitKey(0)

        size_, gap = 120, 30
        height, width = depth.shape[0], depth.shape[1]
        height = height - height % size_
        width = width - width % size_

        for h in range(0, height-size_, gap):
            for w in range(0, width-size_, gap):
                if np.sum(border[h:h+size_, w:w+size_]) == 0:
                    # temp = depth[h:h+size_, w:w+size_]//65535*255
                    # cv2.imshow('result', temp)
                    # cv2.waitKey(0)
                    # print("./clip/%s%s-%s.png" % (lines[-2].strip(), h, w))
                    imageio.imwrite("./clip_nohole/%s-%s%s_%s.png" % (line.strip().split('\\')[-3], line.strip().split('\\')[-1], h, w), depth[h:h+size_, w:w+size_])


def depth2pcd(depth_path):
    depth = cv2.imread(depth_path, flags=-1)

    # depth = np.expand_dims(depth, -1)
    # depth = imresize(depth, size=[depth.shape[0]*4, depth.shape[1]*4], interp='bicubic')
    # imageio.imwrite('C:\\Users\Administrator\Desktop\\222222.png',
    #                 depth.astype(np.uint16))

    # cv2.imshow('border', depth)
    # cv2.waitKey(0)
    # depth = get_depth_fn(depth_path)
    image = cv2.imread(cp, flags=-1)
    image = image.astype(int)

    height, width = depth.shape[0], depth.shape[1]
    # image = imresize(image, size=[height, width, 3], interp='bicubic')

    mask = depth != 0
    mask = mask.astype(np.uint16)
    # mask = scipy.misc.imresize(mask, size=[height*4, width*4], interp='nearest', mode='F')
    #
    # depth = depth * mask

    point_num = mask.sum()
    PREFIX1 = "# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n" \
              "WIDTH %s\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %s\nDATA ascii\n" % (point_num, point_num)
    result_path = depth_path.strip().split('.')[0]+'.pcd'
    f = open(result_path, mode='x')

    f.write(PREFIX1)

    constant = 351.3
    MM_PER_M = 1092.5
    # print(height, width)

    for h in range(height):
        for w in range(width):
            # print(h, w)
            # if depth[h, w] >= 3000:

            f.write("%s %s %s %s\n" % (h*depth[h, w]/constant/MM_PER_M, w*depth[h, w]/constant/MM_PER_M,  depth[h, w]/MM_PER_M, (image[h, w, 0] << 16 | image[h, w, 1]<<8 | image[h, w, 2])))
            # else:
            #     f.write("%s %s nan\n" % (
            #     h * depth[h, w] / constant / MM_PER_M, w * depth[h, w] / constant / MM_PER_M))

    f.close()

fp = 'C:\\Users\Administrator\Desktop\\Valid_step_1601-scene_14_00648-.png'
fp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_10\\00004-depth.png'
cp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_14\\00640-color.png'
cp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_14\\00640-color.png'
fp2 = 'C:\\Users\Administrator\Desktop\\111114.png'
r1 = 'C:\\Users\Administrator\Desktop\\11119.png'
# rp = 'C:\\Users\Administrator\Desktop\\0002_result.pcd'
cp = 'D:\Downloads\FirefoxDownload\\color_200.png'
fp3 = 'C:\\Users\Administrator\Desktop\\result_0003.png'
# cp = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_14\\00648-color.png'
p1 = 'D:\Projects\Datasets\\nyu_depth_v2_raw\\bathroom_0046\\r-1315333127.196258-1604187005.pgm'
p = 'C:\\Users\Administrator\Desktop\\Valid_step_701-scene_14_00648-.png'
fp1 = 'D:\Projects\\rgbd-scenes-v2\imgs\scene_14\\00648-depth.png'
nyu_fp = 'D:\Downloads\FirefoxDownload\\depth_200.png'
# depth2pcd(nyu_fp)
# pred = cv2.imread(fp, flags=-1)
# real = cv2.imread(fp1, flags=-1)
# print(' ')
# r, g, b = 109, 114, 134
# rgb = r<<16|g<<8|b
#
# print()
# import scipy.io as scio
# import h5py
# import cv2
# mat_path = 'C:\zdj\Projects2\\nyu_depth_v2_labeled.mat'
# # # m = scio.loadmat(mat_path)
# print(datetime.datetime.now())
# m = h5py.File(mat_path)
# num = 0
# _images = m['images'][:]
# _images = np.transpose(_images, axes=[0, 3, 2, 1])
# # _gray = cv2.cvtColor(_images[num, :, :, :], cv2.COLOR_BGR2GRAY)
# _depth = m['depths'][:]
# _depth = np.expand_dims(np.transpose(_depth, axes=[0, 2, 1]), axis=-1)*1000
# _depth = _depth.astype(np.uint16)
# # _instances = m['instances'][:]
# # _instances = np.expand_dims(np.transpose(_instances, axes=[0, 2, 1]), -1)
# # _instances = _instances.astype(np.uint8)
# # _depth_split = _depth*_instances.astype(np.uint16)
# _rawdepth = m['rawDepths'][:]
# _rawdepth = np.expand_dims(np.transpose(_rawdepth, axes=[0, 2, 1]), axis=-1)*1000
# _rawdepth = _rawdepth.astype(np.uint16)
# f=open('data2017/nyu_list.txt', mode='w')
# for i in range(_rawdepth.shape[0]):
#     f.write("dataset\\%05d-.png\n" % i)
#     # print('dataset/%05d_depth.png'%i)
#     # imageio.imwrite("dataset\\%05d-color.png" % i, _images[i, :, :, :])
#     # imageio.imwrite("dataset\\%05d-depth.png" % i, _depth[i, :, :, 0])
#     # imageio.imwrite("dataset\\%05d-rawDepth.png" % i, _rawdepth[i, :, :, 0])
#     # print('pause')
# # file_path = m['rawDepthFilenames']
# # st = file_path[0]
# # # print(st.value)
# # for s in st:
# #     obj = m[s]
# #     str = "".join(chr(i) for i in obj[:])
# #     print('dataset/'+str)
#
#
# print(datetime.datetime.now())
# # print(file_path[0])
# # plt.subplot(221)
# # plt.imshow(_images[num, :, :, :])
# # plt.axis('off')
# # plt.subplot(222)
# # # plt.imshow(np.squeeze(_instances[num, :, :]))
# # plt.imshow(cv2.cvtColor(_images[num, :, :, :], cv2.COLOR_BGR2GRAY), cmap='gray')
# # plt.axis('off')
# # plt.subplot(223)
# # plt.imshow(np.squeeze(_depth[num, :, :]), cmap='gray')
# # plt.axis('off')
# # plt.subplot(224)
# # # plt.imshow(np.squeeze(_depth_split[100, :, :]), cmap='gray')
# # plt.imshow(np.squeeze(_rawdepth[num, :, :]), cmap='gray')
# # plt.axis('off')
# # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
# # plt.show()
# # # imageio.imwrite("D:\Downloads\FirefoxDownload\\depth_%s.png"%num, _depth[num, :, :])
# # # imageio.imwrite("D:\Downloads\FirefoxDownload\\color_%s.png"%num, _images[num, :, :, :])
# #
# # # cv2.imshow('depth', _depth[100, :, :])
# # # cv2.imshow('image', _images[0, :, :,:])
# # # cv2.imshow('instance', _instances[100, :, :])
# # # cv2.waitKey(0)
# # print(' ')
# # p = 'C:\zdj\Projects2\SRGAN-master\\result\\step_1301_0.113542005.png'
# # i = cv2.imread(p, -1)
# print(' ')
# p = 'C:\zdj\Projects2\SRGAN-master\\result\\step_201_9316196.0.png'
# _img = scipy.misc.imread(p)
# print(' ')