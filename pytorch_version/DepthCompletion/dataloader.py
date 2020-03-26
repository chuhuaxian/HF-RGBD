import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageStat
from io import BytesIO
import random
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import transforms as tfs
import PIL.ImageEnhance
from sklearn.model_selection import train_test_split

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomVerticalFlipMask(object):
    def __call__(self, mask):

        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return mask


class RandomHorizontalFlipMask(object):
    def __call__(self, mask):

        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return mask


# class RandomRotateMask(object):
#     def __call__(self, mask):
#
#         if not _is_pil_image(mask):
#             raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
#         if not _is_pil_image(mask):
#             raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
#
#         if random.random() < 0.5:
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#
#         return mask
from itertools import permutations

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


class RandomRemoveDepth(object):

    def __call__(self, sample, mask):
        image, depth = sample['image'], sample['depth']
        depth = np.array(depth) * np.array(mask)
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))

        return {'image': image, 'depth': Image.fromarray(depth)}


def real_load_fn(path):
    # 640, 480
    cp, dp, rp = path.strip().split(',')[:3]
    color = Image.open(cp)
    depth = Image.open(dp)
    # raw = Image.open(rp)

    if random.random() < 0.5:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    indices = list(permutations(range(3), 3))
    color = np.array(color)
    if random.random() < 0.5:
        color = np.array(color)[..., list(indices[random.randint(0, len(indices) - 1)])]
        # color = Image.fromarray(color[..., list(indices[random.randint(0, len(indices) - 1)])])
    depth = np.array(depth)
    return color.astype(np.float32) / 255., depth.astype(np.float32) / 1000.


def mask_load_fn(path):
    mask = Image.open(path.strip())

    if random.random() < 0.5:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    return np.asarray(mask) / 255.


def syn_load_fn(path):
    # 960, 540
    cp, dp = path.strip().split(',')
    res_wid, res_hei = 640, 480

    color = Image.open(cp)
    depth = Image.open(dp)

    img_wid, img_hei = color.size
    wid_range = img_wid - res_wid
    hei_range = img_hei - res_hei

    start_wid, start_hei = random.randint(0, wid_range-1), random.randint(0, hei_range-1)
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    # print()
    color = color.crop(box)
    depth = depth.crop(box)

    if random.random() < 0.5:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    indices = list(permutations(range(3), 3))
    color = np.array(color)
    if random.random() < 0.5:
        color = color[..., list(indices[random.randint(0, len(indices) - 1)])]
        # color = Image.fromarray(color[..., list(indices[random.randint(0, len(indices) - 1)])])
    depth = np.array(depth)
    return color.astype(np.float32) / 255., depth.astype(np.float32) / 1000.


def sun_load_fn(path):
    cp, rp = path.strip().split(',')

    color = Image.open(cp)
    raw = Image.open(rp)
    res_wid, res_hei = color.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = color.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    # print()
    color = color.crop(box)
    raw = raw.crop(box)

    color = np.asarray(color).astype(np.float32) / 255.
    raw = np.asarray(raw).astype(np.float32) / 1000.

    return color, raw

def nyuV2_sparse_load_fn(path):
    cp, dp, rp = path.strip().split(',')[:3]
    color = Image.open(cp)
    depth = Image.open(dp)
    raw = Image.open(rp)

    res_wid, res_hei = color.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = color.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    # print()
    color = color.crop(box)
    raw = raw.crop(box)
    depth = depth.crop(box)

    color = np.asarray(color).astype(np.float32) / 255.
    depth = np.asarray(depth).astype(np.float32) / 1000.
    raw = np.asarray(raw).astype(np.float32) / 1000.

    return color, depth, raw


class NyuV2SparseDataset(Dataset):

    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):

        _color, _depth, _raw = nyuV2_load_fn(self.file_list[index])


        # plt.subplot(131), plt.imshow(_color)
        # plt.subplot(132), plt.imshow(_depth)
        # plt.subplot(133), plt.imshow(_raw)
        # plt.show()

        _color = _color.transpose((2, 0, 1))

        _input = np.concatenate([_color, np.expand_dims(_raw, axis=0)], axis=0)
        _label = np.expand_dims(_depth, axis=0)

        # return torch.from_numpy(_input[:, 16:-16, 16:-16]), torch.from_numpy(_label[:, 16:-16, 16:-16])
        return torch.from_numpy(_input), torch.from_numpy(_label)

    def __len__(self):
        return len(self.file_list)


def nyuV2_load_fn(path):
    cp, dp, rp = path.strip().split(',')[:3]
    color = Image.open(cp)
    depth = Image.open(dp)
    raw = Image.open(rp)

    res_wid, res_hei = color.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = color.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    # print()
    color = color.crop(box)
    raw = raw.crop(box)
    depth = depth.crop(box)

    color = np.asarray(color).astype(np.float32) / 255.
    depth = np.asarray(depth).astype(np.float32) / 1000.
    raw = np.asarray(raw).astype(np.float32) / 1000.

    return color, depth, raw


class NyuV2Dataset(Dataset):

    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):

        _color, _depth, _raw = nyuV2_load_fn(self.file_list[index])


        # plt.subplot(131), plt.imshow(_color)
        # plt.subplot(132), plt.imshow(_depth)
        # plt.subplot(133), plt.imshow(_raw)
        # plt.show()

        _color = _color.transpose((2, 0, 1))

        _input = np.concatenate([_color, np.expand_dims(_raw, axis=0)], axis=0)
        _label = np.expand_dims(_depth, axis=0)

        # return torch.from_numpy(_input[:, 16:-16, 16:-16]), torch.from_numpy(_label[:, 16:-16, 16:-16])
        return torch.from_numpy(_input), torch.from_numpy(_label)

    def __len__(self):
        return len(self.file_list)


def nyuV2_rand_remove_load_fn(path):
    cp, dp, rp, ddcp = path.strip().split(',')[:4]
    color = Image.open(cp)
    depth = Image.open(dp)
    raw = Image.open(rp)
    ddc = Image.open(ddcp)

    res_wid, res_hei = color.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = color.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域
    # print()
    color = color.crop(box)
    raw = raw.crop(box)
    depth = depth.crop(box)
    ddc = ddc.crop(box)

    color = np.asarray(color).astype(np.float32) / 255.
    depth = np.asarray(depth).astype(np.float32) / 1000.
    raw = np.asarray(raw).astype(np.float32) / 1000.
    ddc = np.asarray(ddc).astype(np.float32) / 1000.

    return color, depth, raw, ddc


class NyuV2RandRemoveDataset(Dataset):

    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):

        _color, _depth, _raw, _ddc = nyuV2_rand_remove_load_fn(self.file_list[index])


        # plt.subplot(131), plt.imshow(_color)
        # plt.subplot(132), plt.imshow(_depth)
        # plt.subplot(133), plt.imshow(_raw)
        # plt.show()

        _color = _color.transpose((2, 0, 1))

        _input = np.concatenate([_color, np.expand_dims(_raw, axis=0)], axis=0)
        _label = np.expand_dims(_depth, axis=0)
        _ddc = np.expand_dims(_ddc, axis=0)

        # return torch.from_numpy(_input[:, 16:-16, 16:-16]), torch.from_numpy(_label[:, 16:-16, 16:-16])
        return torch.from_numpy(_input), torch.from_numpy(_label), torch.from_numpy(_ddc)

    def __len__(self):
        return len(self.file_list)


class SunDataset(Dataset):

    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):

        # if 'realsense' not in self.file_list[index]:
        #     return []
        _color, _raw = sun_load_fn(self.file_list[index])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # plt.imshow(_raw)
        # plt.sh
        # nonzeros = np.nonzero(_raw)
        #
        # nonzeros = np.concatenate([np.expand_dims(nonzeros[0], axis=-1), np.expand_dims(nonzeros[1], axis=-1)], axis=-1)
        # samples, _ = train_test_split(nonzeros, test_size=1000, random_state=42)
        # samples = (samples[:, 0], samples[:, 1])
        # input = _raw.copy()
        # input[samples] = 0

        # plt.imshow(input)
        # plt.show()

        # for i in range(1):
        #     _raw = cv2.morphologyEx(_raw, cv2.MORPH_OPEN, kernel, iterations=1)
        #     dilate_img = cv2.dilate(_raw, kernel)
        #     erode_img = cv2.erode(_raw, kernel)
        #
        #     absdiff_img = cv2.absdiff(dilate_img, erode_img)
        #     absdiff_img = np.where(absdiff_img>1, 0, 1)
        #     _raw = _raw*absdiff_img


        # plt.imshow(absdiff_img)
        # plt.show()
        # retval, threshold_img = cv2.threshold(absdiff_img, 40, 255, cv2.THRESH_BINARY)
        # result = cv2.bitwise_not(threshold_img)

        # if 'realsense' in self.file_list[index]:
        #     plt.subplot(121), plt.imshow(_raw)
        #
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #
        #     closed1 = cv2.morphologyEx(_raw, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1
        #     closed2 = cv2.morphologyEx(_raw, cv2.MORPH_CLOSE, kernel, iterations=3)  # 闭运算2
        #     opened1 = cv2.morphologyEx(_raw, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算1
        #     opened2 = cv2.morphologyEx(_raw, cv2.MORPH_OPEN, kernel, iterations=3)
        #
        #     raw = cv2.dilate(opened1, np.ones(shape=(3,3)))
        #     plt.subplot(122), plt.imshow(raw)
        #     plt.show()

        # plt.subplot(131), plt.imshow(_color)
        # plt.subplot(132), plt.imshow(_depth)
        # plt.subplot(133), plt.imshow(_raw)
        # plt.show()

        _color = _color.transpose((2, 0, 1))

        _input = np.concatenate([_color, np.expand_dims(_raw, axis=0)], axis=0)
        # _label = np.expand_dims(_depth, axis=0)

        return torch.from_numpy(_input)

    def __len__(self):
        return len(self.file_list)


class TripleDataset(Dataset):

    def __init__(self, real_list,syn_list, mask_list, transforms=None):
        self.syn_list = syn_list

        self.real_list = real_list

        self.mask_list = mask_list

        self.transforms = transforms

        self.real_len = len(self.real_list)
        self.syn_len = len(self.syn_list)
        self.mask_len = len(self.mask_list)//2

    def __getitem__(self, index):
        _syn_color, _syn_depth = syn_load_fn(self.syn_list[(index+random.randint(0, self.syn_len)) % self.syn_len])
        _real_color, _real_depth = real_load_fn(self.real_list[(index+random.randint(0, self.real_len)) % self.real_len])
        _mask_1 = mask_load_fn(self.mask_list[index % self.mask_len])
        _mask_2 = mask_load_fn(self.mask_list[index % self.mask_len + self.mask_len])

        # plt.subplot(231), plt.imshow(_syn_color)
        # plt.subplot(232), plt.imshow(_syn_depth)
        # plt.subplot(234), plt.imshow(_real_color)
        # plt.subplot(235), plt.imshow(_real_depth)
        # plt.subplot(233), plt.imshow(_mask_1)
        # plt.subplot(236), plt.imshow(_real_depth*_mask_1)
        # plt.show()

        _syn_color = _syn_color.transpose((2, 0, 1))
        _real_color = _real_color.transpose((2, 0, 1))

        _syn_raw = _syn_depth * _mask_1
        _real_raw = _real_depth * _mask_2

        _syn_input = np.concatenate([_syn_color, np.expand_dims(_syn_raw, axis=0)], axis=0)
        _real_input = np.concatenate([_real_color, np.expand_dims(_real_raw, axis=0)], axis=0)

        _syn_label = np.expand_dims(_syn_depth, axis=0)
        _real_label = np.expand_dims(_real_depth, axis=0)

        # print()
        # if self.transforms is not None:
        #     _syn_color = self.transforms(_syn_color)
        #     _real_color = self.transforms(_real_color)

        # print(self.syn_list[index % self.syn_len])
        # print(self.real_list[index % self.real_len])
        # # print(self.mask_list[index])


        # if self.transforms is not None:
        #     _data1 = self.transforms(_syn_data)
        #     _data2 = self.transforms(_real_data)
        #     _data3 = self.transforms(_mask_data)

        return torch.from_numpy(_syn_input), torch.from_numpy(_real_input), torch.from_numpy(_syn_label), torch.from_numpy(_real_label)

    def __len__(self):
        return len(self.mask_list)//2


def get_image_light_mean(img):
    im = img.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def random_gamma_transform(img, gamma):
    if get_image_light_mean(img) <= 30:
        return img
    res = PIL.ImageEnhance.Brightness(img).enhance(gamma)
    if get_image_light_mean(res) <= 30:
        return img
    return res


class RandomGammaTransform(object):
    def __init__(self, bright_low=0.5, bright_high=1.5):
        self.bright_low = bright_low
        self.bright_high = bright_high

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))

        bright = np.random.uniform(self.bright_low, self.bright_high)
        image = random_gamma_transform(image, bright)
        return {'image': image, 'depth': depth}

camera_cx = 256.0
camera_cy = 256.0
camera_fx = 548.9937
camera_fy = 548.9937


row_map = np.ones(shape=(512, 512), dtype=np.float)*np.arange(0, 512)
col_map = np.transpose(row_map)
row_map = np.expand_dims(row_map, axis=-1)-camera_cx
col_map = np.expand_dims(col_map, axis=-1) - camera_cy


def depth2position(depth_map):
    depth_map = 1 - depth_map
    xx = 0.5+np.multiply(row_map, depth_map) / camera_fx
    yy = 0.35-np.multiply(col_map, depth_map) / camera_fy
    return np.concatenate([yy, xx, depth_map], axis=-1)


class NYUDataset(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch, x_mean, x_var, y_mean, y_var):
        self.one_batch = one_batch
        self.x_mean = x_mean
        self.x_var = x_var
        self.y_mean = y_mean
        self.y_var = y_var

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = np.array(self.one_batch['X'][index])
        label = np.array(self.one_batch['Y'][index])

        input = (input - self.x_mean) / self.x_var
        label = (label - self.y_mean) / self.y_var
        input = np.reshape(input, (-1, ))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch['Y'])


class ScanNetDataset(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch, x_mean, x_var, y_mean, y_var):
        self.one_batch = one_batch
        self.x_mean = x_mean
        self.x_var = x_var
        self.y_mean = y_mean
        self.y_var = y_var

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = np.array(self.one_batch['X'][index])
        label = np.array(self.one_batch['Y'][index])

        input = (input - self.x_mean) / self.x_var
        label = (label - self.y_mean) / self.y_var
        input = np.reshape(input, (-1, ))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch['Y'])


class IRSDataset(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch, x_mean, x_var, y_mean, y_var):
        self.one_batch = one_batch
        self.x_mean = x_mean
        self.x_var = x_var
        self.y_mean = y_mean
        self.y_var = y_var

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = np.array(self.one_batch['X'][index])
        label = np.array(self.one_batch['Y'][index])

        input = (input - self.x_mean) / self.x_var
        label = (label - self.y_mean) / self.y_var
        input = np.reshape(input, (-1, ))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch['Y'])


class RGBDDataset(Dataset):
    def __init__(self, datalist, transform=None):
        self.datalist = datalist
        self.transform = transform

    def __getitem__(self, idx):
        cn, dn = self.datalist[idx]
        image = Image.open(cn)
        depth = Image.open(dn)
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.datalist)


class SynDataset(Dataset):
    def __init__(self, datalist, transform=None):
        self.datalist = datalist
        self.transform = transform

    def __getitem__(self, idx):
        cn, dn = self.datalist[idx]
        image = Image.open(cn)
        depth = Image.open(dn)
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.datalist)


class RealDataset(Dataset):
    def __init__(self, datalist, transform=None):
        self.datalist = datalist
        self.transform = transform

    def __getitem__(self, idx):
        cn, dn = self.datalist[idx]
        image = Image.open(cn)
        depth = Image.open(dn)
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.datalist)





class MaskDataset(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch):
        self.one_batch = one_batch

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = np.array(self.one_batch['X'][index])
        label = np.array(self.one_batch['Y'][index])

        input = np.reshape(input, (-1, ))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch['Y'])







