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

import PIL.ImageEnhance
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

# img = Image.open('E:\\Projects\\Datasets\\NYU_V2\\data\\nyu2_train\\bedroom_0138_out\\9.jpg')


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

        left = np.ones(shape=(480, 320), dtype=np.float32)
        right = np.zeros(shape=(480, 320), dtype=np.float32)
        self.mask = np.concatenate([left, right], axis=1) + 1e-10
        # self.eps = np.concatenate([left, right], axis=1)

    def __call__(self, sample):

        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # img = numpy.array(im)

        image, depth = sample['image'], sample['depth']

        image = np.asarray(image)
        # image = image[16:-16, 16:-16, :]

        # image = self.to_tensor(image)

        # plt.subplot(121), plt.imshow(depth)
        # depth = depth.resize((320, 240))
        # depth = np.asarray(depth)
        input_depth = np.asarray(depth.copy())*self.mask
        input = np.concatenate([image, np.expand_dims(input_depth, axis=-1)], axis=-1)
        input = input[16:-16, 16:-16, :]

        # plt.imshow(input)
        # plt.show()



        input = self.to_tensor(input)

        depth = np.asarray(depth)[16:-16, 16:-16]
        depth = np.expand_dims(depth, axis=-1)


        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000
        # depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': input, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)
    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])


def getDefaultTrainTransform():
    return transforms.Compose([RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()])


def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem('E:\\Projects\\Datasets\\nyu_data.zip')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=True)

def getTestingData(batch_size):
    data, nyu2_train = loadZipToMem('E:\\Projects\\Datasets\\nyu_data.zip')

    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(DataLoader(transformed_testing, batch_size, shuffle=True))
