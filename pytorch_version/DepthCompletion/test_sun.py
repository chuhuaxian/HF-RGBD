import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from dataloader import SunDataset
from model import Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize
from config import config
import os
from Myloss import Normal
import matplotlib.pyplot as plt
import numpy as np
from Myloss import Sobel
from torch.utils.data import DataLoader
import cv2

Norm = Normal(shape=[config.bs, 1, 240, 320]).cuda()
sobel_ = Sobel(winsize=3).cuda()


def depth2normal_cv(depth, win_size=15):
    depth = np.squeeze(depth)

    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    # depth = np.expand_dims(depth, axis=-1)
    # zy, zx = np.gradient(depth)
    # depth = depth
    zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=win_size)
    zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=win_size)

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)

    # n = np.sqrt(np.sum(np.square(normal), axis=-1))
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255
    nm = normal.astype('uint8')

    # plt.imshow(nm)
    # plt.show()
    return nm

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    args = parser.parse_args()

    # Create model
    model = Model().cuda()

    print('Model created.')

    batch_size = 1
    # prefix = 'densenet_' + str(batch_size)

    test_list = open('sun_list.txt').readlines()

    test_dataset = SunDataset(test_list)

    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name))
    niter = int(config.model_name.split('_')[-3])
    print('load success -> %s' % config.model_name)

    # Loss
    l1_criterion = nn.L1Loss()
    # Start testing...
    idx = 0
    for _input in test_loader:

        model.eval()
        # image = torch.autograd.Variable(_input.float().cuda())

        # Normalize depth
        # depth_n = DepthNorm(depth)

        # print()

        # if epoch == 0:
        # writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
        # if epoch == 0



        # output = model(image)
        #
        # # nms2 = []
        # # for i in range(output.shape[0]):
        # #     normal = depth2normal_cv(output[i, :, :, :].detach().cpu().numpy())
        # #     nms2.append(torch.from_numpy(np.expand_dims(np.transpose(normal.astype(np.float32), (2, 0, 1)), axis=0)))
        # # nms2 = torch.cat(nms2, dim=0)
        # # nms2 = vutils.make_grid(nms2[:, :, :, :].data, nrow=6, normalize=False).detach().cpu().numpy().astype(np.uint8)
        # #
        # # plt.subplot(211), plt.imshow(np.transpose(nms1,(1,2,0)))
        # # plt.subplot(212), plt.imshow(np.transpose(nms2, (1, 2, 0)))
        # # plt.show()
        #
        #
        # raw = colorize(vutils.make_grid(image[:,3:4,:,:].data * 1000., nrow=6, normalize=False))
        # res = colorize(vutils.make_grid(output.data*1000., nrow=6, normalize=False))
        #
        # rgb = vutils.make_grid(image[:,:3,:,:].data * 255., nrow=6, normalize=False).detach().cpu().numpy().astype(np.uint8)
        #
        # # print()
        # out = np.concatenate([rgb, raw, res], axis=1)
        # out = np.transpose(out, (1, 2, 0))
        # plt.imsave('C:\\Users\\39796\Desktop\\sun\\%05d.png'%idx, out)
        # idx += 1




        # print()
        # plt.imshow(out), plt.axis('off')
        # plt.show()
        # plt.subplot(411), plt.imshow(np.transpose(rgb, (1, 2, 0))), plt.axis('off')
        # plt.subplot(412), plt.imshow(np.transpose(raw, (1, 2, 0))), plt.axis('off')
        # plt.subplot(413), plt.imshow(np.transpose(ddc, (1, 2, 0))), plt.axis('off')
        # plt.subplot(414), plt.imshow(np.transpose(res, (1, 2, 0))), plt.axis('off')
        # plt.show()


        # Compute the loss
        # l_sobel = nn.L1Loss()(edge, pred_edge)
        # l_depth = nn.L1Loss()(output, depth)
        # l_ssim = torch.clamp((1 - ssim(output, depth, val_range=10.0)) * 0.5, 0, 1)




if __name__ == '__main__':
    main()
#