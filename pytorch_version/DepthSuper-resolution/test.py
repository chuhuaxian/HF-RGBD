import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils

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

import cv2

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=config.TRAIN_EPOCH, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create model
    model = Model().cuda()

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = 1
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    print(len(train_loader), len(test_loader))

    train_loader = None
    # Logging
    # writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)


    niter = 0
    if config.load_model and config.model_name != '':
        model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name))
        niter = int(config.model_name.split('_')[-3])
        print('load success -> %s' % config.model_name)

    bicubic_total_loss = 0
    our_total_loss = 0

    bicubic_total_loss_rel = 0
    our_total_loss_rel = 0

    bicubic_total_loss_rms = 0
    our_total_loss_rms = 0

    bicubic_total_loss = 0
    our_total_loss = 0


    count = 0
    model.eval()
    for i, sample_batched in enumerate(test_loader):
        if i> 1500:
            break
        # optimizer.zero_grad()
        print(i)
        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        rgb = sample_batched['image'][:, :3, :, :].numpy().squeeze().transpose((1, 2, 0))*255.
        rgb = rgb.astype(np.uint8)

        raw = sample_batched['image'][:, 3:, :, :].numpy().squeeze()
        raw = raw * 1000.
        # raw = raw.astype(np.uint8)


        # plt.subplot(141), plt.imshow(rgb)
        # plt.show()

        # Normalize depth
        depth_n = DepthNorm(depth)
        gt = depth_n.detach().cpu().numpy().squeeze()
        gt = DepthNorm(gt)
        # Predict
        output = model(image)
        output = output.detach().cpu().numpy().squeeze()
        output = DepthNorm(output)
        # raw = DepthNorm(raw)

        raw = raw / 100.
        output = output / 100.
        gt = gt /100.


        our_diff = np.mean(np.abs(gt-output))
        bic_diff = np.mean(np.abs(gt-raw))

        our_total_loss_rel += our_diff/gt
        bicubic_total_loss_rel += bic_diff/gt

        our_total_loss_rms += np.mean(np.square(our_diff))
        bicubic_total_loss_rms += np.mean(np.square(bic_diff))
        count += 1

        # print(np.max(raw), np.max(output), np.max(gt))
        res = np.concatenate([raw, output, gt], axis=1)
        res = depth2color(res, cmap='plasma')
        res = np.split(res, 3, axis=1)
        res = make_grid([rgb, res[0], res[1], res[2]])


        BASE_DIR = 'E:\Experimrnt\\SR'
        # cv2.imwrite('%s/%04d-pseudo.png' % (BASE_DIR, i), res[:, :, (2,1,0)])

        # plt.subplot(142), plt.imshow(raw, cmap='plasma')
        # plt.subplot(143), plt.imshow(output, cmap='plasma')
        # plt.subplot(144), plt.imshow(gt, cmap='plasma')
        # plt.show()
    print(our_total_loss_rel/count, bicubic_total_loss_rel/count)
    print(np.sqrt(our_total_loss_rms/count), np.sqrt(bicubic_total_loss_rms/count))


def LogProgress(model, writer, test_loader, global_step):
    model.eval()

    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    # Normalize depth
    depth_n = DepthNorm(depth)

    # print()

    # if epoch == 0:
    # writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    # if epoch == 0:

    # writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), global_step)
    output = model(image)

    plt.imshow(output[0].detach().cpu().numpy().squeeze(), cmap='plasma')
    plt.show()

    # Predict
    # output = model(image)

    # edge = sobel_(output)
    # pred_edge = sobel_(depth_n)

    # Compute the loss
    # l_sobel = nn.L1Loss()(edge, pred_edge)
    # l_depth = nn.L1Loss()(output, depth_n)
    # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

    # writer.add_scalar('Test/L1', l_depth.item(), global_step)
    # writer.add_scalar('Test/SSIM', l_ssim.item(), global_step)
    # writer.add_scalar('Test/EDGE', l_sobel.item(), global_step)

    # normal = Norm(output)
    # normal = (normal + 1.0) / 2.
    # normal = normal[3, :, :, :]
    # normal = normal.detach().cpu().numpy().astype(np.uint8)
    # normal = np.transpose(normal, (1, 2, 0))
    # print()
    # plt.imshow(normal)
    # plt.show()
    output = DepthNorm(output)
    # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
    # writer.add_image('Test.2.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), global_step)
    # writer.add_image('Test.3.Diff', colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)),
    #                  global_step)
    del image
    del depth
    del output
    # del edge
    # del pred_edge
    # del normal


if __name__ == '__main__':
    main()
