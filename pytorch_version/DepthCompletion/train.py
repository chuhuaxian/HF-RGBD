import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

from model import Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize
from config import config
import os
from Myloss import Normal
import matplotlib.pyplot as plt
import numpy as np
from  Myloss import Sobel
Norm = Normal(shape=[config.bs, 1, 240, 320]).cuda()
sobel_ = Sobel(winsize=3).cuda()

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
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    niter = 0
    if config.load_model and config.model_name != '':
        model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name))
        niter = int(config.model_name.split('_')[-3])
        print('load success -> %s' % config.model_name)

    # Start training...
    for epoch in range(0, args.epochs):



        # batch_time = AverageMeter()
        loss_l1 = AverageMeter()
        loss_ssim = AverageMeter()
        loss_edge = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()


        # end = time.time()

        for i, sample_batched in enumerate(train_loader):

            if niter % config.save_interval == 0:
                if not os.path.exists('Checkpoints\\%s' % config.save_name):
                    os.makedirs('Checkpoints\\%s' % config.save_name)

                save_name = '%s\\net_params_%s_%s.pkl' % (
                config.save_name, niter, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

                torch.save(model.state_dict(), 'Checkpoints\\%s' % save_name)
                print('save success -> %s' % save_name)

            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm(depth)

            # Predict
            output = model(image)
            edge = sobel_(output)
            pred_edge = sobel_(depth_n)
            # Compute the loss
            l_sobel = nn.L1Loss()(edge, pred_edge)
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

            # print('nan', torch.sum(torch.isnan(image)).item())
            # print('inf', torch.sum(torch.isinf(image)).item())
            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            # if torch.isnan(l_sobel) or torch.isnan(l_depth) or torch.isnan(l_ssim) or torch.isinf(l_sobel) or torch.isinf(l_depth) or torch.isinf(l_ssim):
            #     print(0)
            #     print('nan', torch.sum(torch.isnan(image)).item())
            #     print('inf', torch.sum(torch.isinf(image)).item())
            #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))
            #     print(torch.isinf(l_sobel), torch.isinf(l_depth), torch.isinf(l_ssim))
            #     return

            # Update step
            loss_l1.update(l_depth.data.item(), image.size(0))
            loss_ssim.update(l_ssim.data.item(), image.size(0))
            loss_edge.update(l_sobel.data.item(), image.size(0))

            loss.backward()
            optimizer.step()

            # Measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            # eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter += 1
            if i % 5 == 0:
                # Print to console
                # print('Epoch: [{0}][{1}/{2}]\t'
                # 'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                # 'ETA {eta}\t'
                # 'Loss {loss.val:.4f} ({loss.avg:.4f})'
                # .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/L1', loss_l1.val, niter)
                writer.add_scalar('Train/SSIM', loss_ssim.val, niter)
                writer.add_scalar('Train/EDGE', loss_edge.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        # writer.add_scalar('Train/Loss.avg', losses.avg, epoch)


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


    writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), global_step)
    output = model(image)


    # Predict
    # output = model(image)

    edge = sobel_(output)
    pred_edge = sobel_(depth_n)

    # Compute the loss
    l_sobel = nn.L1Loss()(edge, pred_edge)
    l_depth = nn.L1Loss()(output, depth_n)
    l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))


    writer.add_scalar('Test/L1', l_depth.item(), global_step)
    writer.add_scalar('Test/SSIM', l_ssim.item(), global_step)
    writer.add_scalar('Test/EDGE', l_sobel.item(), global_step)

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
    writer.add_image('Test.2.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), global_step)
    writer.add_image('Test.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), global_step)
    del image
    del depth
    del output
    del edge
    del pred_edge
    # del normal


if __name__ == '__main__':
    main()
