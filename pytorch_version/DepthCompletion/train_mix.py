import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from dataloader import TripleDataset, NyuV2Dataset
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


# Norm = Normal(shape=[config.bs, 1, 240, 320]).cuda()
# sobel_ = Sobel(winsize=5).cuda()

def make_grid(arr_lst, pad_value=1, pad_size=10):
    shape_ = arr_lst[0].shape
    pad_ = np.ones((shape_[0], pad_size, 3), dtype=np.uint8)*pad_value*255
    out = []
    for i in range(len(arr_lst)):
        out.append(arr_lst[i])
        if i < len(arr_lst)-1:
            out.append(pad_)
    return np.concatenate(out, axis=1)

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

    nyu_list = open('nyu_list.txt').readlines()
    scannet_list = open('scannet_list.txt').readlines()
    real_list = nyu_list + scannet_list

    irs_list = open('irs_list.txt').readlines()
    mask_list = open('mask_list.txt').readlines()

    test_list = open('test_list.txt').readlines()

    train_dataset = TripleDataset(real_list, irs_list, mask_list, None)
    test_dataset = NyuV2Dataset(test_list)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs, num_workers=8, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=8, shuffle=True)

    # Load data
    # train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss().cuda()

    niter = 0
    if config.load_model and config.model_name != '':
        checkpoint = torch.load('Checkpoints\\%s' % config.model_name)

        model.load_state_dict(checkpoint['net'])

        optimizer.load_state_dict(checkpoint['optimizer'])

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
        idx = 0
        for syn_input, real_input, syn_label, real_label in (train_loader):

            if niter % config.save_interval == 0:
                if not os.path.exists('Checkpoints\\%s' % config.save_name):
                    os.makedirs('Checkpoints\\%s' % config.save_name)

                save_name = '%s\\net_params_%s_%s.pkl' % (
                    config.save_name, niter, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, 'Checkpoints\\%s' % save_name)
                print('save success -> %s' % save_name)

            optimizer.zero_grad()

            _input = torch.cat([syn_input, real_input], dim=0)
            _label = torch.cat([syn_label, real_label], dim=0)

            # Prepare sample and target
            image = torch.autograd.Variable(_input.float().cuda())
            depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))

            # Normalize depth
            # depth_n = DepthNorm(depth)

            # Predict
            output = model(image)
            # pred_edge = sobel_(output)
            # edge = sobel_(depth)

            # pred_edge = torch.where(pred_edge>0.2, pred_edge, torch.full_like(pred_edge, 0))
            # edge = torch.where(edge > 0.2, edge, torch.full_like(pred_edge, 0))

            # Compute the loss
            # l_sobel = nn.L1Loss()(edge, pred_edge)
            l_depth = l1_criterion(output, depth)
            l_ssim = torch.clamp((1 - ssim(output, depth, val_range=10.0)) * 0.5, 0, 1)

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
            # loss_edge.update(l_sobel.data.item(), image.size(0))

            loss.backward()
            optimizer.step()

            # Measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            # eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            # Log progress
            niter += 1
            if idx % 50 == 0:
                # Print to console
                # print('Epoch: [{0}][{1}/{2}]\t'
                # 'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                # 'ETA {eta}\t'
                # 'Loss {loss.val:.4f} ({loss.avg:.4f})'
                # .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/L1', loss_l1.val, niter)
                writer.add_scalar('Train/SSIM', loss_ssim.val, niter)
                # writer.add_scalar('Train/EDGE', loss_edge.val, niter)

            if idx % 1000 == 0:
                LogProgress(model, writer, test_loader, niter)
            idx += 1

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        # writer.add_scalar('Train/Loss.avg', losses.avg, epoch)


def LogProgress(model, writer, test_loader, global_step):
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        _input, _label = next(iter(sequential))

        image = torch.autograd.Variable(_input.float().cuda())
        depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))

        # Normalize depth
        # depth_n = DepthNorm(depth)

        # print()

        # if epoch == 0:
        # writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
        # if epoch == 0:

        output = model(image)

        # Predict
        # output = model(image)

        # edge = sobel_(output)
        # pred_edge = sobel_(depth)

        # Compute the loss
        # l_sobel = nn.L1Loss()(edge, pred_edge)
        l_depth = nn.L1Loss()(output, depth)
        l_ssim = torch.clamp((1 - ssim(output, depth, val_range=10.0)) * 0.5, 0, 1)

        # if torch.isnan(l_sobel):
        #     print(1)
        #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

        writer.add_scalar('Test/L1', l_depth.item(), global_step)
        writer.add_scalar('Test/SSIM', l_ssim.item(), global_step)
        # writer.add_scalar('Test/EDGE', l_sobel.item(), global_step)

        # normal = Norm(output)
        # normal = (normal + 1.0) / 2.
        # normal = normal[3, :, :, :]
        # normal = normal.detach().cpu().numpy().astype(np.uint8)
        # normal = np.transpose(normal, (1, 2, 0))
        # print()
        # plt.imshow(normal)
        # plt.show()
        # output = DepthNorm(output)
        # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
        _fig= colorize(image.data, depth.data, output.data)
        # _ = vutils.make_grid(_, nrow=1, padding=0, normalize=False)
        # res1 = torch.cat([image[0:1, 3:4].data, depth[0:1].data, output[0:1].data], dim=0)
        # xx = colorize(vutils.make_grid(res1, nrow=6, padding=0, normalize=False))
        # rgb = image[0:1, :3, :, :].data.cpu().numpy() * 255.
        # rgb = rgb.astype(np.uint8).squeeze()
        # out = np.concatenate([rgb, xx], axis=2)
        # print()
        writer.add_image('Train.2.test', _fig.transpose((2, 0, 1)), global_step)
        # writer.add_image('Train.2.Raw', colorize(vutils.make_grid(image[:,3:4,:,:].data, nrow=6, normalize=False)),
        #                  global_step)
        # writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), global_step)
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
