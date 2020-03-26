import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
from dataloader import NyuV2RandRemoveDataset
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


def main():
    with torch.no_grad():
        # Arguments
        parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
        args = parser.parse_args()

        # Create model
        model = Model().cuda()

        print('Model created.')

        batch_size = 1
        # prefix = 'densenet_' + str(batch_size)
        DATA_DIR = 'E:\Projects\Datasets\\realsense3'
        file_list = ['%s/%s'%(DATA_DIR, i) for i in os.listdir(DATA_DIR)]
        color_list = [i for i in file_list if 'col' in i]
        depth_list = [i for i in file_list if 'gt' in i]
        raw_list = [i for i in file_list if 'dep' in i]
        ddc_list = [i for i in file_list if 'ddc' in i]
        len_ = len(color_list)
        total_list = ['%s,%s,%s,%s' % (color_list[i], depth_list[i], raw_list[i], ddc_list[i]) for i in range(len_)]
        print()
        # test_list = open('test_list.txt').readlines()
        test_list = total_list
        test_dataset = NyuV2RandRemoveDataset(test_list)

        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name))
        # model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name)['net'])
        niter = int(config.model_name.split('_')[-3])
        print('load success -> %s' % config.model_name)

        # Loss
        l1_criterion = nn.L1Loss()
        # Start testing...
        idx = 0
        # total_ours = 0
        # total_ddc = 0
        our_total_rel_loss = 0
        our_total_rmse_loss = 0
        our_total_log_loss = 0
        our_total_sigma1_loss = 0
        our_total_sigma2_loss = 0
        our_total_sigma3_loss = 0

        ddc_total_rel_loss = 0
        ddc_total_rmse_loss = 0
        ddc_total_log_loss = 0
        ddc_total_sigma1_loss = 0
        ddc_total_sigma2_loss = 0
        ddc_total_sigma3_loss = 0

        f = open('nyu_rand_remove_eval_2-09_90.txt', mode='w')
        count = 0
        for _input, _label, _ddc in test_loader:

            model.eval()
            image = torch.autograd.Variable(_input.float().cuda())
            depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))
            ddc = torch.autograd.Variable(_ddc.float().cuda(non_blocking=True))
            # Normalize depth
            # depth_n = DepthNorm(depth)

            # print()

            # if epoch == 0:
            # writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
            # if epoch == 0
            output = model(image)
            output = output.detach().cpu().numpy().squeeze()*1000.
            output = output.astype(np.uint16)
            cv2.imwrite('%s/%05d-ours.png' % ('E:\Experimrnt\Completion\Rand', count), output)
            count += 1
            continue

            # plt.subplot(221), plt.imshow(image[:,:3,:,:].data.cpu().numpy().squeeze().transpose((1, 2, 0)))
            # plt.subplot(222), plt.imshow(depth.data.cpu().numpy().squeeze())
            # plt.subplot(223), plt.imshow(ddc.data.cpu().numpy().squeeze())
            # plt.subplot(224), plt.imshow(output.cpu().numpy().squeeze())
            # plt.show()
            # raw = image[:,3:4,:,:].data
            gt = depth.data
            ddc = ddc.data

            # mask = raw > 0
            # raw = raw[mask]
            # ddc = ddc[mask]
            # output = output[mask]
            our_abs_diff = (output - gt).abs()
            ddc_abs_diff = (ddc - gt).abs()

            rel_loss = our_abs_diff.mean()
            rmse_loss = torch.pow(our_abs_diff, 2).mean()
            log10_loss = (torch.log10(output)-torch.log10(gt)).abs().mean()
            maxRatio = torch.max(output / gt, gt / output)
            sigma1_loss = (maxRatio < 1.25).float().mean().float()
            sigma2_loss = (maxRatio < 1.25**2).float().mean().float()
            sigma3_loss =(maxRatio < 1.25**3).float().mean().float()

            f.write('our,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(), sigma3_loss.item()))
            # print()
            rel_loss = ddc_abs_diff.mean()
            rmse_loss = torch.pow(ddc_abs_diff, 2).mean()
            log10_loss = (torch.log10(gt)-torch.log10(ddc)).abs().mean()
            maxRatio = torch.max(gt / ddc, ddc / gt)
            sigma1_loss = (maxRatio < 1.25).float().mean().float()
            sigma2_loss = (maxRatio < 1.25**2).float().mean().float()
            sigma3_loss =(maxRatio < 1.25**3).float().mean().float()

            f.write('ddc,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (
            rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(),
            sigma3_loss.item()))
            # loss_our = torch.mean(torch.abs(output-gt))
            # loss_ddc = torch.mean(torch.abs(ddc - gt))
            # print(loss_our.item(), loss_ddc.item())
            # total_ours += loss_our.item()
            # total_ddc += loss_ddc.item()
            idx +=1
            print(idx)
        f.close()
        # print('ours=', np.sqrt(total_ours/idx))
        # print('ddc=', np.sqrt(total_ddc / idx))
            # nms1 = []
            # for i in range(output.shape[0]):
            #     normal = depth2normal_cv(depth[i, :, :, :].detach().cpu().numpy())
            #     nms1.append(torch.from_numpy(np.expand_dims(np.transpose(normal.astype(np.float32), (2, 0, 1)), axis=0)))
            # nms1 = torch.cat(nms1, dim=0)
            # nms1 = vutils.make_grid(nms1[:, :, :, :].data, nrow=6, normalize=False).detach().cpu().numpy().astype(np.uint8)
            #
            # nms2 = []
            # for i in range(output.shape[0]):
            #     normal = depth2normal_cv(output[i, :, :, :].detach().cpu().numpy())
            #     nms2.append(torch.from_numpy(np.expand_dims(np.transpose(normal.astype(np.float32), (2, 0, 1)), axis=0)))
            # nms2 = torch.cat(nms2, dim=0)
            # nms2 = vutils.make_grid(nms2[:, :, :, :].data, nrow=6, normalize=False).detach().cpu().numpy().astype(np.uint8)
            # #
            # # plt.subplot(211), plt.imshow(np.transpose(nms1,(1,2,0)))
            # # plt.subplot(212), plt.imshow(np.transpose(nms2, (1, 2, 0)))
            # # plt.show()
            #
            #
            # raw = colorize(vutils.make_grid(image[:,3:4,:,:].data * 1000., nrow=6, normalize=False))
            # ddc = colorize(vutils.make_grid(depth.data * 1000., nrow=6, normalize=False))
            # res = colorize(vutils.make_grid(output.data*1000., nrow=6, normalize=False))
            #
            # rgb = vutils.make_grid(image[:,:3,:,:].data * 255., nrow=6, normalize=False).detach().cpu().numpy().astype(np.uint8)
            #
            # # print()
            # out = np.concatenate([rgb, raw, ddc, res], axis=1)
            # out = np.transpose(out, (1, 2, 0))
            # # plt.imsave('C:\\Users\\39796\Desktop\\result\\%03d.png'%idx, out)
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