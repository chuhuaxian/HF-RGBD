import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
from dataloader import NyuV2SparseDataset
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

        BASE_DIR = 'E:\Projects\Datasets\\realsense4'

        file_list = ['%s/%s'%(BASE_DIR, i) for i in os.listdir(BASE_DIR)]
        c_list = [i for i in file_list if 'col' in i]
        d_list = [i for i in file_list if 'gt' in i]
        r_list = [i for i in file_list if 'spa' in i]
        len_  = len(c_list)
        test_list = ['%s,%s,%s\n' % (c_list[i], d_list[i], r_list[i]) for i in range(len_)]
        # test_list = open('test_list.txt').readlines()

        test_dataset = NyuV2SparseDataset(test_list)

        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        model.load_state_dict(torch.load('Checkpoints\\%s' % config.model_name))
        niter = int(config.model_name.split('_')[-3])
        print('load success -> %s' % config.model_name)

        # Loss
        l1_criterion = nn.L1Loss()
        # Start testing...
        idx = 0
        total_ours = 0
        total_ddc = 0
        time_count = 0
        sum_ = 0
        from datetime import datetime
        for _input, _label in test_loader:

            model.eval()
            image = torch.autograd.Variable(_input.float().cuda())
            depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))

            # Normalize depth
            # depth_n = DepthNorm(depth)

            # print()

            # if epoch == 0:
            # writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
            # if epoch == 0

            start = datetime.now()
            output = model(image)
            end = datetime.now()
            print((end-start).microseconds)


            if sum_ > 1:
                time_count += (end - start).microseconds
                break

            if sum_ > 100:
                break
            sum_ += 1
            continue
        print(time_count / (sum_-1) / 1000 / 1000)


            # # output = output * 1000.
        #             # # cv2.imwrite('%s/%05d-ours.png' % (BASE_DIR, idx), output.cpu().numpy().squeeze().astype(np.uint16))
        #             # # idx += 1
        #             # # continue
        #             # raw = image[:,3:4,:,:].data
        #             # rgb = image[:,:3,:,:].data.cpu().numpy().squeeze().transpose(1, 2, 0)*255
        #             # rgb = rgb.astype(np.uint8)
        #             # # plt.subplot(131), plt.imshow(raw.cpu().numpy().squeeze(), cmap='jet')
        #             # # plt.subplot(132), plt.imshow(output.cpu().numpy().squeeze(), cmap='jet')
        #             # # plt.subplot(133), plt.imshow(depth.cpu().numpy().squeeze(), cmap='jet')
        #             #
        #             # raw = raw.cpu().numpy().squeeze()
        #             # output = output.cpu().numpy().squeeze()
        #             # gt = depth.cpu().numpy().squeeze()
        #             #
        #             # raw = depth2color(raw).astype(np.uint8)
        #             # output = depth2color(output).astype(np.uint8)
        #             # gt = depth2color(gt).astype(np.uint8)
        #             #
        #             #
        #             # RES =make_grid([rgb, raw,output,gt])
        #             #
        #             # plt.imshow(RES)
        #             # plt.show()
        #             # cv2.imwrite('%s/%05d-pseudo.png' % (BASE_DIR, idx), RES[:, :, (2, 1, 0)])
        #             # idx += 1
        #             # continue
            # xx = depth2color(output.cpu().numpy())
            # plt.imshow(RES)
            # plt.show()

            # raw = image[:,3:4,:,:].data
            # ddc = depth.data
            #
            # mask = raw > 0
            #
            # raw = raw[mask]
            # ddc = ddc[mask]
            # output = output[mask]

        #     loss_our = torch.mean(torch.abs(output-depth))
        #     # loss_ddc = torch.mean(torch.abs(ddc - raw))
        #     total_ours += loss_our.item()
        #     # total_ddc += loss_ddc.item()
        #
        #     print(idx)
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