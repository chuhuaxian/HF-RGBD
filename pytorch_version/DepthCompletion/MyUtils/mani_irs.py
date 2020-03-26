import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pyexr
import OpenEXR
import numpy
import Imath


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if CNum > 1:
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in Channels]
    hdr = numpy.zeros((Size[1],Size[0],CNum),dtype=numpy.float32)
    if CNum == 1:
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = numpy.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = numpy.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

focal_length = 480
baseline = 0.1

IRS_DIR = 'F:\Datasets\IRS'
RES_DIR = ''
scene_lst = [os.path.join(IRS_DIR, i) for i in os.listdir(IRS_DIR) if not i.endswith('ini')]

count = 0

xx = 0
for scene in scene_lst:
    room_lst = [os.path.join(scene, i) for i in os.listdir(scene) if not i.endswith('Dark')]
    for room in room_lst:
        file_lst = os.listdir(room)
        print(room)
        # print(room[3:])
        save_room = 'E:\Projects\\' + room[3:]
        # d_lst = [i for i in file_lst]
        len_ = len(file_lst) // 4
        sum_ = 0
        for i in range(1, len_+1):

            lpath = '%s\\l_%s.png' % (room, i)
            npath = '%s\\n_%s.exr' % (room, i)
            dpath = '%s\\d_%s.exr' % (room, i)
            if not os.path.exists(dpath):
                continue


            disp = exr2hdr(dpath)
            disp += 1e-10
            #
            disp = baseline * focal_length / disp
            # # print(np.max(disp))
            if np.max(disp) > 60:
                continue
            disp = disp * 1000.
            normal = exr2hdr(npath)
            normal = normal*2-1

            nx, ny, nz = np.split(normal, 3, axis=2)
            ny = -ny
            nz = -nz
            # nx = -nx
            normal = np.concatenate([nx, ny, nz], axis=-1)
            normal = (normal+1) / 2
            normal = normal[:, :, (0, 1, 2)]

            rgb = cv2.imread(lpath)

            if not os.path.exists(save_room):
                os.makedirs(save_room)

            cv2.imwrite('%s\\%s-color.jpg' % (save_room, sum_), rgb)
            disp = disp.astype(np.uint16)
            cv2.imwrite('%s\\%s-depth.png' % (save_room, sum_), disp)
            normal = normal * 255.
            normal = normal.astype(np.uint8)
            cv2.imwrite('%s\\%s-normal.png' % (save_room, sum_), normal[:, :, (2,1,0)])

            sum_ += 1

            # plt.subplot(131), plt.imshow(rgb.squeeze()[:, :, (2,1,0)])
            # plt.subplot(132), plt.imshow(normal.squeeze())
            # plt.subplot(133), plt.imshow(disp.squeeze())
            # plt.show()

            # # mask = disp < 0.5

                # plt.imshow(disp.squeeze())
                # plt.show()
    # print(i)
print(count)