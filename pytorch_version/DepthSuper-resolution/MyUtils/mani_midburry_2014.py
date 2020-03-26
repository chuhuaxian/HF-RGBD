from pathlib import Path
import numpy as np
import csv
import re
import cv2
import os
import pyexr
import matplotlib.pyplot as plt

def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #
    # img = np.reshape(dispariy, newshape=(height, width, channels))
    # img = np.flipud(img).astype('uint8')
    #
    # show(img, "disparity")

    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib=None):
    dispariy, [shape, scale] = read_pfm(pfm_file_path)

    print()

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))

        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])
        # print(fx, ',         ', base_line, ',       ',  doffs, ',       ', scale)
        # scale factor is used here
        # plt.imshow(np.reshape(dispariy, newshape=shape).squeeze())
        # plt.show()
        depth_map = fx * base_line/10 / (dispariy / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        # print(np.unique(depth_map).shape)
        depth_map = np.flipud(depth_map)

        return depth_map


def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)


def main():
    BASE_DIR = 'E:\\Projects\\Datasets\\MIDDLEBURY\\2014'
    file_lst = [os.path.join(BASE_DIR, i) for i in os.listdir(BASE_DIR)]
    for pfm_file_dir in file_lst:

        # print(pfm_file_dir)
        # if pfm_file_dir != 'E:\Projects\Datasets\MIDDLEBURY\\2014\Shopvac-imperfect':
        #     continue
        # 0
        calib_file_path = os.path.join(pfm_file_dir, 'calib.txt')
        disp_left = os.path.join(pfm_file_dir, 'disp0.pfm')

        # calibration information
        calib = read_calib(calib_file_path)
        # create depth map
        depth_map_left = create_depth_map(disp_left, calib)
        depth_map_left = depth_map_left * 1000.
        depth_map_left = depth_map_left.astype(np.uint16)
        plt.imshow(depth_map_left.squeeze())
        plt.show()

        # print(np.max(depth_map_left))




        # cv2.imwrite('%s/disp0.png' % pfm_file_dir, depth_map_left.astype(np.uint8))
        # pyexr.write('%s/disp0.exr' % pfm_file_dir, depth_map_left)
        # cv2.imwrite('%s/disp0_16.png ' % pfm_file_dir, (depth_map_left*255.).astype(np.uint16))







        # min_ = np.min(depth_map_left)
        # print(min_)
        #
        # if min_<0:
        #     min_ = -min_
        #     depth_map_left += np.clip(min_, 0, 10)
        #
        #     depth_map_left = np.where(depth_map_left>500, 0, depth_map_left)
        #     depth_map_left = np.where(depth_map_left < 0, 0, depth_map_left)
        #     # depth_map_left -= 10
        #     # plt.imshow(depth_map_left.squeeze())
        #     # plt.show()
        #
        #     cv2.imwrite('%s/disp0_1.png' % pfm_file_dir, depth_map_left.astype(np.uint8))
        #     pyexr.write('%s/disp0_1.exr' % pfm_file_dir, depth_map_left)
        #     min_ = np.min(depth_map_left)
        #     max_ = np.max(depth_map_left)
        #     depth_map_left = np.clip(65535 * (depth_map_left - min_) / (max_ - min_), 0, 65535)
        #     cv2.imwrite('%s/disp0_1_16_%.02f_%.02f.png ' % (pfm_file_dir,  max_, min_), (depth_map_left).astype(np.uint16))
        #
        # else:
        #     cv2.imwrite('%s/disp0_1.png' % pfm_file_dir, depth_map_left.astype(np.uint8))
        #     pyexr.write('%s/disp0_1.exr' % pfm_file_dir, depth_map_left)
        #     min_ = np.min(depth_map_left)
        #     max_ = np.max(depth_map_left)
        #     depth_map_left = np.clip(65535 * (depth_map_left - min_) / (max_ - min_), 0, 65535)
        #     cv2.imwrite('%s/disp0_1_16_%.02f_%.02f.png ' % (pfm_file_dir, max_, min_), (depth_map_left).astype(np.uint16))
        #
        # print()
        # # 1
        # calib_file_path = os.path.join(pfm_file_dir, 'calib.txt')
        # disp_left = os.path.join(pfm_file_dir, 'disp1.pfm')
        #
        # # calibration information
        # calib = read_calib(calib_file_path)
        # # create depth map
        # depth_map_left = create_depth_map(disp_left, calib)
        # # cv2.imwrite('%s/disp1.png' % pfm_file_dir, depth_map_left.astype(np.uint8))
        # # pyexr.write('%s/disp1.exr' % pfm_file_dir, depth_map_left)
        # # cv2.imwrite('%s/disp1_16.png' % pfm_file_dir, (depth_map_left*255.).astype(np.uint16))
        #
        # depth_map_left = np.where(depth_map_left>500, 0, depth_map_left)
        # cv2.imwrite('%s/disp1_1.png' % pfm_file_dir, depth_map_left.astype(np.uint8))
        # pyexr.write('%s/disp1_1.exr' % pfm_file_dir, depth_map_left)
        # min_ =  np.min(depth_map_left)
        # max_ = np.max(depth_map_left)
        # depth_map_left = np.clip(65535*(depth_map_left-min_)/(max_-min_), 0, 65535)
        # cv2.imwrite('%s/disp1_1_16_%.02f_%.02f.png ' % (pfm_file_dir,  max_, min_), (depth_map_left).astype(np.uint16))
        #
        # print()

    # show(depth_map_left, "depth_map")

if __name__ == '__main__':
    main()
