import numpy as np
import csv
import re

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
    dispariy = np.reshape(dispariy, newshape=(height, width, channels))
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
        depth_map = fx * base_line / (dispariy / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        # print(np.unique(depth_map).shape)
        depth_map = np.flipud(depth_map)

        return depth_map