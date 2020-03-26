import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from tools import read_pfm
from PIL import Image

BASE_DIR = 'E:\\Projects\\Datasets\\MIDDLEBURY\\2001'
focal_length = 3740
baseline = 0.16
min_ = 200

scene_lst = os.listdir(BASE_DIR)

for scene in scene_lst:
    files = os.listdir('%s/%s' % (BASE_DIR, scene))

    disp_lst = [i for i in files if 'disp' in i]
    color_lst = [i for i in files if 'disp' not in i]
    for index in range(len(disp_lst)):
        pdisp = '%s/%s/%s' % (BASE_DIR, scene, disp_lst[index])
        pcolor = '%s/%s/%s' % (BASE_DIR, scene, color_lst[index])
        print(pdisp)
        disp = np.array(Image.open(pdisp))
        mask = disp != 0
        disp = disp + 1e-10
        depth = focal_length*baseline/disp
        depth = depth * mask
        depth = depth * 1000.
        depth = depth.astype(np.uint16)
        plt.imshow(depth)
        plt.show()
