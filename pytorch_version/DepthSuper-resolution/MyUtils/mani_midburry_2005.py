import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools import read_pfm

BASE_DIR = 'E:\\Projects\\Datasets\\MIDDLEBURY\\2005'

scene_lst = os.listdir(BASE_DIR)
focal_length = 3740
baseline = 0.16

for scene in scene_lst:
    files = os.listdir('%s/%s' % (BASE_DIR, scene))

    disp_lst = [i for i in files if 'disp' in i]
    dmin = [i for i in files if 'txt' in i][0]
    min_ = float(open('%s/%s/dmin.txt'% (BASE_DIR, scene)).readline().strip())
    print()
    # color_lst = [i for i in files if 'disp' not in i]
    for index in range(len(disp_lst)):
        pdisp = '%s/%s/%s' % (BASE_DIR, scene, disp_lst[index])
        # pcolor = '%s/%s/%s' % (BASE_DIR, scene, color_lst[index])
        print(pdisp)
        disp = cv2.imread(pdisp)[:, :, 0]
        mask = disp != 0

        disp = disp + 1e-10
        disp = focal_length*baseline/disp
        disp = disp*mask
        # print(np.max(disp), np.min(disp), np.mean(disp))
        disp  = disp * 1000
        disp = disp.astype(np.uint16)
        # cv2.imwrite('C:\\Users\\39796\Desktop\\xxxxxx.png',disp)
        # disp = disp.astype(np.uint16)
        plt.imshow(disp)
        plt.show()
