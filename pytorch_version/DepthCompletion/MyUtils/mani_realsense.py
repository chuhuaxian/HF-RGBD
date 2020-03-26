import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path = 'C:\\Users\\39796\Desktop\SunRGBD\\realsense'
files = [i for i in os.listdir(path) if 'depth' in i]
for i in files:
    filename = '%s/%s' % (path, i)
    _raw = cv2.imread(filename, -1)
    # plt.subplot(121),plt.imshow(_raw)
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # for i in range(4):
    # _raw = cv2.morphologyEx(_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    # dilate_img = cv2.dilate(_raw, kernel)
    erode_img = cv2.erode(_raw, kernel)
    #
    # absdiff_img = cv2.absdiff(dilate_img, erode_img)
    # absdiff_img = np.where(absdiff_img > 1, 0, 1)
    # _raw = _raw * absdiff_img

    # _raw = cv2.morphologyEx(_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    # dilate_img = cv2.dilate(_raw, kernel)
    # erode_img = cv2.erode(_raw, kernel)
    #
    # absdiff_img = cv2.absdiff(dilate_img, erode_img)
    # absdiff_img = np.where(absdiff_img > 1, 0, 1)
    # _raw = _raw * absdiff_img
    # _raw = _raw.astype(np.uint16)
    # plt.subplot(122), plt.imshow(erode_img)
    # plt.show()

    cv2.imwrite(filename, erode_img.astype(np.uint16))
    print(filename)