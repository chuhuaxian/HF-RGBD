import pyexr

import cv2

import matplotlib.pyplot as plt
import numpy as np
# img_low = pyexr.open('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\samples\\1-hbao_low.exr').get()
# img_high = pyexr.open('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\samples\\1-hbao_high.exr').get()
#
# img_low = np.clip(img_low, 0., 1.)*255
# img_low = img_low.astype(np.uint8)
# cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\samples\\1-hbao_low.png', img_low)
#
# img_high = np.clip(img_high, 0., 1.)*255
# img_high = img_high.astype(np.uint8)
# cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\samples\\1-hbao_high.png', img_high)

#
# plt.subplot(121), plt.imshow(img_low)
# plt.subplot(122), plt.imshow(img_high)
# plt.show()

img_low = pyexr.open('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\radius\\1-nnao_256.exr').get()
img_high = pyexr.open('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\radius\\1-nnao_15.exr').get()

img_low = np.clip(img_low, 0., 1.)*255
img_low = img_low.astype(np.uint8)
cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\radius\\1-nnao_256.png', img_low)

img_high = np.clip(img_high, 0., 1.)*255
img_high = img_high.astype(np.uint8)
cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\\radius\\1-nnao_15.png', img_high)


# plt.subplot(121), plt.imshow(img_low)
# plt.subplot(122), plt.imshow(img_high)
# plt.show()