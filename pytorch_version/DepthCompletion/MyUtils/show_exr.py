import pyexr
import cv2
import numpy as np

ao = pyexr.open('C:\\Users\\39796\Desktop\\3-gt.exr').get()[:, :, 0]
ao = ao*255.

ao = np.clip(ao, 0, 255)
ao = ao.astype(np.uint8)
print(ao.shape)
cv2.imwrite('C:\\Users\\39796\Desktop\\3-gt.png', ao)