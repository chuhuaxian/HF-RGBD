import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# depth = cv2.imread('E:\Projects\Datasets\InStereo2K\\train\part1\\000276\\left_disp.png', -1)
# mask = depth!=0
#
# depth = depth + 1e-10
# depth = 100*1000* 0.1*480/depth
#
# depth = depth * mask
#
# depth = depth.astype(np.uint16)
# cv2.imwrite('C:\\Users\\39796\Desktop\\0-depth.png', depth)
# cv2.imwrite('C:\\Users\\39796\Desktop\\0-rawDepth-open.png', depth)
#
# plt.imshow(depth)
#
# # print(depth.shape)
# # input_depth = np.asarray(depth.copy().resize((2442//8, 452//8), Image.NEAREST).resize((2442, 452), Image.BICUBIC))
# # plt.imsave('C:\\Users\\39796\\Desktop\\individualImage2-input_1.png', input_depth)
#
# # plt.subplot(122), plt.imshow(input_depth)
# plt.show()

print(1.05**23)

def psnr1(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)
import math
print(10* math.log10(255.0**2/0.07579632109705702), 10* math.log10(255.0**2/0.037052201873323906))