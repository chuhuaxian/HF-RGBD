import cv2

import matplotlib.pyplot as plt


order =22
img = cv2.imread('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\Experiment\\%s-nnao.png' % order)

img_median = cv2.medianBlur(img, 3)

# cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\Experiment\\%s-nnao_blur.png' %order, img_median)
# img_mean = cv2.blur(img, (5,5))
img_bilater = cv2.bilateralFilter(img,18,85,85)

cv2.imwrite('C:\\Users\\39796\Desktop\Ambient Occlosion Paper\Experiment\\%s-nnao_blur_bilater.png' %order, img_bilater)
# img_Guassian = cv2.GaussianBlur(img,(5,5),0)
# plt.subplot(121)
# plt.imshow(img)
#
# plt.subplot(122)
# plt.imshow(img_bilater)
# plt.show()