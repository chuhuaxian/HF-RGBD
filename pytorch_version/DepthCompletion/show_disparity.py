import pyexr
import matplotlib.pyplot as plt
import cv2
import numpy
# i = 1
for i in range(952, 1000, 10):
    # d = pyexr.open('F:\Datasets\IRS\Restaurant\DinerEnvironment\\d_%s.exr' % i).get().squeeze()
    # n = pyexr.open('F:\Datasets\IRS\Restaurant\DinerEnvironment\\n_%s.exr' % i).get()
    l = cv2.imread('F:\Datasets\IRS\Restaurant\DinerEnvironment\\l_%s.png'% i)[:, :, (2,1, 0)]

    print()
    # print()
    # plt.subplot(132), plt.imshow(d)
    # plt.subplot(131), plt.imshow(l)
    # plt.subplot(133), plt.imshow(n)
    # plt.show()