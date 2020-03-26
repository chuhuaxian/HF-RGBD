from Myloss import Sobel, Normal
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
sobel_ = Sobel(winsize=3).cuda()

Norm = Normal(shape=[1, 1, 480, 640]).cuda()
depth = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\0006-depth.png', -1)
plt.subplot(221), plt.imshow(depth)
# depth = cv2.imread('E:\Projects\Datasets\IRS\Home_ArchVizInterior03Data\\0-depth.png', -1)
depth = cv2.bilateralFilter(src=depth.astype(np.float32), d=0, sigmaColor=100, sigmaSpace=55)
depth = cv2.bilateralFilter(src=depth.astype(np.float32), d=0, sigmaColor=100, sigmaSpace=55)
depth = cv2.bilateralFilter(src=depth.astype(np.float32), d=0, sigmaColor=100, sigmaSpace=55)
plt.subplot(222), plt.imshow(depth)

depth = np.expand_dims(np.expand_dims(depth, 0), 0).astype(np.float32) /1000.


depth = torch.from_numpy(depth).float().cuda()
# depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))
with torch.no_grad():
    norm = Norm(depth)
    norm = norm.detach().cpu().numpy().squeeze()
    norm = (norm + 1.) /2.
    norm = np.clip(norm*255., 0, 255)
    norm = norm.astype(np.uint8)
    plt.subplot(223), plt.imshow(norm.transpose((1, 2, 0)))
    # edge = np.where(edge>0.15, edge, 0)
    # plt.subplot(223), plt.imshow(edge)
    plt.show()
    print()