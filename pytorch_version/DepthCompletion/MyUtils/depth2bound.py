from Myloss import Sobel
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
sobel_ = Sobel(winsize=3).cuda()

depth = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\1236-depth.png', -1)
# depth = cv2.imread('E:\Projects\Datasets\IRS\Home_ArchVizInterior03Data\\0-depth.png', -1)
blur_depth = cv2.GaussianBlur(depth, (3, 3), 0)
plt.subplot(224), plt.imshow(blur_depth)
plt.subplot(221), plt.imshow(depth)
depth = np.expand_dims(np.expand_dims(depth, 0), 0).astype(np.float32) /1000.


depth = torch.from_numpy(depth).float().cuda()
# depth = torch.autograd.Variable(_label.float().cuda(non_blocking=True))
with torch.no_grad():
    edge = sobel_(depth)
    edge = edge.detach().cpu().numpy().squeeze()

    plt.subplot(222), plt.imshow(edge)
    edge = np.where(edge>0.15, edge, 0)
    plt.subplot(223), plt.imshow(edge)
    plt.show()
    print()