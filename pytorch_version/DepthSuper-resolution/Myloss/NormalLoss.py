import numpy as np
import torch
import torch.nn as nn


class Gradient(nn.Module):
    def __init__(self, win_size=5):
        super(Gradient, self).__init__()
        sobel_operator = {3: np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                          5: -1 * np.array(
                              [[1, 2, 0, -2, -1], [4, 8, 0, -8, -4], [6, 12, 0, -12, -6], [4, 8, 0, -8, -4],
                               [1, 2, 0, -2, -1]], dtype=np.float32)}
        self.conv_x = nn.Conv2d(1, 1, kernel_size=win_size, stride=1, padding=win_size//2)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=win_size, stride=1, padding=win_size//2)
        conv_x = sobel_operator[win_size]
        conv_y = conv_x.transpose()
        self.conv_x.weight = nn.Parameter(torch.from_numpy(conv_x).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(conv_y).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def getGrd(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        return -grd_x, -grd_y


class Normal(torch.nn.Module):
    def __init__(self, win_size=5, shape=[]):

        super(Normal, self).__init__()
        self.gradient = Gradient()
        self.shape = shape

    def forward(self, X):
        gx, gy = self.gradient.getGrd(X)
        normal = torch.cat([gx, gy, torch.ones(size=self.shape, dtype=torch.float32).cuda()], dim=1)
        norm = torch.sqrt(torch.sum(torch.pow(normal, 2.0), dim=1, keepdim=True))
        normal = normal / norm
        # normal = (normal + 1) / 2 * 255
        return normal










