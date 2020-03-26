import matplotlib
import matplotlib.cm
import numpy as np


def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def colorize(value, vmin=500, vmax=40000, cmap='jet'):
#     value = value.cpu().numpy()[0, :, :]
#
#     vmax = None
#     vmin = None
#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin != vmax:
#         value = (value - vmin) / (vmax - vmin)  # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value*0.
#     # squeeze last dim if it exists
#     #value = value.squeeze(axis=0)
#
#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value,bytes=True)  # (nxmx4)
#
#     img = value[:, :, :3]
#
#     return img.transpose((2, 0, 1))

import matplotlib.pyplot as plt
def colorize(input, ddc, ours, cmap='jet'):
    input = input.cpu().numpy()
    ddc = ddc.cpu().numpy().squeeze()
    ours = ours.cpu().numpy().squeeze()

    rgb = input[:, :3]*255.
    rgb = rgb.astype(np.uint8)
    raw = input[:, 3]

    out = []

    for i in range(rgb.shape[0]):
        c = np.transpose(rgb[i], (1, 2, 0))
        r = raw[i]
        d = ddc[i]
        o = ours[i]

        res = np.concatenate([r, d, o], axis=1)
        min_ = np.min(res)
        max_ = np.max(res)

        res = (res-min_)/(max_-min_)
        # plt.imshow(res.squeeze(), cmap='jet')
        # plt.show()

        cmapper = matplotlib.cm.get_cmap(cmap)
        res = cmapper(res,bytes=True)  # (nxmx4)
        res = np.concatenate([c, res[:, :, :3]], axis=1)
        out.append(res)
        # plt.imshow(res.squeeze())
        # plt.show()
    out = np.concatenate(out, axis=0)
    # plt.imshow(out)
    # plt.show()

    # value = value.cpu().numpy()[0, :, :]
    #
    # vmax = None
    # vmin = None
    # # normalize
    # vmin = value.min() if vmin is None else vmin
    # vmax = value.max() if vmax is None else vmax
    # if vmin != vmax:
    #     value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    # else:
    #     # Avoid 0-division
    #     value = value*0.
    # # squeeze last dim if it exists
    # #value = value.squeeze(axis=0)
    #
    # cmapper = matplotlib.cm.get_cmap(cmap)
    # value = cmapper(value,bytes=True)  # (nxmx4)
    #
    # img = value[:, :, :3]

    return out