import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _get_sobel_kernel(winsize=3) -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    sobel_operator = {3: np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                      5: -1 * np.array(
                          [[1, 2, 0, -2, -1], [4, 8, 0, -8, -4], [6, 12, 0, -12, -6], [4, 8, 0, -8, -4],
                           [1, 2, 0, -2, -1]], dtype=np.float32)}
    if winsize == 3:
        return torch.tensor(sobel_operator[3], requires_grad=False)
    elif winsize == 5:
        return torch.tensor(sobel_operator[5], requires_grad=False)


class SpatialGradient(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`
    """

    def __init__(self, winsize=3) -> None:
        super(SpatialGradient, self).__init__()
        self.kernel: torch.Tensor = self.get_sobel_kernel(winsize=winsize)

    @staticmethod
    def get_sobel_kernel(winsize=3) -> torch.Tensor:
        kernel_x: torch.Tensor = _get_sobel_kernel(winsize=winsize)
        kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
        return torch.stack([kernel_x, kernel_y])

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with sobel kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        return F.conv3d(input[:, :, None], kernel_flip, padding=1, groups=c)


class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    """

    def __init__(self, winsize=3) -> None:
        super(Sobel, self).__init__()
        self.winsize=winsize

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges: torch.Tensor = spatial_gradient(input, winsize=self.winsize)

        # unpack the edges
        gx: torch.Tensor = edges[:, :, 0]
        gy: torch.Tensor = edges[:, :, 1]

        # compute gradient maginitude
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy)


        return magnitude



# functiona api

def spatial_gradient(input: torch.Tensor, winsize=3) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return SpatialGradient(winsize=winsize)(input)


def sobel(input: torch.Tensor, winsize=3) -> torch.Tensor:
    r"""Computes the Sobel operator and returns the magnitude per channel.

    See :class:`~kornia.filters.Sobel` for details.
    """
    return Sobel(winsize=winsize)(input)

# import cv2
#
# color = np.expand_dims(cv2.imread('E:\Projects\Datasets\\NYU_V2\\data\\nyu2_train\\basement_0001a_out\\1.png', flags=-1), axis=-1)
# color = np.expand_dims(np.transpose(color ,axes=(2, 0, 1)), axis=0).astype(np.float32)
# color = torch.from_numpy(color)
#
# res = sobel(color, winsize=3).detach().numpy()
# res = np.squeeze(res)
# # res = np.where(res>0, 1, 0)
# import matplotlib.pyplot as plt
# plt.imshow(res, cmap='gray')
# plt.show()
# print(res.shape)
# # print(color.shape)