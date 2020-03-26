import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pyexr
import OpenEXR
import numpy
import Imath


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
    	Channels = ['R', 'G', 'B']
    	CNum = 3
    else:
    	Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in Channels]
    hdr = numpy.zeros((Size[1],Size[0],CNum),dtype=numpy.float32)
    if (CNum == 1):
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
    else:
	    hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
	    hdr[:,:,1] = numpy.reshape(Pixels[1],(Size[1],Size[0]))
	    hdr[:,:,2] = numpy.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr


RGBD_DIR = 'E:\Projects\Datasets\RGBD'
IRS_DIR = 'F:\Datasets\IRS\Office\ModernModularOffice'
focal_length = 480
baseline = 0.1
depth = cv2.imread(os.path.join(RGBD_DIR, '0.png'), -1)


# disp = pyexr.open(os.path.join(IRS_DIR, 'n_2.exr')).get()

disp = exr2hdr(os.path.join(IRS_DIR, 'd_366.exr'))
disp = baseline * focal_length / disp
disp = np.where(disp>60, 20, disp)
print(np.max(disp))
print(np.min(disp))

# disp = disp*2-1
#
# nx, ny, nz = np.split(disp, 3, axis=2)
# ny = -ny
# nz = -nz
# # nx = -nx
# normal = np.concatenate([nx, ny, nz], axis=-1)
# normal = (normal+1) / 2
# normal = normal[:, :, (0,1,2)]
plt.imshow(disp.squeeze())
plt.show()