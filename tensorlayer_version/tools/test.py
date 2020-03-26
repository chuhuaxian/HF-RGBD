# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
# from tqdm import tqdm
#
#
# print('')
# def gray2pseudo(gray):
#     """
#     灰度图转换成伪彩色图
#     :param gray: 灰度图
#     :return: 伪彩色图
#     """
#     gray = gray.astype(np.float32)
#     gray = ((gray-np.min(gray)) / (np.max(gray)-np.min(gray)))
#     sc = np.ones(shape=[gray.shape[0], gray.shape[1], 1], dtype=np.float32) * 1.
#     vc = np.ones(shape=[gray.shape[0], gray.shape[1], 1], dtype=np.float32) * 1.
#     # gray = gray*60.
#
#     for h in range(gray.shape[0]):
#         for w in range(gray.shape[1]):
#             v = gray[h, w]
#             if v <= 0.55:
#                 v = v * 110.
#                 vc[h, w] = vc[h, w]*0.2+0.7
#                 # v = 0
#             # elif 1./3. < v <= 2./3.:
#             #     # v = v*180. + 60.
#             #     v = 0
#             else:
#                 v = (v-0.5)*220.+150.
#             gray[h, w] = v
#     gray = np.expand_dims(gray, -1)
#     gray = np.concatenate([gray, sc, vc], axis=-1)
#     return cv2.cvtColor(gray, cv2.COLOR_HSV2BGR)
#
#
# def normalize(img):
#     img = img.astype(np.float32)
#     return (img-np.min(img))/(np.max(img)-np.min(img))
#
#
# def getDiffMap(raw, depth):
#     mask = raw == 0
#     mask = mask.astype(np.float32)
#
#     rawMax = np.max(raw)
#     raw = raw.astype(np.float32) + mask * rawMax
#
#     diff = np.abs(normalize(raw)-normalize(depth))
#     diff = diff*np.abs(mask-1)
#     return diff
#
#
# def validFill(img):
#     mask = img <=500
#     mask = mask.astype(np.float32)
#     rawMax = np.max(img)
#     result = img.astype(np.float32) + mask * rawMax
#     return result
#
#
# def validRemove(img, mask_):
#     mask = np.expand_dims(mask_, -1)
#     mask = np.concatenate([mask, mask, mask], -1)
#     mask_r = np.abs(mask - 1)
#     result = img * mask_r
#     return result
#
#
# def turnPseduo(depthPath):
#     raw = cv2.imread(depthPath, -1)
#     mask = raw <=500
#     raw = validFill(raw)
#     raw = gray2pseudo(raw)
#     raw = validRemove(raw, mask)
#     return raw
#
# order = '09205'
# # order = '11110'
# step = '501'
# path = 'C:\zdj\project\python\RGBD\dataset\RGBD-SCENCES-V2\\raw\\%s-depth.png' % order
# ourpath = 'C:\zdj\project\python\RGBD\SR\\result\RGBD_SR\\Test_step_901-09589-result.png'
# srganpath = 'C:\zdj\project\python\RGBD\SR\\result\SRGAN_g\\Test-%s-srgan.png' % order
# rdbspath = 'C:\zdj\project\python\RGBD\SR\\result\RDBs_Network\\Test-%s-rdb.png' % order
#
# depth = cv2.imread(path, -1)
# our = cv2.imread(ourpath, -1)
# srgan = cv2.imread(srganpath, -1)
# rdb = cv2.imread(rdbspath, -1)
#
# # depth = turnPseduo(path)
# # our = turnPseduo(ourpath)
# # srgan = turnPseduo(srganpath)
# # rdb = turnPseduo(rdbspath)
#
# depth_hr = cv2.resize(depth, dsize=(depth.shape[1]*4, depth.shape[0]*4), interpolation=cv2.INTER_NEAREST)
# plt.subplot(231), plt.imshow(depth[41:-41, 18:-38], cmap='gray'), plt.axis('off')
# plt.subplot(232), plt.imshow(depth_hr[41*4:-41*4, 18*4:-38*4], cmap='gray'), plt.axis('off')
# # plt.subplot(236), plt.imshow(our[41*4:-41*4, 18*4:-38*4], cmap='gray'), plt.axis('off')
# plt.subplot(234), plt.imshow(srgan[41*4:-41*4, 18*4:-38*4], cmap='gray'), plt.axis('off')
# # plt.subplot(235), plt.imshow(rdb[41*4:-41*4, 18*4:-38*4], cmap='gray'), plt.axis('off')
# plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.02, wspace=0.05)
# plt.show()

from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run --multi_gpu=False.')

print(num_gpus)

