import os
import cv2
import matplotlib.pyplot as plt
base_dir = 'E:\\Projects\\Datasets\\ScanNet\\scene0002_01\\color'

c_lst = os.listdir(base_dir)

count = 0

for i in range(len(c_lst)):

    c_name = os.path.join(base_dir, '%s.png' % i)

    # if count % 10 == 0:
        # print(count)
    color = cv2.imread(c_name, -1)
    image_size = [480, 640]
    # color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    res_name = 'E:\\Projects\\Datasets\\ScanNet\\scene0002_01\\color1\\%s.jpg' % (count*10)
    print(res_name)
    cv2.imwrite(res_name, color)

    count += 1

    # print()
    # print(color.shape)
    # cv2.imshow('', color)
    # cv2.waitKey()
    # print(c_name)