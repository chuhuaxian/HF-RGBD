import os
import shutil

def generate_test_list():
    test_dir = 'E:\Projects\Datasets\\NyuV2Dataset'
    res_dir = 'E:\Projects\Datasets\\realsense'
    file_lst = os.listdir(test_dir)

    raw_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'raw' in i]
    depth_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'depth' in i]
    color_lst = ['%s/%s' % (test_dir, i) for i in file_lst if 'color' in i]

    f = open('realsense_list.txt', mode='w')
    for idx in range(len(raw_lst)):
        shutil.copy(color_lst[idx], '%s/%s.jpg' % (res_dir, idx))
        shutil.copy(raw_lst[idx], '%s/%s.png' % (res_dir, idx))

        # out = '%s,%s,%s\n' % (color_lst[idx], depth_lst[idx], raw_lst[idx])
        f.write('%s\n'%idx)
            # print(out)
    f.close()
generate_test_list()