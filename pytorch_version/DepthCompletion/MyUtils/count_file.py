import os

BASE_DIR = 'E:\Projects\Datasets\IRS'

scene_lst = [os.path.join(BASE_DIR, i) for i in os.listdir(BASE_DIR) if not i.endswith('ini')]

count = 0

for i in scene_lst:
    count += len(os.listdir(i))

print(count)