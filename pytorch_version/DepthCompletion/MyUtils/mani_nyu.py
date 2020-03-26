import os

nyu_dir = 'E:\Projects\Datasets\\Nyu_sub'

scene_lst = os.listdir(nyu_dir)

count = 0
for i in scene_lst:
    # if i.endswith('ini'):
    #     os.remove(os.path.join(nyu_dir, i))
    #     continue
    scene_name = os.path.join(nyu_dir, i)
    count += len(os.listdir(scene_name))

print(count)
