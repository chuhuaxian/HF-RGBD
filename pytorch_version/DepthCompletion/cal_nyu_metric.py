import os
import numpy as np
f = open('nyu_rand_remove_eval_2-09_90.txt').readlines()

# xx = [i.strip() for i in f if 'ddc' in i]
# print(len(xx))
# idx = [i for i in range(len(xx)) if 'inf' not in xx[i]]
# print(len(idx))

ours = [[float(j) for j in i.strip().split(',')[1:]] for i in f if 'ou' in i]
ddc = [[float(j) for j in i.strip().split(',')[1:]] for i in f if 'ddc' in i]

# ours = ours[idx]
# ddc = ddc[idx]

print(len(ours), len(ddc))

# our_30 = np.mean(np.array(ours[::3]), axis=0)
# ddc_30 = np.mean(np.array(ddc[::3]), axis=0)

our_50 = np.array(ours)
ddc_50 = np.array(ddc)
idx = np.mean(ddc_50, axis=1) !=np.inf
print(idx.shape)
our_50 = np.mean(our_50[idx], axis=0)
ddc_50 = np.mean(ddc_50[idx], axis=0)
#
# our_70 = np.array(ours[2::3])
# ddc_70 = np.array(ddc[2::3])
# idx = np.mean(ddc_70, axis=1) !=np.inf
# print(idx.shape)
# our_70 = np.mean(our_70[idx], axis=0)
# ddc_70 = np.mean(ddc_70[idx], axis=0)


# print('our30:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (our_30[0],np.sqrt(our_30[1]),our_30[2],our_30[3],our_30[4],our_30[5]))
#
# print('ddc30:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (ddc_30[0],np.sqrt(ddc_30[1]),ddc_30[2],ddc_30[3],ddc_30[4],ddc_30[5]))
# print('\n')
print('our50:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (our_50[0],np.sqrt(our_50[1]),our_50[2],our_50[3],our_50[4],our_50[5]))
print('ddc50:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (ddc_50[0],np.sqrt(ddc_50[1]),ddc_50[2],ddc_50[3],ddc_50[4],ddc_50[5]))
# print('\n')
# print('our70:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (our_70[0],np.sqrt(our_70[1]),our_70[2],our_70[3],our_70[4],our_70[5]))
# print('ddc70:', '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (ddc_70[0],np.sqrt(ddc_70[1]),ddc_70[2],ddc_70[3],ddc_70[4],ddc_70[5]))

# print('ddc30:', np.mean(ddc_30, axis=0))
# print('our50:', np.mean(our_50, axis=0))
# print('ddc50:', np.mean(ddc_50, axis=0))
# print('our70:', np.mean(our_70, axis=0))
# print('ddc70:', np.mean(ddc_70, axis=0))
print()