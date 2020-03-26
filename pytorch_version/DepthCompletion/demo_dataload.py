import os
import torch
from dataloader import TripleDataset, NyuV2Dataset
from torch.utils.data import DataLoader
from config import config

nyu_list = open('nyu_list.txt').readlines()
scannet_list = open('scannet_list.txt').readlines()
real_list = nyu_list+scannet_list

irs_list = open('irs_list.txt').readlines()
mask_list = open('mask_list.txt').readlines()

test_list = open('test_list.txt').readlines()

train_dataset = TripleDataset(real_list, irs_list, mask_list, None)
test_dataset = NyuV2Dataset(test_list)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs,  shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4,  shuffle=True)

for syn_input, real_input, syn_label, real_label in train_loader:
    _input = torch.cat([syn_input, real_input], dim=0)
    _label = torch.cat([syn_label, real_label], dim=0)
#
#     print()


for _input, _label in test_loader:

    print()


print()

