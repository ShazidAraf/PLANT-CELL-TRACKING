import cmpnn
import data
from utils import str2bool, to_cuda
import argparse
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import os
import pdb

batch_size=5
dataset = data.RandomGraphDataset(10,15, 0,5)
pt1, knn_pt1 , pt2, knn_pt2, mask, gt = dataset.__getitem__(10)

dataloader = DataLoader(dataset,batch_size,shuffle=True,num_workers=1)
print(len(dataloader))

model = cmpnn.graph_matching.feature_network(with_residual=True, with_global=True,input_dim=3)


for ibatch, data_batch in tqdm(enumerate(dataloader)):
	pdb.set_trace()
	pt1, nn_idx1, pt2, nn_idx2, mask, gt = to_cuda(data_batch, 0)
	pt1 = torch.rand(batch_size,pt1.shape[1],3)
	pt2 = torch.rand(batch_size,pt2.shape[1],3)
	
	print(pt1.shape)
	print(nn_idx1.shape)
	print(pt2.shape)
	print(nn_idx2.shape)
	print(mask.shape)
	print(gt.shape)

	feature = model(pt1.permute(0, 2, 1), nn_idx1, mask)
	print('feature',feature.shape)

print('End')
