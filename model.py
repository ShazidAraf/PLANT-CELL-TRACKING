import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import cmpnn as cmpnn
# import graph_sim.data as data
from graph_sim.utils import str2bool, to_cuda
import argparse
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class C3D(nn.Module):
    def __init__(self,ip_shape):

        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3)))
        self.group2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3)))

        self.group3 = nn.Sequential(
            nn.Conv3d(16,32, kernel_size=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3)))


        k = ip_shape//(3*3*3)+1

        self.fc1 = nn.Sequential(
            nn.Linear((32*k*k*k) , 128),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(64,1))

        self.sigmoid = nn.Sigmoid()
 

       

    def forward(self, x):

        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.sigmoid(2*out)

        return out




class Graph_Pair_Model(torch.nn.Module):

    def __init__(self, input_dim,n_points,n_edges, output_dim):
        super(Graph_Pair_Model, self).__init__()

        self.input_dim = input_dim
        self.n_points = n_points
        self.n_edges = n_edges
        self.output_dim = output_dim

        self.graph_feature = cmpnn.graph_matching.feature_network(with_residual=True, with_global=True,input_dim=self.input_dim,n_edges = self.n_edges, output_dim=self.output_dim)

        self.flatten_layer = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


        self.bn_1d = nn.BatchNorm1d(self.n_points)

        self.l1 = nn.Linear(self.n_points, 1)

        self.bn_1 = nn.BatchNorm1d(256)

        self.dist_func = torch.nn.PairwiseDistance()
        self.avg_pool = torch.nn.AvgPool1d(self.n_points)


    def forward(self, pt1, nn_idx1 , pt2, nn_idx2,mask):


        f1 = self.graph_feature(pt1.permute(0, 2, 1), nn_idx1, mask)
        f2 = self.graph_feature(pt2.permute(0, 2, 1), nn_idx2, mask)

        sim = torch.sum(torch.mul(f1,f2),dim=1)
        out1 = self.sigmoid(3*sim)
        x = self.l1(out1)
        out2 = self.sigmoid(x)

        return out1,out2



class Graph_Similarity_Model(torch.nn.Module):

    def __init__(self, input_dim,n_points,n_edges, output_dim):
        super(Graph_Similarity_Model, self).__init__()

        self.input_dim = input_dim
        self.n_points = n_points
        self.n_edges = n_edges
        self.output_dim = output_dim

        self.graph_feature = cmpnn.graph_matching.feature_network(with_residual=True, with_global=True,input_dim=self.input_dim,n_edges = self.n_edges, output_dim=self.output_dim)

        self.flatten_layer = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


        self.bn_1d = nn.BatchNorm1d(self.n_points)

        self.l1 = nn.Linear(self.n_points, 1)

        self.bn_1 = nn.BatchNorm1d(256)

        self.dist_func = torch.nn.PairwiseDistance()
        self.avg_pool = torch.nn.AvgPool1d(self.n_points)


    def forward(self, pt1, nn_idx1 , pt2, nn_idx2,mask):


        f1 = self.graph_feature(pt1.permute(0, 2, 1), nn_idx1, mask)
        f2 = self.graph_feature(pt2.permute(0, 2, 1), nn_idx2, mask)



        sim = torch.sum(torch.mul(f1,f2),dim=1)
        out1 = self.sigmoid(4*(sim - 0.5))
        x = self.l1(out1)
        out2 = self.sigmoid(x)

        return out1,out2


def conv_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.MaxPool2d(2),
        nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*11*11, 64),
        nn.Linear(64, 1),
        nn.Sigmoid()
        # nn.Softmax(dim=1)
        )

class deepseed(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(deepseed, self).__init__()
        
        self.in_dim = in_dim
        activation = nn.ReLU(inplace=True)
        self.cnn = conv_block_2d(in_dim,out_dim,activation)


    
    def forward(self, x):

        out = self.cnn(x)
        return out
