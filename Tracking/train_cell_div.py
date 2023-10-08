# python train_cell_div.py --epoch 200 --lr 1e-3 --batch_size 20
# python train_cell_div.py --pretrained checkpoint_cell_div/best_score_ckp/best_ckp.pt --epoch 200 --lr 1e-3 --batch_size 20
import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


import numpy as np
from numpy import random
import pickle, json
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py, scipy

import torch.nn as nn
import torch.nn.functional as F
from graph_sim.utils import str2bool, to_cuda


import matplotlib.pyplot as plt

from model import *
from data_loaders_training import *
from utils_tracking_2 import *



SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--ckp', type=str, default='checkpoint_cell_div')
    parser.add_argument('--v_size', type=int, default=100)
    parser.add_argument('--path', type=str, default= '/home/eegrad/mdislam/SAM/Graph_similarity_new/data/')

    parser.add_argument('--sched', type=str)
    args = parser.parse_args()

    return args

args = parse_args()
print(args)



def loss_val_curve(out,label):


    match = []
    not_match = []

    out = out.cpu().detach().numpy()

    for i in range(len(label)):
        if label[i]==1:
            match.append(out[i])

        elif label[i]==0:
            not_match.append(out[i])

    plt.figure().clear()
    plt.plot(np.array(match),np.arange(len(match)), '.', label='match',color='blue')
    plt.plot(np.array(not_match),np.arange(len(not_match)), '.', label='not match',color='red')

    plt.savefig("train_cell_div_graph.png")


    # print('point mean: ', np.mean(match),  np.mean(not_match))
    # print('point std: ', np.std(match),  np.std(not_match))



def train(epoch, train_loader, model, optimizer, scheduler=None):



    train_loader = tqdm(train_loader)
    criterion = torch.nn.BCELoss()

    model.train()

    for i, data_batch in enumerate(train_loader):

        model.zero_grad()
        optimizer.zero_grad()

        diff,label = to_cuda(data_batch,1)
        diff = torch.unsqueeze(diff,dim=1)
        label = label.view(-1)

        # print(diff.shape)

        out = model(diff)
        out = out.view(-1) 

        loss = criterion(out,label)

        loss_val_curve(out,label)

        loss.backward()
        

        if scheduler is not None:
            scheduler.step()
        optimizer.step()


        lr = optimizer.param_groups[0]['lr']

        train_loader.set_description(
            (
                f'epoch: {epoch}; Loss: {loss.item():.5f}; '
                f'lr: {lr:.5f}'
            )
        )

    
        
        
def evaluate(loader, model,th):

    model.eval()
    loader = tqdm(loader)

    match = []
    all_cells = []

    with torch.no_grad():

        for i, data_batch in enumerate(loader):

            diff,label = to_cuda(data_batch, 1)
            diff = torch.unsqueeze(diff,dim=1)

            label = label.view(-1)

            out = model(diff)

            out = out.view(-1)

            match.append(calc_match(out,label,th))

            all_cells.append(len(label))

    accuracy = np.sum(match)*100/np.sum(all_cells)


    return accuracy


def calc_match(pred,gt,th):

    pred = pred>th
    match = (torch.sum(pred==gt)).cpu().detach().numpy()

    return match

    
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # print('Initialize Conv2D')
        torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # print('Initialize Batch Normalization')
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # print('Initialize Batch Linear Layer')
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    
    
if __name__ == '__main__':



    
    MAIN_DIR = args.path

    model = C3D(ip_shape=args.v_size)
    model.apply(initialize_weights)
    model = model.cuda()
    


    if not args.pretrained == None:
        print('Loading pretrained weights...')
        model.load_state_dict(torch.load(args.pretrained))
        best_score = np.load('{0}/best_score_ckp/best_scores_val.npy'.format(args.ckp))
        print('Previous best score ', best_score)
    else:
        print('Training from scratch')
        best_score = 0.5

   
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001 )

    scheduler = None

    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    
    accuracy = []
       

    os.makedirs('{0}/best_score_ckp'.format(args.ckp), exist_ok=True)
    os.makedirs('{0}/all_saved_ckp'.format(args.ckp), exist_ok=True)

    for i in range(args.epoch):

        val_idx = [i%5]
        train_idx = list(set([0,1,2,3,4])-set(val_idx))
        train_data = Cell_Div_loader(MAIN_DIR,split = 'train',plant_idx=train_idx,v_size=args.v_size)
        val_data = Cell_Div_loader(MAIN_DIR,split = 'val',plant_idx=val_idx,v_size=args.v_size)


        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=2,shuffle=True)



        train(i, train_loader, model, optimizer, scheduler)
        torch.save(model.module.state_dict(), f'{args.ckp}/all_saved_ckp/ckp_epoch_{str(i + 1).zfill(3)}.pt')
 

        accuracy_val = evaluate(val_loader, model,th = 0.5)
        print('Validation Accuracy: ',accuracy_val)
        accuracy.append(accuracy_val)

        if accuracy_val>best_score:
            best_score = accuracy_val
            torch.save(model.module.state_dict(),'{0}/best_score_ckp/best_ckp.pt'.format(args.ckp))

            NAMES  = np.array(['Epoch No', 'Val Accuracy'])
            values = np.array([ i    ,accuracy_val   ])
            DAT =  np.column_stack((NAMES, values))
            np.savetxt('{0}/best_score_ckp/best_scores.txt'.format(args.ckp), DAT, delimiter=" ", fmt="%s")
            np.save('{0}/best_score_ckp/best_scores_val.npy'.format(args.ckp),accuracy_val)


        np.savetxt('{0}/accuracy.csv'.format(args.ckp),accuracy, delimiter = ",")


        index = [l for l in range(i+1)]

        plt.figure().clear()
        plt.plot(index,accuracy)
        plt.savefig('{0}/accuracy.png'.format(args.ckp))

