# python inference_on_stack.py --cell_div_th 0.5

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
from itertools import combinations

from model import *
from data_loaders_inference import *
from utils_tracking_2 import *




SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ckp', type=str, default='checkpoints')
    parser.add_argument('--path', type=str, default= '/home/eegrad/mdislam/SAM/Graph_similarity_new/data/')
    parser.add_argument('--n_points', type=int, default= 15)
    parser.add_argument('--k', type=int, default= 6)
    parser.add_argument('--n_cand', type=int, default= 20)
    parser.add_argument('--max_dist', type=int, default= 15)
    parser.add_argument('--v_size', type=int, default= 100)
    parser.add_argument('--cell_div_th', type=float, default= 0.7)
    parser.add_argument('--sched', type=str)
    args = parser.parse_args()

    return args

args = parse_args()
print(args)

r_th = args.cell_div_th
print(r_th)

def find_point_knn(c,cen):

    a = np.linalg.norm(cen-c,axis=1)
    knn_idx = np.argsort(a)


    return knn_idx

def get_multi_point_info(p_n):

    main_dir = 'data/{0}'.format(p_n)

    text_dir = os.path.join(main_dir,'tracking_data')
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])

    T = np.load('data/transformations/{0}_all_transformations.npy'.format(p_n))
    all_corr_labels = []

    for i in range(T.shape[0]):
        text_path = os.path.join(text_dir,text_files[i])
        prop_t0,prop_t1,mother_gt,daughter_gt = read_PNAS_gt(text_path)
        prop_t1 = prop_t1[:,0]
        all_corr_labels.append(np.array([prop_t0,prop_t1]).T)

    all_corr_labels = np.array(all_corr_labels)

    all_tracks = combine_tracks(all_corr_labels,start=0,finish=all_corr_labels.shape[0])


    return all_tracks,T






def tracker(loader):

    
    loader = tqdm(loader)
    corr_label_1 = []
    corr_label_2 = []
    score = []

    
    with torch.no_grad():

        for i, data_batch in enumerate(loader):

            PT0,KNN_PT0,PT1,KNN_PT1,MASK, label,cand = to_cuda(data_batch,1)
            
            label = label.cpu().detach().numpy()
            cand = cand.cpu().detach().numpy()

            p = torch.zeros(label.shape[0],args.n_cand+1).cuda()


            for j in range(args.n_cand):

                pt0 = PT0[:,j,:,:]
                knn_pt0 = KNN_PT0[:,j,:,:]
                pt1 = PT1[:,j,:,:]
                knn_pt1 = KNN_PT1[:,j,:,:]
                mask = MASK[:,j,:]

                out1,out2 = model_track(pt0, knn_pt0, pt1, knn_pt1,mask)

                p[:,j] = out2.view(-1)

            non_p = 1 - torch.max(p[:,0:-1],dim=1)[0]
            p[:,-1] = non_p

            pred_idx = torch.argmax(p,dim = 1)
            pred_score = torch.max(p,dim = 1)[0]

            pred_idx = pred_idx.cpu().detach().numpy()
            pred_score = pred_score.cpu().detach().numpy()


            # print(label.shape,cand.shape,pred_idx.shape)

            for k in range(label.shape[0]):
                
                if pred_idx[k]==args.n_cand:
                    continue

                corr_label_1.append(label[k][0])
                corr_label_2.append(cand[k,pred_idx[k]])
                score.append(pred_score[k])


    corr_label_1 = np.array(corr_label_1)
    corr_label_2 = np.array(corr_label_2)
    score = np.array(score)


    tr1 = []
    tr2 = []
    s = []
    for i in range(len(corr_label_2)):

        

        if corr_label_2[i] in tr2:
            continue

        loc = np.where(corr_label_2==corr_label_2[i])[0]

        if score[loc[0]]>=0.80:

            if len(loc)==1:
                tr1.append(corr_label_1[loc[0]])
                tr2.append(corr_label_2[loc[0]])
                s.append(score[loc[0]])

            else:
                d = score[loc]
                ind = np.argmax(d)
                tr1.append(corr_label_1[loc[ind]])
                tr2.append(corr_label_2[loc[ind]])
                s.append(score[loc[ind]])



    tr1 = np.array(tr1)
    tr2 = np.array(tr2)
    s = np.array(s)

    sorted_idx = np.flip(np.argsort(np.array(s)))

    corr_label_1 = np.array([ tr1[ff] for ff in sorted_idx])
    corr_label_2 = np.array([ tr2[ff] for ff in sorted_idx])
    all_scores = np.array([ s[ff] for ff in sorted_idx])



    return corr_label_1,corr_label_2,all_scores


def evaluate_track(tr1,tr2,plant_data):
    match = 0
    found = 0
    for i in range(len(plant_data.gt_l0)):
        idx1 = np.where(tr1==plant_data.gt_l0[i])[0]

        if len(idx1)>0:
            found+=1

            l = idx1[0]
            
            if tr2[l]== plant_data.gt_l1[i]:
                match+=1

    
    # match/(len(plant_data.gt_l0)+1e-20)
    return match,found, len(plant_data.gt_l0)

def evaluate_division(mother,daughter,cell_division_data):
    match = 0
    found = 0

    for i in range(len(cell_division_data.gt_mother)):

        idx1 = np.where(mother==cell_division_data.gt_mother[i])[0]

        if len(idx1)>0:
            found+=1

            l = idx1[0]
            d_out = np.sort(daughter[l])
            d_gt = np.sort(cell_division_data.gt_daughter[i])
            if d_out[0]== d_gt[0] and d_out[1]== d_gt[1]:
                match+=1

    
    # match/(len(plant_data.gt_l0)+1e-20)
    return match,found, len(cell_division_data.gt_mother)

def track_rest(plant_data,corr_label_0,corr_label_1,track_scores):

    corr_label_0 = corr_label_0.tolist() 
    corr_label_1 = corr_label_1.tolist()
    track_scores = track_scores.tolist()

    label0 = plant_data.label_0
    label1 = plant_data.label_1
    cen_0 = plant_data.cen_0
    cen_1 = plant_data.cen_1
    # neighbour0 = plant_data.neighbour0
    # neighbour1 = plant_data.neighbour1
    neighbour0 = get_neighbouring_structure_knn(cen_0,150)
    neighbour1 = get_neighbouring_structure_knn(cen_1,150)

    aa = 0
    while aa<len(corr_label_1):


        idx0 = np.where(label0==corr_label_0[aa])[0][0]
        idx1 = np.where(label1==corr_label_1[aa])[0][0]

        # c0 = cen_0[idx0]
        # c1 = cen_1[idx1]

        nbr0 = copy.deepcopy(neighbour0[idx0])
        nbr1 = copy.deepcopy(neighbour1[idx1])

        cord_c0 = create_graph_knn(cen_0,nbr0,T =np.eye(4))
        cord_c1 = create_graph_knn(cen_1,nbr1,T =np.eye(4))

        # cord_c0 = cord_c0 - c0
        # cord_c1 = cord_c1 - c1

        c00,c11,corr0,corr1,_ = distance_matrix(cord_c0,cord_c1,8)


            
        for z in range(len(corr0)):

            x0 = label0[nbr0[corr0[z]]]
            x1 = label1[nbr1[corr1[z]]]

            if (x0 not in corr_label_0) and (x1 not in corr_label_1):
                corr_label_0.append(x0)
                corr_label_1.append(x1)
                # track_scores.append(track_scores[aa])
                track_scores.append(0.5)

        aa+=1

    corr_label_0 = np.array(corr_label_0)
    corr_label_1 = np.array(corr_label_1)
    track_scores = np.array(track_scores)

    
    return corr_label_0, corr_label_1, track_scores




def voxelization(pc,n):
   
    c = np.mean(pc,axis=0)
    pc0 = pc - c+n//2
    pc1 = np.round_(pc0)
    pc2 = np.clip(pc1,0,n-1).astype(int)

    box = np.zeros((n,n,n))
    box[pc2[:,0],pc2[:,1],pc2[:,2]] = 1
   
    return box


def get_nbr_score(data_batch):



    data_batch = [torch.unsqueeze(torch.from_numpy(temp),dim=0) for temp in data_batch]

    with torch.no_grad():
        pt0,knn_pt0,pt1,knn_pt1,mask,_ = to_cuda(data_batch,1)
        _,nbr_score = model_pair(pt0, knn_pt0, pt1, knn_pt1,mask)

        nbr_score = nbr_score.cpu().detach().numpy()
        nbr_score = nbr_score[0][0]

    return nbr_score



def cell_division_detection(cell_div_data):

    mother = []
    daughter = []
    div_scores = []
    all_d = []

    n = cell_div_data.__len__()

    for i in tqdm(range(n)):

        l0,daughter_cand = cell_div_data.__getitem__(i)

        loc0 = np.array(np.where(seg0==l0)).T
        tt = points_to_pcd(loc0).transform(cell_div_data.T)
        pc0 = np.array(tt.points)
        box0 = np.max(pc0,axis=0) - np.min(pc0,axis=0)
        vox_0 = voxelization(pc0,args.v_size)

        div_score = []


        for j in range(daughter_cand.shape[0]):

            l1 = daughter_cand[j,0]
            l2 = daughter_cand[j,1]

            loc1 = np.array(np.where(seg1==l1)).T
            loc2 = np.array(np.where(seg1==l2)).T

            pc1 = np.concatenate((loc1,loc2),axis=0)
            box1 = np.max(pc1,axis=0) - np.min(pc1,axis=0)
            delta = np.divide(np.abs(box1-box0),box0)
            th = np.max(delta)

            if th>0.5:
                continue

            vox_1 = voxelization(pc1,args.v_size)
            diff = vox_0-vox_1
            diff = diff.astype(np.float32)
            diff = torch.from_numpy(diff ) 
            diff = torch.unsqueeze(diff,dim=0)
            diff = torch.unsqueeze(diff,dim=0)
            diff = diff.cuda()
            div_probability = model_div(diff)
            div_probability = div_probability.cpu().detach().numpy()


            data_batch = graph_pair_data .__getitem__(cell_div_data.t1,cell_div_data.t2,int(l0),int(l1))
            nbr_score_1 = get_nbr_score(data_batch)

            data_batch = graph_pair_data .__getitem__(cell_div_data.t1,cell_div_data.t2,int(l0),int(l2))
            nbr_score_2 = get_nbr_score(data_batch)

            w = 0.5
            out = w*div_probability+(1-w)*(nbr_score_1+nbr_score_2)/2

            div_score.append(out)

        if len(div_score)==0:
            continue
        div_score = np.array(div_score)
        s = np.max(div_score)
        # print(s)

        if s>r_th:
            idx = np.argmax(div_score)
            mother.append(l0[0])
            daughter.append(daughter_cand[idx])
            div_scores.append(s)
            all_d.append(daughter_cand[idx,0])
            all_d.append(daughter_cand[idx,1])

            # print(mother, daughter,div_scores)
    
    mother = np.array(mother)
    mother = np.resize(mother,(len(mother),1))
    daughter = np.array(daughter)
    div_scores = np.array(div_scores)
    div_scores = np.resize(div_scores,(len(mother),1))

    # print(mother.shape,daughter.shape,div_scores.shape)


    all_div_info = np.concatenate((mother,daughter,div_scores),axis=1)
    print(all_div_info.shape)

    print('mother',len(mother))
    return all_div_info,all_d





    
MAIN_DIR = args.path


seg_dir = '/home/eegrad/mdislam/SAM/Graph_similarity_new/data/plant_18/seg'
seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])

model_track = Graph_Similarity_Model(input_dim=3,n_points=args.n_points,n_edges = 16, output_dim = 512)
path = 'checkpoints/graph_multi_ckp.pt'
model_track.load_state_dict(torch.load(path))
model_track = nn.DataParallel(model_track).cuda()
model_track.eval()

model_pair = Graph_Pair_Model(input_dim=3,n_points=args.n_points,n_edges = 16, output_dim = 512)
path = 'checkpoints/graph_pair_ckp.pt'
model_pair.load_state_dict(torch.load(path))
model_pair = nn.DataParallel(model_pair).cuda()
model_pair.eval()


model_div = C3D(ip_shape=args.v_size)
path = 'checkpoints/cell_div_ckp.pt'
model_div.load_state_dict(torch.load(path))
model_div = nn.DataParallel(model_div).cuda()
model_div.eval()



    
accuracy = []
   
plant_idx = 5

n = 21

accuracy1 = []
accuracy2 = []
match_number = []
gt_number = []

d = 1

all_matches = 0
all_found = 0
all_gt = 0

all_matches_div = 0
all_found_div = 0
all_gt_div = 0


seg_dir = '/home/eegrad/mdislam/SAM/Learning_based_tracking/data/plant_18/seg'
seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])


start = 0

for t1 in tqdm(range(start,n-d)):

    seg0 = tifffile.imread(os.path.join(seg_dir,seg_files[t1]))
    seg1 = tifffile.imread(os.path.join(seg_dir,seg_files[t1+d]))

    cell_div_data = mother_daughter_pair_loader(MAIN_DIR,plant_idx,t1,t1+d,n_cand = args.n_cand)
    graph_pair_data = Graph_pair_loader(MAIN_DIR,plant_idx= plant_idx, n_points=args.n_points-1,k=args.k,max_dist=args.max_dist)

    plant_data = Plant_data_loader(MAIN_DIR,plant_idx= plant_idx,t1 = t1,t2 = t1+d, n_points=args.n_points-1,k=args.k,n_cand = args.n_cand,max_dist=args.max_dist)
    plant_loader = DataLoader(plant_data, batch_size=args.batch_size, num_workers=2,shuffle=True)

    n1 = len(plant_data.gt_l0)
    n2 = plant_data.__len__()

    all_div_info, _ = cell_division_detection(cell_div_data)
    # pdb.set_trace()
    # all_div_info = np.load('/home/eegrad/mdislam/SAM/Learning_based_tracking/final_results_May28,2023/results/graph_3D/all_divisions_stack_{0}.npy'.format(t1))
    mother = all_div_info[:,0]
    daughter = all_div_info[:,1:3]
    all_d = (copy.deepcopy(daughter)).flatten()
    # mother = []
    # daughter = []
    # all_d = []
    # div_scores = []


    corr_label_0,corr_label_1,track_scores = tracker(plant_loader)
    print('Primary: ',len(corr_label_0))
    corr_label_0,corr_label_1,track_scores = track_rest(plant_data,corr_label_0,corr_label_1,track_scores)
    print('Secondary: ',len(corr_label_0))
    

    tr_0 = []
    tr_1 = []
    tr_score = []

    for kk in range(len(corr_label_0)):

        if (corr_label_0[kk] in mother) or (corr_label_1[kk] in all_d):
            continue

        tr_0.append(corr_label_0[kk])
        tr_1.append(corr_label_1[kk])
        tr_score.append(track_scores[kk])




    tr_0  = np.array(tr_0)
    tr_0 = np.resize(tr_0,(len(tr_0),1))
    
    tr_1  = np.array(tr_1)
    tr_1 = np.resize(tr_1,(len(tr_1),1))

    tr_score  = np.array(tr_score)
    tr_score = np.resize(tr_score,(len(tr_score),1))

    all_tracks = np.concatenate((tr_0,tr_1,tr_score),axis=1)
    print('Final: ',len(tr_0))

    # all_tracks = []
    # all_divisions = []
    # all_tracks.append(np.array([tr_0,tr_1,tr_score]))
    # all_divisions.append([mother,daughter,div_scores])

    np.save('result3/all_tracks_stack_{0}.npy'.format(t1),all_tracks)
    np.save('result3/all_divisions_stack_{0}.npy'.format(t1),all_div_info)


    match_div,found_div,gt_div = evaluate_division(mother,daughter,cell_div_data)
    match,found,gt = evaluate_track(tr_0,tr_1,plant_data)

    print(match_div,found_div,gt_div,match_div/(1e-30+found_div),match_div/(1e-30+gt_div))
    print(match,found,gt,match/found,match/gt)

    all_matches_div+=match_div
    all_found_div+=found_div
    all_gt_div+=gt_div

    all_matches+=match
    all_found+=found
    all_gt+=gt

acc1 = all_matches*100/(all_gt+1e-30)
acc2 = all_matches*100/(all_found+1e-30)

acc1_div = all_matches_div*100/(all_gt_div+1e-30)
acc2_div = all_matches_div*100/(all_found_div+1e-30)

print('time gap {0}: {1}'.format(d,acc1))
print('time gap {0}: {1}'.format(d,acc2))


print('div time gap {0}: {1}'.format(d,acc1_div))
print('div time gap {0}: {1}'.format(d,acc2_div))


    # match_number.append(all_matches)
    # gt_number.append(all_gt)

    # accuracy1.append(acc1)
    # accuracy2.append(acc2)


    # np.save('accuracy1.npy',accuracy1)
    # np.save('accuracy2.npy',accuracy2)

    # np.save('match_number.npy',match_number)
    # np.save('gt_number.npy',gt_number)
