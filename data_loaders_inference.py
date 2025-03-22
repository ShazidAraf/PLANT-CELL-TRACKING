import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import cvxpy
import open3d as o3d
import pickle
import json

from utils_tracking_2 import *

import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# plant_name = ['plant_1','plant_2','plant_4','plant_13','plant_15','plant_18']

def find_point_knn(c,cen):

    a = np.linalg.norm(cen-c,axis=1)
    knn_idx = np.argsort(a)


    return knn_idx

class Plant_data_loader(Dataset):
    
    def __init__(self,main_dir,plant_name,plant_idx,t1,t2,Reg_Mtx,n_points,k,n_cand,max_dist,segmentation):

        self.plant_name = plant_name
        self.main_dir = main_dir
        self.plant_idx = plant_idx
        self.t1 = t1
        self.t2 = t2
        self.segmentation = segmentation

        self.n_points = n_points
        self.k = k
        self.n_cand = n_cand
        self.max_dist = max_dist

        self.all_plant_pairs = []
        self.all_plant_pairs_len = []

        # Directory 

        self.seg_dir = os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'seg')
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith('.tif')])


        self.T = Reg_Mtx


        self.label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t1])))
        self.label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t2])))

        cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t1])))
        self.cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t2])))

        tt = points_to_pcd(cen_0).transform(self.T)
        self.cen_0 = np.array(tt.points)

        self.neighbour0 = get_neighbouring_structure_knn(self.cen_0,self.n_points)
        self.neighbour1 = get_neighbouring_structure_knn(self.cen_1,self.n_points)

    def __len__(self):

        return len(self.label_0)

    

    def __getitem__(self, idx):


        knn_idx = find_point_knn(self.cen_0[idx],self.cen_1)
        knn_idx = knn_idx[0:self.n_cand]

        label = np.array([self.label_0[idx]])
        cand = np.array(self.label_1[knn_idx])

        # cord_0, _ = create_graph_knn(self.cen_0,self.neighbour0[idx],shift= 'make_center_zero',T =np.eye(4),k=5,n_points = 200,draw_graph=False)
        cord_0 = create_graph_knn(self.cen_0,self.neighbour0[idx],T =np.eye(4))


        PT0 = []
        PT1 = []
        KNN_PT0 = []
        KNN_PT1 = []
        MASK = []

        for temp_idx in knn_idx:

            # cord_1, _ = create_graph_knn(self.cen_1,self.neighbour1[temp_idx],shift= 'make_center_zero',T =np.eye(4),k=5,n_points = 200,draw_graph=False)
            
            cord_1 = create_graph_knn(self.cen_1,self.neighbour1[temp_idx],T =np.eye(4))
            cord_00,cord_1,label_point,gt_mtx = prepare_distance_gt(copy.deepcopy(cord_0),cord_1,dist = self.max_dist)

            pt0 = pc_normalize(cord_00).astype(np.float32)
            knn_pt0 = knn(pt0, self.k)

            pt1 = pc_normalize(cord_1).astype(np.float32)
            knn_pt1 = knn(pt1, self.k)

            pt0 = pt0.astype(np.float32)
            knn_pt0 = knn_pt0.astype(np.int64)
            pt1 = pt1.astype(np.float32)
            knn_pt1 = knn_pt1.astype(np.int64)
            mask = np.ones(self.n_points+1).astype(np.float32)

            pt0 = pt0 - pt0[0]
            pt1 = pt1 - pt1[0]

            PT0.append(pt0)
            PT1.append(pt1)
            KNN_PT0.append(knn_pt0)
            KNN_PT1.append(knn_pt1)
            MASK.append(mask)


        PT0 = np.array(PT0)
        PT1 = np.array(PT1)
        KNN_PT0 = np.array(KNN_PT0)
        KNN_PT1 =  np.array(KNN_PT1)
        MASK = np.array(MASK)
        label = label.astype(np.int64)
        cand = cand.astype(np.int64)

        return PT0,KNN_PT0,PT1,KNN_PT1,MASK,label,cand


class Graph_pair_loader(Dataset):
    
    def __init__(self,main_dir,plant_name,plant_idx,Reg_Mtx,n_points=15,k=6,max_dist=15,segmentation=None):

        self.plant_name = plant_name
        self.main_dir = main_dir
        self.plant_idx = plant_idx
        self.n_points = n_points
        self.k = k
        self.max_dist = max_dist
        self.segmentation = segmentation
        self.Reg_Mtx = Reg_Mtx


        self.seg_dir = os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'seg')
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith('.tif')])
        # self.all_t = np.load('{0}/transformations/{1}_all_transformations.npy'.format(self.main_dir,self.plant_name[self.plant_idx]))

    def __getitem__(self, t1,t2, l1,l2):


        T = self.Reg_Mtx

        label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t1])))
        label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t2])))

        cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t1])))
        cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t2])))

        tt = points_to_pcd(cen_0).transform(T)
        cen_0 = np.array(tt.points)

        neighbour0 = get_neighbouring_structure_knn(cen_0,self.n_points)
        neighbour1 = get_neighbouring_structure_knn(cen_1,self.n_points)

        idx0 = np.where(label_0==l1)[0][0]
        idx1 = np.where(label_1==l2)[0][0]


        dist_cen = np.array([np.linalg.norm(cen_0[idx0]-cen_1[idx1])])

        # cord_0, _ = create_graph_knn(cen_0,neighbour0[idx0],shift= 'make_center_zero',T =np.eye(4),k=5,n_points = 200,draw_graph=False)
        # cord_1, _ = create_graph_knn(cen_1,neighbour1[idx1],shift= 'make_center_zero',T =np.eye(4),k=5,n_points = 200,draw_graph=False)

        cord_0 = create_graph_knn(cen_0,neighbour0[idx0],T =np.eye(4))
        cord_1 = create_graph_knn(cen_1,neighbour1[idx1],T =np.eye(4))

        cord_0,cord_1,label_point,gt_mtx = prepare_distance_gt(cord_0,cord_1,dist = self.max_dist)

        pt0 = pc_normalize(cord_0).astype(np.float32)
        knn_pt0 = knn(pt0, self.k)

        pt1 = pc_normalize(cord_1).astype(np.float32)
        knn_pt1 = knn(pt1, self.k)

        pt0 = pt0.astype(np.float32)
        knn_pt0 = knn_pt0.astype(np.int64)

        pt1 = pt1.astype(np.float32)
        knn_pt1 = knn_pt1.astype(np.int64)

        mask = np.ones(self.n_points+1).astype(np.float32)

        pt0 = pt0 - pt0[0]
        pt1 = pt1 - pt1[0]


        return pt0,knn_pt0,pt1,knn_pt1,mask,dist_cen





def points_to_pcd(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def voxelization(pc,n):
   
    c = np.mean(pc,axis=0)
    pc0 = pc - c+n//2
    pc1 = np.round_(pc0)
    pc2 = np.clip(pc1,0,n-1).astype(int)

    box = np.zeros((n,n,n))
    box[pc2[:,0],pc2[:,1],pc2[:,2]] = 1
   
    return box




class mother_daughter_pair_loader(Dataset):
    
    def __init__(self,main_dir,plant_name,plant_idx,t1,t2,Reg_Mtx,n_cand,segmentation):

        self.plant_name = plant_name

        self.main_dir = main_dir
        self.plant_idx = plant_idx
        self.t1 = t1
        self.t2 = t2
        self.n_cand = n_cand
        self.segmentation = segmentation

        self.seg_dir = os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'seg')
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith('.tif')])

        self.T = Reg_Mtx


        self.label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t1])))
        self.label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/label_{0}.npy'.format(self.seg_files[t2])))

        cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t1])))
        self.cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx],self.segmentation,'stack_info/cen_{0}.npy'.format(self.seg_files[t2])))

        tt = points_to_pcd(cen_0).transform(self.T)
        self.cen_0 = np.array(tt.points)


    def __len__(self):

        return len(self.label_0)

    

    def __getitem__(self, idx):


        knn_idx = find_point_knn(self.cen_0[idx],self.cen_1)
        knn_idx = knn_idx[0:self.n_cand]

        label = np.array([self.label_0[idx]])

        cand = np.array(self.label_1[knn_idx])

        daughter_cand_pairs = []
        comb = combinations(cand, 2)


        for i in list(comb):

            l1 = i[0]
            l2 = i[1]

            idx1 = np.where(self.label_1==l1)[0]
            idx2 = np.where(self.label_1==l2)[0]

            c1 = self.cen_1[idx1]
            c2 = self.cen_1[idx2]
            dist = np.linalg.norm(c1-c2)

            if dist>20:
                continue

            daughter_cand_pairs.append(np.array([l1,l2]))

        daughter_cand_pairs = np.array(daughter_cand_pairs)

        return label,daughter_cand_pairs



