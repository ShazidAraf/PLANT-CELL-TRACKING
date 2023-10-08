import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import cvxpy
import open3d as o3d
import pickle
import json

# from model_div import *
# from model_graph_pair import *
from utils_tracking_2 import *

import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



class Graph_loader(Dataset):

    def __init__(self,main_dir,split,plant_idx,n_points,k,n_cand,max_dist):

        self.plant_name = ['plant_1','plant_2','plant_4','plant_13','plant_15','plant_18']
        self.main_dir = main_dir
        self.plant_idx = np.array(plant_idx)
        self.split = split
        self.n_points = n_points
        self.k = k
        self.n_cand = n_cand
        self.max_dist = max_dist

        self.ALL_T = []

        self.all_plant_pairs = []
        self.all_plant_pairs_len = []
        self.l_temp = []


        for i_temp in plant_idx:

            all_pairs = np.load('{0}/all_pairs/{1}.npy'.format(self.main_dir,self.plant_name[i_temp]))
            T = np.load('{0}/transformations/{1}_all_transformations.npy'.format(self.main_dir,self.plant_name[i_temp]))

            self.all_plant_pairs.append(all_pairs)
            self.all_plant_pairs_len.append(all_pairs.shape[0])
            self.ALL_T.append(T)

        self.all_plant_pairs = np.array(self.all_plant_pairs)
        self.cum_len = np.cumsum(self.all_plant_pairs_len)
        print(self.cum_len)



    def __len__(self):

        if self.split=='train':
            return self.cum_len[-1]
        elif self.split=='test':
            return self.cum_len[-1]

        else:
            return self.cum_len[-1]



    def __getitem__(self, idx):

        flag = -1


        while flag==-1:

            # print('idx',idx)

            if self.split=='train': 

                for plant_id in range(len(self.cum_len)):
                    if self.cum_len[plant_id ]>idx:
                        break


                if plant_id==0:
                    pair_id = idx
                else:
                    pair_id = idx - self.cum_len[plant_id-1]

                # print(idx,plant_id,pair_id)



                # plant_id = np.random.choice(len(self.plant_idx),1)[0]
                # pair_id = np.random.choice(self.all_plant_pairs_len[plant_id],1)[0]
            else:
                plant_id = 0
                pair_id = idx


            all_t = self.ALL_T[plant_id]


            pair_info = copy.deepcopy(self.all_plant_pairs[plant_id][pair_id])
            pair_info = [np.int(x_temp) for x_temp in pair_info]

            T = np.eye(4)
            for i_temp in range(pair_info[0],pair_info[1]):
                T = np.dot(T,all_t[i_temp])


            label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(pair_info[0])))
            label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(pair_info[1])))

            cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(pair_info[0])))
            cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(pair_info[1])))

            tt = points_to_pcd(cen_0).transform(T)
            cen_0 = np.array(tt.points)

            k0 = np.where(label_0==pair_info[2])[0]
            k1 = np.where(label_1==pair_info[3])[0]


            if len(k0)==0 or len(k1)==0:
                idx+=1
                continue

            k0 = k0[0]
            k1 = k1[0]

            knn_idx = find_point_knn(cen_0[k0],cen_1)

            LABEL = np.where(knn_idx==k1)[0].astype(np.int64)



            knn_idx = knn_idx[0:self.n_cand]
            if LABEL[0]>=self.n_cand:
                # idx+=1
                # continue
                LABEL = np.array([self.n_cand])

            # print('label: ',LABEL)

            flag = 1

            neighbour0 = get_neighbouring_structure_knn(cen_0,self.n_points)
            neighbour1 = get_neighbouring_structure_knn(cen_1,self.n_points)

            cord_0 = create_graph_knn(cen_0,neighbour0[k0],T =np.eye(4))


            PT0 = []
            PT1 = []
            KNN_PT0 = []
            KNN_PT1 = []
            MASK = []
            LABEL_POINT = []

            count = 0
            for temp_idx in knn_idx:

                cord_1 = create_graph_knn(cen_1,neighbour1[temp_idx],T =np.eye(4))
                cord_00,cord_1,label_point,gt_mtx = prepare_distance_gt(copy.deepcopy(cord_0),cord_1,dist = self.max_dist)

                if count!=LABEL[0]:
                    label_point = np.zeros(cord_0.shape[0])

                count+=1
                pt0 = pc_normalize(cord_00).astype(np.float32)
                knn_pt0 = knn(pt0, self.k)

                pt1 = pc_normalize(cord_1).astype(np.float32)
                knn_pt1 = knn(pt1, self.k)

                pt0 = pt0.astype(np.float32)
                knn_pt0 = knn_pt0.astype(np.int64)
                pt1 = pt1.astype(np.float32)
                knn_pt1 = knn_pt1.astype(np.int64)
                mask = np.ones(self.n_points+1).astype(np.float32)

                label_point = label_point.astype(np.float32)

                pt0 = pt0 - pt0[0]
                pt1 = pt1 - pt1[0]

                PT0.append(pt0)
                PT1.append(pt1)
                KNN_PT0.append(knn_pt0)
                KNN_PT1.append(knn_pt1)
                MASK.append(mask)
                LABEL_POINT.append(label_point)


        PT0 = np.array(PT0)
        PT1 = np.array(PT1)
        KNN_PT0 = np.array(KNN_PT0)
        KNN_PT1 =  np.array(KNN_PT1)
        MASK = np.array(MASK)
        LABEL_POINT = np.array(LABEL_POINT)

        # print(PT0.shape,KNN_PT0.shape,PT1.shape,KNN_PT1.shape,MASK.shape, LABEL.shape,LABEL_POINT.shape)



        return PT0,KNN_PT0,PT1,KNN_PT1,MASK, LABEL,LABEL_POINT



class Graph_Pair_loader(Dataset):

    def __init__(self,main_dir,split,plant_idx,n_points,k,max_dist):

        self.plant_name = ['plant_1','plant_2','plant_4','plant_13','plant_15','plant_18']
        self.main_dir = main_dir
        self.plant_idx = np.array(plant_idx)
        self.split = split
        self.n_points = n_points
        self.k = k
        # self.transform = transformation
        self.max_dist = max_dist

        self.ALL_T = []

        self.all_plant_pairs = []
        self.all_plant_pairs_len = []


        for i_temp in plant_idx:

            all_pairs = np.load('{0}/all_pairs/{1}.npy'.format(self.main_dir,self.plant_name[i_temp]))
            T = np.load('{0}/transformations/{1}_all_transformations.npy'.format(self.main_dir,self.plant_name[i_temp]))

            self.all_plant_pairs.append(all_pairs)
            self.all_plant_pairs_len.append(all_pairs.shape[0])
            self.ALL_T.append(T)

        self.all_plant_pairs = np.array(self.all_plant_pairs)
        self.cum_len = np.cumsum(self.all_plant_pairs_len)

    def __len__(self):

        if self.split=='train':
            return 10000
        elif self.split=='test':
            return self.cum_len[-1]

        else:
            return 1000



    def __getitem__(self, idx):

        flag = -1


        while flag==-1:

            # print('idx',idx)

            if self.split=='train': 
                plant_id = np.random.choice(len(self.plant_idx),1)[0]
                pair_id = np.random.choice(self.all_plant_pairs_len[plant_id],1)[0]
            else:
                plant_id = 0
                pair_id = idx//2


            all_t = self.ALL_T[plant_id]


            pair_info = copy.deepcopy(self.all_plant_pairs[plant_id][pair_id])
            pair_info = [np.int(x_temp) for x_temp in pair_info]

            T = np.eye(4)
            for i_temp in range(pair_info[0],pair_info[1]):
                T = np.dot(T,all_t[i_temp])
            # print('pair_info',pair_info)


            label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(pair_info[0])))
            label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(pair_info[1])))

            cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(pair_info[0])))
            cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(pair_info[1])))


            tt = points_to_pcd(cen_0).transform(T)
            cen_0 = np.array(tt.points)

            k0 = np.where(label_0==pair_info[2])[0]
            k1 = np.where(label_1==pair_info[3])[0]


            if len(k0)==0 or len(k1)==0:
                idx+=1
                continue


            k0 = k0[0]
            k1 = k1[0]

            neighbour0 = get_neighbouring_structure_knn(cen_0,self.n_points)
            neighbour1 = get_neighbouring_structure_knn(cen_1,self.n_points)

            s1 = idx%2

            cord_0 = create_graph_knn(cen_0,neighbour0[k0],T =np.eye(4))

            if s1!=0:

                cord_1 = create_graph_knn(cen_1,neighbour1[k1],T =np.eye(4))
                cord_0,cord_1,label_point,gt_mtx = prepare_distance_gt(cord_0,cord_1,dist = self.max_dist)

                if np.sum(label_point)>self.n_points//2:
                    flag = 1
                    label = np.array([1])

            else:

                # cord_1, _ = create_graph_knn(cen_1,neighbour1[neighbour1[k1][-1]],shift = 'make_center_zero',T=np.eye(4),k=5,n_points = 200,draw_graph=False)
                cord_1 = create_graph_knn(cen_1,neighbour1[neighbour1[k1][-1]],T=np.eye(4))
                cord_0,cord_1,label_point,gt_mtx = prepare_distance_gt(cord_0,cord_1,dist = self.max_dist)

                label_point = np.zeros(cord_0.shape[0])
                label = np.array([0])


                flag = 1

            if flag==-1:
                idx+=1
                continue


            n_temp = int(np.sum(label_point))
            vis = 0

            if n_temp>0 and vis==1:

                print(label)


                matching_idx = np.arange(n_temp)
                # print(matching_idx)
                non_matching_idx = [0]+list(np.arange(n_temp,cord_0.shape[0]))
                # print(non_matching_idx)

                super_impose_graph(cord_0[matching_idx],cord_1[matching_idx])
                super_impose_graph(cord_0[non_matching_idx],cord_1[non_matching_idx])




            pt0 = pc_normalize(cord_0).astype(np.float32)
            knn_pt0 = knn(pt0, self.k)

            pt1 = pc_normalize(cord_1).astype(np.float32)
            knn_pt1 = knn(pt1, self.k)

            pt0 = pt0.astype(np.float32)
            knn_pt0 = knn_pt0.astype(np.int64)
            pt1 = pt1.astype(np.float32)
            knn_pt1 = knn_pt1.astype(np.int64)
            mask = np.ones(self.n_points+1).astype(np.float32)

            label = label.astype(np.float32)
            label_point = label_point.astype(np.float32)

            pt0 = pt0 - pt0[0]
            pt1 = pt1 - pt1[0]


        return pt0,knn_pt0,pt1,knn_pt1,mask, label,label_point


class Cell_Div_loader(Dataset):

    def __init__(self,main_dir,split,plant_idx,v_size):

        self.plant_name = ['plant_1','plant_2','plant_4','plant_13','plant_15','plant_18']
        self.main_dir = main_dir
        self.plant_idx = np.array(plant_idx)
        self.split = split
        self.v_size = v_size


        self.ALL_T = []

        self.all_plant_pairs_prop = []
        self.all_plant_pairs_div = []

        self.all_plant_pairs_prop_len = []
        self.all_plant_pairs_div_len = []
        self.l_temp = []


        for i_temp in plant_idx:

            all_pairs_prop = np.load('{0}/all_pairs/{1}.npy'.format(self.main_dir,self.plant_name[i_temp]))
            all_pairs_div = np.load('{0}/all_div/{1}.npy'.format(self.main_dir,self.plant_name[i_temp]))

            T = np.load('{0}/transformations/{1}_all_transformations.npy'.format(self.main_dir,self.plant_name[i_temp]))

            self.all_plant_pairs_prop.append(all_pairs_prop)
            self.all_plant_pairs_prop_len.append(all_pairs_prop.shape[0])

            self.all_plant_pairs_div.append(all_pairs_div)
            self.all_plant_pairs_div_len.append(all_pairs_div.shape[0])

            self.ALL_T.append(T)

        self.all_plant_pairs_prop = np.array(self.all_plant_pairs_prop)
        self.all_plant_pairs_div = np.array(self.all_plant_pairs_div)

        self.cum_len_prop = np.cumsum(self.all_plant_pairs_prop_len)
        self.cum_len_div = np.cumsum(self.all_plant_pairs_div_len)

        # print(self.cum_len_prop)
        # print(self.cum_len_div)


    def __len__(self):

        if self.split=='train':
            return 10000
            # return 2*self.cum_len_div[-1]
        elif self.split=='test':

            return 2*self.cum_len_div[-1]

        elif self.split=='val':
            # return 200
            return 2*self.cum_len_div[-1]



    def __getitem__(self, idx):


        if self.split=='train': 

            plant_id = np.random.choice(len(self.plant_idx),1)[0]


            if idx%5==0:
                div_flag = 1
                pair_id = np.random.choice(self.all_plant_pairs_div_len[plant_id],1)[0]
            else:
                div_flag = 0
                pair_id = np.random.choice(self.all_plant_pairs_prop_len[plant_id],1)[0]
        else:
            div_flag = 1
            plant_id = 0
            pair_id = idx//2

        seg_dir = os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'seg')
        seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])

        all_t = self.ALL_T[plant_id]

        if div_flag==1:
            pair_info = copy.deepcopy(self.all_plant_pairs_div[plant_id][pair_id])
        else:
            pair_info = copy.deepcopy(self.all_plant_pairs_prop[plant_id][pair_id])

        pair_info = [np.int(x_temp) for x_temp in pair_info]

        t1 = pair_info[0]
        t2 = pair_info[1]

        T = np.eye(4)
        for i_temp in range(t1,t2):
            T = np.dot(T,all_t[i_temp])

        # print(t1,t2,self.plant_name[self.plant_idx])

        label_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(t1)))
        label_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/label_{0}.npy'.format(t2)))

        cen_0 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(t1)))
        cen_1 = np.load(os.path.join(self.main_dir,self.plant_name[self.plant_idx[plant_id]],'stack_info/cen_{0}.npy'.format(t2)))

        seg0 = tifffile.imread(os.path.join(seg_dir,seg_files[t1]))
        seg1 = tifffile.imread(os.path.join(seg_dir,seg_files[t2]))



        if div_flag == 0:
            l0 = pair_info[2]
            l1 = pair_info[3]

            s1 = idx%2

            if s1==0:

                loc_x = np.where(label_1==l1)[0][0]
                rand_flag=0

                while rand_flag==0:
                    loc_y = random.randint(len(label_1))
                    if loc_x!=loc_y:
                        rand_flag=1
                l1 = label_1[loc_y]
                gt = np.array([0])

            else:

                gt = np.array([1])

            loc0 = np.array(np.where(seg0==l0)).T
            tt = points_to_pcd(loc0).transform(T)
            pc0 = np.array(tt.points)
            pc1 = np.array(np.where(seg1==l1)).T


            vox_0 = voxelization(pc0,self.v_size)
            vox_1 = voxelization(pc1,self.v_size)
            diff = vox_0-vox_1 


            diff = diff.astype(np.float32)
            gt = gt.astype(np.float32)

            return diff,gt
            print("Error ")


        else:
            l0 = pair_info[2]
            l1 = pair_info[3]
            l2 = pair_info[4]

            loc0 = np.array(np.where(seg0==l0)).T
            tt = points_to_pcd(loc0).transform(T)
            pc0 = np.array(tt.points)

            box0 = np.max(pc0,axis=0) - np.min(pc0,axis=0)


            tt = points_to_pcd(cen_0).transform(T)
            cen_0 = np.array(tt.points)
            idx = np.where(label_0==l0)[0][0]



            s1 = idx%2



            if s1==1:



                loc1 = np.array(np.where(seg1==l1)).T
                loc2 = np.array(np.where(seg1==l2)).T

                pc1 = np.concatenate((loc1,loc2),axis=0)
                gt = np.array([1])



            else:


                n_cand = 10

                knn_idx = find_point_knn(cen_0[idx],cen_1)
                knn_idx = knn_idx[0:n_cand]

                cand = np.array(label_1[knn_idx])
                comb = combinations(cand, 2)

                xx0 = np.sort(np.array([l1,l2]))


                c_temp = 0
                for i in list(comb):

                    c_temp+=1


                    xx1 = np.sort(np.array([i[0],i[1]]))

                    if xx0[0]==xx1[0] and xx0[1]==xx1[1]:
                        continue


                    idx1 = np.where(label_1==i[0])[0][0]
                    idx2 = np.where(label_1==i[1])[0][0]

                    c0 = cen_1[idx1]
                    c1 = cen_1[idx2]
                    dist = np.linalg.norm(c0-c1)


                    if dist>25 and c_temp < len(list(comb)):
                        continue

                    loc1 = np.array(np.where(seg1==i[0])).T
                    loc2 = np.array(np.where(seg1==i[1])).T
                    pc1 = np.concatenate((loc1,loc2),axis=0)

                    box1 = np.max(pc1,axis=0) - np.min(pc1,axis=0)

                    delta = np.divide(np.abs(box1-box0),box0)
                    th = np.max(delta)
                    # print(th)

                    if th<0.5:
                        break

                gt = np.array([0])


            vox_0 = voxelization(pc0,self.v_size)
            vox_1 = voxelization(pc1,self.v_size)
            diff = vox_0-vox_1 

            # tifffile.imwrite('sample/{0}_0.tif'.format(idx),vox_0)
            # tifffile.imwrite('sample/{0}_1.tif'.format(idx),vox_1)
            # tifffile.imwrite('sample/{0}_diff.tif'.format(idx),diff)


            diff = diff.astype(np.float32)
            gt = gt.astype(np.float32)


            return diff,gt
