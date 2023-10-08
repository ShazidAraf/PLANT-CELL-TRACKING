import numpy as np
import copy, os, time
import tifffile
import pickle, json
import open3d as o3d
from tqdm import tqdm
from itertools import combinations
from numpy import random
import h5py
import scipy
from scipy.spatial.distance import directed_hausdorff
color = random.rand(2000,3)


class Tracklets:
    
    def __init__(self, t_start,label):
        
        self.t_start = t_start
        self.t_end = t_start
        self.labels = [label]
        self.score = -1
        self.div_score = 0
        self.div_flag = 0
        self.mother = []


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





def load_data(f_name):
    try:
        with open("{0}".format(f_name), 'rb') as f:
            data = pickle.load(f)
    except:
        data = []
    return data

def save_data(data,f_name):
    with open("{0}".format(f_name), "wb") as f:
        pickle.dump(data, f)


def voxelization(pc,n):
   
    c = np.mean(pc,axis=0)
    pc0 = pc - c+n//2
    pc1 = np.round_(pc0)
    pc2 = np.clip(pc1,0,n-1).astype(int)

    box = np.zeros((n,n,n))
    box[pc2[:,0],pc2[:,1],pc2[:,2]] = 1
   
    return box




def plot_cells_3D(img,seg,label,color,division=False,write_label=False):
    
    
    font_size = 0.3
    thickness = 1
    w = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    for i in tqdm(range(len(label))):
        
        l = label[i]
        
        if l==0:
            continue
        
        loc3D = np.where(seg==l)
        n_z = np.unique(loc3D[0])
        
        
        j = (n_z[0]+n_z[-1])//2
#         print(idx)
         
        loc = np.where(seg[j]==l)
        c = determine_centroid(loc[0],loc[1])
        position =(int(c[1]),int(c[0]))
        
#         temp = np.clip([j-10,j+10],0,lim)
#         for  k in range(temp[0],temp[1]): 
        for  k in n_z: 
            if write_label:
                img[k] = cv2.putText(img[k] , "{0}".format(l),position , font, font_size, color, thickness, cv2.LINE_AA)
            elif division:
                img[k] = cv2.putText(img[k] , "{0}".format(i//2),position , font, font_size, color, thickness, cv2.LINE_AA)
            else:
                img[k] = cv2.putText(img[k] , "{0}".format(i),position , font, font_size, color, thickness, cv2.LINE_AA)
    
    return img    


def determine_centroid(x,y):
    centroid = np.array([sum(x) / (len(x)+1e-10), sum(y) / (len(y)+1e-10)])
    return centroid


def combine_tracks(all_corr_labels,start,finish):
    

    all_tracks = np.zeros((100000,all_corr_labels.shape[0]+1))
    

    for i in range(start,finish):

        l0 = all_corr_labels[i][:,0]
        l1 = all_corr_labels[i][:,1]

        if i==start:

            all_tracks[0:len(l0),i] = l0
            all_tracks[0:len(l1),i+1] = l1
            vacant_idx = np.min(np.where(all_tracks[:,i+1]==0)[0])

            continue



        for j in range(len(l0)):

            idx = np.where(all_tracks[:,i]==l0[j])[0]

            if len(idx)>0:

                all_tracks[idx[0],i+1] = l1[j]

            else:
                all_tracks[vacant_idx,i] = l0[j]
                all_tracks[vacant_idx,i+1] = l1[j]
                vacant_idx+=1




    for i in range(all_tracks.shape[0]):

        if sum(all_tracks[i])==0:
            end = i
            break

    all_tracks_final = all_tracks[0:end]
    
    return all_tracks_final



def visualize1(point_cloud,neighbour,n_samp = 10):
    
    for p in range(n_samp):

        nbr = neighbour[p]
        all_p = []

        for k in range(len(nbr)):

            all_p.append(point_cloud[nbr[k]])

        o3d.visualization.draw_geometries(all_p)




def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def adj_mat(pc):
    pc_square = np.sum(pc**2, axis=1, keepdims=True)
    ptp = -2 * np.matmul(pc, np.transpose(pc, [1, 0]))
    pt_square = np.transpose(pc_square, [1, 0])
    res = pc_square + pt_square + ptp
    return res


def knn(pc, k):
    adj = adj_mat(pc)
    distance = np.argsort(adj, axis=1)
    return distance[:, :k]



def get_knn_centroid(cen,idx):
    c00 = cen[idx[0]]
    all_c = []
    for i in range(len(idx)):
        all_c.append(cen[idx[i]])

    all_c = np.array(all_c) - c00


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_c)

    return pcd

def get_k_nearest_candidates(center,all_cord,k):
    
    a = np.linalg.norm(all_cord - center,axis=1)
    idx = np.argsort(a)[1:k+1]
    
    return idx






def create_graph_knn(cen,idx,T):
    

    all_c = []

    for i in range(len(idx)):
        
        c = cen[idx[i]]
        all_c.append(c)

    all_c = np.array(all_c)
    pcd_c = points_to_pcd(all_c).transform(T)

    shift = -np.array(pcd_c.points[0])

    points = np.array(pcd_c.points)+ shift

    return points



def get_neighbouring_structure_knn(center,k):
    
    neighbour = [] 
    
    for i in range(len(center)):
        
        a = np.linalg.norm(center - center[i],axis=1)
        nbr = np.argsort(a)[0:k+1].tolist()
        neighbour.append(nbr)    
        
    return neighbour



def prepare_distance_gt(cord1,cord2,dist=10):
    n = cord1.shape[0]
    
    mat = np.zeros((cord1.shape[0],cord2.shape[0]))
    
    for i in range(cord1.shape[0]):
        
        mat[i] = np.linalg.norm(cord1[i]-cord2,axis=1)
        

    min_val = np.min(mat,axis=1)
    min_idx = np.argmin(mat,axis=1)
    
    
    
    corr_1 = []
    corr_2 = []
    for i in range(len(min_val)):
        
        if min_val[i]<dist:
            
            if (min_idx[i] not in corr_2):
                
                loc = np.where(min_idx==min_idx[i])[0]
                
                if len(loc)==1:                
                    corr_1.append(i)
                    corr_2.append(min_idx[i])
                else:
                    d = min_val[loc]
                    ind = np.argmin(d)
                    
                    corr_1.append(i)
                    corr_2.append(min_idx[loc[ind]])
                
    a = np.arange(cord1.shape[0])

    b1 = list(set(a)-set(corr_1))
    b2 = list(set(a)-set(corr_2))

    m_pcd1 = cord1[corr_1]
    m_pcd2 = cord2[corr_2]

    n1 = len(corr_1)

    um_pcd1 = cord1[b1]
    um_pcd2 = cord2[b2]

    pcd_1 = np.concatenate((m_pcd1, um_pcd1), axis=0)
    pcd_2 = np.concatenate((m_pcd2, um_pcd2), axis=0)


    gt_mtx = np.zeros((n,n))
    gt_mtx[0:n1,0:n1] = 1

    gt = np.zeros(n)
    gt[0:n1] = 1
    



    return pcd_1,pcd_2,gt,gt_mtx



def read_PNAS_gt(text_file_path,lines=None):


    if lines==None:
        with open(os.path.join(text_file_path)) as f:
            lines = f.readlines()


    data = []

    for i in range(len(lines)):
        data.append(lines[i].split(':'))

#     data_t0 = np.array([int(d[0]) for d in data])
    data_t0 = [int(d[0]) for d in data]
    data_t1 = [d[1].split(',') for d in data]

    for i in range(len(data_t1)):

        d = copy.deepcopy(data_t1[i])

        for j in range(len(d)):
            d[j] = int(d[j])

        data_t1[i] = copy.deepcopy(d)
       
    l0 = []
    l1 = []
    mother = []
    daughter = []
    
    for i in range(len(data_t0)):
        
        if len(data_t1[i])==1:
            l0.append(data_t0[i])
            l1.append(data_t1[i])
        elif len(data_t1[i])==2:
            mother.append(data_t0[i])
            daughter.append(np.array(data_t1[i]))
            
    
    return np.array(l0),np.array(l1),np.array(mother),np.array(daughter)



def determine_accuracy(text_path,corr_label_gt0,corr_label_gt1,mother_label_gt,daughter_label_gt):

    correct_propagation_count = 0
    correct_division_count = 0
    pn = 0
    dn = 0
    t = 0
    u = 0

    prop_t0,prop_t1,mother_gt,daughter_gt = read_PNAS_gt(text_path)

    pn = len(prop_t0)
    dn = len(mother_gt)
                    

    for q in range(len(corr_label_gt0)):

        loc = np.where(prop_t0==corr_label_gt0[q])[0]

        if len(loc)>0:

            k2 = loc[0]
            l2 = prop_t1[k2]

            t+=1
            if corr_label_gt1[q]==l2[0]:
                correct_propagation_count+=1


    for q in range(len(mother_label_gt)):

        loc = np.where(mother_gt==mother_label_gt[q])[0]

        if len(loc)>0:

            k2 = loc[0]
            l2 = daughter_gt[k2]

            u+=1

            dau = np.sort(daughter_label_gt[q])
            l2 = np.sort(l2)

            if np.all(l2==dau):
                correct_division_count+=1


    propagation_accuracy = correct_propagation_count*100/pn
    detected_propagation_accuracy = correct_propagation_count*100/(t+1e-40)

    # print(propagation_accuracy)

    division_accuracy = correct_division_count*100/dn
    detected_division_accuracy = correct_division_count*100/(u+1e-40)



    return [correct_propagation_count,pn,t,propagation_accuracy,detected_propagation_accuracy,correct_division_count,dn,u,division_accuracy,detected_division_accuracy]



def get_label(source_seg,target_seg,source_label):
    
    loc = np.where(source_seg==source_label)
    if len(loc[0])==0:
        print(len(loc[0]),source_label)
        return -1

    cen = [int(np.mean(loc[0])),int(np.mean(loc[1])),int(np.mean(loc[2]))]
    target_label = target_seg[cen[0],cen[1],cen[2]]

    # target_label,cen,len(loc[0])
    
    return target_label


def distance_matrix(cord1,cord2,dist=6):
    
    mat = np.zeros((cord1.shape[0],cord2.shape[0]))
    
    for i in range(cord1.shape[0]):
        
        mat[i] = np.linalg.norm(cord1[i]-cord2,axis=1)
        
    global_min_val =  np.min(mat,axis=None)
    global_min_idx = np.unravel_index(np.argmin(mat, axis=None), mat.shape)

    min_val = np.min(mat,axis=1)
    min_idx = np.argmin(mat,axis=1)
    
    
    
    corr_1 = []
    corr_2 = []
    for i in range(len(min_val)):
        
        if min_val[i]<dist:
            
            if (min_idx[i] not in corr_2):
                
                loc = np.where(min_idx==min_idx[i])[0]
                
                if len(loc)==1:                
                    corr_1.append(i)
                    corr_2.append(min_idx[i])
                else:
                    d = min_val[loc]
                    ind = np.argmin(d)
                    
                    corr_1.append(i)
                    corr_2.append(min_idx[loc[ind]])
                
    pcd1 = cord1[corr_1]
    pcd2 = cord2[corr_2]
    
    return pcd1,pcd2,corr_1,corr_2,[global_min_val,global_min_idx]