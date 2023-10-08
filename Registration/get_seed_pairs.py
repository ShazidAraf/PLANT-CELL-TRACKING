import numpy as np
import copy, os, time
import tifffile
import pickle, json
#import open3d as o3d
from tqdm import tqdm
from numpy import random
import h5py
color = random.rand(2000,3)
#from utils import *



print('Define all directories.....')

main_dir = 'data/plant18'

text_dir = os.path.join(main_dir,'tracking_data')
text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
# print(text_files)

data_dir = os.path.join(main_dir,'data')
data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
# print(data_files)

stack_dir =   os.path.join(main_dir,'cellpose3D/stacks')
stack_files = sorted([f for f in os.listdir(stack_dir) if f.endswith('.pkl')])

seg_dir = os.path.join(main_dir,'cellpose3D/seg')
seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])
# print(seg_files)


# visual_dir = os.path.join(main_dir,'visual')
# os.makedirs(visual_dir,exist_ok=True)

seg_gt_dir = os.path.join(main_dir,'seg_gt')
seg_gt_files = sorted([f for f in os.listdir(seg_gt_dir) if f.endswith('.tif')])
# print(seg_gt_files)

seed_dir = os.path.join(main_dir,'cellpose3D/seed_pairs')
seed_files = sorted([f for f in os.listdir(seed_dir ) if f.endswith('.npy')])
n_interval = len(seed_files)//2

point_cloud_dir = os.path.join(main_dir,'cellpose3D/point_cloud_corr_cells')
os.makedirs(point_cloud_dir,exist_ok=True)



def get_seed_pairs(a,score):

    b = np.unique(a.flatten())
    c = -np.sort(-b) 
    idx  = c>score
    c = c[idx]
    d = [np.where(a==l) for l in  c]

#     l = 0
    a1 = []
    b1 = []
    
#     while len(a1)<5:    
    for l in range(len(c)):
        a1= a1+d[l][0].tolist()
        b1= b1+d[l][1].tolist()

        l+=1
        
    return a1,b1



print('Preparing Point Clouds of Correspondant cells.....')
    
thresh = 0.98
weight = 0.8


for i in tqdm(range(n_interval)):



    pc_save_dir = os.path.join(point_cloud_dir,'interval_{0}'.format(i))
    os.makedirs(pc_save_dir,exist_ok=True)
    
    
    f0 = os.path.join(stack_dir,stack_files[i])
    f1 = os.path.join(stack_dir,stack_files[i+1])

    with (open(f0, "rb")) as f:
        stack0 = pickle.load(f)

    with (open(f1, "rb")) as f:
        stack1 = pickle.load(f)

#     seed_dir = os.path.join(main_dir,'seed_pairs1')

    neighbouring_similarity_matrix = np.load(os.path.join(seed_dir,'neighbouring_similarity_{0}.npy'.format(str(i).zfill(2))))
    cell_similarity_matrix = np.load(os.path.join(seed_dir,'cell_similarity_{0}.npy'.format(str(i).zfill(2))))
    
    
    assert neighbouring_similarity_matrix.shape[0]==len(stack0)
    assert neighbouring_similarity_matrix.shape[1]==len(stack1)
    
    similarity_score = weight*neighbouring_similarity_matrix+(1-weight)*cell_similarity_matrix
    a1,b1 = get_seed_pairs(similarity_score,thresh)
    seed_pairs = []
    
    
    for l in range(len(a1)):
        seed_pairs.append(np.array([stack0[a1[l]]["center"],stack1[b1[l]]["center"],similarity_score[a1[l],b1[l]]]))

    seed_pairs = np.array(seed_pairs)

    seed = seed_pairs
    score = seed[:,2]

    idx = []
    for j in range(len(score)):
        if score[j]>=thresh:
            idx.append(j)

    idx = np.array(idx)


    label0 = seed[idx ,0]
    label1 = seed[idx ,1]

    seg0 = tifffile.imread(os.path.join(seg_dir,seg_files[i]))
    seg1 = tifffile.imread(os.path.join(seg_dir,seg_files[i+1]))

    print(len(label0))
    point_cloud0 = np.empty((0,3))
    point_cloud1 = np.empty((0,3))


    for j in range(len(label0)):
        

        p0 = np.where(seg0==label0[j])
        p1 = np.where(seg1==label1[j])

        point_cloud0 = np.append(point_cloud0,np.array(p0).T,axis=0)
        point_cloud1 = np.append(point_cloud1,np.array(p1).T,axis=0)


    print(point_cloud0.shape)
    print(point_cloud1.shape)
    
    np.save(os.path.join(pc_save_dir,'pc0.npy'), point_cloud0)
    np.save(os.path.join(pc_save_dir,'pc1.npy'), point_cloud1)


#     pcd0 = o3d.geometry.PointCloud()
#     pcd0.points = o3d.utility.Vector3dVector(point_cloud0)
#     o3d.io.write_point_cloud(os.path.join(pc_save_dir,'pc0.ply'), pcd0)

#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
#     o3d.io.write_point_cloud(os.path.join(pc_save_dir,'pc1.ply'), pcd1)

