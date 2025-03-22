import os
import copy
import json
import pickle
import numpy as np
import tifffile
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from model import deepseed
from deepseed_utils import *
import pdb
from scipy.optimize import linear_sum_assignment

transform = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(50),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def neighbour_similarity_score(g1,g2):
    
    dist = np.zeros((g1["feature"].shape[0],g2["feature"].shape[0]))+100
    
    for i in range(g1["feature"].shape[0]):
        for j in range(g2["feature"].shape[0]):
            
            dist[i,j] = np.linalg.norm(g1["feature"][i,:]-g2["feature"][j,:])
            
    min_dist = np.min(dist)
    ns_score = np.exp(-min_dist)

    return ns_score


def cell_similarity_score(model,cell1,cell2):

    img1 = 1.0*torch.unsqueeze(torch.from_numpy(copy.deepcopy(cell1)),dim=0)
    img1 = torch.cat((img1, img1, img1), 0)
    img1 = transform(img1)


    img2 = 1.0*torch.unsqueeze(torch.from_numpy(copy.deepcopy(cell2)),dim=0)
    img2 = torch.cat((img2, img2, img2), 0)
    img2 = transform(img2)


    img1 = (img1.type(torch.FloatTensor)).cuda()
    img2 = (img2.type(torch.FloatTensor)).cuda()

    with torch.no_grad():
        cs_score = model(torch.unsqueeze(img1-img2,0))[0][0]

    # print(similarity_score)
    return cs_score.detach().cpu().numpy()


def get_seed_pairs(similarity_matrix, threshold):
    """
    Perform one-to-one matching on a similarity matrix while considering only values above a threshold.

    Parameters:
    - similarity_matrix: 2D numpy array where higher values indicate higher similarity.
    - threshold: Minimum similarity score required for consideration.

    Returns:
    - row_indices: List of row indices of matched pairs.
    - col_indices: List of column indices of matched pairs.
    """
    # Mask values below threshold
    cost_matrix = -similarity_matrix
    
    # Solve assignment problem
    # pdb.set_trace()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter out invalid matches (those originally below threshold)
    valid_indices = []
    for i in range(len(row_ind)):
        if similarity_matrix[row_ind[i], col_ind[i]] >= threshold:
            valid_indices.append(i)


    row_indices = [row_ind[i] for i in valid_indices]
    col_indices = [col_ind[i] for i in valid_indices]
    
    return row_indices, col_indices



# Registration

def prepare_dataset(source,target,voxel_size,trans_init=None):


    if trans_init is not None:
        source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
#     print(np.array(source_down.points).shape,source_fpfh)
#     print(np.array(target_down.points).shape,target_fpfh.shape)
    
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    

    radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=1000))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size,k=1.5):
    
    distance_threshold = voxel_size * k
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def registration(pc0,pc1,voxel_size,k):
    
    pc0, pc1, pc0_down, pc1_down, pc0_fpfh, pc1_fpfh = prepare_dataset(pc0,pc1,voxel_size)
    
#     print(np.array(pc0_down.points).shape,np.array(pc1_down.points).shape)
    result = execute_global_registration(pc0_down, pc1_down,
                                                pc0_fpfh, pc1_fpfh,
                                                voxel_size,k)
    
    return result



def get_Registration_Matrix(args,file_dir, i,j, thresh=0.9,weight =0.6):


    # Load the deep learning model for cell similarity scoring
    model = deepseed(in_dim=3, out_dim=32)
    model = model.cuda()
    model.eval()

    # pretrained_path = 'checkpoints/Registration_ckp.pt'
    pretrained_path = args.checkpoint_Registraton
    print(f'Loading pretrained weights from {pretrained_path}...')
    model.load_state_dict(torch.load(pretrained_path))

    seg_dir = os.path.join(file_dir,'seg')
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])
    
    stack_dir = os.path.join(file_dir,'Registration_info')
    stack_files = sorted([f for f in os.listdir(stack_dir) if f.endswith('.pkl')])
    
    seed_pairs = []

    print(seg_files[i],seg_files[j])
    print(stack_files[i],stack_files[j])

    f0, f1 = os.path.join(stack_dir, stack_files[i]), os.path.join(stack_dir, stack_files[j])
    with open(f0, "rb") as f:
        stack0 = pickle.load(f)
    with open(f1, "rb") as f:
        stack1 = pickle.load(f)

    seg0 = tifffile.imread(os.path.join(seg_dir, seg_files[i]))
    seg1 = tifffile.imread(os.path.join(seg_dir, seg_files[j]))

    cell_similarity_matrix = np.zeros((len(stack0), len(stack1)))
    neighbouring_similarity_matrix = np.zeros((len(stack0), len(stack1)))
    # len(stack0)
    for j1 in tqdm(range(len(stack0))):
        for k in range(len(stack1)):
            cell_similarity_matrix[j1, k] = cell_similarity_score(model,stack0[j1]["cell"], stack1[k]["cell"])
            neighbouring_similarity_matrix[j1, k] = neighbour_similarity_score(stack0[j1], stack1[k])

    similarity_score = weight *cell_similarity_matrix + (1 - weight) * neighbouring_similarity_matrix 


    a1, b1 = get_seed_pairs(similarity_score, thresh)
    # seed_pairs = [np.array([stack0[a1[l]]["center"], stack1[b1[l]]["center"], similarity_score[a1[l], b1[l]]]) for l in range(len(a1))]

    pc0 = [np.array(np.where(seg0 == stack0[lx]["center"])).T for lx in a1]
    pc0 = np.concatenate(pc0,axis=0)
    pc1 = [np.array(np.where(seg1 == stack1[lx]["center"])).T for lx in b1]
    pc1 = np.concatenate(pc1,axis=0)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pc0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    result_ransac = registration(pcd0,pcd1,voxel_size=5,k=1.5)
    print("Fitness: ",result_ransac.fitness)
    print("Transformation Matrix : ", result_ransac.transformation)

    return result_ransac.transformation



def plot_transformed_point_clouds(pcd1, pcd2, transformation_matrix):
    """
    Plots two 3D point clouds using Open3D, applying a transformation matrix to the first point cloud.

    :param pcd1: numpy array of shape (N, 3), representing the first point cloud
    :param pcd2: numpy array of shape (M, 3), representing the second point cloud
    :param transformation_matrix: 4x4 numpy array representing the transformation matrix for pcd1
    """
    # Convert numpy arrays to Open3D point clouds
    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)

    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)

    # Apply the transformation matrix to the first point cloud
    pcd1_o3d.transform(transformation_matrix)

    # Assign colors for visualization (pcd1: red, pcd2: blue)
    pcd1_o3d.paint_uniform_color([1, 0, 0])  # Red
    pcd2_o3d.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize the point clouds
    o3d.visualization.draw_geometries([pcd1_o3d, pcd2_o3d])


