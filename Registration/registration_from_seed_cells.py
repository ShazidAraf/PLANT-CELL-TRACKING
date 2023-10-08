import sys,os,argparse,copy, pdb
import numpy as np
from numpy import random
import pickle, json
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py, scipy
import tifffile, random
# from utils import *


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



point_cloud_path = '/home/eegrad/mdislam/SAM/Registration/reg_from_cell_similarity/data/plant18/cellpose3D/experiment_3/point_cloud_corr_cells/'

Transformation = []


for i in range(1,20):


    point_cloud0 = np.load(os.path.join(point_cloud_path,'interval_{0}'.format(i),'pc0.npy'))
    point_cloud1 = np.load(os.path.join(point_cloud_path,'interval_{0}'.format(i),'pc1.npy'))

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(point_cloud0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)


    # draw_registration_result(pcd0,pcd1, np.eye(4))

    result_ransac = registration(pcd0,pcd1,voxel_size=5,k=1.5)
    print('Fitness',result_ransac.fitness)

   
#     draw_registration_result(pcd0,pcd1, result_ransac.transformation)
   
    Transformation.append(result_ransac.transformation)

    

   
Transformation = np.array(Transformation)
print(Transformation.shape)
# np.save('data/plant_F1_all_transformations.npy',Transformation)


