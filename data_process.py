import sys,os,argparse,copy

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from utils_tracking import *

import numpy as np
from numpy import random
import pickle, json
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py,scipy, pdb

################


import os
import pickle
import numpy as np
import tifffile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from skimage import morphology
from deepseed_utils import get_cell_info, get_pairs, triangle_feature, extract_cell, determine_centroid


class STACK_INF0_3D(Dataset):
    
	def __init__(self,main_dir,plant_name,plant_idx,segmentation=None):

		self.plant_name = plant_name
		self.main_dir = main_dir
		self.plant_idx = np.array(plant_idx)

		self.ALL_TRACKS = []
		self.ALL_T = []

		self.all_plant_pairs = []
		self.all_plant_pairs_len = []


		for i_temp in plant_idx:

			seg_dir = os.path.join(self.main_dir,self.plant_name[i_temp],segmentation,'seg')
			seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])
			save_dir = os.path.join(self.main_dir,self.plant_name[i_temp],segmentation,'stack_info')
			os.makedirs(save_dir,exist_ok=True)

			for i in tqdm(range(len(seg_files))):


				seg = tifffile.imread(os.path.join(seg_dir,seg_files[i]))
				labels = []
				centroids = []

				label_temp = np.unique(seg)[1:]

				for j in tqdm(range(len(label_temp))):

					loc = np.where(seg==label_temp[j])

					if len(loc[0])<100:
						continue

					centroids.append(np.array([sum(loc[0]) / (len(loc[0])+1e-10), sum(loc[1]) / (len(loc[1])+1e-10), sum(loc[2]) / (len(loc[2])+1e-10)]))
					labels.append(label_temp[j])


				centroids = np.array(centroids)
				labels = np.array(labels)

				# np.save(f'{save_dir}/cen_{i}.npy',np.array(centroids))
				# np.save(f'{save_dir}/label_{i}.npy',np.array(labels))

				np.save(f'{save_dir}/cen_{seg_files[i]}.npy',np.array(centroids))
				np.save(f'{save_dir}/label_{seg_files[i]}.npy',np.array(labels))







class STACK_INF0_Registration(Dataset):
    def __init__(self, main_dir, plant_name,plant_idx, segmentation=None):
        self.main_dir = main_dir
        self.plant_name = plant_name
        self.plant_idx = plant_idx
        self.segmentation = segmentation
        # self.save_dir = os.path.join(self.main_dir,self.plant_name[i_temp],segmentation,'Registration_info')
        # os.makedirs(self.save_dir, exist_ok=True)
        
        for i_temp in self.plant_idx:

            seg_dir = os.path.join(self.main_dir,self.plant_name[i_temp],segmentation,'seg')
            seg_files = sorted(f for f in os.listdir(seg_dir) if f.endswith('.tif'))

            save_dir = os.path.join(self.main_dir,self.plant_name[i_temp],segmentation,'Registration_info')
            os.makedirs(save_dir, exist_ok=True)
            
            for i in tqdm(range(len(seg_files)), desc=f'Processing Plant {i_temp} Segmentation Files'):
                seg = tifffile.imread(os.path.join(seg_dir, seg_files[i]))
                ws = self.get_membrane_mask(seg)
                # tifffile.imwrite('x.tif',ws)
                # pdb.set_trace()
                
                print(f'Processing: {seg_files[i]}')
                
                labels = np.unique(seg)[1:]
                all_cell_info = []
                
                for label in tqdm(labels, desc=f'Extracting Cells from {seg_files[i]}'):
                    loc = np.where(seg == label)
                    nz = np.unique(loc[0])
                    idx = (nz[0] + nz[-1]) // 2
                    
                    seg_slice, ws_slice = seg[idx], ws[idx]

                    loc_x = np.where(seg_slice==label)
                    if len(loc_x[0])==0:
                        continue
                    slice_labels, c, area, nbr = get_cell_info(seg_slice, label, background=0, bx=1)
                    
                    if area[0] > 100 and len(nbr) >= 4:
                        pair = get_pairs(nbr.tolist())
                        features = [triangle_feature(c, area, slice_labels, label, p[0], p[1]) for p in pair]
                        cell = extract_cell(ws_slice, seg_slice, label)
                        center_cord = determine_centroid(*np.where(seg_slice == label))
                        
                        record = {
                            "center": label,
                            "center_cord": center_cord,
                            "nbr": nbr,
                            "pair": np.array(pair),
                            "feature": np.array(features),
                            "area": area[0],
                            "cell": cell,
                        }
                        all_cell_info.append(record)
                
                # save_path = os.path.join(save_dir, f'stack_{str(i).zfill(2)}.pkl')
                print(len(all_cell_info))
                save_path = os.path.join(save_dir, f'stack_{seg_files[i]}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(all_cell_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_membrane_mask(self, instance_mask, background=0):
        """Generate membrane mask using morphological dilation."""
        membrane_mask = morphology.dilation(instance_mask, footprint=morphology.ball(1)) - instance_mask

        return (membrane_mask != background).astype(np.float32)


if __name__ == "__main__":    
    
	MAIN_DIR = 'data'
	plant_name = ['plant_1','plant_2','plant_4','plant_13','plant_15','plant_18','test_plant']
	STACK_INF0_3D(MAIN_DIR, plant_name,plant_idx=[6],segmentation='cellpose3d')
	STACK_INF0_Registration(MAIN_DIR, plant_name, plant_idx=[6],segmentation='cellpose3d')

