import tifffile, cv2
import numpy as np
import scipy.io, math
import os, copy, json, pickle
from tqdm import tqdm
import torch
import matplotlib
from  matplotlib.pyplot import imshow
matplotlib.rcParams['figure.figsize'] = [20, 20]
from torchvision import datasets, transforms
from model import deepseed
from deepseed_utils import *


main_dir = 'data/plant18/cellpose3D'
membrane_dir = os.path.join(main_dir,'membrane')
seg_dir = os.path.join(main_dir,'seg')
save_dir =   os.path.join(main_dir,'stacks')
os.makedirs(save_dir ,exist_ok=True)

membrane_files = sorted([f for f in os.listdir(membrane_dir) if f.endswith('.tif')])
# print(membrane_files)

seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.tif')])
# print(seg_files)


for i in tqdm(range(len(seg_files))):

	seg = tifffile.imread(os.path.join(seg_dir,seg_files[i]))
	ws = tifffile.imread(os.path.join(membrane_dir,membrane_files[i]))

	print(seg_files[i],membrane_files[i])

	label = np.unique(seg)[1:]
	all_cell_info =  []
	 

	for j in tqdm(range(len(label))):
		loc = np.where(seg==label[j])
		nz = np.unique(loc[0])

		# a = []

		# for k in nz:
		# 	x = np.where(seg[k]==label[j])
		# 	a.append(len(x[0]))

		# a = np.array(a)
		# idx = np.where(a==np.max(a))[0][0]

		idx = (nz[0]+nz[-1])//2

		seg_slice = seg[idx]
		ws_slice = ws[idx]

		a2 = np.where(seg_slice==label[j])



		slice_labels,c, area,nbr =  get_cell_info(seg_slice,label[j],background=0, bx = 1)

		if area[0]>100 and len(nbr)>=4:

			pair = get_pairs(nbr.tolist())
			F = []
			for k in range(len(pair)):
				# print(label[j])
				# print(slice_labels)
				F.append(triangle_feature(c,area,slice_labels,label[j],pair[k][0],pair[k][1]))

			cell = extract_cell(ws_slice,seg_slice,label[j])

			loc = np.where(seg_slice ==label[j])
			center_cord = determine_centroid(loc[0],loc[1]) 


			record = {}
			record["center"] = label[j]
			record["center_cord"] = center_cord
			record["nbr"] = nbr
			record["pair"] = np.array(pair)
			record["feature"] = np.array(F)
			record["area"] = area[0]
			record["cell"] = cell 
			all_cell_info.append(record)


	# with open(os.path.join(save_dir,'stack_{0}.pkl'.format(str(i).zfill(2))), 'wb') as f:
	#     pickle.dump(all_cell_info,f, protocol=pickle.HIGHEST_PROTOCOL)






