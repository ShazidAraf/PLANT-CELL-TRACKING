# python result_analysis.py --result_path results/Ablation_study/two_step/n_cand_15/
# python result_analysis.py --result_path results/Ablation_study/only_pp/one_step/





import sys,os,argparse,copy
sys.path.insert(0,'graph_sim')


import numpy as np
from numpy import random

# from model import *
from data_loaders_inference import *
from utils_tracking_2 import *


def evaluate_track(tr1,tr2,plant_data):
    match = 0
    found = 0
    
    assert len(np.unique(tr1))==len(np.unique(tr2))

    for i in range(len(plant_data.gt_l0)):
        idx1 = np.where(tr1==plant_data.gt_l0[i])[0]

        if len(idx1)==1:
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



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ckp', type=str, default='check_points')
    parser.add_argument('--path', type=str, default= '/home/eegrad/mdislam/SAM/Graph_similarity_new/data')
    parser.add_argument('--result_path', type=str, default= 'result3')
    parser.add_argument('--n_points', type=int, default= 15)
    parser.add_argument('--k', type=int, default= 6)
    parser.add_argument('--n_cand', type=int, default= 20)
    parser.add_argument('--max_dist', type=int, default= 15)
    parser.add_argument('--v_size', type=int, default= 100)
    parser.add_argument('--cell_div_th', type=float, default= 0.8)
    parser.add_argument('--sched', type=str)
    args = parser.parse_args()

    return args

args = parse_args()
# args.result_path = os.path.join('data',args.result_path)
print(args)


plant_idx = 5
all_stack_idx = range(20)
# all_stack_idx = np.array([0,1,7,9,15,18,19])

MAIN_DIR = args.path
result_dir = args.result_path

TP_track_all = 0
TP_div_all = 0
FP_track_all = 0
FP_div_all = 0
FN_track_all = 0
FN_div_all = 0




for t1 in tqdm(all_stack_idx):

	# print("Time: ", t1 )

	cell_div_data = mother_daughter_pair_loader(MAIN_DIR,plant_idx,t1,t1+1,n_cand = args.n_cand)
	plant_data = Plant_data_loader(MAIN_DIR,plant_idx= plant_idx,t1 = t1,t2 = t1+1, n_points=args.n_points-1,k=args.k,n_cand = args.n_cand,max_dist=args.max_dist)


	n1 = len(plant_data.gt_l0)
	n2 = plant_data.__len__()

	all_div_info = np.load('{0}/all_divisions_stack_{1}.npy'.format(result_dir,t1))

	# print(all_div_info.shape)



	mother = all_div_info[:,0]
	daughter = all_div_info[:,1:3]

	all_track_info = np.load('{0}/all_tracks_stack_{1}.npy'.format(result_dir,t1))

	# print(all_track_info.shape)

	tr_0 = all_track_info[:,0]
	tr_1 = all_track_info[:,1]


	TP_div,found_div,gt_div = evaluate_division(mother,daughter,cell_div_data)
	# print('DIVISION: ', TP_div,found_div,gt_div)
	TP_track,found_track,gt_track = evaluate_track(tr_0,tr_1,plant_data)
	# print('Track: ', TP_track,found_track,gt_track)


	FP_div = found_div - TP_div
	FP_track = found_track - TP_track


	FN_div = gt_div - found_div
	FN_track = gt_track - found_track


	precision_track = TP_track/(TP_track+FP_track+1e-30)
	precision_div = TP_div/(TP_div+FP_div+1e-30)

	recall_track = TP_track/(TP_track+FN_track+1e-30)
	recall_div = TP_div/(TP_div+FN_div+1e-30)

	F1_track = 2*precision_track*recall_track/(precision_track + recall_track+1e-30)
	F1_div = 2*precision_div*recall_div/(precision_div + recall_div+1e-30)

	# print('precision track',precision_track)
	# print('recall track',recall_track)
	# print('F1 score track',F1_track)


	# print('precision div',precision_div)
	# print('recall div',recall_div)
	# print('F1 score div',F1_div)



	TP_track_all += TP_track
	TP_div_all  +=TP_div
	FP_track_all += FP_track
	FP_div_all += FP_div
	FN_track_all += FN_track
	FN_div_all += FN_div





precision_track_all = TP_track_all/(TP_track_all+FP_track_all+1e-30)
precision_div_all = TP_div_all/(TP_div_all+FP_div_all+1e-30)

recall_track_all = TP_track_all/(TP_track_all+FN_track_all+1e-30)
recall_div_all = TP_div_all/(TP_div_all+FN_div_all+1e-30)

F1_track_all = 2*precision_track_all*recall_track_all/(precision_track_all + recall_track_all+1e-30)
F1_div_all = 2*precision_div_all*recall_div_all/(precision_div_all + recall_div_all+1e-30)


print('precision track all',precision_track_all)
print('recall track all',recall_track_all)
print('F1 score track all',F1_track_all)


print('precision div all',precision_div_all)
print('recall div all',recall_div_all)
print('F1 score div all',F1_div_all)
