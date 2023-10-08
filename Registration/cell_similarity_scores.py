import tifffile, cv2
import numpy as np
import scipy.io, math
import os, copy, json, pickle
# import pickle5 as pickle
from tqdm import tqdm
import torch
import matplotlib
from  matplotlib.pyplot import imshow
matplotlib.rcParams['figure.figsize'] = [20, 20]
from torchvision import datasets, transforms
from model import deepseed
from deepseed_utils import *


model = deepseed(in_dim = 3,out_dim = 32)
model = model.cuda()
model.eval()
pretrained_path = 'trained_model/best_ckp.pt'

print('Loading pretrained weights from {0}...'.format(pretrained_path))
model.load_state_dict(torch.load(pretrained_path))

transform = transforms.Compose(
[
    transforms.Resize(50),
    transforms.CenterCrop(50),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])

])

def neighbour_similarity_score(g1,g2):
    
    dist = np.zeros((g1["feature"].shape[0],g2["feature"].shape[0]))+100
    
    for i in range(g1["feature"].shape[0]):
        for j in range(g2["feature"].shape[0]):
            
            dist[i,j] = np.linalg.norm(g1["feature"][i,:]-g2["feature"][j,:])
            
    min_dist = np.min(dist)
    ns_score = np.exp(-min_dist)

    return ns_score


def cell_similarity_score(cell1,cell2):

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



main_dir = 'data/plant18'
seg_dir = os.path.join(main_dir,'seg')
stack_dir =   os.path.join(main_dir,'cellpose3D/stacks')
save_dir =   os.path.join(main_dir,'cellpose3D/seed_pairs')
os.makedirs(save_dir,exist_ok=True)

stack_files = sorted([f for f in os.listdir(stack_dir) if f.endswith('.pkl')])
# print(stack_files)



# for i in tqdm(range(len(stack_files)-1)):
for i in tqdm(range(len(stack_files)-1)):


    seed_pairs = []


    f0 = os.path.join(stack_dir,stack_files[i])
    f1 = os.path.join(stack_dir,stack_files[i+1])
    print(f0)
    print(f1)
    with (open(f0, "rb")) as f:
        stack0 = pickle.load(f)

    with (open(f1, "rb")) as f:
        stack1 = pickle.load(f)


    similarity_score = np.zeros((len(stack0),len(stack1)))
    cell_similarity_matrix = np.zeros((len(stack0),len(stack1)))
    neighbouring_similarity_matrix = np.zeros((len(stack0),len(stack1)))

    for j in tqdm(range(len(stack0))):
        for k in range(len(stack1)):

            cell1 = 1.0*stack0[j]["cell"]
            cell2 = 1.0*stack1[k]["cell"]

            cell_similarity_matrix[j,k] = cell_similarity_score(cell1,cell2)
            neighbouring_similarity_matrix[j,k] = neighbour_similarity_score(stack0[j],stack1[k]) 

#             similarity_score[j,k] = 0.67*neighbour_similarity_score(stack0[j],stack1[k])+0.33*cell_similarity_score(cell1,cell2)


    # similarity_score = 0.67*neighbouring_similarity_matrix+0.33*cell_similarity_matrix
    # a1,b1 = get_seed_pairs(similarity_score,0.93)

    # for l in range(len(a1)):
        # seed_pairs.append(np.array([stack0[a1[l]]["center"],stack1[b1[l]]["center"],similarity_score[a1[l],b1[l]]]))

    # seed_pairs = np.array(seed_pairs)
    # print(seed_pairs.shape)

    # np.save(os.path.join(save_dir ,'seed_{0}.npy'.format(str(i).zfill(2))), seed_pairs)
    np.save(os.path.join(save_dir ,'cell_similarity_{0}.npy'.format(str(i).zfill(2))), cell_similarity_matrix)
    np.save(os.path.join(save_dir ,'neighbouring_similarity_{0}.npy'.format(str(i).zfill(2))), neighbouring_similarity_matrix)
