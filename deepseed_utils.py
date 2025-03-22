import tifffile, cv2
import numpy as np
import scipy.io, math
import os, copy, json, pickle
from tqdm import tqdm
import torch
import matplotlib
from  matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [20, 20]
from torchvision import datasets, transforms
from model import deepseed
# from utils import *
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff
import pdb

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def get_border_points(points):
    
        
    edges = alpha_shape(points, alpha=0.8, only_outer=True)
    idx = []
        
    for i, j in edges:
        idx.append(i)
        idx.append(j)
        
    idx = np.unique(idx)
    
    border = np.array([np.array([points[k,0],points[k,1]]) for k in idx])
#     print(border.shape)
    
    return border

def plot_slice(stack,seg,idx, show_cell_no = 1, label = None, save_path= None):

    image_slice = stack[idx]
    seg_slice = seg[idx]
    
    if label is None:
        label = np.unique(seg_slice)[1:]

    
    size = stack[idx].shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.3
    color = (0, 255, 0)
    thickness = 1
    w = 1


    img = np.zeros((size,size,3))
    img[:,:,0] = image_slice
    img[:,:,1] = image_slice
    img[:,:,2] = image_slice
    
    c = []
    for i in range(len(label)):

        loc = np.where(seg_slice==label[i])
#         print(label[i], len(loc[0]),len(loc[1]))
        c.append(determine_centroid(loc[0],loc[1]))

    c = np.array(c)


    n = c.shape[0]

    if show_cell_no==1:

        for i in range(n):

            # if i<10:
            #     color = (0, 255, 0)
            # else:
            #     color = (255, 0, 0)

            position =(int(c[i,1])-10,int(c[i,0]))
            img = cv2.putText(img , "{0}".format(label[i]),position , font, font_size, color, thickness, cv2.LINE_AA)

    else:

        for i in range(c.shape[0]):
            img[int(c[i,0])-w:int(c[i,0])+w, int(c[i,1])-w:int(c[i,1])+w] = [255,0,0]

    if save_path is None:
        imshow(img)
    else:
        plt.imshow(img)
        plt.savefig(save_path)

    return img



def determine_centroid(x,y):
    centroid = np.array([sum(x) / (len(x)+1e-10), sum(y) / (len(y)+1e-10)])
    return centroid



def get_box(x,y):
    return [np.min(x), np.max(x), np.min(y), np.max(y)]


def extract_cell(data,seg,l):
    
    box_scale = 0.5
    loc = np.where(seg==l)
    box = get_box(loc[0], loc[1])

    W,H = seg.shape 

    w = box[1] - box[0]
    h = box[3] - box[2]
    dw = int (box_scale*w/2)
    dh = int (box_scale*h/2)
    
    xl = np.clip(box[0]-dw,0,W)
    xh = np.clip(box[1]+dw,0,W)
    yl = np.clip(box[2]-dh,0,H)
    yh = np.clip(box[3]+dh,0,H)
    
    
    cell = data[xl:xh,yl:yh]
    
    return cell



def get_pairs(nbr):
    temp = copy.deepcopy(nbr)

    pairs = []
    if len(nbr)>2:
        temp.append(nbr[0])
        n = len(nbr)
    elif len(nbr)==2:
        n = 1     

    for i in range(n):
        pairs.append([temp[i],temp[i+1]])

        
    return pairs

def get_slice_info(seg_slice, background,bx):




    label = np.unique(seg_slice)
    label = np.array(list(set(label) - set([background])))

    area = []
    c = []
    neighbour = []


    for i in range(len(label)):

        loc = np.where(seg_slice==label[i])
        area.append(len(loc[0]))
        c.append(determine_centroid(loc[0],loc[1]))
        

    c = np.array(c)
    area = np.array(area)
    
    
    for i in range(len(label)):
        neighbour.append(get_neighbour(seg_slice, label[i],background,bx))

        
    return label,c,area,neighbour


def get_cell_info(seg_slice,label,background,bx):

    neighbour = get_neighbour(seg_slice, label,background,bx)
    # print(neighbour.tolist(),label)
    area = []
    c = []
    slice_labels = [label]+neighbour.tolist()
    # print(slice_labels)

    for i in range(len(slice_labels)):

        loc = np.where(seg_slice==slice_labels[i])
        area.append(len(loc[0]))
        c.append(determine_centroid(loc[0],loc[1]))
    
        
    return np.array(slice_labels),c,area,neighbour



def sort_counterclockwise(points, centre = None):
    if centre is not None:
        centre_x = centre[0]
        centre_y = centre[1]
    else:
        centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)

    angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points

def get_neighbour(seg_slice, cell_label,background,bx):
    
    size = seg_slice.shape[1]
    
    loc = np.where(seg_slice==cell_label)    
    points = np.array([np.array([loc[0][k],loc[1][k]]) for k in range(len(loc[0]))])

    boundary = np.vstack([points-[bx,0],points+[bx,0],points-[0,bx],points+[0,bx]])
    b = (np.clip(boundary[:,0],0,size),np.clip(boundary[:,1],0,size))
    nbr = np.unique(seg_slice[b])

    discard_nbr = []

    # nbr2 = copy.deepcopy(nbr)
    # for i in range(len(nbr2)):
    #     loc = np.where(seg_slice==nbr2[i])
    #     if len(loc[0])<20:
    #         nbr = np.delete(nbr,np.where(nbr==nbr2[i]))
    
    nbr = np.delete(nbr,np.where(nbr==cell_label))
    nbr = np.delete(nbr,np.where(nbr==background))
    
    
    loc_c = np.where(seg_slice==cell_label)
    cen = determine_centroid(loc_c[0],loc_c[1])
    
    c = []
    for i in range(len(nbr)):
        
        loc = np.where(seg_slice==nbr[i])
        a = determine_centroid(loc[0],loc[1])
        if seg_slice[int(a[0]),int(a[1])]!=nbr[i]:
            a = np.array([loc[0][0],loc[1][0]]) 
        
        assert seg_slice[int(a[0]),int(a[1])]==nbr[i]
        
        c.append([int(a[0]),int(a[1])])
        
        
    sorted_c = sort_counterclockwise(c,centre = cen)

    sorted_nbr = np.array([seg_slice[sorted_c[i][0],sorted_c[i][1]] for i in range(len(sorted_c))])
    

    assert np.all(nbr==np.sort(sorted_nbr))
    
    return sorted_nbr






def get_angle(a,b,c):
    x = b-a
    y = c-a
#     print(x,y)
#     print(np.dot(x,y)/( np.linalg.norm(x)* np.linalg.norm(y)))
    t = round(np.dot(x,y)/( np.linalg.norm(x)* np.linalg.norm(y)),2)
    ang = math.acos(t)
    if ang<0:
        ang = math.pi-ang

    return ang*180/math.pi


def triangle_feature(cord,area,label,c,a,b):

    assert c!=a
    assert c!=b
    assert a!=b

    # print(label)
    # print(c)
    
    idx_c = np.where(label==c)[0][0]
    idx_a = np.where(label==a)[0][0]
    idx_b = np.where(label==b)[0][0]
    
    angle = get_angle(cord[idx_c],cord[idx_a],cord[idx_b])*math.pi/180
    
    l1 = np.linalg.norm(cord[idx_c]-cord[idx_a])
    l2 = np.linalg.norm(cord[idx_c]-cord[idx_b])
    ration_l = l1/l2
    ratio_area = area[idx_a]/area[idx_b]


    F = np.array([angle,ration_l,ratio_area ]).T
    
    return F






# def neighbour_similarity_index(g1,g2):
    
#     dist = np.zeros((g1["feature"].shape[0],g2["feature"].shape[0]))
    
#     for i in range(g1["feature"].shape[0]):
#         for j in range(g2["feature"].shape[0]):
            
#             dist[i,j] = np.linalg.norm(g1["feature"][i,:]-g2["feature"][j,:])
    
#     dist_score = np.exp(-dist)
    
#     min_dist = np.min(dist)
    
#     ns_score = np.exp(-min_dist)
# #     print(dist_score)
#     a,b = np.where(dist_score>0.9)

#     return a,b


# def final_check(seg1_slice,seg2_slice,cen1,cen2,cell_a,cell_b):
    
# #     print(cell_a,cell_b)
#     a1 = len(np.where(seg1_slice==cell_a)[0])
#     a2 = len(np.where(seg2_slice==cell_b)[0])
#     ea = np.abs((a1-a2))/a1
# #     print(a1,a2)
    
#     loc = np.where(seg1_slice==cen1)
#     cen_c1 = determine_centroid(loc[0],loc[1])
    
#     loc = np.where(seg2_slice==cen2)
#     cen_c2 = determine_centroid(loc[0],loc[1])
    
    
#     loc = np.where(seg1_slice==cell_a)
#     cen_a = determine_centroid(loc[0],loc[1])
    
#     loc = np.where(seg2_slice==cell_b)
#     cen_b = determine_centroid(loc[0],loc[1])
    
#     l1 = np.linalg.norm(cen_c1-cen_a)
#     l2 = np.linalg.norm(cen_c2-cen_b)
#     el = np.abs((l1-l2))/l1
# #     print(cell_a,cell_b)
# #     print(ea,el)
#     if ea<0.15 and el<0.1 :
#         a = 1
#     else:
#         a = 0
        
#     return a

# def check_pair_correspondance(seg1_slice,seg2_slice,pair_1,pair_2):
    
    
#     for i in range(pair_1.shape[0]):
        
#         p1a = len(np.where(seg1_slice==pair_1[i,0])[0])
#         p2a = len(np.where(seg2_slice==pair_2[i,0])[0])
        
#         p1b = len(np.where(seg1_slice==pair_1[i,1])[0])
#         p2b = len(np.where(seg2_slice==pair_2[i,1])[0])
        
#         ea = np.abs(p1a-p2a)/p1a
#         eb = np.abs(p1b-p2b)/p1b
        
#         if ea>0.15 or eb>0.15:
#             temp = copy.deepcopy(pair_2[i,0])
#             pair_2[i,0] = copy.deepcopy(pair_2[i,1])
#             pair_2[i,1] = copy.deepcopy(temp)
            
            
#     return pair_1,pair_2


# def division_test(seg1_slice,seg2_slice,mother,daughter):
    
#     loc = np.where(seg1_slice==mother)
#     c = determine_centroid(loc[0],loc[1])
#     a = np.array([loc[0],loc[1]]).T - c
# #     a = get_border_points(a)
    
#     area1 =  a.shape[0]
# #     print(a.shape)
#     loc1 = np.where(seg2_slice==daughter[0])
#     loc2 = np.where(seg2_slice==daughter[1])
    
#     loc = []
#     loc.append(np.concatenate([loc1[0],loc2[0]]))
#     loc.append(np.concatenate([loc1[1],loc2[1]]))
    
    
#     c = determine_centroid(loc[0],loc[1])
#     b = np.array([loc[0],loc[1]]).T - c
# #     b = get_border_points(b)
# #     print(b.shape)
#     area2 =  b.shape[0]
    
#     ea = np.abs(area1-area2)/area1
#     ra = len(loc1[0])/len(loc2[0])
#     if ra>1:
#         ra = 1/ra
    
#     return directed_hausdorff(a, b)[0],ea,ra


def neighbour_similarity_index(g1,g2,th_neighbour_similarity):
    
    dist = np.zeros((g1["feature"].shape[0],g2["feature"].shape[0]))
    
    for i in range(g1["feature"].shape[0]):
        for j in range(g2["feature"].shape[0]):
            
            dist[i,j] = np.linalg.norm(g1["feature"][i,:]-g2["feature"][j,:])
    
    dist_score = np.exp(-dist)
    
    min_dist = np.min(dist)
    
    ns_score = np.exp(-min_dist)
#     print(dist_score)
    a,b = np.where(dist_score>th_neighbour_similarity)

    return a,b


def final_check(seg1_slice,seg2_slice,cen1,cen2,cell_a,cell_b,th_area,th_l):
    
#     print(cell_a,cell_b)
    a1 = len(np.where(seg1_slice==cell_a)[0])
    a2 = len(np.where(seg2_slice==cell_b)[0])
    ea = np.abs((a1-a2))/a1
#     print(a1,a2)
    
    loc = np.where(seg1_slice==cen1)
    cen_c1 = determine_centroid(loc[0],loc[1])
    
    loc = np.where(seg2_slice==cen2)
    cen_c2 = determine_centroid(loc[0],loc[1])
    
    
    loc = np.where(seg1_slice==cell_a)
    cen_a = determine_centroid(loc[0],loc[1])
    
    loc = np.where(seg2_slice==cell_b)
    cen_b = determine_centroid(loc[0],loc[1])
    
    l1 = np.linalg.norm(cen_c1-cen_a)
    l2 = np.linalg.norm(cen_c2-cen_b)
    el = np.abs((l1-l2))/l1
#     print(cell_a,cell_b)
#     print(ea,el)
    if ea<th_area and el<th_l :
        a = 1
    else:
        a = 0
        
    return a

def check_pair_correspondance(seg1_slice,seg2_slice,pair_1,pair_2,th_area):
    
    
    for i in range(pair_1.shape[0]):
        
        p1a = len(np.where(seg1_slice==pair_1[i,0])[0])
        p2a = len(np.where(seg2_slice==pair_2[i,0])[0])
        
        p1b = len(np.where(seg1_slice==pair_1[i,1])[0])
        p2b = len(np.where(seg2_slice==pair_2[i,1])[0])
        
        ea = np.abs(p1a-p2a)/p1a
        eb = np.abs(p1b-p2b)/p1b
        
        if ea>th_area or eb>th_area:
            temp = copy.deepcopy(pair_2[i,0])
            pair_2[i,0] = copy.deepcopy(pair_2[i,1])
            pair_2[i,1] = copy.deepcopy(temp)
            
            
    return pair_1,pair_2


def division_test(seg1_slice,seg2_slice,mother,daughter):
    
    loc = np.where(seg1_slice==mother)
    c = determine_centroid(loc[0],loc[1])
    a = np.array([loc[0],loc[1]]).T - c
#     a = get_border_points(a)
    
    area1 =  a.shape[0]
#     print(a.shape)
    loc1 = np.where(seg2_slice==daughter[0])
    loc2 = np.where(seg2_slice==daughter[1])
    
    loc = []
    loc.append(np.concatenate([loc1[0],loc2[0]]))
    loc.append(np.concatenate([loc1[1],loc2[1]]))
    
    
    c = determine_centroid(loc[0],loc[1])
    b = np.array([loc[0],loc[1]]).T - c
#     b = get_border_points(b)
#     print(b.shape)
    area2 =  b.shape[0]
    
    ea = np.abs(area1-area2)/area1
    ra = len(loc1[0])/len(loc2[0])
    if ra>1:
        ra = 1/ra
    
    return directed_hausdorff(a, b)[0],ea,ra

def match_left_cells(cen1,cen2,nbr1,nbr2,left_cells1,left_cells2,seg1_slice,seg2_slice,th_left_cells):


    F1 = []
    for l1 in left_cells1:

        a = len(np.where(seg1_slice==l1)[0])
        idx = np.where(nbr1==l1)[0][0]
        idx = [idx-1,idx,(idx+1)%len(nbr1)]
#         print(idx)
#         print(nbr1)
        trio_1 = nbr1[idx]
        ang1 = get_angle_v2(seg1_slice,cen1,trio_1[0],trio_1[1])
        ang2 = get_angle_v2(seg1_slice,cen1,trio_1[1],trio_1[2])

        f = np.array([a,ang1,ang2])
        F1.append(f)

    F1 = np.array(F1)



    F2 = []
    for l1 in left_cells2:

        a = len(np.where(seg2_slice==l1)[0])
        idx = np.where(nbr2==l1)[0][0]
        idx = [idx-1,idx,(idx+1)%len(nbr2)]
        trio_1 = nbr2[idx]
        ang1 = get_angle_v2(seg2_slice,cen2,trio_1[0],trio_1[1])
        ang2 = get_angle_v2(seg2_slice,cen2,trio_1[1],trio_1[2])

        f = np.array([a,ang1,ang2])
        F2.append(f)

    F2 = np.array(F2)

    match_mtx = np.zeros((F1.shape[0],F2.shape[0]))
    for i in range(F1.shape[0]):
        for j in range(F2.shape[0]):
            d = F1[i,:] - F2[j,:]
            d = np.abs(np.divide(d,(F1[i,:]+1e-10)))
            match_mtx[i,j] = d[0]+d[1]+d[2]


    if match_mtx.shape[0]> match_mtx.shape[1]:
        min_idx = np.argmin(match_mtx ,0)
        match_cand = [min_idx,range(match_mtx.shape[1])]

    else:
        min_idx = np.argmin(match_mtx ,1)
        match_cand = [range(match_mtx.shape[0]),min_idx]
#     print(match_cand)

    match = [[],[]]
#     print(match_mtx)
    for i in range(len(match_cand[0])):

        if match_mtx[match_cand[0][i],match_cand[1][i]]<th_left_cells:

            match[0].append(match_cand[0][i])
            match[1].append(match_cand[1][i])


    match1 = left_cells1[match[0]]
    match2 = left_cells2[match[1]]
    
    return match1,match2

def get_angle_v2(seg_slice,a,b,c):
    
    loc = np.where(seg_slice==a)
    ac = determine_centroid(loc[0],loc[1])
    
    loc = np.where(seg_slice==b)
    bc = determine_centroid(loc[0],loc[1])
    
    loc = np.where(seg_slice==c)
    cc = determine_centroid(loc[0],loc[1])
    
    
    ang = get_angle(ac,bc,cc)
    
    return ang    



def track_cells_by_seeds(seg,stack_info,stack_id,slice_id,seed_pair,threshold_values,background,bx):
    
    th_neighbour_similarity = threshold_values[0]
    th_area = threshold_values[1]
    th_l = threshold_values[2]
    th_left_cells = threshold_values[3]
    th_hausdorff_dist = threshold_values[4]
    th_ra = threshold_values[5]
    th_area_div = threshold_values[6]

    seg1_slice  = seg[stack_id][slice_id]
    seg2_slice  = seg[stack_id+1][slice_id]


    slice1_info = stack_info[stack_id][slice_id]
    slice2_info = stack_info[stack_id+1][slice_id]

    c1 = np.array([slice1_info[p]['center'] for p in range(len(slice1_info))])
    c2 = np.array([slice2_info[p]['center'] for p in range(len(slice2_info))])


    corr1_label = [seed_pair[0]]
    corr2_label = [seed_pair[1]]


    mother_label = []
    daughter_cells = []
    used_label = []

    
    cen1= seed_pair[0]
    cen2 = seed_pair[1]

    idx1 = np.where(c1==cen1)[0][0]
    idx2 = np.where(c2==cen2)[0][0]

    nbr1 = slice1_info[idx1]['nbr']
    nbr2 = slice2_info[idx2]['nbr']

#     print('Neighbour')
#     print(nbr1)
#     print(nbr2)



    # Propagration

    a,b = neighbour_similarity_index(slice1_info[idx1],slice2_info[idx2],th_neighbour_similarity)


    mc1 = np.array([slice1_info[idx1]['pair'][p1] for p1 in a])
    mc2 = np.array([slice2_info[idx2]['pair'][p1] for p1 in b])


#     mc1,mc2 = check_pair_correspondance(seg1_slice,seg2_slice,mc1,mc2,th_area)

    mc1 = mc1.flatten()
    mc2 = mc2.flatten()


    for i in range(len(mc1)):

        conf = final_check(seg1_slice,seg2_slice,cen1,cen2,mc1[i],mc2[i],th_area,th_l)
        if conf==1 and mc1[i] not in corr1_label:
            corr1_label.append(mc1[i])
            corr2_label.append(mc2[i])

    # Division
    
    div_cand1 = np.setdiff1d(nbr1,corr1_label)
    div_cand2 = np.setdiff1d(nbr2,corr2_label)
    
    # print('Division Candidate')
    # print(div_cand1)
    # print(div_cand2)
    
    division_pair = []
    daughter_label = []
    for l1 in div_cand2:

        try:
            idx = np.where(c2==l1)[0][0]
            nbr = slice2_info[idx]['nbr']
            temp = np.intersect1d(nbr, div_cand2)
            if len(temp)>0:
                for l2 in temp:
                    if [l1,l2] and [l2,l1] not in division_pair:
                        division_pair.append([l1,l2])


            z1 = []
            z2 = []

            for m1 in range(len(div_cand1)):
                for m2 in range(len(division_pair)):
                    hausdorff_dist, ea, ra = division_test(seg1_slice,seg2_slice,div_cand1[m1],division_pair[m2])                  
                    if (hausdorff_dist<th_hausdorff_dist and ea<th_area_div and ra>th_ra):
                        z1.append(m1)
                        z2.append(m2)


            m_cell = [div_cand1[p1] for p1 in z1]
            d_cells = [division_pair[p1] for p1 in z2]



            for m1 in range(len(m_cell)):
                if m_cell[m1] not in mother_label:
                    mother_label.append(m_cell[m1])
                    daughter_cells.append(d_cells[m1])
                    daughter_label.append(d_cells[m1][0])
                    daughter_label.append(d_cells[m1][1])



    #     print('Propagration')
    #     print(corr1_label)
    #     print(corr2_label)

    #     print('Division Candidate')
    #     print(mother_label)
    #     print(daughter_cells)

    #     print('Cells Left')
    #     print(left_cells1)
    #     print(left_cells2)

        except:
            xx = np.where(seg2_slice==l1)
            nbr_x = get_neighbour(seg2_slice, l1,background,bx)
            print('{0} not found which has area of {1} and nbr no : {2}'.format(l1,len(xx[0]),len(nbr_x)))

    left_cells1 = np.setdiff1d(div_cand1,mother_label)
    left_cells2 = np.setdiff1d(div_cand2,daughter_label)
    error_th=0.1
    
    while len(left_cells1)>0 and len(left_cells2) and error_th<th_left_cells:
        m_left_cells1, m_left_cells2 = match_left_cells(cen1,cen2,nbr1,nbr2,left_cells1,left_cells2,seg1_slice,seg2_slice,error_th)

        for i in range(len(m_left_cells1)):
            corr1_label.append(m_left_cells1[i])
            corr2_label.append(m_left_cells2[i])

        left_cells1 = np.setdiff1d(left_cells1,m_left_cells1)
        left_cells2 = np.setdiff1d(left_cells2,m_left_cells2)
        
        error_th = error_th+0.1
        
        if error_th>th_left_cells:
            break
#     print(error_th)
        
        
    
#     print('Propagration')
#     print(corr1_label)
#     print(corr2_label)

#     print('Division')
#     print(mother_label)
#     print(daughter_cells)
    
#     print('Cells Left')
#     print(left_cells1)
#     print(left_cells2)
            
    return corr1_label,corr2_label,mother_label,daughter_cells,left_cells1,left_cells2