import numpy as np
import copy, os, time
import tifffile
import pickle, json
import open3d as o3d
from tqdm import tqdm
from itertools import combinations
from numpy import random
import h5py
import scipy, pdb
from scipy.spatial.distance import directed_hausdorff
color = random.rand(2000,3)



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
    
    # pdb.set_trace()
    # all_tracks = np.zeros((100000,all_corr_labels.shape[0]+1))
    all_tracks = np.zeros((100000,len(all_corr_labels)+1))
    

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



def visualize2(cen0,cen1,point_cloud0,point_cloud1,neighbour0,neighbour1,n_samp = 10):


    # Visiualize t0 cells with corresponding t1 neighbours

    cand_t1 = []
#     len(point_cloud0)
    for i in tqdm(range(n_samp)):

        all_p = []
        nbr = neighbour0[i]
        
        for k in range(len(nbr)):
            all_p.append(point_cloud0[nbr[k]])

#         o3d.visualization.draw_geometries(all_p)


        a = np.linalg.norm(cen_1 - cen_0[i],axis=1)
        y = []
        
        for k in range(len(point_cloud1)):
            if a[k]<60:
                y.append(k)
                
                
        for j in range(len(y)):
            
            nbr = neighbour1[y[j]]
            for k in range(len(nbr)):
                
                pcd = copy.deepcopy(point_cloud1[nbr[k]])
                pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)-np.array([300,0,0]))
                
                all_p.append(pcd)
            

        o3d.visualization.draw_geometries(all_p)


        
def visualize3(point_cloud,neighbour,idx):
    

    nbr = neighbour[idx]
    all_p = []

    for k in range(len(nbr)):

        all_p.append(point_cloud[nbr[k]])

    o3d.visualization.draw_geometries(all_p)


def get_feature(corr_polar,l):
    
    p0 = copy.deepcopy(corr_polar[l[0]])
    p1 = copy.deepcopy(corr_polar[l[1]])

    f0 = p1[0]/(p0[0]+1e-20)
    f1 = p1[1]-p0[1]
    if f1<0:
        f1 = np.pi+f1

    f2 = p1[2]-p0[2]

    if f2<-np.pi:
        f2 = f2+2*np.pi
    elif f2> np.pi:
        f2 = f2-2*np.pi
        
    f = np.array([f0,f1,f2])
#     r = np.array([p1[0]/(p0[0]+1e-20), p1[1]/(p0[1]+1e-20), p1[2]/(p0[2]+1e-20) ])
#     e = np.array([(p1[0]-p0[0])/(p0[0]+1e-20), (p1[1]-p0[1])/(p0[1]+1e-20), (p1[2]-p0[2])/(p0[2]+1e-20) ])
    
    return f

    
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



def stack_point_cloud(seg,th_size=10,transformation=None,label=None):
    
    seg_temp=copy.deepcopy(seg)
    
    if transformation is None:
        transformation = np.eye(4)
    
    
    if label is None:
        label = np.unique(seg_temp)[1:]
    
    point_cloud = []
    new_label = []
    
    for i in tqdm(range(len(label))):
        
        pc = np.where(seg_temp==label[i])
        
        if len(pc[0])>th_size:
            pc = np.array(pc).T
            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.paint_uniform_color(color[i])
            pcd.transform(transformation)
            point_cloud.append(pcd)
            
            new_label.append(label[i])

    

    return point_cloud,new_label


def concatenate_point_cloud(seg,label,transformation=None):
    
    seg_temp=copy.deepcopy(seg)
    
    if transformation is None:
        transformation = np.eye(4)
    
    
    if label is None:
        label = np.unique(seg_temp)[1:]
    
    point_cloud = np.empty((0,3))
    
    for i in range(len(label)):
        
        pc = np.where(seg_temp==label[i])
        point_cloud = np.append(point_cloud,np.array(pc).T,axis=0)
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.transform(transformation)

        
    return pcd



def points_to_pcd(points):
    
    point_cloud = []
    boundary = []
    
    for i in range(len(points)):
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i])
        pcd.paint_uniform_color(color[i])
        
        min_points = np.min(points[i],axis=0)
        max_points = np.max(points[i],axis=0)

        
        
        point_cloud.append(pcd)
        boundary.append(np.array([min_points,max_points]))
        
    return point_cloud, boundary

def get_box_points(boundary):
    
    box = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                box.append(np.array([boundary[i,0],boundary[j,1],boundary[k,2]]))
            
    box = np.array(box)
    return box


def get_center_cloud_and_volume(point_cloud):
    
    center_cloud = np.empty((0,3))
    V = []
    for i in range(len(point_cloud)):
        
        p = np.array(point_cloud[i].points)
        pc = np.mean(p,axis=0)
        pc = np.resize(pc,(1,3))
        center_cloud = np.append(center_cloud,pc,axis=0)
        V.append(p.shape[0])
    
    return center_cloud, np.array(V)




# def get_neighbouring_structure_dist(point_cloud,center,d):
    
#     neighbour = [] 
    
#     for i in tqdm(range(len(point_cloud))):
        
#         a = np.linalg.norm(center - center[i],axis=1)
#         nbr = []
#         for j in range(len(point_cloud)):
#             if i!=j:
            
#                 if a[j]<d:
#                     nbr.append(j)

#         nbr.append(i)
#         neighbour.append(nbr)    
        
#     return neighbour


# def create_graph(cen,idx,shift=[0,0,0],n_points = 200):
    
    
#     all_c = []
#     edges = np.empty((0,3)) 
    
#     c00 = cen[idx[-1]]

#     for i in range(len(idx)):
        
#         c = cen[idx[i]]
#         all_c.append(c)
        
#         ed = np.linspace((c[0],c[1],c[2]),(c00[0],c00[1],c00[2]),n_points)
#         edges = np.append(edges,ed,axis=0)
        

     
#     all_c = np.array(all_c)
#     pcd_c = o3d.geometry.PointCloud()
#     pcd_c.points = o3d.utility.Vector3dVector(all_c+shift)
#     pcd_c.paint_uniform_color([1,0,0])
    
    
#     pcd_e = o3d.geometry.PointCloud()
#     pcd_e.points = o3d.utility.Vector3dVector(edges+shift)
#     pcd_e.paint_uniform_color([0,0,1])
    
#     all_pcd = [pcd_c,pcd_e]
#     return all_pcd

def get_neighbouring_structure_dist(point_cloud,center,d):
    
    neighbour = [] 
    
    for i in tqdm(range(len(point_cloud))):
        
        a = np.linalg.norm(center - center[i],axis=1)
        nbr = []
        for j in range(len(point_cloud)):
            
            if a[j]<d:
                nbr.append(j)

        neighbour.append(nbr)    
        
    return neighbour


def create_graph(cen,idx,shift=[0,0,0],n_points = 200):
    
    
    all_c = []
    edges = np.empty((0,3)) 
    
    c00 = cen[idx[0]]

    for i in range(len(idx)):
        
        c = cen[idx[i]]
        all_c.append(c)
        
        ed = np.linspace((c[0],c[1],c[2]),(c00[0],c00[1],c00[2]),n_points)
        edges = np.append(edges,ed,axis=0)
        

     
    all_c = np.array(all_c)
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(all_c+shift)
    pcd_c.paint_uniform_color([1,0,0])
    
    
    pcd_e = o3d.geometry.PointCloud()
    pcd_e.points = o3d.utility.Vector3dVector(edges+shift)
    pcd_e.paint_uniform_color([0,0,1])
    
    all_pcd = [pcd_c,pcd_e]
    return all_pcd




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


def point_cloud_to_mesh(pcd):


    pcd.estimate_normals()

    # to obtain a consistent normal orientation
    pcd.orient_normals_towards_camera_location(pcd.get_center())

    # or you might want to flip the normals to make them point outward, not mandatory
#     pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

    # surface reconstruction using Poisson reconstruction
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = mesh.vertices
    
    
#     keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_mesh)
    

    
#     pcd_mesh.estimate_normals()
    
    return pcd_mesh



def detect_division(info,daughter_cand,dist_hd):
    
    b0,b1,point_cloud1,V0,V1,idx0 = info
    
    mother =  []
    daughter = []
    score = []
    x = []

    pcd_c0 = copy.deepcopy(b0[idx0])

    v0 = V0[idx0]
    
    comb = combinations(daughter_cand, 2)
    # print(daughter_cand)
    for pair in list(comb):

        # pront(pair)
        
        idx1 = pair[0]
        idx2 = pair[1]

        
        v1 = V1[idx1]
        v2 = V1[idx2]
        
        if v1/v0<0.9 and v2/v0<0.9 and (v1+v2)/v0<1.45:
            
            pcd_c1 = copy.deepcopy(b1[idx1])
            pcd_c2 = copy.deepcopy(b1[idx2])

            pcd_c12 = np.concatenate((pcd_c1,pcd_c2),axis=0)


            # pcd_c0 = pcd_c0 - np.mean(pcd_c0,axis=0)
            # pcd_c12 = pcd_c12 - np.mean(pcd_c12,axis=0)
            # dist = directed_hausdorff(pcd_c0,pcd_c12)[0]

            dist1 = determine_haussdorff_distance(pcd_c0,pcd_c12)

            if dist1<5:

                mesh1 = b0[idx0]

                pcd11 = point_cloud1[idx1]
                pcd12 = point_cloud1[idx2]

                mesh2 = point_cloud_to_mesh(pcd11+pcd12)
                mesh2 = np.array(mesh2.points)

                dist2 = determine_haussdorff_distance(mesh1,mesh2)

                if dist2<dist_hd:
                    score.append(dist2)
                    x.append(np.array([pair[0],pair[1]]))
                
                
            

    if len(score)>0:
        ind = np.argmin(score)
        mother = np.array([idx0])
        daughter = x[ind]
        score = np.min(score)
        
        
    return mother, daughter, score


def determine_haussdorff_distance(b0,b1):
    
    # b0 = np.array(mesh0.points)
    # b1 = np.array(mesh1.points)
    
    b0 = b0 - np.mean(b0,axis=0)
    b1 = b1 - np.mean(b1,axis=0)

                               
    
    dist = directed_hausdorff(b0,b1)[0]
                               
    
    return dist


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

def create_graph_knn(cen,idx,shift=[0,0,0],T = np.eye(4),k=5,n_points = 200,draw_graph=False):
    

    all_c = []

    for i in range(len(idx)):
        
        c = cen[idx[i]]
        all_c.append(c)

    all_c = np.array(all_c)
    pcd_c = points_to_pcd(all_c)
    pcd_c,_ = pcd_operation(pcd_c,T, shift)
    all_pcd = []


    if draw_graph:
        all_pcd = draw_graphs(all_c,T,'make_center_zero',k,n_points)

    return np.array(pcd_c.points), all_pcd






# def create_graph_knn(cen,idx,shift=[0,0,0],T = np.eye(4), n_points = 200,k=5,draw_graph=False):
    
    
    
#     star_graph_edges = np.empty((0,3))
#     # c00 = cen[idx[0]]
#     all_c = []

#     for i in range(len(idx)):
        
#         c = cen[idx[i]]
#         all_c.append(c)

#         if draw_graph:
#             ed = np.linspace((c[0],c[1],c[2]),(c00[0],c00[1],c00[2]),n_points)
#             star_graph_edges = np.append(star_graph_edges,ed,axis=0)

#     if draw_graph:
        
#     all_c = np.array(all_c)
#     edges = np.empty((0,3))
    
#     if graph:
#         for i in range(len(all_c)):
            
#             c00 = all_c[i]
#             k_idx = get_k_nearest_candidates(c00,all_c,k)
            
#             for j in range(len(k_idx)):
                    
#                 c = all_c[k_idx[j]]
#                 ed = np.linspace((c[0],c[1],c[2]),(c00[0],c00[1],c00[2]),n_points)
#                 edges = np.append(edges,ed,axis=0)
            
#     pcd_c = points_to_pcd(all_c)
#     pcd_c,shift = pcd_operation(pcd_c,T, 'make_center_zero')
#     # print(shift)


#     all_pcd = [pcd_c]

#     if graph:

#         pcd_e = points_to_pcd(edges)
#         pcd_e,_ = pcd_operation(pcd_e,T,shift)

#         star_pcd_e = points_to_pcd(star_graph_edges)
#         star_pcd_e,_ = pcd_operation(star_pcd_e,T,shift)

#         all_pcd.append(pcd_e)
#         all_pcd.append(star_pcd_e)

#     return all_pcd

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


    # gt = np.zeros(n)-100
    # gt[0:n1] = np.arange(n1)

    gt_mtx = np.zeros((n,n))
    gt_mtx[0:n1,0:n1] = 1

    gt = np.zeros(n)
    gt[0:n1] = 1
    



    return pcd_1,pcd_2,gt,gt_mtx

def super_impose_graph(points1,points2):

    graph_0 = draw_graphs(points1,np.eye(4),np.array([0,0,0]),5,200)
    graph_1 = draw_graphs(points2,np.eye(4),np.array([0,0,0]),5,200)

    # pcd_c1 = points_to_pcd(points1)
    # pcd_c2 = points_to_pcd(points2)

    graph_0[0].paint_uniform_color([0,0,1])
    graph_0[1].paint_uniform_color([0,0,1])
    graph_0[2].paint_uniform_color([0,0,1])

    graph_1[0].paint_uniform_color([1,0,0])
    graph_1[1].paint_uniform_color([1,0,0])
    graph_1[2].paint_uniform_color([1,0,0])

    o3d.visualization.draw_geometries([graph_0[0],graph_1[0]])
    o3d.visualization.draw_geometries([graph_0[1],graph_1[1]])
    o3d.visualization.draw_geometries([graph_0[2],graph_1[2]])

def draw_graphs(all_c,T,shift,k,n_points):

    edges = np.empty((0,3))
    star_graph_edges = np.empty((0,3))
    c0 = all_c[0]


    for i in range(all_c.shape[0]):
        
        c00 = all_c[i]

        ed_s = np.linspace((c0[0],c0[1],c0[2]),(c00[0],c00[1],c00[2]),n_points)
        star_graph_edges = np.append(star_graph_edges,ed_s,axis=0)

        k_idx = get_k_nearest_candidates(c00,all_c,k)
        
        for j in range(len(k_idx)):
                
            c = all_c[k_idx[j]]
            ed = np.linspace((c[0],c[1],c[2]),(c00[0],c00[1],c00[2]),n_points)
            edges = np.append(edges,ed,axis=0)
            
    pcd_c = points_to_pcd(all_c)
    pcd_c,shift = pcd_operation(pcd_c,T, shift)

    

    pcd_e = points_to_pcd(edges)
    pcd_e,_ = pcd_operation(pcd_e,T,shift)

    star_pcd_e = points_to_pcd(star_graph_edges)
    star_pcd_e,_ = pcd_operation(star_pcd_e,T,shift)

    all_pcd = [pcd_c,pcd_e,star_pcd_e]

    return all_pcd






def pcd_operation(pcd,T,shift):

    pcd.transform(T)

    if shift=='make_center_zero':
        shift = -pcd.points[0]

    points = pcd.points+ shift

    return points_to_pcd(points),shift


def points_to_pcd(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd



def get_neighbouring_structure_knn(center,k):
    
    neighbour = [] 
    
    for i in range(len(center)):
        
        a = np.linalg.norm(center - center[i],axis=1)
        nbr = np.argsort(a)[0:k+1].tolist()
        neighbour.append(nbr)    
        
    return neighbour



def point_to_boundary(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    # to obtain a consistent normal orientation
    pcd.orient_normals_towards_camera_location(pcd.get_center())


    # surface reconstruction using Poisson reconstruction
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = mesh.vertices
    
    
    return np.array(pcd_mesh.points)