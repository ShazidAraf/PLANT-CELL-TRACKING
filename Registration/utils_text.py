import numpy as np
import tifffile, os, cv2, copy
from tqdm import tqdm


def read_PNAS_gt(text_file,text_dir=None,lines=None):


    if lines==None:
        if text_dir==None:
            with open(text_file) as f:
                lines = f.readlines()
        else:
            with open(os.path.join(text_dir,text_file)) as f:
                lines = f.readlines()


    data = []

    for i in range(len(lines)):
        data.append(lines[i].split(':'))

#     data_t0 = np.array([int(d[0]) for d in data])
    data_t0 = [int(d[0])-1 for d in data]
    data_t1 = [d[1].split(',') for d in data]

    for i in range(len(data_t1)):

        d = copy.deepcopy(data_t1[i])

        for j in range(len(d)):
            d[j] = int(d[j])-1

        data_t1[i] = copy.deepcopy(d)
        
    return data_t0,data_t1


def one_timepoint_info(tra_t0,tra_t1,i):

    tracking_info = []
    m_d = []
    propagate = []
    for j in tqdm(range(len(tra_t0))):

#         loc0 = np.where(segm_t0==tra_t0[j])

        if len(tra_t1[j])==1:
            d = [tra_t0[j],i,i+1,0]
            tracking_info.append(copy.deepcopy(d))
            propagate.append([tra_t0[j],tra_t1[j][0]])

        else:
            d = [tra_t0[j],i,i,0]
            tracking_info.append(copy.deepcopy(d))

            for k in range(len(tra_t1[j])):
                d = [tra_t1[j][k],i+1,i+1,tra_t0[j] ]
                m_d.append(copy.deepcopy(d))
                
                
    return tracking_info,propagate,m_d

def change_label(img,old_label,new_label):
    
    loc = np.where(img==old_label)
    img[loc[0],loc[1],loc[2]] = new_label
    
    return img
    
    
def text_to_label(text_files,text_dir,segm_id):
    
    
    if segm_id==0:
        a0,b0 = read_PNAS_gt(text_files[segm_id],text_dir)
        labels = a0
        # print(a0)
    elif segm_id>0 and segm_id<len(text_files):
        a0,b0 = read_PNAS_gt(text_files[segm_id-1],text_dir)
        a1,b1 = read_PNAS_gt(text_files[segm_id],text_dir)
        
        labels = []
        for j in range(len(b0)):
            for k in range(len(b0[j])):
                labels.append(b0[j][k])
                
        for j in range(len(a1)):
            labels.append(a1[j])
            
    elif segm_id==len(text_files):
        # print(i)

        a0,b0 = read_PNAS_gt(text_files[segm_id-1],text_dir)        
        labels = []
        for j in range(len(b0)):
            for k in range(len(b0[j])):
                labels.append(b0[j][k])

    labels = np.unique(labels)
    
    return labels
    
    
    
    
def create_text(a,b):
    
    lines = []
    for i in range(len(a)):
        
        x = ','.join([str(f) for f in b[i]])
#         print(x)
        y = '{0}: {1}'.format(a[i],x)
#         print(y)
        lines.append(y)
#     print(lines)
    return lines

def replace_b(b,old_label,new_label):
    
    for i in range(len(b)):
        for j in range(len(b[i])):
            if b[i][j]==old_label:
                b[i][j] = copy.deepcopy(new_label)
                
                
    return b
 
    
def replace_a(a,old_label,new_label):
    
    for i in range(len(a)):
        if a[i]==old_label:
            a[i] = copy.deepcopy(new_label)
                
                
    return a


def find_val(list_2D,col_idx,val):
    
    k = np.where(np.array(list_2D)[:,col_idx]==val)[0]
    if len(k)==0:
        k = -1
    else:
        k = k[0]
    
    return k
    
    
