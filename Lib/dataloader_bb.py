# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 02:04:07 2019

@author: Nazari
"""
import os.path
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.io
import glob as G

class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass
    
    
    @property
    def data(self):
        return self._data
    
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


Path="/home/nazari/Desktop/compcars/data/data"

FileNametype="/misc/attributes.txt"
FileName2="/train_test_split/classification/train.txt"
with open(Path+FileNametype) as f:
    car_data = f.readlines()
    
car_data.pop(0)
car_data = [x.strip() for x in car_data]
car_data = [x.split() for x in car_data]
car_data = [[x[0],x[5]] for x in car_data]
car_data = dict(car_data)

#/home/nazari/Desktop/compcars/data/data/train_test_split/classification
FileName="/train_test_split/classification/train.txt"
with open(Path+FileName) as f:
    fnames = f.readlines()
fnames = [x.strip() for x in fnames]

idx=np.arange(0, len(fnames))
dataset = Dataset(idx)

def get_data(batch_size=32,fnames=fnames,Path=Path,car_data=car_data):
    images =[]
    recs = np.zeros((batch_size,4), dtype=np.float32)
    poses = np.zeros((batch_size), dtype=np.int32)
    car_types = np.zeros((batch_size), dtype=np.int32)
    
    crop_img_fvpn = np.zeros((batch_size,60,60,3))
    crop_img_aln  = np.zeros((batch_size,224,224,3))
    
    batch_idx=dataset.next_batch(batch_size)
    idx = 0
    for b in batch_idx:
        Fname = fnames[b]   
        tmp=Fname.split('/')
        tmp_car_type=int(car_data[tmp[1]])
        
        if (tmp_car_type in [0,1,2,3,4]) or (tmp_car_type==7):
            car_type=tmp_car_type
            if (tmp_car_type==7):
                car_type=5
        else:
            car_type=6
        
        
        
        
        
        fn1=Path+"/image/"+Fname
        fn2=Path+"/label/"+Fname[:-3]+"txt"
        with open(fn2) as f:
            data = f.readlines()
        data = [x.strip() for x in data]
        pose=int(data[0])-1
        rec=(np.array(data[2].split())).astype(np.uint16)
        img = cv2.imread(fn1)
        images.append(list(img))
        crop_img = img[rec[1]:rec[1]+rec[3], rec[0]:rec[0]+rec[2]]
        crop_img_1 = cv2.resize(crop_img,(60,60))
        crop_img_2 = cv2.resize(crop_img,(224,224))
        
        rec=rec.astype(np.float32)
        rec[1]=rec[1]/img.shape[0]
        rec[3]=rec[3]/img.shape[0]
        rec[0]=rec[0]/img.shape[1]
        rec[2]=rec[2]/img.shape[1]
        
        
        recs[idx,...] = rec
        poses[idx,...] = pose
        car_types[idx,...] = car_type
        crop_img_fvpn[idx,...] =  crop_img_1
        crop_img_aln[idx,...]  =  crop_img_2             
        idx += 1
        
    return images,crop_img_fvpn,crop_img_aln,poses,car_types,recs


mat_file = '/home/nazari/Desktop/compcars/data/data/sh/color_list.mat'
mat = scipy.io.loadmat(mat_file,struct_as_record=False, squeeze_me=True)

tmp = mat['color_list']
color_matrix = dict(tmp)
pth = '/home/nazari/Desktop/compcars/sv_data/sv_data/image/*/*.jpg'
color_files = G.glob(pth)
def get_color_data(batch_size=32,color_matrix=color_matrix,color_files=color_files):
    colors = np.zeros((batch_size), dtype=np.int32)
    images = np.zeros((batch_size,224,224,3))
    idxs = np.random.choice(len(color_files),batch_size,replace=False)
    fns = np.array(color_files)[idxs]
    idx = 0
    for g in fns:
#       while not os.path.isfile(g):
#           tmp_idx = np.random.choice(len(color_files),1,replace=False)
#           g = np.array(color_files)[tmp_idx]
       tst_loop = True
#       t_img = cv2.imread(g)
       while tst_loop:
           try:
               t_img = cv2.imread(g)
               t_img = cv2.resize(t_img,(224,224))
               tst_loop = False
           except:
               tmp_idx = 0#np.random.choice(len(color_files),1,replace=False)
               g = np.array(color_files)[tmp_idx]
           
           
           
       tmp=g.split('/')
       fn_srch = tmp[-2]+'/'+tmp[-1]
       color = color_matrix[fn_srch]
       
       
       if color<0:
           color = 10
       colors[idx,...] = color
       images[idx,...] = t_img
       idx += 1 
    return images,colors

pth2 = '/home/nazari/Desktop/compcars/images_street/*.*'
background_files = G.glob(pth2)

def get_background_data(batch_size=32,background_files=background_files):
    
    images =[]
    recs = np.zeros((batch_size,4), dtype=np.float32)
    car_types = np.zeros((batch_size), dtype=np.int32)    
    crop_img_fvpn = np.zeros((batch_size,60,60,3))
    crop_img_aln  = np.zeros((batch_size,224,224,3))
    poses = np.zeros((batch_size), dtype=np.int32)

    idxs = np.random.choice(len(background_files),batch_size,replace=False)
    fns = np.array(background_files)[idxs]
    idx = 0
    for g in fns:
        tst_loop = True
        while tst_loop:
           try:
               t_img = cv2.imread(g)
               bk_img_2 = cv2.resize(t_img,(224,224))
               tst_loop = False
           except:
               tmp_idx = 0#np.random.choice(len(color_files),1,replace=False)
               g = np.array(color_files)[tmp_idx]
     
#        t_img = cv2.imread(g)
        images.append(list(t_img))
        bk_img_1 = cv2.resize(t_img,(60,60))
#        bk_img_2 = cv2.resize(t_img,(224,224))        
        crop_img_fvpn[idx,...] =  bk_img_1
        crop_img_aln[idx,...]  =  bk_img_2                   
        idx += 1
       
    return images,crop_img_fvpn,crop_img_aln,poses,car_types,recs

#img,crop_img,pose,car_type=get_data()
#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(crop_img)
#print("pose=> ", pose)
#print("car_type=> ", car_type)
##import time
##time.sleep(3)
#
#for i in range(10):
#    batch_idx=dataset.next_batch(5)
#    batch_file_names=[fnames[b] for b in batch_idx]
#    print(batch_file_names)
#    print('******************************')
    
    
    