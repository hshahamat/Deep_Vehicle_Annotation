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

def chk_type_freq(fnames=fnames,Path=Path,car_data=car_data):
    labels = []
    for b in range(len(fnames)):
        Fname = fnames[b]   
        tmp=Fname.split('/')
        tmp_car_type=int(car_data[tmp[1]])
        
        if (tmp_car_type in [0,1,2,3,4]) or (tmp_car_type==7):
            car_type=tmp_car_type
            if (tmp_car_type==7):
                car_type=5
        else:
            car_type=6
        labels.append(car_type)
        
    return labels

from collections import Counter
labels = chk_type_freq()
Counter(labels)

def chk_pose_freq(fnames=fnames,Path=Path,car_data=car_data):
    labels = []
    for b in range(len(fnames)):
        Fname = fnames[b]   
        fn2=Path+"/label/"+Fname[:-3]+"txt"
        with open(fn2) as f:
            data = f.readlines()
        data = [x.strip() for x in data]
        pose=int(data[0])-1
        labels.append(pose)        
    return labels
labels = chk_pose_freq()
Counter(labels)

def get_data(idxs,fnames=fnames,Path=Path,car_data=car_data):
    batch_size=len(idxs)
    recs = np.zeros((batch_size,4), dtype=np.float32)
    poses = np.zeros((batch_size), dtype=np.int32)
    car_types = np.zeros((batch_size), dtype=np.int32)
    
    crop_img_fvpn = np.zeros((batch_size,60,60,3))
    crop_img_aln  = np.zeros((batch_size,224,224,3))
    
    idx = 0
    for b in idxs:
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

        crop_img = img[rec[1]:rec[3], rec[0]:rec[2],:]
        crop_img_1 = cv2.resize(img,(60,60))
        crop_img_2 = cv2.resize(crop_img,(224,224))
        
        rec=rec.astype(np.float32)
        rec[1]=rec[1]/img.shape[0]
        rec[3]=rec[3]/img.shape[0]
        rec[0]=rec[0]/img.shape[1]
        rec[2]=rec[2]/img.shape[1]
        
        
        recs[idx,...] = rec
        poses[idx,...] = pose
        car_types[idx,...] = car_type
        crop_img_fvpn[idx,...] =  crop_img_1/255.
        crop_img_aln[idx,...]  =  crop_img_2/255.        
        idx += 1
        
    return [crop_img_fvpn,crop_img_aln,poses,car_types,recs]


pth2 = '/home/nazari/Desktop/compcars/images_street/*.*'
background_files = G.glob(pth2)

def get_background_data(batch_size=32,background_files=background_files):
    
    recs = np.zeros((batch_size,4), dtype=np.float32)
    car_types = np.zeros((batch_size), dtype=np.int32)    
    crop_img_fvpn = np.zeros((batch_size,60,60,3))
    crop_img_aln  = np.zeros((batch_size,224,224,3))
    poses = np.zeros((batch_size), dtype=np.int32)

    idxs = np.random.choice(len(background_files),batch_size,replace=False)
    fns = np.array(background_files)[idxs]
    idx = 0
    for g in fns:
        # tst_loop = True
        # while tst_loop:
        try:
            t_img = cv2.imread(g)
            bk_img_2 = cv2.resize(t_img,(224,224))
            # tst_loop = False
        except:
            tmp_idx = 0 #np.random.choice(len(background_files),1,replace=False)
            g = np.array(background_files)[tmp_idx]
            t_img = cv2.imread(g)
            bk_img_2 = cv2.resize(t_img,(224,224))
    
        bk_img_1 = cv2.resize(t_img,(60,60))
        crop_img_fvpn[idx,...] =  bk_img_1/255.
        crop_img_aln[idx,...]  =  bk_img_2/255.                   
        idx += 1
      
    return [crop_img_fvpn,crop_img_aln,poses,car_types,recs]
