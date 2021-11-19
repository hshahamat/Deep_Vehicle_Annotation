#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:17:58 2020

@author: nazari
"""
import os.path
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.io
import glob as G


mat_file = '/home/nazari/Desktop/compcars/sv_data/sv_data/color_list.mat'
mat = scipy.io.loadmat(mat_file,struct_as_record=False, squeeze_me=True)

tmp = mat['color_list']
color_matrix = dict(tmp)
pth = '/home/nazari/Desktop/compcars/sv_data/sv_data/image/*/*.jpg'
color_files = G.glob(pth)

color_files = color_files[:5000]
def get_color_data(idxs,color_matrix=color_matrix,color_files=color_files):
    batch_size=len(idxs)
    colors = np.zeros((batch_size), dtype=np.int32)
    images = np.zeros((batch_size,224,224,3))
#    idxs = np.random.choice(len(color_files),batch_size,replace=False)
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
       
#       print('color=====>',color)
#       if color<0:
#           color = 10
       if color in [0,1,2,4,9]:
           if color == 4:
               color = 3
           if color == 9:
               color = 4
                   
       else:
           color = 5
#       print('color=====>',color)
       colors[idx,...] = color
       images[idx,...] = t_img/255.
       idx += 1
    
    return images,colors