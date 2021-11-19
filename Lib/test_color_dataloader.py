# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 02:04:07 2019

@author: Nazari
"""
import os.path
import numpy as np 
np.random.seed(2020)
import cv2
import matplotlib.pyplot as plt
import scipy.io
import glob as G


mat_file = '/home/nazari/Desktop/compcars/data/data/sh/color_list.mat'
mat = scipy.io.loadmat(mat_file,struct_as_record=False, squeeze_me=True)

tmp = mat['color_list']
color_matrix = dict(tmp)
pth = '/home/nazari/Desktop/compcars/sv_data/sv_data/image/*/*.jpg'
color_files = G.glob(pth)
def get_test_color_data(batch_size=32,color_matrix=color_matrix,color_files=color_files):
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

