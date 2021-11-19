#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:19:56 2019

@author: nazari
"""
import numpy as np
import cv2
from random import randint

def agument_images(img, true_size = 60, enlarge_size = 70, is_train=True):
    if is_train:
         resize_to = enlarge_size
    else:
         resize_to = true_size
  
    _imgs = np.zeros((img.shape[0],true_size,true_size, img.shape[3])) 
    
    tl_h = randint(0,resize_to-true_size)
    tl_w = randint(0,resize_to-true_size)
    flipflag = False#((randint(0,1)>0) and (is_train))
        
    for i in range(img.shape[0]):
        A = img[i,...]#*2.0-1.0
        
        rn_=np.random.rand()
        if rn_>0.4 and rn_<=0.7:
            A=A+0.2
            A[A>1]=1
        if rn_>0.7 and rn_<=1.0:
            A=A-0.2
            A[A<0]=0
            
        rn_=np.random.rand()
        if rn_>0.5:
            rn_fs=1-(np.random.rand()*0.8)
            _rs1=int(A.shape[0]*rn_fs)
            _rs0=int(A.shape[1]*rn_fs)
            A = cv2.resize(A,(_rs0,_rs1))
            
            
        A = cv2.resize(A,(resize_to,resize_to)) 
        _imgs[i,...] = flip_image(A[tl_h: tl_h+true_size, tl_w: tl_w+true_size, :],flipflag)

    return _imgs

def flip_image(img,flipflag):
  if flipflag:
    return np.fliplr(img)
  else:
    return img

