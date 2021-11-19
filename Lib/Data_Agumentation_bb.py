#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:19:56 2019

@author: nazari
"""
import numpy as np
#import tensorflow as tf
#from random import randint
import cv2

def agument_images_bb(img, true_size = 60):
    resize_to = true_size
  
    _imgs = np.zeros((img.shape[0],true_size,true_size, img.shape[3])) 
    
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
        _imgs[i,...] = A

    return _imgs


