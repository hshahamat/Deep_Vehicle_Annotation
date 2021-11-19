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


img_pth = '/home/nazari/Desktop/compcars/sv_data/sv_data/image/'


mat_file = '/home/nazari/Desktop/compcars/sv_data/sv_data/color_list.mat'
mat = scipy.io.loadmat(mat_file,struct_as_record=False, squeeze_me=True)

color_matrix = mat['color_list']

print('color_matrix.shape==>',color_matrix.shape)
new_color_matrix = []

for i in range(len(color_matrix)):
    if os.path.exists(img_pth + color_matrix[i][0]):
#        print(i,'yes')
        new_color_matrix.append(color_matrix[i])
#    else:
#        print(i,'No')

new_color_matrix = np.array(new_color_matrix)    
print('new_color_matrix.shape==>',new_color_matrix.shape)

#
#hist, bin_edges = np.histogram(new_color_matrix[:,1]+1,bins=list_of_color)
#print(bin_edges)
#print(hist)

from collections import Counter
_Count = Counter(new_color_matrix[:,1])

_min = _Count[0]
for _i in [0,1,2,4,9]:
    if _Count[_i]<_min:
        _min = _Count[_i]

balanced_color_matrix = []
tmp_class = new_color_matrix[:,1]        
for _i in [0,1,2,4,9]:
    tmp_data = new_color_matrix[tmp_class==_i,:]
    tmp_data = tmp_data[:_min,:]
    balanced_color_matrix.extend(tmp_data)

balanced_color_matrix = np.array(balanced_color_matrix)    
#_Count = Counter(balanced_color_matrix[:,1])

color_matrix = balanced_color_matrix


def get_color_data(idxs,color_matrix=color_matrix,img_pth=img_pth):
    batch_size=len(idxs)
    colors = np.zeros((batch_size), dtype=np.int32)
    images = np.zeros((batch_size,224,224,3))

    idx = 0
    for _i in range(len(idxs)):

       c_i = idxs[_i]
#       t_img = cv2.imread(g)
       g = img_pth + color_matrix[c_i,0]
       color = color_matrix[c_i,1]
       t_img = cv2.imread(g)
       t_img = cv2.resize(t_img,(224,224))

#       if color in [0,1,2,4,9]:
       if color == 4:
           color = 3
       if color == 9:
           color = 4
                   
#       else:
#           color = 5
#       print('color=====>',color)
       colors[idx,...] = color
       images[idx,...] = t_img/255.
       idx += 1
    
    return images,colors