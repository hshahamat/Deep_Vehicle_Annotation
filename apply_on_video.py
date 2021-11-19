#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:11:55 2020

@author: nazari
"""

import argparse
from tqdm import tqdm
import numpy as np
import cv2


from YOLO.myopt import opt
from YOLO.YoloBBX import get_bb

import torch
from Lib.load_lisa import get_frame, get_frame_label
from Lib.annotation import annotate_img

import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


vfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/march9.avi"
dfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/pos_annot.dat"

# vfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/jan28.avi"
# dfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/pos_annot.dat"


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

print('*************************************')
model_name_ = "type.h5"#"aln_class.h5"
model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')


type_dict = ['other', 'MPV', 'SUV', 'sedan', 
             'hatchback','minibus',
             'pickup' ]

pose_dict = ['front', 'rear', 'side', 'frontside', 'rearside']

color_dict = [ 'black', 'white', 'blue','red','silver']


image_size = 224

_res_dict=type_dict


_res_type =[]

BBXS =[]

with torch.no_grad():

    for idx in range(0,300,1):
    #    idx = 5
        frm = get_frame(idx, vfile=vfile)
# image = np.expand_dims(image, axis=0)
        bbxs=get_bb(frm)
        print(bbxs)
        
        BBXS.append(bbxs)
        
        
        if bbxs!=[]:
            
            for _nop in range(len(bbxs)):
                _bbx = bbxs[_nop]
                _x1=_bbx[0]
                _x2=_bbx[1]
                _y1=_bbx[2]
                _y2=_bbx[3]
                                   
                marg = 0
                row_start = max(0,_y1-marg)
                row_end = min(frm.shape[0],_y2+marg)
                
                col_start = max(0,_x1-marg)
                col_end = min(frm.shape[1],_x2+marg)
                
                main_size_x = col_end - col_start
                main_size_y = row_end - row_start
                
                _image = frm[row_start:row_end ,
                            col_start:col_end ,
                            :]
                
                _image = cv2.resize(_image,(image_size,image_size))/255.
                _image = _image[np.newaxis,...]
                
                
                _res = model.predict(_image)
                
                _res_type.append(_res.argmax())
                # plt.imshow(_image[0])
                # plt.pause(0.001)
                # print('class=>',_res.argmax(),'=>',_res_dict[_res.argmax()])
    
    
                # cv2.rectangle(frm, (col_start, row_start),
                #               (col_end, row_end),
                #               (0, 0, 255), 2)
                
                
            # frm = (frm*255).astype(np.uint8)   
            # res =  cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            # plt.imshow(res)
            # plt.pause(0.001)


predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

print('*************************************')
model_name_ = "pose.h5"#"aln_class.h5"
model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')


from tqdm import tqdm

_res_pose = []

with torch.no_grad():

    for idx in tqdm(range(0,300,1)):
    #    idx = 5
        frm = get_frame(idx, vfile=vfile)
        
        bbxs = BBXS[idx]
        
        
        if bbxs!=[]:
            
            for _nop in range(len(bbxs)):
                _bbx = bbxs[_nop]
                _x1=_bbx[0]
                _x2=_bbx[1]
                _y1=_bbx[2]
                _y2=_bbx[3]
                                   
                marg = 0
                row_start = max(0,_y1-marg)
                row_end = min(frm.shape[0],_y2+marg)
                
                col_start = max(0,_x1-marg)
                col_end = min(frm.shape[1],_x2+marg)
                
                main_size_x = col_end - col_start
                main_size_y = row_end - row_start
                
                _image = frm[row_start:row_end ,
                            col_start:col_end ,
                            :]
                
                _image = cv2.resize(_image,(image_size,image_size))/255.
                _image = _image[np.newaxis,...]
                
                
                _res = model.predict(_image)
                
                _res_pose.append(_res.argmax())
                
                
                
                
                
                
                
                
                
                
                
                
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

print('*************************************')
model_name_ = "color.h5"#"aln_class.h5"
model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')


_res_color = []

with torch.no_grad():

    for idx in tqdm(range(0,300,1)):
    #    idx = 5
        frm = get_frame(idx, vfile=vfile)
        
        bbxs = BBXS[idx]
        
        
        if bbxs!=[]:
            
            for _nop in range(len(bbxs)):
                _bbx = bbxs[_nop]
                _x1=_bbx[0]
                _x2=_bbx[1]
                _y1=_bbx[2]
                _y2=_bbx[3]
                                   
                marg = 0
                row_start = max(0,_y1-marg)
                row_end = min(frm.shape[0],_y2+marg)
                
                col_start = max(0,_x1-marg)
                col_end = min(frm.shape[1],_x2+marg)
                
                main_size_x = col_end - col_start
                main_size_y = row_end - row_start
                
                _image = frm[row_start:row_end ,
                            col_start:col_end ,
                            :]
                
                _image = cv2.resize(_image,(image_size,image_size))/255.
                _image = _image[np.newaxis,...]
                
                
                _res = model.predict(_image)
                
                _res_color.append(_res.argmax())
                
                


Cap = cv2.VideoCapture(vfile)

width  = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(Cap.get(cv2.CAP_PROP_FPS))
fnum = int(Cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('width => ', width)
print('height => ', height)
print('fps => ', fps)
print('num frames => ', fnum)

outv = cv2.VideoWriter('annotation_result.mp4',
					  cv2.VideoWriter_fourcc(*'MP4V'), 
					  fps, (width,height))


for idx in tqdm(range(0,300,1)):
#    idx = 5
    frm = get_frame(idx, vfile=vfile)
    
    bbxs = BBXS[idx]
    
    
    if bbxs!=[]:
        _type =  _res_type[idx]
        _color = _res_color[idx]
        _pose = _res_pose[idx]
        res = annotate_img(frm,bbxs[0],_type=_type,_pose=_pose,_color=_color)
        outv.write(res)
        res =  cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        plt.imshow(res)
        plt.pause(0.001)
        
outv.release()
            
            
                
                
                