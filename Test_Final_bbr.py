#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:02:42 2020

@author: nazari
"""

#*********************************************
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
#predictions = Dense(5, activation='softmax')(x)
#
## this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
#
#model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False

#
#model.compile(optimizer='rmsprop',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#**********************************************
#**********************************************

def create_fvpn_network():
    input = Input(shape=(None, None, 3), name='input')
    conv1 = Conv2D(filters=32, kernel_size=(5,5), 
                   strides=(1, 1), padding='valid', 
                   activation='relu', name='conv1')(input)
    
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(5,5), 
                   strides=(1, 1), padding='valid', 
                   activation='relu', name='conv2')(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2)
    
    conv3 = Conv2D(filters=64, kernel_size=(3,3), 
                   strides=(1, 1), padding='valid', 
                   activation='relu', name='conv3')(pool2)   
    
    net = Dropout(0.2)(conv3)
    
    Conv_fc_bbr = Conv2D(filters=4, kernel_size=(10,10), 
                         strides=(1, 1), padding='valid', 
                         activation='sigmoid', name='Conv_fc_bbr')(net)
    logits_bbr = GlobalAveragePooling2D()(Conv_fc_bbr)
    
    model = Model(inputs=input, outputs=logits_bbr)
    
    model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    return model
    
    
#**********************************************
#**********************************************
# compile the model (should be done *after* setting layers to non-trainable)

fvpn_model = create_fvpn_network()


fvpn_model.summary()
print('*************************************')
model_name_ = "bbr.h5"
fvpn_model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')
   
       
from Lib.dataloader import get_data,fnames
import numpy as np
import matplotlib.pyplot as plt
import cv2


bs = 8
for i in range(2):   
    idxs = np.random.choice(len(fnames),bs,replace=False)
    tmp_data = get_data(idxs)
    batch_x = tmp_data[0]
    batch_y = tmp_data[4]
    preds = fvpn_model.predict(batch_x)
    for j in range(bs):
        pred = preds[j]
        img = batch_x[j]
        bb = (pred*60).astype(np.int)
        res = cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)
        plt.imshow(res)
        plt.pause(0.001)
        print('real_BBX=>',batch_y[j,:])
        print('obtained_BBX=>',pred) 
