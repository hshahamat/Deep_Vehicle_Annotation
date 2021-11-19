#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:22:31 2020

@author: nazari
"""

#from keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import math


from Lib.dataloader import get_data,fnames,get_background_data

from Lib.Data_Agumentation import agument_images
from Lib.Data_Agumentation_bb import agument_images_bb

dataset_info = fnames

print('len(dataset_info)=>',len(dataset_info))
class TGenerator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, O=3, I=1, dataset_info=dataset_info, batch_size=32):
        self.I=I
        self.O=O
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.indices = np.arange(len(self.dataset_info))

    def __len__(self):
        return math.ceil(len(self.dataset_info) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#        print(inds)
        tmp_data = get_data(inds)
        batch_x = tmp_data[self.I]
        if self.O != 5:
            batch_y = tmp_data[self.O]
            
        if self.I == 1:
            batch_x = agument_images(batch_x,224,256)
            if self.O == 5:
                batch_x_bk = get_background_data(self.batch_size)[1]
                batch_x_bk = agument_images(batch_x_bk,224,256)
                batch_y = np.concatenate((np.ones(len(batch_x)),
                                          np.zeros(len(batch_x_bk))),axis=0)
                batch_x = np.concatenate((batch_x,batch_x_bk),axis=0)
                
            
        if self.I == 0:
            batch_x = agument_images_bb(batch_x,60)
            if self.O == 5:
                batch_x_bk = get_background_data(self.batch_size)[0]
                batch_x_bk = agument_images(batch_x_bk,60,90)
                batch_y = np.concatenate((np.ones(len(batch_x)),
                                          np.zeros(len(batch_x_bk))),axis=0)
                batch_x = np.concatenate((batch_x,batch_x_bk),axis=0)
                
            
        
        # print(batch_y)
        return batch_x,batch_y 
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
