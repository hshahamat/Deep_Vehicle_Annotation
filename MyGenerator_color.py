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


from Lib.color_dataloader import get_color_data,color_matrix

from Lib.Data_Agumentation import agument_images

dataset_info = color_matrix

print('len(dataset_info)=>',len(dataset_info))
class TGenerator_color(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, dataset_info=dataset_info, batch_size=32):
        
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.indices = np.arange(len(self.dataset_info))

    def __len__(self):
        return math.ceil(len(self.dataset_info) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#        print(inds)
        batch_x,  batch_y = get_color_data(inds)
#        batch_x = agument_images(batch_x,224,256)
#        print(batch_y)
        return batch_x,batch_y 
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
