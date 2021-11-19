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
    
    Conv_fc_class = Conv2D(filters=2, kernel_size=(10,10), 
                         strides=(1, 1), padding='valid', 
                         activation='softmax', name='Conv_fc_class')(net)
    logits_class = GlobalAveragePooling2D()(Conv_fc_class)
    
    model = Model(inputs=input, outputs=logits_class)
    
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model
    
    
#**********************************************
#**********************************************
# compile the model (should be done *after* setting layers to non-trainable)

fvpn_model = create_fvpn_network()


fvpn_model.summary()
print('*************************************')
model_name_ = "fvpn_class.h5"
fvpn_model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')
   
       
from Lib.dataloader import get_data,fnames,get_background_data
import numpy as np
import matplotlib.pyplot as plt
import cv2


# bs = 8
# for i in range(2):   
#     idxs = np.random.choice(len(fnames),bs,replace=False)
#     tmp_data = get_data(idxs)
#     batch_x = tmp_data[0]
#     batch_x_bk = get_background_data(bs)[0]
#     batch_y = np.concatenate((np.ones(len(batch_x)),
#                               np.zeros(len(batch_x_bk))),axis=0)
#     batch_x = np.concatenate((batch_x,batch_x_bk),axis=0)

#     preds = fvpn_model.predict(batch_x)
#     for j in range(len(preds)):
#         pred = preds[j]
#         img = batch_x[j]
#         plt.imshow(img)
#         plt.pause(0.001)
#         print('real_class=>',batch_y[j])
#         print('obtained_class=>',pred.argmax())
        
# multi_car_img = cv2.imread('/home/nazari/Desktop/compcars/img_dave2.jpg')/255.

# plt.imshow(multi_car_img)
# plt.pause(0.001) 

# multi_car_img = np.expand_dims(multi_car_img, axis=0)
        
Conv_fc_class_fvpn = fvpn_model.get_layer('Conv_fc_class').output
Conv_fc_class_fvpn_model = Model(inputs=fvpn_model.input,
                                  outputs=Conv_fc_class_fvpn)
# Conv_fc_class_fvpn_output = Conv_fc_class_fvpn_model.predict(multi_car_img)

# img = Conv_fc_class_fvpn_output

# plt.imshow(img[0,:,:,0])
# plt.pause(0.001)        
# plt.imshow(img[0,:,:,1])
# plt.pause(0.001)        

##########################################################################
##########################################################################

multi_car=cv2.imread('/home/nazari/Desktop/compcars/img_dave2.jpg')/255.
multi_car=cv2.resize(multi_car,(2048,1024))
(r,c)=multi_car.shape[0:2]

_for_new_siz = np.linspace(0.4, 1.0, num=20)

##############################################################
##############################################################
_resized_hms = []
_new_img = multi_car
for siz_idx in range(3):
    
    _tmp_multi_car = _new_img[np.newaxis,...]
    print("siz_idx => ", siz_idx, "     New Size => ", _tmp_multi_car.shape)
    tmp_heatmap = Conv_fc_class_fvpn_model.predict(_tmp_multi_car)
    tmp_heatmap = np.squeeze(tmp_heatmap)
    _heatmap = tmp_heatmap[...,1]

    _heatmap = (_heatmap-np.min(_heatmap))/(np.max(_heatmap)-np.min(_heatmap))
    
    if  siz_idx==0:
        for_last_test = cv2.resize(_heatmap,(c,r))
    
    _resized_hms.append(cv2.resize(_heatmap,(c,r)))
    _hm_res = (_heatmap*255).astype(np.uint8)
    _new_img = cv2.pyrDown(_new_img)
    plt.title("Heatmap of size => " + str(siz_idx))
    plt.imshow(_hm_res)
    plt.pause(0.1)
    
_resized_hms = np.array(_resized_hms)
print("Shape of Heatmaps Matrix=> " , _resized_hms.shape)

_final_hms = np.max(_resized_hms, axis=0)
print("Shape of Final Heatmap Matrix=> " , _final_hms.shape)
plt.title("Final Heatmap")
plt.imshow(_final_hms)
plt.pause(0.001)

_final_hms_idx = np.argmax(_resized_hms, axis=0)
print("Shape of Final Heatmap Index Matrix=> " , _final_hms_idx.shape)
plt.title("Final Heatmap idx")
plt.imshow(_final_hms_idx)
plt.pause(0.001)



_final_hms_thresh = _final_hms
_final_hms_thresh[_final_hms<0.75]=0
plt.title("Final Heatmap after applying Threshold")
plt.imshow(_final_hms_thresh)
plt.pause(0.001)


from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float

image_max = ndi.maximum_filter(_final_hms_thresh, size=10, mode='constant')
coordinates = peak_local_max(image_max, min_distance=50)

plt.imshow(_final_hms,cmap=plt.cm.gray)
#plt.autoscale(False)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')



from Lib.my_nms import nms
bounding_boxes = []
confidence_score = []
_sz=250
_sz2=int(_sz/2)
for _bb in range(len(coordinates)):
    c = coordinates[_bb,:]
    start_x = c[1]-_sz2
    start_y = c[0]-_sz2
    end_x = c[1]+_sz2
    end_y = c[0]+_sz2
    bounding_boxes.append([start_x,start_y,end_x,end_y]) 
    confidence_score.append(for_last_test[c[0],c[1]])


# Read image
image = multi_car#cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

# Copy image as original
org = image.copy()



# Draw bounding boxes and confidence score
for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
    # cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    # cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
    
    

# Run non-max suppression algorithm
threshold = 0.2
picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    
    
    
cv2.imshow('Original', org)
cv2.imshow('NMS', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
##########################################################################
##########################################################################