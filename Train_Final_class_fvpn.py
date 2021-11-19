#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:02:42 2020

@author: nazari
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:59:13 2020

@author: nazari
"""

from MyGenerator import TGenerator
train_data_generator = TGenerator(O=5,I=0)
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
# train the model on the new data for a few epochs
fvpn_model.fit_generator(train_data_generator,
          steps_per_epoch=500,
          epochs=5)

fvpn_model.save('fvpn_class.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#
## we chose to train the top 2 inception blocks, i.e. we will freeze
## the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
#   layer.trainable = False
#for layer in model.layers[249:]:
#   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#from tensorflow.keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
## we train our model again (this time fine-tuning the top 2 inception blocks
## alongside the top Dense layers
#model.fit(...)
#
##*******************************************************



