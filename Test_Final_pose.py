

from Lib.load_lisa import get_frame, get_frame_label


#vfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/jan28.avi"
#dfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/pos_annot.dat"

vfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/march9.avi"
dfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/pos_annot.dat"


    
    

############################################################################
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

print('*************************************')
model_name_ = "pose.h5"
model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')
   
       
        
from Lib.dataloader import get_data,fnames
import numpy as np
import matplotlib.pyplot as plt


pose_dict = ['front', 'rear', 'side', 'frontside', 'rearside']


bs = 32
for i in range(2):   
    idxs = np.random.choice(len(fnames),bs,replace=False)
    tmp_data = get_data(idxs)
    batch_x = tmp_data[1]
    batch_y = tmp_data[2]
    preds = model.predict(batch_x)
    for j in range(bs):
        pred = preds[j]

        plt.imshow(batch_x[j])
        plt.pause(0.001)
        print('real_pose=>',batch_y[j],'=>',pose_dict[batch_y[j]])
        print('obtained_pose=>',pred.argmax(),'=>',pose_dict[pred.argmax()]) 
        