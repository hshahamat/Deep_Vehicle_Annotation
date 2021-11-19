

from Lib.load_lisa import get_frame, get_frame_label


vfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/jan28.avi"
dfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/pos_annot.dat"

#vfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/march9.avi"
#dfile = "/home/nazari/Desktop/LISA/Urban_LISA_2/Urban/pos_annot.dat"


    
    

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
   
       
        
#from Lib.dataloader import get_data,fnames
import numpy as np
import matplotlib.pyplot as plt
import cv2

pose_dict = ['front', 'rear', 'side', 'frontside', 'rearside']

image_size = 224
for idx in range(0,300,30):
#    idx = 5
    frm = get_frame(idx, vfile=vfile)
    
    res = (frm * 255).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    
    lbl = get_frame_label(idx, dfile=dfile)
    
    num_of_cars = int(lbl[1])

    for _car in range(num_of_cars):
        _tmp = lbl[_car+2].split()
        _rect = np.array(_tmp,dtype=np.uint16)
        xmin= _rect[0]
        ymin= _rect[1]
        xmax= _rect[0]+_rect[2]
        ymax= _rect[1]+_rect[3]
        res = cv2.rectangle(res,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        

        croped_img = frm[ymin:ymax,
                         xmin:xmax, :]
        tfrm = cv2.resize(croped_img,(image_size,image_size))
        tfrm = tfrm[np.newaxis,...]    
        _pose = model.predict(tfrm)
        cv2.putText(res,'Pose=>'+pose_dict[_pose.argmax()],(xmin,ymin-30),cv2.FONT_HERSHEY_PLAIN,2, (0,255,0),3)
    plt.imshow(res)
    plt.pause(0.001)
#    print('pose=>',_pose.argmax(),'=>',pose_dict[_pose.argmax()])
    
    
    
    
    
    
    
    
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#model.summary()

print('*************************************')
model_name_ = "color.h5"
model.load_weights(model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************') 
        

color_dict = [ 'black', 'white', 'blue','red','silver']
image_size = 224
for idx in range(0,300,30):
#    idx = 5
    frm = get_frame(idx, vfile=vfile)
    res = (frm * 255).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    
    lbl = get_frame_label(idx, dfile=dfile)
    
    num_of_cars = int(lbl[1])

    for _car in range(num_of_cars):
        _tmp = lbl[_car+2].split()
        _rect = np.array(_tmp,dtype=np.uint16)
        xmin= _rect[0]
        ymin= _rect[1]
        xmax= _rect[0]+_rect[2]
        ymax= _rect[1]+_rect[3]
        res = cv2.rectangle(res,(xmin,ymin),(xmax,ymax),(0,255,0),2)

        croped_img = frm[ymin:ymax,
                         xmin:xmax, :]
        tfrm = cv2.resize(croped_img,(image_size,image_size))
        tfrm = tfrm[np.newaxis,...]
    
        _color = model.predict(tfrm)
        
        cv2.putText(res,'Color=>'+color_dict[_color.argmax()],(xmin,ymin-50),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),3)
    plt.imshow(res)
    plt.pause(0.001)        
#        print('color=>',_color.argmax(),'=>',color_dict[_color.argmax()])   