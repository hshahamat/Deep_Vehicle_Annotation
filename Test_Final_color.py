

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
model_name_ = "color.h5"
model.load_weights(model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')
   

####################################################################################    
#################################################################################    
#   -1: None,0:"black",1: "white",2: "red",3: "yellow",4: "blue",5: "green",6: "purple",7: "brown",8
#color_dict = [ 'black', 'white', 'red', 'yellow',
#              'blue','green','purple','brown','champagne','silver']
#import matplotlib.pyplot as plt
#import numpy as np
#import cv2
#
#image_size = 224
#color_dict = [ 'black', 'white', 'blue','red','silver']#,'other']
#
#import glob
#G = glob.glob('/home/nazari/Desktop/compcars/sv_data/sv_data/image/1/*.jpg')
#
#for g in G:
#    frm = cv2.imread(g)/255.
#    plt.imshow(frm)
#    plt.pause(0.001)
#        
#        
#    frm = cv2.resize(frm,(image_size,image_size))
#    frm = frm[np.newaxis,...]
#    pred = model.predict(frm)
#    
#    print('color=>',pred,'=>',color_dict[pred.argmax()])   
#        
        
from Lib.color_dataloader import get_color_data,color_matrix
import numpy as np
import matplotlib.pyplot as plt


color_dict = [ 'black', 'white', 'blue','red','silver']
bs = 32
for i in range(2):   
    idxs = np.random.choice(len(color_matrix),bs,replace=False)
    batch_x, batch_y = get_color_data(idxs)    
    preds = model.predict(batch_x)
    for j in range(bs):
        pred = preds[j]

        plt.imshow(batch_x[j])
        plt.pause(0.001)
        print('real_color=>',batch_y[j],'=>',color_dict[batch_y[j]])
        print('obtained_color=>',pred.argmax(),'=>',color_dict[pred.argmax()]) 
        
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#####################################################################
#multi_car = get_frame(idx, vfile=vfile)
#    
##multi_car=cv2.imread('/home/nazari/Desktop/compcars/img_dave2.jpg')
##multi_car=cv2.resize(multi_car,(2048*2,1024*2))
#
#(r,c)=multi_car.shape[0:2]
#
#coordinates = coordinates.astype(np.float)
#coordinates[:, 0] = coordinates[:, 0]/float(c)
#coordinates[:, 1] = coordinates[:, 1]/float(r)
#
##_for_new_siz = np.linspace(0.4, 1.0, num=20)
#
###############################################################
###############################################################
#    
#_tmp_multi_car = multi_car[np.newaxis,...]
#my_dict_hm = {input_images_fvpn : _tmp_multi_car,
#              dropout_keep_prob : 1.0}
#_bbr = sess.run(Conv_fc_bbr,feed_dict=my_dict_hm)
#    
#    
#coordinates[:, 0] = coordinates[:, 0]*_bbr.shape[2]
#coordinates[:, 1] = coordinates[:, 1]*_bbr.shape[1]
#coordinates = coordinates.astype(np.int)
#
#_r = coordinates[:, 1]
#_c = coordinates[:, 0]
#print(_bbr[0,_r,_c,:])

