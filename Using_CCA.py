from Lib.load_lisa import get_frame, get_frame_label


vfile = "/home/Desktop/LISA/Urban_LISA_2/Urban/march9.avi"
dfile = "/home/Desktop/LISA/Urban_LISA_2/Urban/pos_annot.dat"


    
    

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
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

print('*************************************')
model_name_ = "pose.h5"
model.load_weights  (model_name_) 
print(model_name_+" has been loaded . . . ")
print('*************************************')
   
feature_layer = model.layers[-2].output
new_model = Model(inputs=model.input,
                  outputs=feature_layer)
       
 
#########Prepare Train Data       
from Lib.dataloader import get_data,fnames
dataset_info = fnames

import numpy as np
import matplotlib.pyplot as plt
from my_embedding import word_embedding


pose_dict = ['uncertain','front', 'rear', 'side', 'frontside', 'rearside']
embedded_labels = []
for _lbl in pose_dict:
    embedded_labels.append(word_embedding(_lbl))
embedded_labels  = np.array(embedded_labels)

bs =32
_features = []
_labels = []
for i in range(0,len(dataset_info),bs):
    min_idx = i
    max_idx = i+bs
    max_idx = min(max_idx,len(dataset_info)-1)
    print("min_idx,max_idx::>",min_idx,max_idx)    
    idxs = np.arange(min_idx,max_idx)
    tmp_data = get_data(idxs)
    batch_x = tmp_data[1]
    batch_y = tmp_data[2]
    _feat = new_model.predict(batch_x)
    _features.extend(_feat)
    _labels.extend(batch_y)
      
_features  = np.array(_features)
_labels  = np.array(_labels)
       
new_labels = np.zeros((_labels.shape[0],embedded_labels.shape[1]))

for _i in range(_labels.shape[0]):
    new_labels[_i,:] = embedded_labels[_labels[_i],:]
    
l_train = _labels
y_train = new_labels
x_train = _features


###################Prepare Test Data

from Lib.test_dataloader import get_data,fnames
dataset_info = fnames

_features = []
_labels = []
for i in range(0,len(dataset_info),bs):
    min_idx = i
    max_idx = i+bs
    max_idx = min(max_idx,len(dataset_info)-1)
    print("min_idx,max_idx::>",min_idx,max_idx)    
    idxs = np.arange(min_idx,max_idx)
    tmp_data = get_data(idxs)
    batch_x = tmp_data[1]
    batch_y = tmp_data[2]
    _feat = new_model.predict(batch_x)
    _features.extend(_feat)
    _labels.extend(batch_y)
      
_features  = np.array(_features)
_labels  = np.array(_labels)
       
new_labels = np.zeros((_labels.shape[0],embedded_labels.shape[1]))

for _i in range(_labels.shape[0]):
    new_labels[_i,:] = embedded_labels[_labels[_i],:]
    
l_test = _labels
y_test = new_labels
x_test = _features


print("train / new input data shape :", x_train.shape)
print("train / new labels shape :", y_train.shape)
print("train / old labels shape :", l_train.shape)

print("test / new input data shape :", x_test.shape)
print("test / new labels shape :", y_test.shape)
print("test / old labels shape :", l_test.shape)



from sklearn.cross_decomposition import CCA
_n_components = 200
cca = CCA(n_components=_n_components)
cca = cca.fit(x_train, y_train)
X_cca, Y_cca = cca.transform(x_train, y_train)

X_test_cca = cca.transform(x_test)
print("new train input data shape :", X_cca.shape)
print("new train labels shape :", Y_cca.shape)

from scipy.spatial.distance import cdist

_cca_res = cdist(X_test_cca, Y_cca, 'correlation')

y_pred = []
for _cca in _cca_res:
    _tmp = _cca.argmin()
    y_pred.append(l_train[_tmp])

y_pred  = np.array(y_pred)

print("Test_Accuracy=>",np.mean((y_pred==l_test)))
