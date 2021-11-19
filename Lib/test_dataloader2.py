import os.path
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.io
import glob as G


class TestDataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass
    
    
    @property
    def data(self):
        return self._data
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed = 1
            return self.data[start:],True
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.data[start:end],False



Path="/home/nazari/Desktop/compcars/data/data"
FileName="/train_test_split/classification/test.txt"
with open(Path+FileName) as f:
    fnames = f.readlines()
fnames = [x.strip() for x in fnames]

FileNametype="/misc/attributes.txt"
with open(Path+FileNametype) as f:
    car_data = f.readlines()
    
car_data.pop(0)
car_data = [x.strip() for x in car_data]
car_data = [x.split() for x in car_data]
car_data = [[x[0],x[5]] for x in car_data]
car_data = dict(car_data)


idx=np.arange(0, len(fnames))
dataset = TestDataset(idx)

def get_data_final_test(batch_size=64,fnames=fnames,Path=Path,car_data=car_data,dataset=dataset):
    images =[]
    recs = np.zeros((batch_size,4), dtype=np.float32)
    poses = np.zeros((batch_size), dtype=np.int32)
    car_types = np.zeros((batch_size), dtype=np.int32)
    
    crop_img_fvpn = np.zeros((batch_size,60,60,3))
    crop_img_aln  = np.zeros((batch_size,224,224,3))
    
    batch_idx,chk_end=dataset.next_batch(batch_size)
    idx = 0
    for b in batch_idx:
        Fname = fnames[b]   
        tmp=Fname.split('/')
        tmp_car_type=int(car_data[tmp[1]])
        
        if (tmp_car_type in [0,1,2,3,4]) or (tmp_car_type==7):
            car_type=tmp_car_type
            if (tmp_car_type==7):
                car_type=5
        else:
            car_type=6        
        
        fn1=Path+"/image/"+Fname
        fn2=Path+"/label/"+Fname[:-3]+"txt"
        with open(fn2) as f:
            data = f.readlines()
        data = [x.strip() for x in data]
        pose=int(data[0])-1
        rec=(np.array(data[2].split())).astype(np.uint16)
        img = cv2.imread(fn1)
        images.append(list(img))
        crop_img = img[rec[1]:rec[1]+rec[3], rec[0]:rec[0]+rec[2]]
        crop_img_1 = cv2.resize(crop_img,(60,60))
        crop_img_2 = cv2.resize(crop_img,(224,224))
        
        rec=rec.astype(np.float32)
        rec[1]=rec[1]/img.shape[0]
        rec[3]=rec[3]/img.shape[0]
        rec[0]=rec[0]/img.shape[1]
        rec[2]=rec[2]/img.shape[1]
        
        
        recs[idx,...] = rec
        poses[idx,...] = pose
        car_types[idx,...] = car_type
        crop_img_fvpn[idx,...] =  crop_img_1#/255.0
        crop_img_aln[idx,...]  =  crop_img_2#/255.0             
        idx += 1
        
    return images,crop_img_fvpn,crop_img_aln,poses,car_types,recs,chk_end