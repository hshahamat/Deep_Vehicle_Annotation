import cv2

#
vfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/jan28.avi"
dfile = "/home/nazari/Desktop/LISA/Dense_LISA_1/Dense/pos_annot.dat"



def get_frame_label(frame_number,dfile=dfile):
    with open(dfile) as f:
        car_data = f.readlines()
        
    car_data = [x.strip() for x in car_data]
    car_data = [x.split('\t') for x in car_data]
    return car_data[frame_number]
    
    
    
    
    
def get_frame(frame_number,vfile=vfile):
    Cap = cv2.VideoCapture(vfile)
    Cap.set(1,frame_number)
    ret, frame = Cap.read()
    frame = frame/255.#cv2.resize(frame,(512,512))
    return frame 