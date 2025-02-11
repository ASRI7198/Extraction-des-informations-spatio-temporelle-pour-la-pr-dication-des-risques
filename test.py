import cv2
import numpy as np
from collections import deque
import os
from build_tools import build_tools


model_tools = build_tools()
network = model_tools.create_network()
network.load_weights('model_weights_032.ckpt') 


collision_path = "_data"
output_video_path = 'output_video1.avi'
output_frame_size=(640, 480) 
output_frame_rate=10.0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, output_frame_rate, output_frame_size)

for subfolder in os.listdir(collision_path):
    subfolder_path = os.path.join(collision_path, subfolder)
    image_seq = deque([],8)
    stat = "safe"
    for image in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 480))
        _image = cv2.resize(image,(210,140))
        image_seq.append(_image)
        if len(image_seq)==8:
            np_image_seqs = np.reshape(np.array(image_seq)/255,(1,8,140,210,3))    
            prediction = network.predict(np_image_seqs)
            stat = ['safe', 'Not Safe'][np.argmax(prediction, 1)[0]]
            print(f'{image_path}: {stat}')
        if stat == 'safe':
            cv2.putText(image, stat, (230, 230), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255, 0), 3)
        else:
            cv2.putText(image, stat, (230, 230), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,0,255),3)        
        cv2.imshow('Frame', image)
        out.write(image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
        