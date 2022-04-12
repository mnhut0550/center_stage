import cv2
from face_detector import face_detector

cam = cv2.VideoCapture(0)
detector = face_detector(model_path='frozen_inference_graph.pb')

border_x = 200
border_y = 200

cam.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)

while cam.isOpened():
    ret,frame = cam.read()
    
    max_frame_height,max_frame_width,channels = frame.shape
    
    bbox,_ = detector.inference(frame,threshold=0.5)
    
    if len(bbox) ==0:
        center_point_x = int(max_frame_width/2)
        center_point_y = int(max_frame_height/2)
        
        start_y,end_y = center_point_y-border_y, center_point_y+border_y
        start_x,end_x = center_point_x-border_x, center_point_x+border_x
        
        img = frame[start_y:end_y, start_x:end_x]
        
    elif len(bbox) == 1:
        bbox = bbox[0]
        
        start_x,start_y,end_x,end_y = bbox
         
        face_height = end_y - start_y
        face_width = end_x - start_x
          
        start_x,start_y,end_x,end_y = start_x-border_x,start_y-border_y,end_x+border_x,end_y +border_y
          
        if start_x < 0:
            start_x = 0
        if start_y <0:
            start_y = 0
            
        if end_x >= max_frame_width:
            end_x = max_frame_width
        if end_y >= max_frame_height:
            end_y = max_frame_height
         
        img = frame[start_y:end_y,start_x:end_x]
    else:
        x1_position = []
        y1_position = []
        
        x2_position = []
        y2_position = []
        
        for box in bbox:
            start_x,start_y,end_x,end_y = box
            
            x1_position.append(start_x)
            y1_position.append(start_y)
            
            x2_position.append(end_x)
            y2_position.append(end_y)
            
        start_x, start_y = min(x1_position),min(y1_position)
        end_x, end_y = max(x2_position),max(y2_position)
        
        start_x,start_y,end_x,end_y = start_x-border_x,start_y-border_y,end_x+border_x,end_y +border_y
          
        if start_x < 0:
            start_x = 0
        if start_y <0:
            start_y = 0
            
        if end_x >= max_frame_width:
            end_x = max_frame_width
        if end_y >= max_frame_height:
            end_y = max_frame_height
         
        img = frame[start_y:end_y,start_x:end_x]
        

    cv2.imshow('t',img)
    cv2.waitKey(1)