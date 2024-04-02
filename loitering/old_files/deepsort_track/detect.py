from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker


import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np

#----------DEEP SORT----------#
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)    
#max_age = controls how long a tracked object is valid if it's not detected in consecutive frames
#it's the max # of frames an object tracked can be maintained w/o corresponding detection before it's considered as expired

video_path = '../images/test5.mp4'

cap = cv2.VideoCapture(video_path)

#Getting video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#Define the codec and create VideoWriter Object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #video codec to be used for video compression. 'mp4v' represents the MPEG-4 Video codec.
    #FourCC is a 4-byte code used to specify video codecs. Different codecs have different FourCC codes.
output_path = 'output2.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

frames = []
unique_track_ids = set() #a set is an unordered collection of unique elements.

i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

while cap.isOpened():   #a loop that continues as long as the video capture object cap is opened and available to read frames.
    success, frame = cap.read()

    if success:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

        model = YOLO("yolov8n.pt")
        #Classes 0 = person, confidence value = 0.8 (only conf >= 0.8 will consider as detected)
        results = model(frame, device = 'cpu', classes = 0, conf = 0.8)
        
        for result in results:
            boxes = result.boxes    #Boxes object for bbox outputs
            probs = result.probs    #Class probabilities for classification outputs
            cls = boxes.cls.tolist()
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh #box with xywh format, (N,4)
            for class_index in cls:
                class_name = classNames[int(class_index)]
                print(class_name)

        #Just transferring into CPU & converting to numpy bc requirement of tracking algorithm
        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)

        #tracker - defined in line 17: tracker = DeepSort(model_path=deep_sort_weights, max_age=70) 
        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        #Iterating through each tracked object by tracker
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()    #Getting bounding box coordinates
            
            w = x2 - x1     #Calculate width
            h = y2 - y1     #Calculate height

            #For plotting bounding boxes
            #Set color values for R, B, G
            red_color = (0, 0, 255) #(B,G,R)
            blue_color = (255, 0, 0) #(B,G,R)
            green_color = (0, 255, 0) #(B,G,R)

            #Determine the color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color
            
            #Drawing a bounding box around the detected object - cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 1)
            text_color = (0,0,0)    #black color for text

            #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), 0, 1, text_color, thickness=1, lineType=cv2.LINE_AA)

            #Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)
        
        #Update the person count based on the number of unique track IDs
        person_count = len(unique_track_ids)

        #Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time
        
        #Draw person count on frame
        cv2.putText(og_frame, f"Person count: {person_count}", (10, 30), 0, 1, text_color, thickness=1, lineType=cv2.LINE_AA)

        #Append the frame to the list
        frames.append(og_frame)
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        cv2.imshow("Image", cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF==ord('1'): #Clicking 1 will stop the program
            break

cap.release()
out.release()
cv2.destroyAllWindows()

