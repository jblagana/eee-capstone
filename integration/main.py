""""
TO DO:
1. Inference deployment
2. Stop when RBP reach threshold? Print how early detected?
3. Benchmarking
"""

import csv
import cv2 as cv
import numpy as np
import os
import random
import sys
import threading
import torch
import time
import math

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def concealment_module(class_list):
    """
    Detects concealment levels and counts instances for each class.

    Args:
        class_list: list of classes from detection results

    Returns:
        A numpy array with four elements representing the counts for each concealment class:
            [high_concealment_count, low_concealment_count, med_concealment_count, no_concealment_count].
    """
    
    concealment_counts = [0, 0, 0, 0]                               # Initialize all class counts to 0
    for class_id in class_list:
        #class_id = int(class_id)
        concealment_counts[int(class_id)] += 1
    return np.array(concealment_counts)

def crowd_density_module(boxes, frame):
    """
    Calculate crowd density based on bounding boxes and the current frame

    Args:
        boxes (list): list of bounding boxes from the detection results
        frame: current frame to be processed
    Output:
        crowd_density (float): crowd density value
    """
    
    segmented_area = 0
    crowd_density = 0

    img = frame
    frame_area = img.shape[0]*img.shape[1]
    existing_boxes = []

    for box in boxes:
        x, y, w, h = box.tolist()
        box_area = w * h
        overlap = 0

        for existing_box in existing_boxes:
            overlap = calculate_overlap((x, y, w, h), existing_box)

        segmented_area += (box_area - overlap) 
        existing_boxes.append((x, y, w, h))

    crowd_density = 100 * segmented_area / frame_area

    #print(f"crowd density: {round(crowd_density, 2)}%")
    return crowd_density

def calculate_overlap(box1, box2):
    # Calculate overlapping area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    
    return overlap

def loitering_module(frame, boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age):
    """
    Updates dwell time of detected objects across frames.
    Args:
        frame: used for annotating the video with the labeled detections
        boxes, track_ids, clss, names: results from YOLO model
        missed_detect: dictionary {Key: track ID, Value: True/False}. False value = not absent in the frame
        misses_cnt: dictionary {Key: track ID, Value: no. of consecutive missed detections}
        dwell_time: dictionary {Key: track ID, Value: dwell time}
        max_age: maximum number of consecutive missed detections used in deleting track IDs

    Output:
        frame: annotated frame
        missed_detect: updated missed_detect
        misses_cnt: updated misses_cnt
        dwell_time: updated dwell_time
        std_val: standard deviation of dwell times

    """
    class_names = ["high","low","med","none"]

    for box, track_id, cls in zip(boxes, track_ids, clss):
        #For dwell time computation- Since ID is present in the frame:
        missed_detect[track_id] = False
        dwell_time[track_id] = dwell_time.get(track_id, 0) + 1 #Increment its dwell time
        misses_cnt[track_id] = 0    #Reset misses_cnt to 0
        
        #Annotate video
        x1, y1, x2, y2 = box
        cls_name = class_names[int(cls)]
        xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
        label = "#{}:{}".format(track_id, dwell_time[track_id])
        annotator = Annotator(frame, line_width=1, example=names)
        annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))


    # Check number of missed detections of each object
    if missed_detect:
        for person_id in list(missed_detect.keys()):
            if person_id not in track_ids:
                misses_cnt[person_id] += 1 #If absent in the current frame, increment misses_cnt
                missed_detect[person_id] = True #Changing to TRUE = absent in the current frame
                #print("ABSENT! ID#{} missed detections = {}".format(person_id, misses_cnt[person_id]))  
                if misses_cnt[person_id] >= max_age:
                    #print("Age of ID#{}: {}>{}".format(person_id, misses_cnt[person_id], max_age))
                    del missed_detect[person_id]
                    del misses_cnt[person_id]
                    del dwell_time[person_id]
                
    #Standard deviation of dwell times
    std_val = np.std(list(dwell_time.values())) if dwell_time else -1
    #print("Dwell time list: ", list(dwell_time.values()), "\nStandard Deviation: ", std_val)

    return frame, missed_detect, misses_cnt, dwell_time, std_val

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Single output neuron for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state for prediction (summary)
        out = self.sigmoid(out) # Apply sigmoid activation
        return out

def infer(input_sequence):
    n_features = 6
    sequence_length = 20
    input_sequence = np.array(input_sequence)

    # Create an instance of the LSTM model
    model = LSTMModel(n_features, hidden_size=64)

    # Load the saved weights
    model.load_state_dict(torch.load('./integration/lstm_model_0.485.pt'))
    model.eval()  # Set the model to evaluation mode

    #input_data = input_sequence[:, 2:].astype(np.float32)
    input_data = input_sequence[:, 1:].astype(np.float32)   #Updated array slicing
    input_data_scaled = scaler.fit_transform(input_data)
    input_data = torch.tensor(input_data_scaled, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))  # Add batch dimension
        RBP = (output).squeeze().cpu().numpy()

    return RBP

def process_video(source):
    #Global variables 
    global yolo_path, bytetrack_path, max_age
    global frame_width, frame_height
    global font_scale, thickness, position, x_text, y_text, WIN_NAME

    #Capture video
    cap = cv.VideoCapture(source)
    fps_start_time = time.time()
    if not cap.isOpened():
        print(f"Error: Could not open {source}. Closing the program.")
        sys.exit()
    
    #Video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    #Display window properties
    font_scale = min(frame_width, frame_height) / 500
    thickness = max(1, int(font_scale * 2))
    x_text, y_text = position = (frame_width - 20, 20)
    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    cv.waitKey(1)

    #Frame variables
    frame_num = 0

    #Crowd density variables

    #Concealment variables

    #Loitering variables
    missed_detect = {} #Dictionary that contains all the tracked object, key:id, value: True/False (false - present in the frame)
    dwell_time = {}
    misses_cnt = {} #Dictionary that stores how many consecutive missed detection for the object

    #Inference variables
    module_result = []  #stores results from the 3 modules for 20 frames
    RBP = 0
    
    if isinstance(source, str):
        fps = int(cap.get(cv.CAP_PROP_FPS))

        #Initialize progress bar
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc=f"Video {source} ")

    
    while cv.waitKey(1) != 27: #ESC key
        has_frame, frame = cap.read()
        if not has_frame:
            break
        
        if isinstance(source, int):
            frame = cv.flip(frame, 1)
        
        frame_num += 1

        # FPS Manual Calculation
        fps_end_time = time.time()
        manual_fps = math.ceil(1 / (fps_end_time - fps_start_time))
        if frame_num == 1:
            average_fps = manual_fps
        else:
            average_fps = math.ceil((average_fps * frame_num + manual_fps) / (frame_num+1))

        # Perform detection & tracking on frame
        results = model.track(frame, persist=True, verbose=False, tracker=bytetrack_path)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu() 
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
        else:
            boxes, track_ids, clss, names = [[] for _ in range(4)]
        
        #Crowd density module
        crowd_density = crowd_density_module(boxes, frame)
        
        #Concealment module
        concealment_counts = concealment_module(clss)

        #Loitering module
        frame, missed_detect, misses_cnt, dwell_time, loitering = loitering_module(frame, boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age)
        
        """
        print([frame_num, 
                  crowd_density, 
                  loitering, 
                  concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])
        """

        if len(module_result) < 20:
            module_result.append([frame_num, 
                  crowd_density, 
                  loitering, 
                  concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])
        else:
            # Make predictions
            RBP = infer(module_result)
            module_result.clear()


        #Progress bar for video
        if isinstance(source, str):
            progress_bar.update(1)

        # Display FPS
        cv.putText(frame, "Average FPS: " + str(average_fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
        #Video annotation
        frame = annotate_video(frame, RBP)
        cv.imshow(WIN_NAME, frame)
    
    if isinstance(source, str):
        progress_bar.close()   
    cap.release()
    cv.destroyAllWindows()


        
def annotate_video(frame, RBP):
    global RBP_threshold, RBP_info
    global font, font_scale, thickness, position, x_text, y_text
    global persist
    
    RBP_text = RBP_info.format(RBP)

    if RBP > RBP_threshold:
            persist = 1
            text_color = (0, 0, 128)  # Red color
    else:
        text_color = (0, 128, 0)   # Green color
        
    size_text = cv.getTextSize(RBP_text, font, font_scale, thickness)[0]
    
    # Calculate the position and size of the rectangle
    x_rect = x_text - 5
    y_rect = y_text - size_text[1] - 5
    width_rect = size_text[0] + 10
    height_rect = size_text[1] + 10

    # Adjust if rectangle goes out of frame
    if x_rect + width_rect > frame.shape[1]:
        x_rect = frame.shape[1] - width_rect
    if y_rect < 0:
        y_rect = 0

    # Draw white background rectangle
    cv.rectangle(frame, (x_rect, y_rect), (x_rect + width_rect, y_rect + height_rect), (255, 255, 255), -1)
    
    # Adjust text position to fit inside the rectangle
    x_text = x_rect + 5
    y_text = y_rect + size_text[1] + 5


    # Add text on top of the rectangle
    cv.putText(frame, RBP_text, (x_text, y_text), font, font_scale, text_color, thickness, cv.LINE_AA)


    # WARNING SIGN
    if persist == 1:
        height, width = frame.shape[0], frame.shape[1]

        # Define the warning text and rectangle properties
        warning_text = "WARNING!"
        warning_font_scale = 3
        warning_font_thickness = 5
        warning_font_color = (255, 255, 255)  # White
        bg_color = (0, 0, 255)  # Red

        # Calculate the text size and position
        w_text_size = cv.getTextSize(warning_text, font, warning_font_scale, warning_font_thickness)[0]
        w_text_x = (width - w_text_size[0]) // 2
        w_text_y = (height + w_text_size[1]) // 2

        # Draw the red background rectangle
        cv.rectangle(frame, (w_text_x - 10, w_text_y - 80), (w_text_x + w_text_size[0] + 10, w_text_y + w_text_size[1] - 50), bg_color, -1)

        # Add the warning text
        cv.putText(frame, warning_text, (w_text_x, w_text_y), font, warning_font_scale, warning_font_color, warning_font_thickness)
   
    return frame

if __name__ == "__main__":
    #---------------YOLO & ByteTrack---------------#
    yolo_path = './yolo_model/best_finalCustom.pt'
    class_names = ["high","low","med","none"]
    try:
        model = YOLO('yolo_model/best_finalCustom.pt')
    except:
        print("Failed to load YOLO model.")
    
    
    bytetrack_path = "./loitering/custom-bytetrack.yaml"
    max_age = 0

    #---------------Video Source---------------#
    #Video Source
    source = "./integration/input-vid" #must be folder name, not file name
    #source = 0
    frame_width = frame_height = fps = 0
    

    #Output file parameters
    #---------------Output Video---------------#
    output_video_path = "out.mp4"
    #TO DO: writing output video    
    #cap_out = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'MP4V'), cap.get(cv.CAP_PROP_FPS),
            #(frame.shape[1], frame.shape[0]))


    #---------------Display window properties---------------#
    RBP_info = ("RBP: {:.2f}")
    RBP_threshold = 0.485
    persist = 0
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = thickness = 0
    x_text = y_text = position = 0
    
    if isinstance(source, int):
        WIN_NAME = "RBP: Camera Feed"
        process_video(source)
    elif isinstance(source, str):
        #List of all video files in the folder_path
        video_files = os.listdir(source)
        for video_file in video_files:
            if video_file.endswith('.mp4'): 
                WIN_NAME = f"RBP: {video_file}"
                video_path = os.path.join(source, video_file)
                process_video(video_path)
            else:
                print("Invalid source.")
                sys.exit()