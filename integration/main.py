import csv
import cv2 as cv
import cProfile
import numpy as np
import os
import pickle
import pstats
import queue
import sys
import threading
import torch
import time

from argparse import ArgumentParser
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import torch
import torch.nn as nn
import numpy as np

def parse_args():
    """Parse command line arguments."""

    parser = ArgumentParser(
        description="Enhancing Robbery Prediction: A Two-Stage System Integrating Human Behavior-Driven Feature Extraction and LSTM-Based Neural Network Inference",
        add_help=True
    )
    
    #YOLO model arguments
    parser.add_argument(
        "--yolo-path",
        type=str,
        default="./yolo_model/best_finalCustom.pt",
        help="Path of YOLO model."
    )

    #Byetrack arguments
        #bytetrack yaml file
    parser.add_argument(
        "--bytetrack-path",
        type=str,
        default = "./loitering/custom-bytetrack.yaml",
        help="Path of Byterack configuration file."
    )

    parser.add_argument(
        "---max-age",
        type=int,
        default=100,
        help="Maximum consecutive missed detections before deleting ID."
    )
    
    #Source input arguments
    parser.add_argument(
        "--source",
        type=str,
        default="./integration/input-vid",
        help="For camera: 0. For video: [Folder path] where video/s is stored."
    )

    #User settings
        #Output video destination
        #Display window
        #Save log file
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=6,
        help="Frames to be skipped during processing."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use."
    )

    parser.add_argument(
        "--save-vid",
        type=str,
        default="0",
        help="Saves annotated video if enabled.",
    )

    parser.add_argument(
        "--no-display",
        action="store_false",
        help="Disables playing of video while processing.",
    )

    parser.add_argument(
        "--no-thread",
        action="store_true",
        help="Disables multi-threading (skipping only).",
    )


    return parser.parse_args()

def annotate_video(frame, RBP):
    global RBP_threshold, RBP_info
    global frame_width, frame_height
    global font, font_scale, thickness, position, x_text, y_text, size_text
    global x_rect, y_rect, width_rect, height_rect
    global warning_text, warning_font_scale, warning_font_thickness, warning_font_color, bg_color
    global w_text_size, w_text_x, w_text_y, w_rect_x, w_rect_y, w_width_rect, w_height_rect
    global persist

    # frame = cv.resize(frame, (frame_width, frame_height))
    
    # Display RBP
    RBP_text = RBP_info.format(RBP)

    if RBP > RBP_threshold:
        persist = 1
        text_color = (0, 0, 128)  # Red color
    else:
        text_color = (0, 128, 0)   # Green color

    # Draw white background rectangle
    cv.rectangle(frame, (x_rect, y_rect), (x_rect + width_rect, y_rect + height_rect), (255, 255, 255), -1)
    
    # Add text on top of the rectangle
    cv.putText(frame, RBP_text, (x_text, y_text), font, font_scale, text_color, thickness, cv.LINE_AA)

    # WARNING SIGN
    if persist:
        # Draw the red background rectangle
        cv.rectangle(frame, (w_rect_x, w_rect_y), (w_rect_x + w_width_rect, w_rect_y + w_height_rect), bg_color, -1)

        # Add the warning text
        cv.putText(frame, warning_text, (w_text_x, w_text_y), font, warning_font_scale, warning_font_color, warning_font_thickness)
   
    return frame

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

def crowd_density_module(boxes):
    """
    Calculate crowd density based on bounding boxes and the current frame

    Args:
        boxes (list): list of bounding boxes from the detection results
    Output:
        crowd_density (float): crowd density value
    """
    global frame_area
    segmented_area = 0
    crowd_density = 0

    #img = frame
    #frame_area = img.shape[0]*img.shape[1]
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

def loitering_module(boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age):
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
        missed_detect: updated missed_detect
        misses_cnt: updated misses_cnt
        dwell_time: updated dwell_time
        std_val: standard deviation of dwell times

    """
    global save_vid, display_vid
    global class_names

    for box, track_id, cls in zip(boxes, track_ids, clss):
        #For dwell time computation- Since ID is present in the frame:
        missed_detect[track_id] = False
        dwell_time[track_id] = dwell_time.get(track_id, 0) + 1 #Increment its dwell time
        misses_cnt[track_id] = 0    #Reset misses_cnt to 0
        
        #Annotate video
        """
        if save_vid or display_vid:
            x1, y1, x2, y2 = box
            cls_name = class_names[int(cls)]
            xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
            label = "#{}:{}".format(track_id, dwell_time[track_id])
            annotator = Annotator(frame, line_width=1, example=names)
            annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))
        """

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

    return missed_detect, misses_cnt, dwell_time, std_val

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
    input_sequence = np.array(input_sequence)

    #input_data = input_sequence[:, 2:].astype(np.float32)
    input_data = input_sequence[:, 1:].astype(np.float32)   #Updated array slicing

    input_data_scaled = scaler.transform(input_data)
    input_data = torch.tensor(input_data_scaled, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        output = lstm_model(input_data.unsqueeze(0))  # Add batch dimension
        RBP = (output).squeeze().cpu().numpy()

    return RBP

def capture_frames(source):
    global capture_thread_done, frame_queue
    global save_vid, display_vid
    global frame_height, frame_width
    global skip

    frame_num = 0
    cap = cv.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open {source}. Closing the program.")
        sys.exit()

    while not thread_interrupt:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        #if isinstance(source, int):
            #frame = cv.flip(frame, 1)

        frame_num += 1
        if frame_num % skip == 0:
            frame = cv.resize(frame, (frame_width,frame_height))
            frame_queue.put((frame_num, frame))
            # to_annotate_queue.put((frame_num, frame))
            # print(f"T1: Adding to buffer: frame# {frame_num}")
        
        #Test
        # if frame_num == 20:
        #     break
    cap.release()
    capture_thread_done = True

    # if thread_interrupt:
    #     print(">>>>>T1: Keyboard Interrupt. Thread terminating.<<<<<")    
    # else:
    #     print("------T1: ALL FRAMES CAPTURED. CAPTURE_FRAME THREAD STOP.------")


def process_frames(filename):
    global yolo_path, bytetrack_path, max_age
    global frame_queue, capture_thread_done, process_thread_done, thread_interrupt
    global frame_width, frame_height
    global WIN_NAME
    global display_vid, save_vid, output_path
    global persist, RBP_threshold

    #Frame variables
    # frames = []
    # frames_cnt = []
    # frames_dict = {}    #id: cnt, val: frame
    
    #Loitering variables
    missed_detect = {}
    misses_cnt = {}
    dwell_time = {}

    #Inference variables
    module_result = []  #stores results from the 3 modules for 20 frames
    RBP = 0

    #Test variables
    # module_result_test = []
    # model_test = YOLO(yolo_path)

    if save_vid:
        cap_out = cv.VideoWriter(
            output_path + "/annotated-" + filename,
            cv.VideoWriter_fourcc(*'MP4V'),
            30.0, #cap.get(cv.CAP_PROP_FPS)     #FPS of output video is set to 30FPS
            (frame_width, frame_height)
        )
    if display_vid:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
        cv.waitKey(1)
    
    fps_start_time = time.perf_counter()
    while not thread_interrupt: #ESC key:
        try:
            #Getting frame from buffer
            frame_num, frame = frame_queue.get(timeout=1)

            #Testing
            # frames.append(frame)
            # frames_cnt.append(frame_num)

            #print(f"    ------T2: Extrated from buffer & start processing: frame# {frame_num}")
            
            #Perform detection & tracking on frame
            results = model.track(frame, conf=0.481, persist=True, verbose=False, tracker=bytetrack_path)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu() 
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                names = results[0].names
            else:
                boxes, track_ids, clss, names = [[] for _ in range(4)]
            
            #Feed detection results to the modules
            crowd_density = crowd_density_module(boxes)
            concealment_counts = concealment_module(clss)
            missed_detect, misses_cnt, dwell_time, loitering = loitering_module(boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age)

            module_result.append([frame_num, 
                crowd_density, 
                loitering, 
                concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])

            # print(f"    ------T2: Done processing: frame# {frame_num}")
            
            if len(module_result) == 20:
                RBP = infer(module_result)
                if RBP > RBP_threshold:
                    warning_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Robbery warning at: {warning_time}")
                # RBP_val = RBP
                module_result = []
                # print(f"                        >>>T2: RBP inference done.<<<")

            # FPS Manual Calculation
            fps_end_time = time.perf_counter()
            time_diff = fps_end_time - fps_start_time
            if time_diff == 0:
                manual_fps = 0.0
            else:
                manual_fps = (skip / time_diff)

            with open(csv_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([filename, frame_num, manual_fps])
            fps_start_time = time.perf_counter()
            
            if save_vid or display_vid:          
                #Video annotation
                annotated_frame = annotate_video(frame, RBP)

                if display_vid:
                    # print(f"    ------T2: Displaying annotated frame #{frame_num}")
                    cv.imshow(WIN_NAME, annotated_frame)
                    key = cv.waitKey(1)
                    if key == 27:
                        #print("Terminating thread2")
                        thread_interrupt = True
                        break
                    elif key == ord("Q") or key == ord("q"):
                        persist = 0
                        #print(f"Q pressed. Persist:{persist}")

                if save_vid:
                    cap_out.write(frame)
        except queue.Empty:
            if capture_thread_done:
                process_thread_done = True
                # print("    ------T2: EMPTY BUFFER. NOT ENOUGH FRAMES FOR PROCESSING.------")
                break
            else:
                continue
    
    if save_vid:
        cap_out.release()
    if display_vid:
        cv.destroyAllWindows()

    process_thread_done = True

    # if thread_interrupt:
    #     print(">>>>>T2: Keyboard Interrupt. Thread terminating.<<<<<")    
    # else:
    #     print("    ------T2: PROCESS_FRAME THREAD STOP.------")

def process_video_thread(source, filename):
# def process_video(source, filename):
    global frame_queue, capture_thread_done, process_thread_done, thread_interrupt
    global persist
    try:
        capture_thread = threading.Thread(target=capture_frames, args=(source,))
        process_thread = threading.Thread(target=process_frames, args=(filename,))

        print("---------------------------------")
        print(f"Process_video: {filename}")
        print("Starting threads. CTRL+C to terminate threads.")
        capture_thread.start()
        process_thread.start()

        while not thread_interrupt:
            if process_thread_done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(">>>>>Process_video: Keyboard interrupt. Stopping threads<<<<<")
        #persist = 0
        thread_interrupt = True


    capture_thread.join()
    process_thread.join()
    print(f"Process_video: {filename} done")

    # Clearing global variables
    with frame_queue.mutex:
        frame_queue.queue.clear()
    
    #Checking if queue is indeed cleared
    if frame_queue.empty():
        print("Frame queue is empty.")
    
    capture_thread_done = False
    process_thread_done = False
    thread_interrupt = False
    persist = 0

def process_video_no_thread(source, filename):
    #Global variables 
    global yolo_path, bytetrack_path, max_age
    global skip
    global frame_width, frame_height
    global font_scale, thickness, position, x_text, y_text, WIN_NAME
    global display_vid, save_vid, output_path, persist

    print("---------------------------------")
    print(f"Process_video: {filename}")

    #Capture video
    cap = cv.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Could not open {source}. Closing the program.")
        sys.exit()
    
    
    if save_vid:
        cap_out = cv.VideoWriter(
            output_path + "/annotated-" + filename,
            cv.VideoWriter_fourcc(*'MP4V'),
            cap.get(cv.CAP_PROP_FPS),
            (frame_width, frame_height)
        )

    if display_vid:
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

    manual_fps = 0.0 
    fps_start_time = time.perf_counter()

    while True:

        has_frame, frame = cap.read()
        if not has_frame:
            break
        
        if isinstance(source, int):
            frame = cv.flip(frame, 1)
        
        frame_num += 1

        if frame_num % skip == 0:
            # Perform detection & tracking on frame
            frame = cv.resize(frame, (frame_width,frame_height))
            results = model.track(frame, conf=0.481, persist=True, verbose=False, tracker=bytetrack_path)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu() 
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                names = results[0].names
            else:
                boxes, track_ids, clss, names = [[] for _ in range(4)]
            
            #Crowd density module
            crowd_density = crowd_density_module(boxes)
            
            #Concealment module
            concealment_counts = concealment_module(clss)

            #Loitering module
            missed_detect, misses_cnt, dwell_time, loitering = loitering_module(boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age)
            
            """
            print([frame_num, 
                    crowd_density, 
                    loitering, 
                    concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])
            """
            
            module_result.append([frame_num, 
                    crowd_density, 
                    loitering, 
                    concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])

                
            # if (len(module_result) == 20) and (not persist):
            if len(module_result) == 20:
                # Make predictions
                # print(module_result) #testing
                RBP = infer(module_result)
                if RBP > RBP_threshold:
                    warning_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Robbery warning at: {warning_time}")
                module_result.clear()

            # FPS Manual Calculation
            fps_end_time = time.perf_counter()
            time_diff = fps_end_time - fps_start_time
            if time_diff == 0:
                manual_fps = 0.0
            else:
                manual_fps = (skip / time_diff)

            with open(csv_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([filename, frame_num, manual_fps])

            fps_start_time = time.perf_counter()


        if save_vid or display_vid:          
                #Video annotation
            annotated_frame = annotate_video(frame, RBP)

            if display_vid:
                cv.imshow(WIN_NAME, annotated_frame)
                key = cv.waitKey(1)
                if key == 27:
                    break
                elif key == ord("Q") or key == ord("q"):
                    persist = 0
                    #print(f"Q pressed. Persist:{persist}")

            if save_vid:
                cap_out.write(frame)
    
    cap.release()
    if save_vid:
        cap_out.release()
    if display_vid:
        cv.destroyAllWindows()
    print(f"Process_video: {filename} done")

if __name__ == "__main__":
    args = parse_args() 
    skip = args.skip_frames

    #---------------YOLO & ByteTrack---------------#
    yolo_path = args.yolo_path
    class_names = ["high","low","med","none"] #Based on roboflow training
    try:
        model = YOLO(yolo_path)
    except:
        print("Failed to load YOLO model.")
        sys.exit()
    
    bytetrack_path = args.bytetrack_path
    max_age = args.max_age

    #---------------RBP Thresholds---------------#
    if skip == 1:
        f1 = 0.7368
        RBP_threshold = 0.514
    elif skip == 2:
        f1 = 0.7234
        RBP_threshold = 0.492
    elif skip == 3:
        f1 = 0.7556
        RBP_threshold = 0.481      
    elif skip == 4:
        f1 = 0.6957
        RBP_threshold = 0.459
    elif skip == 5:
        f1 = 0.7273
        RBP_threshold = 0.478
    elif skip == 6:
        f1 = 0.75
        RBP_threshold = 0.409

    #---------------Inference LSTM Model Loading and Feature Scaling---------------#
    n_features = 6
    sequence_length = 20

    # Create an instance of the LSTM model
    lstm_model = LSTMModel(n_features, hidden_size=64)

    # Load the saved weights
    lstm_model_path = f'./inference/LSTM_v2/conf_0.481/_jetson/lstm_models_jetson/lstm_model_skip{skip}_f1={f1}_th={RBP_threshold:.3f}_jetson.pt'
    lstm_model.load_state_dict(torch.load(lstm_model_path))
    print(f"Loaded LSTM model = lstm_model_skip{skip}_f1={f1}_th={RBP_threshold:.3f}_jetson.pt")
    lstm_model.eval()  # Set the model to evaluation mode

    with open(f'./inference/LSTM_v2/conf_0.481/_jetson/scaler_jetson/scaler_skip{skip}_jetson.pkl','rb') as file:
        scaler = pickle.load(file)  

    #---------------Source---------------#
    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # If conversion fails, it's a string
    #frame_width = frame_height = fps = 0
    
    #---------------Output Video---------------#
    try:
        save_vid = int(args.save_vid)
    except ValueError:
        save_vid = True
        output_path = args.save_vid
        os.makedirs(output_path, exist_ok=True)  # Create the output folder if it doesn't exist

    #---------------Display window properties---------------#
    display_vid = args.no_display
    RBP_info = ("RBP: {:.3f}")

    persist = 0
    font = cv.FONT_HERSHEY_SIMPLEX

    frame_width = 320
    frame_height = 240
    frame_area = frame_height*frame_width
    font_scale = min(frame_width, frame_height) / 500
    thickness = max(1, int(font_scale * 2))
    x_text, y_text = position = (frame_width - 30, 20)
    size_text = (115, 16)

    # Calculate the position and size of the rectangle for RBP
    x_rect = x_text
    y_rect = y_text - size_text[1] - 5
    width_rect = size_text[0]
    height_rect = size_text[1] + 10

    # Adjust if rectangle goes out of frame
    
    if x_rect + width_rect > frame_width:
        x_rect = frame_width - width_rect
    if y_rect < 0:
        y_rect = 0

    # Adjust text position to fit inside the rectangle
    x_text = x_rect + 5
    y_text = y_rect + size_text[1] + 5

    # Define the warning text and rectangle properties
    warning_text = "WARNING!"
    warning_font_scale = font_scale*3
    warning_font_thickness = thickness*2
    warning_font_color = (255, 255, 255)  # White
    bg_color = (0, 0, 255)  # Red

    # Calculate the text size and position
    w_text_size = cv.getTextSize(warning_text, font, warning_font_scale, warning_font_thickness)[0]
    w_text_x = (frame_width - w_text_size[0]) // 2
    w_text_y = (frame_height + w_text_size[1]) // 2

    # Calculate the position and size of the rectangle
    w_rect_x = w_text_x - 5
    w_rect_y = w_text_y - w_text_size[1] - 5
    w_width_rect = w_text_size[0] + 10
    w_height_rect = w_text_size[1] + 10

    #---------------Peformance profiling---------------#
    profiling_folder = 'integration/profiling'  # Define the profiling folder

    # Create the profiling folder if it doesn't exist
    if not os.path.exists(profiling_folder):
        os.makedirs(profiling_folder)

    # CSV file to log fps
    csv_file = os.path.join(profiling_folder, f'fps_log-thread-skip{skip}.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'frame_num', 'fps'])

    #---------------Threading Variables---------------#
    capture_thread_done = False
    process_thread_done = False
    thread_interrupt = False
    frame_queue = queue.Queue(maxsize=1000)  # Buffer size
    no_thread = args.no_thread
    print(f"Default No_thread:{no_thread}")

    #---------------MAIN PROCESSING---------------#
    if isinstance(source, int):
        WIN_NAME = "RBP: Camera Feed"
        with cProfile.Profile() as pr:
            #process_video(source, "Camera") #<<<<<<<<<
            if no_thread:
                print(">>>>>>>>>>>STARTING NO THREAD PROCESS VIDEO")
                process_video_no_thread(0, "Camera")
            else:
                process_video_thread(0, "Camera")
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        #stats.print_stats()
        profile_filename = os.path.join(profiling_folder, f"profiling_cam-thread-skip{skip}.prof")
        stats.dump_stats(filename=profile_filename)

    elif isinstance(source, str):
        #List of all video files in the folder_path
        video_files = os.listdir(source)
        try:
            with cProfile.Profile() as pr:
                for video_file in video_files:
                    if video_file.endswith('.mp4'): 
                        WIN_NAME = f"RBP: {video_file}"
                        video_path = os.path.join(source, video_file)
                        
                        # process_video(video_path, video_file)
                        if no_thread:
                            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Video started at: {start_time}")
                            process_video_no_thread(video_path, video_file)
                        else:
                            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Video started at: {start_time}")
                            process_video_thread(video_path, video_file)
                    else:
                        print("Invalid source.")
                        continue

            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)

            # Save the profiling stats in the profiling folder
            # stats.print_stats()
            profile_filename = os.path.join(profiling_folder, f"profiling_vid-thread-skip{skip}.prof")
            stats.dump_stats(filename=profile_filename)

        except Exception as e:
            print(f"Profiling error: {e}")