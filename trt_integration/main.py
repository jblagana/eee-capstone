import sys
import os
import cv2 as cv
import csv
import cProfile
import pstats
import argparse
import random
import ctypes
import time
import pickle
from argparse import ArgumentParser
from loguru import logger

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import threading

from trt_integration.bytetrack.byte_tracker import BYTETracker
# from trt_integration.bytetrack.byte_tracker_v2 import BYTETracker
from ultralytics.utils.plotting import Annotator, colors

import torch
import torch.nn as nn
import numpy as np

class YoloTRT():
    def __init__(self, library, engine, conf):
        self.CONF_THRESH = conf 
        self.IOU_THRESHOLD = 0.4
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.POSE_NUM = 17 * 3
        self.DET_NUM = 6
        self.SEG_NUM = 32
        self.categories = ["high_conc","low_conc","med_conc","no_conc"]
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        # Deserialize the engine from file
        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        
        # Store
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.det_output_length = host_outputs[0].shape[0]

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv.resize(image, (tw, th))
        image = cv.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def Inference(self, frame):
        img = frame
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]
        
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.PostProcess(output[i * self.det_output_length: (i + 1) * self.det_output_length], origin_h, origin_w)            

        det_res = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            det = dict()
            det["class_id"] = int(result_classid[j])
            det["conf"] = result_scores[j]
            det["box"] = box            
            det_res.append(det)
            # self.PlotBbox(box, img, color = colors(int(result_classid[j]), True), label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]),)
        
        return det_res, img

    def PostProcess(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        num_values_per_detection = self.DET_NUM + self.SEG_NUM + self.POSE_NUM
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimensional ndarray
        pred = np.reshape(output[1:], (-1, num_values_per_detection))[:num, :]
        # Do nms
        boxes = self.NonMaxSuppression(pred, origin_h, origin_w, conf_thres=self.CONF_THRESH, nms_thres=self.IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def NonMaxSuppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections        
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)                
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
    
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h
        return y
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = (np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
                      * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None))
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    
    def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
        # color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
            cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA,)

def parse_args():
    parser = ArgumentParser(
        description="Robbery Prediction",
        add_help=True
    )
    # Input Arguments    
    parser.add_argument(
        "--input",
        default="video",
        help="Input type: 'video' or 0/1 (for CSI camera)"
    )
    # ByteTrack Arguments
    parser.add_argument(
        "--max-age",
        type=int,
        default=500,
        help="Maximum consecutive missed detections before deleting ID."
    )    
    # Display window
    parser.add_argument(
        "--no-display",
        action="store_false",
        help="Disables playing of video while processing. Default = True.",
    )
    # Profiling
    parser.add_argument(
        "--no-profile",
        action="store_false",
        help="Disables profiling of code.",
    )
    # FPS Logging
    parser.add_argument(
        "--no-fps-log",
        action="store_false",
        help="Disables logging of FPS",
    )
    # Annotation
    parser.add_argument(
        "--no-annotate",
        action="store_false",
        help="Disables annotation of frame",
    )    
    # Skip frames
    parser.add_argument(
        "--skip-frames",
        default=1,
        help="Enables skipping of frames by input number. Default = 1 (no skip)",
    )
    args = parser.parse_args()
    return args

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

def loitering_module(frame, boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age):
    """
    Updates dwell time of detected objects across frames.
    Args:
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
    global display_vid
    global class_names

    for box, track_id, cls in zip(boxes, track_ids, clss):
        #For dwell time computation- Since ID is present in the frame:
        missed_detect[track_id] = False
        dwell_time[track_id] = dwell_time.get(track_id, 0) + 1 #Increment its dwell time
        misses_cnt[track_id] = 0    #Reset misses_cnt to 0
        
        # Annotate video
        x1, y1, x2, y2 = box
        cls_name = class_names[int(cls)]
        xyxy = [x1,y1,x2,y2]
        # xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
        label = "#{}:{}".format(track_id, dwell_time[track_id])
        annotator = Annotator(frame, line_width=1, example=names)
        annotator.box_label(xyxy, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))

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
    global skip
    n_features = 6
    sequence_length = 20
    input_sequence = np.array(input_sequence)

    # Create an instance of the LSTM model
    model = LSTMModel(n_features, hidden_size=64)

    # Load the saved weights
    if skip == 1:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip1_0.503.pt'))   #No Skip
    elif skip == 2:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip2_0.484.pt'))        #Skip = 2
    elif skip == 3:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip3_0.484.pt'))        #Skip = 3
    elif skip == 4:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip4_0.507.pt'))        #Skip = 4
    elif skip == 5:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip5_0.500.pt'))        #Skip = 5
    elif skip == 6:
        model.load_state_dict(torch.load('./inference/LSTM_v2/skipping_analysis/lstm_models/lstm_model_skip6_0.370.pt'))        #Skip = 6

    model.eval()  # Set the model to evaluation mode

    #input_data = input_sequence[:, 2:].astype(np.float32)
    input_data = input_sequence[:, 1:].astype(np.float32)   #Updated array slicing

    input_data_scaled = scaler.transform(input_data)
    input_data = torch.tensor(input_data_scaled, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))  # Add batch dimension
        RBP = (output).squeeze().cpu().numpy()

    return RBP

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def process_video(source, filename):
    """Performs real-time object detection and tracking on a video."""

    # Global variables 
    global model, tracker, max_age
    global frame_width, frame_height, capture_width, capture_height
    global font_scale, thickness, position, x_text, y_text, WIN_NAME
    global display_vid, annotate, skip
    
    # Capture source
    if args.input == 'video':    
        cap = cv.VideoCapture(source)
    else:
        cap = cv.VideoCapture(gstreamer_pipeline(sensor_id=source, flip_method=0), cv.CAP_GSTREAMER)
            
    if not cap.isOpened():
        print(f"Error: Could not open {source}. Closing the program.")
        sys.exit()    

    if display_vid:
        cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
        cv.waitKey(1)

    #Frame variables
    frame_num = 0

    # #Loitering variables
    missed_detect = {} # Dictionary that contains all the tracked object, key:id, value: True/False (false - present in the frame)
    dwell_time = {}
    misses_cnt = {} # Dictionary that stores how many consecutive missed detection for the object
    names = {0: 'high_conc', 1: 'low_conc', 2: 'med_conc', 3: 'no_conc'}

    # Inference variables
    module_result = []  #stores results from the 3 modules for 20 frames
    RBP = 0

    if fps_log:
        manual_fps = 0.0 
        fps_start_time = time.perf_counter()

    # Iterate through each frame of the video
    while cv.waitKey(1) != 27: #ESC key
        
        # Read a frame from the video capture
        has_frame, frame = cap.read()
        if not has_frame:
            break  
        frame_num += 1

        # Skip frames
        if frame_num % skip == 0:

            # YOLO Inference with TensorRT
            detections, frame = model.Inference(frame)
            logger.info("Detections: {}".format(detections))

            # If detection results is not empty
            if detections:
                
                # Format Conversion and Filtering
                output = []   
                boxes = []    
                clss = []      
                for detection in detections:      
                    box = detection["box"]
                    conf = detection["conf"]
                    cls = detection["class_id"]
                    output.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), conf])
                    # output.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), conf, cls])
                    boxes.append([box[0], box[1], box[2], box[3]])
                    clss.append(cls)
                output = torch.tensor(output, dtype=torch.float32)
                # output = np.array(output)                                   # defaults to higher precision float64
                boxes = torch.tensor(boxes, dtype=torch.float32)              # xyxy
                if args.input == 'video':
                    info_imgs = img_size = [frame_height, frame_width]
                else:
                    info_imgs = img_size = [capture_height, capture_width]

                # Tracking
                if len(output) != 0 :
                    # Call the ByteTracker.update method with the filtered detections, frame information, and image size.
                    online_targets = tracker.update(output, info_imgs, img_size)

                    # Extracting  information about the tracked objects
                    online_boxes = []
                    online_ids = []    

                    # Iterating through updated tracks
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        online_boxes.append(tlwh)
                        online_ids.append(tid)
                online_boxes = torch.tensor(online_boxes, dtype=torch.float32)

                #Crowd density module
                crowd_density = crowd_density_module(online_boxes)

                #Concealment module
                concealment_counts = concealment_module(clss)

                #Loitering module
                missed_detect, misses_cnt, dwell_time, loitering = loitering_module(frame, boxes, online_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age)

                module_result.append([frame_num, 
                        crowd_density, 
                        loitering, 
                        concealment_counts[3], concealment_counts[1], concealment_counts[2], concealment_counts[0]])

                # Make predictions every 20 frames
                if len(module_result) == 20:
                    RBP = infer(module_result)
                    # Log module results
                    with open(csv_file_module_result, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow([filename, frame_num, output, online_boxes, module_result, RBP])
                    module_result.clear()
                
            elif not detections:
                print("No detections found!")

            # FPS Manual Calculation
            if fps_log:
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

        if display_vid:
                if annotate:
                    frame = annotate_video(frame, RBP)
                cv.imshow(WIN_NAME, frame)

    cap.release()
    if display_vid:
        cv.destroyAllWindows()

def reset_persist():
    """Function to reset the persist variable"""
    global persist
    if persist == 1:
        persist = 0
        print("Persist reset to 0")

def set_persist(value, delay):
    """Function to set persist and start the timer"""
    global persist
    persist = value
    if value == 1:
        print("Persist set to 1")
        timer = threading.Timer(delay, reset_persist)
        timer.start()

def annotate_video(frame, RBP):
    global RBP_threshold, RBP_info
    global frame_width, frame_height
    global font, font_scale, thickness, position, x_text, y_text, size_text
    global x_rect, y_rect, width_rect, height_rect
    global warning_text, warning_font_scale, warning_font_thickness, warning_font_color, bg_color
    global w_text_size, w_text_x, w_text_y, w_rect_x, w_rect_y, w_width_rect, w_height_rect
    global persist

    frame = cv.resize(frame, (frame_width, frame_height))
    
    # Display RBP
    RBP_text = RBP_info.format(RBP)

    if RBP > RBP_threshold:
        if args.input == 'video':
            persist = 1                 # Set persist to 1 (for the duration of the vid)
        else:
            set_persist(1, 3)           # Set persist to 1 and reset it after 3 seconds
        text_color = (0, 0, 128)        # Red color
    else:
        text_color = (0, 128, 0)        # Green color

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

if __name__ == "__main__":

    #---------------ARGS---------------#
    
    args = parse_args()
    # Log the command-line args for reference
    logger.info("Args: {}".format(args))

    #---------------YOLOv8 & TensorRT---------------#

    # Load custom plugin and engine
    PLUGIN_LIBRARY = "./trt_integration/yolo_model/libmyplugins.so"
    engine_file_path = "./trt_integration/yolo_model/best_finalCustom.engine"    

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # Initialize YOLOv8 Detector object using a TensorRT engine file
    model = YoloTRT(library=PLUGIN_LIBRARY, engine=engine_file_path, conf=0.25)

    #---------------ByteTrack---------------#
    args_bytetrack = argparse.Namespace()
    # args_bytetrack.track_thresh = 0.5
    args_bytetrack.track_high_thresh = 0.5  # threshold for the first association
    args_bytetrack.track_low_thresh = 0.1   # threshold for the second association
    args_bytetrack.new_track_thresh = 0.6   # threshold for init new track if the detection does not match any tracks
    args_bytetrack.track_buffer = 500       # buffer to calculate the time when to remove tracks; default = 30
    args_bytetrack.mot20 = False
    args_bytetrack.match_thresh = 0.8       # threshold for matching tracks

    # Initializes a ByteTrack tracker object
    tracker = BYTETracker(args_bytetrack)
    max_age = args.max_age
    class_names = ["high", "low", "med", "none"]
    
    #--------------- Source---------------#
    if args.input == "video":
        source = "./integration/input-vid"
    else:
        source = 0

    # For video
    frame_width = frame_height = fps = 0
    # For camera capture
    capture_width = 1920
    capture_height = 1080

    #---------------Display window properties---------------#

    display_vid = args.no_display
    annotate = args.no_annotate
    skip = int(args.skip_frames)

    RBP_info = ("RBP: {:.2f}")

    if skip == 1:
        RBP_threshold = 0.503
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip1.pkl','rb') as file:
            scaler = pickle.load(file)
    elif skip == 2:
        RBP_threshold = 0.484
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip2.pkl','rb') as file:
            scaler = pickle.load(file)
    elif skip == 3:
        RBP_threshold = 0.484
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip3.pkl','rb') as file:
            scaler = pickle.load(file)
    elif skip == 4:
        RBP_threshold = 0.507
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip4.pkl','rb') as file:
            scaler = pickle.load(file)
    elif skip == 5:
        RBP_threshold = 0.500
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip5.pkl','rb') as file:
            scaler = pickle.load(file)
    elif skip == 6:
        RBP_threshold = 0.370
        with open('./inference/LSTM_v2/skipping_analysis/scaler/scaler_skip6.pkl','rb') as file:
            scaler = pickle.load(file) 

    logger.info("RBP Threshold: {}".format(RBP_threshold))

    persist = 0
    font = cv.FONT_HERSHEY_SIMPLEX

    frame_width = 640       #360p: 640x360 pixels
    frame_height = 360
    frame_area = frame_height*frame_width
    font_scale = min(frame_width, frame_height) / 500
    thickness = max(1, int(font_scale * 2))
    x_text, y_text = position = (frame_width - 20, 20)
    size_text = (115, 16)

    # Calculate the position and size of the rectangle
    x_rect = x_text - 5
    y_rect = y_text - size_text[1] - 5
    width_rect = size_text[0] + 10
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
    
    #---------------Performance Profiling---------------#

    # Peformance profiling
    profiling_folder = './trt_integration/profiling'  # Define the profiling folder

    # Create the profiling folder if it doesn't exist
    if not os.path.exists(profiling_folder):
        os.makedirs(profiling_folder)

    #---------------Log Module Result---------------#
        
    # CSV file to log module result
    csv_file_module_result = os.path.join(profiling_folder, 'module_result_trt.csv')
    with open(csv_file_module_result, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'frame_num', 'boxes', 'online_boxes', 'module_result', 'RBP'])

    #---------------Log FPS---------------#
    fps_log = args.no_fps_log

    if fps_log:
        # CSV file to log fps
        csv_file = os.path.join(profiling_folder, 'fps_log.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'frame_num', 'fps'])
    
    #---------------Processing/Profiling---------------#

    profile_code = args.no_profile 

    # No Profiling
    if not profile_code:
        if isinstance(source, int):
            WIN_NAME = "RBP: Camera Feed"
            process_video(source, 'camera')
            
        elif isinstance(source, str):
            #List of all video files in the folder_path
            video_files = os.listdir(source)                    
            
            for video_file in video_files:
                if video_file.endswith('.mp4'):
                    WIN_NAME = f"RBP: {video_file}"
                    video_path = os.path.join(source, video_file)
                    process_video(video_path, video_file)
                    persist = 0

    # With Profiling
    else:
        if isinstance(source, int):
            try:
                with cProfile.Profile() as pr:
                    WIN_NAME = "RBP: Camera Feed"
                    process_video(source, 'camera')
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                #stats.print_stats()
                profile_filename = os.path.join(profiling_folder, f"profiling_total.prof")
                stats.dump_stats(filename=profile_filename)
                
            except Exception as e:
                print(f"Profiling error: {e}")
                
        elif isinstance(source, str):
            #List of all video files in the folder_path
            video_files = os.listdir(source)

            try:
                with cProfile.Profile() as pr:
                    for video_file in video_files:
                        if video_file.endswith('.mp4'): 
                            WIN_NAME = f"RBP: {video_file}"
                            video_path = os.path.join(source, video_file)
                            process_video(video_path, video_file)
                            persist = 0
                        else:
                            print("Invalid source.")
                            sys.exit()

                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)

                # Save the profiling stats in the profiling folder
                # stats.print_stats()
                profile_filename = os.path.join(profiling_folder, f"profiling_total.prof")
                stats.dump_stats(filename=profile_filename)

            except Exception as e:
                print(f"Profiling error: {e}")
