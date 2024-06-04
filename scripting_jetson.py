"""
TO-DO as of 4/6:
1) Formula for RBP
2) Multi-threading for the 3 modules [In progress - Ysai 4/6]
3) Compare runtime when using single thread vs multithread

NOTES:
    *To run scripting.py (from main directory): python -m inference.scripting
"""
import csv
import cv2
import numpy as np
import os
import random
import threading
import torch

from tqdm import tqdm
# from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from trt_integration.bytetrack.byte_tracker import BYTETracker
import argparse
from argparse import ArgumentParser

from concealment.concealment import concealment_module
from crowd_density.crowd_density import crowd_density_module
from loitering.loitering import loitering_module

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import ctypes
import pycuda.driver as cuda
import time

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
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
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
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
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
        
        return det_res

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


def module_threads():
    return None

def create_csv(csv_filename, field_names):
    """
    Creates CSV file with its header fields

    Args:
        filename: csv file name
        field_names: column headers
    """

    with open(csv_filename, mode="w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()


def vid_processing(folder_path, csv_filename, field_names):
    """
    Opens and processes each video file in "folder_path" directory.
    Each frames is then processed by the 3 modules.

    Args:
        folder_path: folder where the videos are located
        csv_filename: CSV file where outputs from detection modules will be stored
        field_names: CSV header fields
    """
    #---------------INITIALIZING YOLO MODEL---------------#
    try:
        # model = YOLO('yolo_model/best_finalCustom.pt')
        # YOLO Inference with TensorRT
        # Load custom plugin and engine
        PLUGIN_LIBRARY = "./trt_integration/yolo_model/libmyplugins.so"
        engine_file_path = "./trt_integration/yolo_model/best_finalCustom.engine"    

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # Initialize YOLOv8 Detector object using a TensorRT engine file
        model = YoloTRT(library=PLUGIN_LIBRARY, engine=engine_file_path, conf=0.481)

    except:
        print("Failed to load YOLO model.")
    class_names = ["high","low","med","none"]


    #---------------ByteTrack---------------#
    try:
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
    except:
        print("Faled to load ByteTrack")

    #-----LISTING ALL VIDEO FILES IN THE folder_path DIRECTORY-----#
    video_files = os.listdir(folder_path)

    #---------------VIDEO PROCESSING---------------#
    with open(csv_filename, mode="a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        for video_file in video_files: 
            #Loitering parameters all throughout the video
            missed_detect = {} #Dictionary that contains all the tracked object, key:id, value: True/False (false - present in the frame)
            dwell_time = {}
            misses_cnt = {} #Dictionary that stores how many consecutive missed detection for the object
            max_age = 100

            video_path = os.path.join(folder_path, video_file)

            #Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open {video_file}")
                continue

            # Frame varaibles
            frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

            #print("Vid [{}] successfully opened".format(video_file))

            #Initialize progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = tqdm(total=total_frames, desc=f"Video {video_file} ")
            
            #Processing each frame
            frame_num = 0 
            ret, frame = cap.read()
            while ret:
                frame_num += 1

                if frame_num % skip == 0:
                    detections = model.Inference(frame)

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
                        info_imgs = img_size = [frame_height, frame_width]


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
                    
                    else:
                        #No detections
                        #print("No detections")
                        online_boxes, online_ids, clss, class_names = [[] for _ in range(4)]

                    #Crowd density module
                    crowd_density = crowd_density_module(online_boxes, frame)

                    #Loitering module
                    frame, missed_detect, misses_cnt, dwell_time, loitering = loitering_module(frame, online_boxes, online_ids, clss, class_names, missed_detect, misses_cnt, dwell_time, max_age)

                    #Concealment module
                    concealment_counts = concealment_module(clss)

                    # RBP of each video
                    if 'anomaly' in video_file:
                        rbp = 1
                    elif 'normal' in video_file:
                        rbp = 0
                        
                    #print("Video file [{}] Frame #{}: crowd density={}, crowd_count={}, loitering={}, concealment={}".format(video_file, frame_num, crowd_density, crowd_count, loitering, concealment_counts))

                    #Appending data to the CSV file
                    writer.writerow({
                        "video_id": video_file,
                        "frame_num": frame_num,
                        "crowd_density": crowd_density,
                        "loitering": loitering,
                        "no_concealment": concealment_counts[3],
                        "low_concealment": concealment_counts[1],
                        "med_concealment": concealment_counts[2],
                        "high_concealment": concealment_counts[0],
                        "rbp": rbp
                    })

                #Update progress bar
                progress_bar.update(1)

                #Read next frame
                ret, frame = cap.read()
            cap.release()
        progress_bar.close()
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    # skip = 5 # log only every x frames
    skip = 5

    #Declaring CSV filename and header fields
    csv_filename = f"inference/LSTM_v2/conf_0.481/training_csv/train_inf5050_conf0.481_skip{skip}_jetson.csv"
    field_names = ["video_id","frame_num","crowd_density","loitering","no_concealment","low_concealment","med_concealment","high_concealment","rbp"]

    #Declaring folder path of the videos to be processed
    folder_path = r"/home/robbers/Downloads/inference_videos"

    create_csv(csv_filename, field_names)
    vid_processing(folder_path, csv_filename, field_names)
