import sys
import os
import cv2 as cv
import csv
import cProfile
import pstats
import argparse
import shutil
import random
import time
from pathlib import Path
from loguru import logger

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from trt_integration.bytetrack.byte_tracker import BYTETracker
from trt_integration.yoloDet import YoloTRT

from argparse import ArgumentParser
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

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
    # Display window
    parser.add_argument(
        "--no-display",
        action="store_false",
        help="Disables playing of video while processing.",
    )
    args = parser.parse_args()
    return args

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

def process_video(source):

    # Global variables 
    global WIN_NAME
    global display_vid
    
    # Capture source
    if args.input == 'video':    
        cap = cv.VideoCapture(source)
    else:
        cap = cv.VideoCapture(gstreamer_pipeline(sensor_id=source,flip_method=0), cv.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Error: Could not open {source}. Closing the program.")
        sys.exit()    

    if display_vid:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
        cv.waitKey(1)

    # Iterate through each frame of the video        
    while cv.waitKey(1) != 27: #ESC key
        # Read a frame from the video capture
        has_frame, frame = cap.read()
        if not has_frame:
            break  

    cap.release()
    if display_vid:
        cv.destroyAllWindows()
        
if __name__ == "__main__":

    #---------------ARGS---------------#
    
    args = parse_args()
    # Log the command-line args for reference
    logger.info("Args: {}".format(args))
    
    #--------------- Source---------------#
    if args.input == "video":
        source = "integration/input-vid"
    else:
        source = 0    
        
    #---------------Display window properties---------------#

    display_vid = args.no_display
    
    #---------------Processing the source---------------#

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
        