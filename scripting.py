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

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from concealment.concealment import updated_concealment
from crowd_density.density_custom import detect_crowd_density
from loitering.loitering import detect_loitering


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
        model = YOLO('yolo_model/best_finalCustom.pt')
    except:
        print("Failed to load YOLO model.")
    class_names = ["high","low","med","none"]

    
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
            
            #print("Vid [{}] successfully opened".format(video_file))

            #Initialize progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = tqdm(total=total_frames, desc=f"Video {video_file} ")
            
            #Processing each frame
            frame_num = 0 
            ret, frame = cap.read()
            while ret:
                frame_num += 1

                # Perform detection & tracking on frame
                results = model.track(frame, persist=True, conf=0.481, verbose=False, tracker="loitering/custom-bytetrack.yaml")
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu() 
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()
                    names = results[0].names
                else:
                    #No detections
                    #print("No detections")
                    boxes, track_ids, clss, names = [[] for _ in range(4)]

                #Crowd density module
                crowd_density, crowd_count = detect_crowd_density(boxes, frame)

                #Loitering module
                missed_detect, misses_cnt, dwell_time, loitering = detect_loitering(boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age)

                #Concealment module
                concealment_counts = updated_concealment(clss)

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
    #Declaring CSV filename and header fields
    csv_filename = "output_50_50.csv"
    field_names = ["video_id","frame_num","crowd_density","loitering","no_concealment","low_concealment","med_concealment","high_concealment","rbp"]

    #Declaring folder path of the videos to be processed
    folder_path = r"C:\Users\janrh\OneDrive - University of the Philippines\Acads\4TH YEAR (23-24)\2ND SEM\EE 199\DATASETS\inference 50-50"

    create_csv(csv_filename, field_names)
    vid_processing(folder_path, csv_filename, field_names)


