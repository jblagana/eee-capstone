from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import numpy as np

def detect_loitering(boxes, track_ids, clss, names, missed_detect, misses_cnt, dwell_time, max_age):
    class_names = ["high","low","med","none"]
    
    # Iterate through the results
    for box, track_id, cls in zip(boxes, track_ids, clss):
        """
            Since ID is present in the frame:
                1. Update missed_detect to False - meaning that it is NOT absent in the frame
                2. Increment its dwell time
                3. Reset misses_cnt to 0
        """
        missed_detect[track_id] = False
        dwell_time[track_id] = dwell_time.get(track_id, 0) + 1
        misses_cnt[track_id] = 0
        
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
                
    #Final loitering module output: compute variance of dwell times
    std_val = np.std(list(dwell_time.values())) if dwell_time else -1
    #print("Dwell time list: ", list(dwell_time.values()), "\nStandard Deviation: ", std_val)

    return missed_detect, misses_cnt, dwell_time, std_val



def main():

    #Load model
    model = YOLO('best_finalCustom.pt')

    #Load video
    video_path = "./anomaly_ucf_4.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    #Video output
    output_video_path = "./runs/out-3.mp4"    
    cap_out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
            (frame.shape[1], frame.shape[0]))

    #Class names
    class_names = ["high","low","med","none"]

    #Frame count
    frame_cnt = 0

    #Dwell time parameters
    missed_detect = {} #Dictionary that contains all the tracked object, key:id, value: True/False (false - present in the frame)
    dwell_time = {}
    misses_cnt = {} #Dictionary that stores how many consecutive missed detection for the object
    max_age = 100



    while ret:
        #ret, frame = cap.read()
        frame_cnt += 1
        print(frame_cnt)
        # Perform tracking on frame
        results = model.track(frame, persist=True, verbose=False, tracker="custom-bytetrack.yaml")
        
        # Extract bounding boxes, tracking IDs, classes names
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
       
            # Iterate through the results
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = box
                cls_name = class_names[int(cls)]
                xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
                
                """
                Since ID is present in the frame:
                    1. Update missed_detect to False - meaning that it is NOT absent in the frame
                    2. Increment its dwell time
                    3. Reset misses_cnt to 0
                """
                missed_detect[track_id] = False
                dwell_time[track_id] = dwell_time.get(track_id, 0) + 1
                misses_cnt[track_id] = 0
                
                label = "#{}:{}:{}".format(str(track_id), cls_name,dwell_time[track_id])
                annotator = Annotator(frame, line_width=1, example=names)
                annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))
        
        #Check if missed detections of each object > max age
        if missed_detect: #Checking if list is not empty
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
                
        #Final loitering module output: compute variance of dwell times
        std_val = np.std(list(dwell_time.values())) if dwell_time else None
        print("Dwell time list: ", list(dwell_time.values()), "\nStandard Deviation: ", std_val)


        cap_out.write(frame)
        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
    #results = model.track(source="yolo_ucf_81.mp4", show=True, tracker="bytetrack.yaml", conf = 0.4, save=True, persist=True)
    #print(results)

if __name__ == "__main__":
    main()