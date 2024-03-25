import os
import random
import numpy as np

import cv2
from ultralytics import YOLO

from tracker import Tracker

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def dwell_time_update(track_id, object_id_list, dwell_time):
    #Dwell time
    if track_id not in object_id_list:
        object_id_list.append(track_id)
        dwell_time[track_id] = 0
    else:
        dwell_time[track_id] += 1
    """
    #if same ang time && age niya is >max age:
        delete
    """
    
    

def main():
    
    video_path = os.path.join('.', 'data', 'yolo_ucf_81.mp4')
    video_out_path = os.path.join('.','runs', 'custom_out.mp4')

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                            (frame.shape[1], frame.shape[0]))

    #model = YOLO("yolov8n.pt")
    model = YOLO(r"custom_yolo\best_n.pt")

    tracker = Tracker()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    detection_threshold = 0.5

    #Dwell time parameters

    object_id_list = set()
    dtime = {}
    dwell_time = {}
    age_id = {}
    max_age_ids = []
    
    frame_cnt = 0

    while ret:
        frame_cnt += 1
        """
        if frame_cnt == 1000:
            break
        """        
        #print("Frame:" , frame_cnt, "\n")

        #results = model(frame, classes = 0)
        results = model(frame)
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                #print(score)
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score >= detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            """
            for item in dwell time
                tracker.tracks
            
            """
            

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                center = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_id = track.track_id

                #dwell_time_update(track_id, object_id_list, dwell_time)
                if track_id not in object_id_list:
                    object_id_list.add(track_id)
                    age_id[track_id] = 0
                    dwell_time[track_id] = 1
                else:
                    dwell_time[track_id] += 1

                label = "{}: {}".format(track_id, dwell_time[track_id])
                #print(label)

                #Draw center
                cv2.circle(frame, center, radius=5, color=(0, 255, 0), thickness=-1)

                #Bounding box and label
                UI_box(bbox, frame, label=label, color=colors[track_id % len(colors)], line_thickness=2)


            # Check age of each ID in object_id_list is the current tracked ids
            for object_id in list(object_id_list):
                if age_id[object_id] > 100: #max age of deepsort == 100 (change this if max age changes)
                    #max_age_ids.append(object_id)
                    dwell_time.pop(object_id)
                    age_id.pop(object_id)
                    object_id_list.remove(object_id)
                else:
                    id_found = any(track.track_id == object_id for track in tracker.tracks)
                    if not id_found:
                        age_id[object_id] += 1
            
            #Remove IDs that reached maximum age
            """
            for object_id in max_age_ids:
                object_id_list.remove(object_id)
                dwell_time.pop(object_id)
                age_id.pop(object_id)
            
            max_age_ids.clear()
            """
                    
                
                

        #Final loitering module output: compute variance of dwell times
        std_val = np.std(list(dwell_time.values()))

        print("Dwell time list: ", list(dwell_time.values()), "\nStandard Deviation: ", std_val)
        cap_out.write(frame)

        ret, frame = cap.read()
        

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
    print(object_id_list)
    print(dwell_time)

if __name__ == "__main__":
    main()