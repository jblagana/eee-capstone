import numpy as np

from ultralytics.utils.plotting import Annotator, colors

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

