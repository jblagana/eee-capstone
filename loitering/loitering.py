from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import numpy as np

def dwell_time_update(track_id, object_id_list, dwell_time):
    #Dwell time
    if track_id not in object_id_list:
        object_id_list.append(track_id)
        dwell_time[track_id] = 0
    else:
        dwell_time[track_id] += 1

def detect_loitering(boxes, track_ids, clss, names, object_id_list, age_id, dwell_time, max_age):
    # Iterate through the results
    class_names = ["high","low","med","none"]

    for box, track_id, cls in zip(boxes, track_ids, clss):
        #Uncomment next 3 lines if 
        #x1, y1, x2, y2 = box
        #cls_name = class_names[int(cls)]
        #xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
        
        #Dwell time update
        if track_id not in object_id_list:
            object_id_list.add(track_id)
            age_id[track_id] = 0
            dwell_time[track_id] = 1
        else:
            dwell_time[track_id] += 1
        
        #label = "#{}:{}:{}".format(str(track_id), cls_name,dwell_time[track_id])
        #annotator = Annotator(frame, line_width=1, example=names)
        #annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))

    # Check age of each ID in object_id_list is the current tracked ids
    if object_id_list:
        for object_id in list(object_id_list):
            if age_id[object_id] > max_age: 
                dwell_time.pop(object_id)
                age_id.pop(object_id)
                object_id_list.remove(object_id)
            else:
                if track_ids is not None:
                    id_found = any(track == object_id for track in track_ids)
                    if not id_found:
                        age_id[object_id] += 1
                
    #Final loitering module output: compute variance of dwell times
    std_val = np.std(list(dwell_time.values())) if dwell_time else 0
    #print("Dwell time list: ", list(dwell_time.values()), "\nStandard Deviation: ", std_val)

    return object_id_list, age_id, dwell_time, std_val



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
    object_id_list = set()
    dtime = {}
    dwell_time = {}
    age_id = {}
    max_age_ids = []
    max_age = 100



    while ret:
        #ret, frame = cap.read()
        frame_cnt += 1
        print(frame_cnt)
        # Perform tracking on frame
        results = model.track(frame, persist=True, verbose=False, tracker="custom-bytetrack.yaml")
        
        # Extract bounding boxes, tracking IDs, classes names
        """
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        """
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
                
                #Dwell time update
                if track_id not in object_id_list:
                    object_id_list.add(track_id)
                    age_id[track_id] = 0
                    dwell_time[track_id] = 1
                else:
                    dwell_time[track_id] += 1
                
                label = "#{}:{}:{}".format(str(track_id), cls_name,dwell_time[track_id])
                annotator = Annotator(frame, line_width=1, example=names)
                annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))
        
        # Check age of each ID in object_id_list is the current tracked ids
        if object_id_list:
            for object_id in list(object_id_list):
                if age_id[object_id] > max_age: 
                    dwell_time.pop(object_id)
                    age_id.pop(object_id)
                    object_id_list.remove(object_id)
                else:
                    if track_ids is not None:
                        id_found = any(track == object_id for track in track_ids)
                        if not id_found:
                            age_id[object_id] += 1
                
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