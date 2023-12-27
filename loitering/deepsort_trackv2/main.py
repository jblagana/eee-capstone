import cv2
import random
from ultralytics import YOLO 
from tracker import Tracker

video_path = '../images/test10.mp4'
#video_path = '../images/test9.mp4'

cap = cv2.VideoCapture(video_path)

#Getting video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#Define the codec and create VideoWriter Object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #video codec to be used for video compression. 'mp4v' represents the MPEG-4 Video codec.
    #FourCC is a 4-byte code used to specify video codecs. Different codecs have different FourCC codes.
output_path = 'output3.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

tracker = Tracker()
success, frame = cap.read()
model = YOLO("yolov8n.pt")
unique_track_ids = set()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

while success:
    #results = model(frame)
    results = model(frame, device = 'cpu', classes = 0, conf = 0.5)

    #results is the detection in a SINGLE frame
    #Hence, it's length = 1

    for result in results: #for loop only for "aesthetic purposes"
        detections = []
        for r in result.boxes.data.tolist():    
            #print(r)
            x1, y1, x2, y2, conf, class_id = r
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            detections.append([x1, y1, x2, y2, conf])

        tracker.update(frame, detections)

        for track in tracker.tracks:    #Accessing each tracked object
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            unique_track_ids.add(track_id)

            track_color = colors[track_id %  len(colors)]
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #Plotting bounding box of a person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_color, 2)

            #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            cv2.putText(frame, f"ID: {track_id}", (int(x1) + 10, int(y1) - 5), 0, 1, track_color, thickness=2, lineType=cv2.LINE_AA)

    #Draw person count on frame
    crowd_count = len(unique_track_ids)
    cv2.putText(frame, f"Crowd count: {crowd_count}", (10, 30), 0, 1, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('1'): #Clicking 1 will stop the program
            break
    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
