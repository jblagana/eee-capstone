import os
import sys
import cv2 as cv
from ultralytics import YOLO

def process_video(s):
    source = cv.VideoCapture(s)
    while cv.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break

        if isinstance(s, int):
            frame = cv.flip(frame, 1)

        results = model(frame)
        annotated_frame = results[0].plot()
        cv.imshow(WIN_NAME, annotated_frame)
    source.release()
    cv.destroyWindow(WIN_NAME)

def process_image(s):
    img = cv.imread(s)
    results = model(img)
    annotated_frame = results[0].plot()
    cv.imshow(WIN_NAME, annotated_frame)
    cv.waitKey(0)

def detect_objects(s):

    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)

    if isinstance(s, int):
        process_video(s)
    elif isinstance(s, str):
        if s.endswith('.mp4'):
            process_video(s)
        else:
            process_image(s)


if __name__ == "__main__":

    WIN_NAME = "Detections"
    model = YOLO(os.path.join("models", "yolo", "yolov8n-seg.pt"))

    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    detect_objects(s)
