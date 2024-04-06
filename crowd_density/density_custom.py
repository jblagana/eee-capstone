import os
import sys
import cv2 as cv
from ultralytics import YOLO

def detect_crowd_density(boxes, frame):
    segmented_area = 0
    crowd_count = 0
    crowd_density = 0

    img = frame
    frame_area = img.shape[0]*img.shape[1]
    existing_boxes = []

    for box in boxes:
        x, y, w, h = box.tolist()
        box_area = w * h
        for existing_box in existing_boxes:
            iou = calculate_iou((x, y, w, h), existing_box)
            segmented_area += box_area * (1 - iou)  # Consider only non-overlapping area
        existing_boxes.append((x, y, w, h))
        crowd_count += 1

    crowd_density = 100 * segmented_area / frame_area

    #print(f"crowd density: {round(crowd_density, 2)}%")
    return crowd_density, crowd_count


def process_video(s):
    source = cv.VideoCapture(s)
    while cv.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break

        if isinstance(s, int):
            frame = cv.flip(frame, 1)

        global segmented_area
        img = frame
        frame_area = img.shape[0]*img.shape[1]
        results = model(frame)


        boxes = results[0].boxes.xywh.cpu()
        existing_boxes = []
        for box in boxes:
            x, y, w, h = box.tolist()
            box_area = w * h
            for existing_box in existing_boxes:
                iou = calculate_iou((x, y, w, h), existing_box)
                segmented_area += box_area * (1 - iou)  # Consider only non-overlapping area
            existing_boxes.append((x, y, w, h))

        crowd_density = 100 * segmented_area / frame_area

        print(f"crowd density: {round(crowd_density, 2)}%")
        segmented_area = 0

        annotated_frame = results[0].plot()
        cv.imshow(WIN_NAME, annotated_frame)

        # return crowd_density
    
    source.release()
    cv.destroyWindow(WIN_NAME)

def process_image(frame):
    global segmented_area
    img = cv.imread(frame)
    frame_area = img.shape[0]*img.shape[1]
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


def calculate_iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou




if __name__ == "__main__":

    segmented_area = 0

    WIN_NAME = "Detections"
    model = YOLO(r"models\yolo\best_finalCustom.pt")

    s = r"items\yolo_ucf_null.jpg"
    if len(sys.argv) > 1:
        s = sys.argv[1]

    detect_objects(s)
