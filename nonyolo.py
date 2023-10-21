import os
import sys
import cv2 as cv
from ultralytics import YOLO

# Function to load the model based on the MODEL_TYPE
def load_model():
    if MODEL_TYPE == 'tf':
        class_file = os.path.join("models", "coco_class_labels.txt")
        with open(class_file) as fp:
            global labels
            labels = fp.read().split("\n")

        net = cv.dnn.readNetFromTensorflow(
            os.path.join("models", MODEL_TYPE, "frozen_inference_graph.pb"),
            os.path.join("models", MODEL_TYPE, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"))
    
    elif MODEL_TYPE == 'caffe':
        net = cv.dnn.readNetFromCaffe(
            os.path.join("models", MODEL_TYPE, "ssd", "deploy.prototxt"),
            os.path.join("models", MODEL_TYPE, "ssd", "res10_300x300_ssd_iter_140000_fp16.caffemodel"))
    else:
        raise ValueError("Unsupported model type")

    return net

# Function to detect objects
def detect_objects(s):
    net = load_model()
    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)

    if isinstance(s, int):
        source = cv.VideoCapture(s)
        while cv.waitKey(1) != 27:
            has_frame, frame = source.read()
            if not has_frame:
                break
            frame = cv.flip(frame, 1)
            image_processing(frame, net)
        source.release()
        cv.destroyWindow(WIN_NAME)

    elif isinstance(s, str):
        if s.endswith('.mp4'):
            source = cv.VideoCapture(s)
            while cv.waitKey(1) != 27:
                has_frame, frame = source.read()
                if not has_frame:
                    break
                image_processing(frame, net)
            source.release()
            cv.destroyWindow(WIN_NAME)

        else:
            img = cv.imread(s)
            image_processing(img, net)
            cv.waitKey(0)

# Function for image processing
def image_processing(frame, net):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame and run model
    blob = cv.dnn.blobFromImage(frame, 1.0, size=(300,300), mean=MEAN, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    scores = []

    for i in range(detections.shape[2]):
        classId = int(detections[0, 0, i, 1])
        confidence = float(detections[0, 0, i, 2])

        x = int(detections[0, 0, i, 3] * frame_width)
        y = int(detections[0, 0, i, 4] * frame_height)
        w = int(detections[0, 0, i, 5] * frame_width - x)
        h = int(detections[0, 0, i, 6] * frame_height - y)
        
        if confidence > CONF_THRESHOLD:
            boxes.append([x, y, w, h])
            scores.append(confidence)
        
    # Apply Non-Maximum Suppression to use only max confidence detection
    indices = cv.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, 0.4)

    for i in indices:
        classId = int(detections[0, 0, i, 1])
        x, y, w, h = boxes[i]
        confidence = scores[i]

        # Bounding box
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),thickness=2)

        FONT = cv.FONT_HERSHEY_SIMPLEX
        confidence = confidence*100

        if MODEL_TYPE == 'tf':
            label = f"{labels[classId]}{confidence: .2f}%"
        elif MODEL_TYPE == 'caffe':
            label = f"Face{confidence: .2f}%"

        display_text(frame, label, x, y)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
    display_text(frame, label, 0, 20)
    cv.imshow(WIN_NAME, frame)

def display_text(img, text, x, y):
    FONTFACE = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS = 1
    
    # # Get text size
    label_size, base_line = cv.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)

    cv.rectangle(img,(x, y - label_size[1] - base_line),(x + label_size[0], y + base_line),(255, 255, 255),cv.FILLED)
    cv.putText(img, text, (x, y), FONTFACE, FONT_SCALE, (0, 0, 0), THICKNESS, cv.LINE_AA)


if __name__ == "__main__":

    # Define constants
    MODEL_TYPE = 'tf'
    WIN_NAME = "Detections"
    CONF_THRESHOLD = 0.5
    # MEAN = (104, 117, 123) # lower accuracy but faster
    MEAN = (0, 0, 0) # higher accuracy but slower

    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    detect_objects(s)
