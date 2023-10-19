#%%
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

#%%
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets...")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done.")

    except Exception as e:
        print("\nInvalid file.", e)


#%%
def load_model():
    # Download if asset does not exists.
    if model == 'tf':
        URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"
        asset_path = os.path.join(os.getcwd(), "opencv_tf.zip")

        if not os.path.exists(asset_path):
            download_and_unzip(URL, asset_path)

        classFile  = "coco_class_labels.txt"
        with open(classFile) as fp:
            global labels
            labels = fp.read().split("\n")

        modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
        configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
        # Read the Tensorflow network
        net = cv.dnn.readNetFromTensorflow(modelFile, configFile)

    elif model == 'caffe':
        URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
        asset_path = os.path.join(os.getcwd(), "opencv_caffe.zip")

        if not os.path.exists(asset_path):
            download_and_unzip(URL, asset_path)

        net = cv.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    return net

#%%
def detect_objects(im, dim = 300):
    global net
    net = load_model()
    # Create a blob from the image
    blob = cv.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=False, crop=False)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects


#%%
def display_text(im, text, x, y):
    FONTFACE = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1
    # Get text size
    textSize = cv.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv.rectangle(im,(x, y - dim[1] - 2*baseline),(x + dim[0], y + baseline),(0, 0, 0),cv.FILLED)

    # Display text inside the rectangle
    cv.putText(im,text,(x, y - 5),FONTFACE,FONT_SCALE,(0, 255, 255),THICKNESS,cv.LINE_AA)

#%%
def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    boxes = []
    scores = []

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        # print(classId)
        # print(score)

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)


        # Check if the detection is of good quality
        if score > threshold:
            boxes.append([x, y, w, h])
            scores.append(score)

    # Apply Non-Maximum Suppression to use only max confidence detection
    indices = cv.dnn.NMSBoxes(boxes, scores, threshold, 0.4)

    # Display the remaining bounding boxes
    for i in indices:
        classId = int(objects[0, 0, i, 1])
        x, y, w, h = boxes[i]
        score = scores[i]

        if model == 'tf':
            display_text(im, "{} {}%".format(labels[classId], round(score * 100, 2)), x, y)
        elif model == 'caffe':
            display_text(im, "Confidence: {}%".format(round(score * 100, 2)), x, y)
            
        cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
    display_text(im, label, 0, 30)
    # cv.putText(im, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return im
    # cv.imshow('Detected Image', im)
    # cv.waitKey(0)

def display_video(s):
    source = cv.VideoCapture(s)
    win_name = "Camera Preview"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    alive = True

    while alive:
        has_frame, frame = source.read()
        if not has_frame:
            break
        frame = cv.flip(frame, 1)

        # Perform object detection
        objects = detect_objects(frame)
        img = display_objects(frame, objects)
        cv.imshow(win_name, img)

        if cv.waitKey(1) == 27:
            alive = False

    source.release()
    cv.destroyAllWindows()


#%%
def display_image(s):
    # img = cv.imread(os.path.join("images", img))
    img = cv.imread(s)
    objects = detect_objects(img)
    img = display_objects(img, objects)
    cv.imshow('Detected Image', img)
    cv.waitKey(0)


if __name__ == "__main__":
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    """
    Models available:
    1. Tensorflow (tf) - object detection
    2. Caffe (caffe) - face detection
    """

    model = 'caffe' # tf, caffe

    # Webcam sources
    if isinstance(s, int):
        display_video(s) 
    elif s.endswith('.mp4'):
        display_video(s)
    elif s.endswith('.jpg'):
        display_image(s)








# # Import libraries
# import os
# import cv2
# import sys
# import numpy as np
# import matplotlib.pyplot as plt

# from zipfile import ZipFile
# from urllib.request import urlretrieve

# # Load the TensorFlow object detection model using OpenCV's DNN module
# def load_tf_model():

#     # Load class labels from file
#     classFile = "coco_class_labels.txt"
#     with open(classFile) as fp:
#         labels = fp.read().split("\n")

#     modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
#     configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

#     net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
#     return net, labels

# # Perform object detection using TensorFlow model
# def object_detection(frame, net):
#     # Define the input blob for the network
#     blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
#     net.setInput(blob)

#     # Run forward pass to get detections
#     detections = net.forward()

#     frame_height = frame.shape[0]
#     frame_width = frame.shape[1]

#     # Process the detections and draw bounding boxes
#     for i in range(detections.shape[2]):
#         classId = int(detections[0, 0, i, 1])
#         confidence = float(detections[0, 0, i, 2])
#         if confidence > conf_threshold:
#             x = int(detections[0, 0, i, 3] * frame_width)
#             y = int(detections[0, 0, i, 4] * frame_height)
#             w = int(detections[0, 0, i, 5] * frame_width)
#             h = int(detections[0, 0, i, 6] * frame_height)

#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             label = f"{labels[classId]} Confidence: %.4f" % confidence
#             label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

#             cv2.rectangle(
#                 frame,
#                 (x, y - label_size[1]),
#                 (x+ label_size[0], y + base_line),
#                 (255, 255, 255),
#                 cv2.FILLED,
#             )
#             cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#     return frame

# if __name__ == "__main__":
#     # Load the TensorFlow object detection model
#     net, labels = load_tf_model()

#     s = 0
#     if len(sys.argv) > 1:
#         s = sys.argv[1]

#     source = cv2.VideoCapture(s)

#     win_name = "Camera Preview"
#     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#     # Model parameters
#     in_width = 300
#     in_height = 300
#     mean = [104, 117, 123]
#     conf_threshold = 0.7 # detection sensitivity

#     alive = True

#     while alive:
#         has_frame, frame = source.read()
#         if not has_frame:
#             break
#         frame = cv2.flip(frame, 1)

#         # Perform object detection
#         frame = object_detection(frame, net)

#         t, _ = net.getPerfProfile()
#         label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
#         cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#         cv2.imshow(win_name, frame)

#         key = cv2.waitKey(1)
#         if key == ord("Q") or key == ord("q") or key == 27:
#             alive = False

#     source.release()
#     cv2.destroyAllWindows()
