#Load YOLOv8 nano pre-trained model
model = YOLO("yolov8n.pt")
results = model("../images/test5.jpg", save=True)

#COCO Dataset Classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#ctr = 0;
for result in results:

    #print(f"Result {ctr}: {result}")
    
    
    boxes = result.boxes
    probs = result.probs
    cls = boxes.cls.tolist()
    #print(f"Class {ctr}: {cls}")

    xyxy = boxes.xyxy
    #print(f"Xyxy {ctr}: {xyxy}")

    xywh = boxes.xywh
    conf = boxes.conf

    for class_index in cls:
        if class_index == 0:
            print("Class:", classNames[int(class_index)])