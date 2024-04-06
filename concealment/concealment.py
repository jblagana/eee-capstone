import cv2
from ultralytics import YOLO
import numpy as np

def updated_concealment(class_list):
    concealment_counts = [0, 0, 0, 0]                               # Initialize all class counts to 0
    for class_id in class_list:
        #class_id = int(class_id)
        concealment_counts[int(class_id)] += 1
    return np.array(concealment_counts)

def detect_concealment(frame):
    """
    Detects concealment levels and counts instances for each class.

    Args:
        frame: A numpy array representing the video frame

    Returns:
        A numpy array with four elements representing the counts for each concealment class:
            [no_concealment_count, low_concealment_count, medium_concealment_count, high_concealment_count].
    """

    results = model(frame)[0]
    concealment_counts = [0, 0, 0, 0]                               # Initialize all class counts to 0
       
    for person in results.boxes.data.tolist():                      # Iterate through each detected object
        x1, y1, x2, y2, confidence, class_id = person               # Extract bounding box coordinates, confidence score and class ID
        class_id = int(class_id)
        
        if confidence > threshold:

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)                
            cv2.putText(frame, results.names[int(class_id)], (int(x1), int(y1 - 5)),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

            # Add to concealment tensor
            concealment_counts[class_id -1] += 1

    return np.array(concealment_counts)


if __name__ == "__main__":

    # Define video and model paths
    video_path = "concealment/videos/test.mp4"
    video_path_out = '{}_out.mp4'.format(video_path)
    model_path = "concealment/model/best_n.pt"
    
    # Initialize video capture and output writer
    cap = cv2.VideoCapture(video_path)                      
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    
    model = YOLO(model_path)                                        # Load model                                   
    threshold = 0                                                   # Set threshold

    # Loop through each frame in the video
    while ret:
        
        concealment_tensor = detect_concealment(frame)              # Perform detection on frame
        print(concealment_tensor)
        
        # Display the resulting frame
        # cv2.imshow('Concealment Detection', frame)                  
        # if cv2.waitKey(1) & 0xFF == ord('q'):                       
        #    break        
        
        out.write(frame)                                            # Write the processed frame to the output video
        ret, frame = cap.read()                                     # Read the next fram
        
    cap.release()
    cv2.destroyAllWindows()