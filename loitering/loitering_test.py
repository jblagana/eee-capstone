from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import numpy as np

def bytetrack_test(video_path, yolo_model):
    """
    Args:
        video_path: path of input video to be processed
        yolo_model: PT file of the yolo model
    Output:
        Video showing the results from the yolo model + bytetrack
    """
    # Load the YOLOv8 model
    model = YOLO(yolo_model)

    # Open the video file
    #video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.481, verbose=False, tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "./input-vid/yolo_ucf_81.mp4"
    yolo_model = "yolov8n.pt"
    bytetrack_test(video_path, yolo_model)