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

        global segmented_area
        img = frame
        frame_area = img.shape[0]*img.shape[1]
        results = model(frame)
        mask_coords = results[0].masks.xy
        for item in mask_coords:
            segmented_area += compute_area(item)
        crowd_density = 100 * segmented_area / frame_area

        print(f"crowd density: {round(crowd_density, 2)}%")
        segmented_area = 0
        # return crowd_density

        annotated_frame = results[0].plot(boxes=False)
        cv.imshow(WIN_NAME, annotated_frame)
    source.release()
    cv.destroyWindow(WIN_NAME)

def process_image(frame):
    global segmented_area
    img = cv.imread(frame)
    frame_area = img.shape[0]*img.shape[1]
    results = model(img)
    mask_coords = results[0].masks.xy
    for item in mask_coords:
        segmented_area += compute_area(item)
    crowd_density = 100 * segmented_area / frame_area
    # return crowd_density
    
    print(f"crowd density: {round(crowd_density, 2)}%")


    annotated_frame = results[0].plot(boxes=False)
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


def compute_area(coordinates):
  
    area = 0.0
    n = len(coordinates)
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[j][0] * coordinates[i][1]
    area = abs(area) / 2.0
    return area

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, } }




if __name__ == "__main__":

    segmented_area = 0

    WIN_NAME = "Detections"
    model = YOLO(os.path.join("models", "yolo", "yolov8n-seg.pt"))

    s = r"C:\Users\janrh\OneDrive - University of the Philippines\Acads\4TH YEAR (23-24)\2ND SEM\EE 199\eee-capstone\crowd_density\group3.mp4"
    if len(sys.argv) > 1:
        s = sys.argv[1]

    detect_objects(s)
