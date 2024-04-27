def crowd_density_module(boxes, frame):
    """
    Calculate crowd density based on bounding boxes and the current frame

    Args:
        boxes (list): list of bounding boxes from the detection results
        frame: current frame to be processed
    Output:
        crowd_density (float): crowd density value
    """
    
    segmented_area = 0
    crowd_density = 0

    img = frame
    frame_area = img.shape[0]*img.shape[1]
    existing_boxes = []

    for box in boxes:
        x, y, w, h = box.tolist()
        box_area = w * h
        overlap = 0

        for existing_box in existing_boxes:
            overlap = calculate_overlap((x, y, w, h), existing_box)

        segmented_area += (box_area - overlap) 
        existing_boxes.append((x, y, w, h))

    crowd_density = 100 * segmented_area / frame_area

    #print(f"crowd density: {round(crowd_density, 2)}%")
    return crowd_density

def calculate_overlap(box1, box2):
    # Calculate overlapping area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    
    return overlap