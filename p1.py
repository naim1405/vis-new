# Block 1 : YOLOv8 nano
"""
Input : 1 Frame
Output: a list of (x,y,h,w) which points the rectangle over the persons of the frame
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


# Process a single frame for person detection (tracking not included)
def process_frame(
    frame,
):
    # frame is actually a picture
    model = YOLO("yolov8n.pt")
    results = model(frame)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf)
        cls = int(box.cls)

        # class 0 = person in COCO
        if cls == 0 and conf > 0.6:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"yolo/{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            detections.append(
                [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf]
            )
    # print("🚀 detections : ", detections)
    return detections
