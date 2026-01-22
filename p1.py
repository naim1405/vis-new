# Block 1 : YOLOv8 nano
"""
Input : 1 Frame
Output: a list of (x,y,h,w) which points the rectangle over the persons of the frame
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

model_path = "./models/yolov8n.pt"
modelp1 = YOLO(model_path)

# Process a single frame for person detection (tracking not included)
def process_frame(
    frame,
):
    # frame is actually a picture
    model = modelp1
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

video_url = "C:\\Users\\LENOVO\\OneDrive\\Pictures\\strict_mode.mp4"

video_cap = cv2.VideoCapture(video_url)
while True:
    ret, frame = video_cap.read()
    if not ret or frame is None:
        break

    detections = process_frame(frame)

    for det in detections:
        bounding, score = det
        x, y, w, h = bounding
        cv2.rectangle(
            frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"{score:.2f}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
