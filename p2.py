# Block 2: DeepSort (ResNet50)
"""
Input : a list of (x,y,h,w) from block 1 and a frame
Output: a map of the form id:(x,y,h,w)
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
import cv2

tracker = DeepSort(
    max_age=90, embedder="torchreid", embedder_model_name="resnet50", half=True
)


def tracking(detections, frame):
    dets = []

    for bbox, conf in detections:
        x, y, w, h = bbox
        dets.append(([int(x), int(y), int(x + w), int(y + h)], float(conf)))

    # Update tracker
    tracks = tracker.update_tracks(dets, frame=frame)

    track_map = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        w = x2 - x1
        h = y2 - y1

        track_map[track_id] = (x1, y1, w, h)

    # returns {id:(x,y,h,w)}
    # id => person id
    # coor => bounding box coordinates
    return track_map
