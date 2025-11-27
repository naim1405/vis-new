import cv2
from ultralytics import download
from p1 import process_frame
from p2 import tracking
from p3 import block3
from p4 import predict_normality

video_url = "./media/sample.mp4"


video_cap = cv2.VideoCapture(video_url)
b3 = block3(seq_len=30)
block_process_frame = b3["process_frame"]
get_buffer = b3["get_buffer"]
close_block3 = b3["close"]


def do_work(cap):
    model_input = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # p1 person detection returns list of [[x,y,w,h]]
        detections = process_frame(frame)
        # 🚀 detections: [[[550.223388671875, 97.72137451171875, 634.3145751953125, 967.6641235351562], 0.933281421661377]]
        # 🚀 detections: [[[x1,y1,w,h], confidence]]
        # print("🚀 detections:", detections)
        # p2 deep sort tracking returns {id:(x1,y1,x2,y2)}
        tracking_data = tracking(detections, frame)
        print("🚀 tracking_data : ", tracking_data)
        # 🚀 tracking_data :  {'1': (578, 100, 1106, 1065)}
        # Convert from (x1,y1,x2,y2) to (x,y,w,h) for block3
        tracking_data_xywh = {}
        for tid, bbox in tracking_data.items():
            x1, y1, x2, y2 = bbox
            tracking_data_xywh[tid] = (x1, y1, x2 - x1, y2 - y1)
        # p3 block3 processing
        input_i = block_process_frame(tracking_data_xywh, frame)
        # print("🚀 input_i : ", len(input_i))
        print("🚀 input_i : ", len(input_i))
        # with open("logs/output_input_i.txt", "a") as f:
        #     f.write(str(input_i) + "\n")
        model_input.append(input_i)
        if len(model_input) >= 30:
            # cal p4
            print("model input ready", model_input)
            normality_result = predict_normality(model_input)
            print("normality_result ", normality_result)
            model_input = []


do_work(video_cap)
#
# # person detection
# detections = process_frame(frame)
# border_color = (0, 255, 0)
#
# # anomaly detection and logging
# is_anomaly = anomaly_detection(detections, frame)
# if is_anomaly:
#     border_color = (0, 0, 255)
#     self.anomaly_log.write(
#         "Anomaly detected at timestamp: {}\n".format(self._timestamp)
#     )
#
# for x, y, w, h in detections:
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
# # Ensure frame is uint8 numpy array
# frame_uint8 = np.asarray(frame, dtype=np.uint8)
#
# # Convert OpenCV frame (BGR numpy array) to av.VideoFrame
# video_frame = VideoFrame.from_ndarray(frame_uint8, format="bgr24")
#
# # Set timestamp
# video_frame.pts = self._timestamp
# video_frame.time_base = Fraction(1, 90000)
#
# self._timestamp += 3000  # Increment for ~30fps (90000/30 = 3000)
#
# return video_frame
