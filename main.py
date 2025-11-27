import cv2
import os
from datetime import datetime
from ultralytics import download
from p1 import process_frame
from p2 import tracking
from p3 import block3
from p4 import predict_normality

video_url = "./media/sample.mp4"

# Create directories if they don't exist
os.makedirs("anomaly_frames", exist_ok=True)
os.makedirs("normal_frames", exist_ok=True)

video_cap = cv2.VideoCapture(video_url)
b3 = block3(seq_len=30)
block_process_frame = b3["process_frame"]
get_buffer = b3["get_buffer"]
close_block3 = b3["close"]


def do_work(cap):
    model_input = []
    frame_count = 0
    anomaly_count = 0
    normal_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        # p1 person detection returns list of [[x,y,w,h]]
        detections = process_frame(frame)
        # 🚀 detections: [[[550.223388671875, 97.72137451171875, 634.3145751953125, 967.6641235351562], 0.933281421661377]]
        # 🚀 detections: [[[x1,y1,w,h], confidence]]
        # print("🚀 detections:", detections)
        # p2 deep sort tracking returns {id:(x1,y1,x2,y2)}
        tracking_data = tracking(detections, frame)
        # print("🚀 tracking_data : ", tracking_data)
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

        # input_i is a dict: {track_id: np.array(seq_len, num_joints, 3)}
        # Only process when we have ready sequences
        if len(input_i) > 0:
            # Predict normality for each tracked person
            normality_result = predict_normality(input_i)
            print("🚀 normality_result:", normality_result)

            # Separate people by anomaly score
            anomalous_ids = []
            normal_ids = []
            for track_id, score in normality_result.items():
                if score < 0:
                    anomalous_ids.append(track_id)
                else:
                    normal_ids.append(track_id)

            # Save frame with anomalous people (score < 0)
            if len(anomalous_ids) > 0:
                anomaly_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"anomaly_frames/frame_{frame_count:06d}_{timestamp}_ids_{'_'.join(anomalous_ids)}.jpg"

                # Draw red bounding boxes for anomalies
                annotated_frame = frame.copy()
                for tid in anomalous_ids:
                    if tid in tracking_data:
                        x1, y1, x2, y2 = tracking_data[tid]
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),
                            3,
                        )
                        score_text = f"ID:{tid} Score:{normality_result[tid]:.2f}"
                        cv2.putText(
                            annotated_frame,
                            score_text,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                cv2.imwrite(filename, annotated_frame)
                print(f"✅ Saved anomaly frame: {filename}")

            # Save frame with normal people (score >= 0)
            if len(normal_ids) > 0:
                normal_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"normal_frames/frame_{frame_count:06d}_{timestamp}_ids_{'_'.join(normal_ids)}.jpg"

                # Draw green bounding boxes for normal behavior
                annotated_frame = frame.copy()
                for tid in normal_ids:
                    if tid in tracking_data:
                        x1, y1, x2, y2 = tracking_data[tid]
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (95, 147, 222),
                            3,
                        )
                        score_text = f"ID:{tid} Score:{normality_result[tid]:.2f}"
                        cv2.putText(
                            annotated_frame,
                            score_text,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                cv2.imwrite(filename, annotated_frame)
                print(f"✅ Saved normal frame: {filename}")

    print(
        f"\n📊 Summary: Processed {frame_count} frames, saved {anomaly_count} anomaly frames and {normal_count} normal frames"
    )


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
