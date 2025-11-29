import cv2
import os
from datetime import datetime
from torch.nn import init
from ultralytics import download
from ultralytics.models.yolo import model
from p1 import process_frame
from p2 import tracking
# from p3 import block3
from p4 import predict_normality
import numpy as np
from p3_k import FrameBufferManager
from inference_only.json_inference import JSONAnomalyDetector


video_url = "./media/01.mp4"
pose_model_path = "./models/yolov8n-pose.pt"

# Create directories if they don't exist
os.makedirs("anomaly_frames", exist_ok=True)
os.makedirs("ok", exist_ok=True)

video_cap = cv2.VideoCapture(video_url)

manager = FrameBufferManager(
    pose_model_path=pose_model_path,
    sequence_length=30,
    frame_digits=4,
    device="cpu",
)

detector = JSONAnomalyDetector(
    checkpoint_path="models/stg_nf_trained.pth",  # Use trained checkpoint from Nov25_0118
    threshold=0.0,  # Adjust based on EER from training
)


def save_np_array_to_file(array, filename):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"data/{timestamp}.txt"
    with open(filename, "w") as f:
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write(np.array2string(array))


def do_work(cap):
    frame_count = 0
    anomaly_count = 0
    normal_count = 0
    feature_count = 18
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        
        
        
        
        # p1 person detection returns list of [[x,y,w,h]]
        detections = process_frame(frame)
        if detections is None or len(detections) == 0:
            continue
        
        #verificaiton of p1
        
        # for det  in detections:
        #     bounding, score = det
        #     a,b,c,d = bounding
        #     cv2.rectangle(frame, (int(a), int(b)), (int(a + c), int(b + d)), (0, 255, 0), 2)
        #     cv2.imshow("Detections", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #verificaiton end 
        # print(frame_count, "process 1 ended")
        


            
        # p2 deep sort tracking returns {id:(x1,y1,x2,y2)}
        tracking_data = tracking(detections, frame)
        
        #verificaiton of p2
        # print(tracking_data)
        # for key, (x, y, w, h) in tracking_data.items():
        # #     cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0,255,0), 2)
        #     cv2.putText(frame, key, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # cv2.imshow("Detections", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break       
       

        out, multi = manager.update(frame, tracking_data)
        if len(out) > 0:
            results = detector.predict_from_dict(out, scene_id="01", clip_id="0222")
            print("🚀 result : ", results)

            # Process all detected persons
            has_anomaly = False
            frame_with_overlay = frame.copy()

            for result in results:
                person_id = result["person_id"]
                score = result["score"]
                classification = result["classification"]
                confidence = result["confidence"]
                is_abnormal = result["is_abnormal"]

                if is_abnormal:
                    has_anomaly = True

                # Get person bounding box from tracking_data
                if str(person_id) in tracking_data or person_id in tracking_data:
                    bbox_key = (
                        str(person_id) if str(person_id) in tracking_data else person_id
                    )
                    bbox = tracking_data[bbox_key]
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[0]+bbox[2]),
                        int(bbox[1]+bbox[3]),
                    )

                    # Choose color based on classification
                    color = (
                        (0, 0, 255) if is_abnormal else (0, 255, 0)
                    )  # Red for anomaly, Green for normal

                    # Draw bounding box
                    cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 3)

                    # Prepare text overlay
                    text_lines = [
                        f"ID: {person_id}",
                        f"Score: {score:.2f}",
                        f"{classification}",
                        f"Conf: {confidence}",
                    ]

                    # Draw text background and text
                    y_offset = y1 - 10
                    for i, text in enumerate(text_lines):
                        text_y = y_offset - (len(text_lines) - i) * 25
                        # Background rectangle
                        (text_w, text_h), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            frame_with_overlay,
                            (x1, text_y - text_h - 5),
                            (x1 + text_w + 10, text_y + 5),
                            color,
                            -1,
                        )
                        # Text
                        cv2.putText(
                            frame_with_overlay,
                            text,
                            (x1 + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

            # Save frame to appropriate folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if has_anomaly:
                anomaly_count += 1
                filename = f"anomaly_frames/frame_{frame_count:06d}_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_overlay)
                print(f"✅ Saved anomaly frame: {filename}")
            else:
                normal_count += 1
                filename = f"ok/frame_{frame_count:06d}_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_overlay)
                print(f"✅ Saved normal frame: {filename}")

    print(
        f"\n📊 Summary: Processed {frame_count} frames, saved {anomaly_count} anomaly frames and {normal_count} normal frames"
    )


do_work(video_cap)

# for i in range(1, 7):
#     video_url = f"./media/sample{i}.mp4"
#     # print("🚀 video_url : ", video_url)
#     # video_cap = cv2.VideoCapture(video_url)
#     # do_work(video_cap)
