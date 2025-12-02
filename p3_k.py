# %%
from ultralytics import YOLO

model_path = "models/yolov8n-pose.pt"

model = YOLO(model_path)  # should download or load without error
print("Model loaded successfully")


# %%
import os
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO  # pip install ultralytics
import cv2
import json
from typing import Dict, Tuple, Any, Optional, List
import logging

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class FrameBufferManager:
    """
    Maintains per-person, per-frame keypoint buffer.
    For each incoming frame, call `update(frame, bbox_map, frame_id=None)`.
    - frame: HxWx3 BGR numpy array (cv2 image)
    - bbox_map: dict{person_id: (x, y, w, h)} where x,y is top-left in pixels
    - frame_id: optional integer frame number (if None, manager will increment)

    When any person reaches `sequence_length`, their sequence is removed
    from internal buffer and returned as part of output dict.
    """

    def __init__(
        self,
        pose_model_path: str = model_path,
        sequence_length: int = 24,
        frame_digits: int = 4,
        crop_padding: float = 0.01,
        device: str = "cpu",
    ):
        # load pose model (YOLOv8 pose COCO 17)
        self.model = YOLO(pose_model_path, verbose=False)
        # ensure running on given device if supported
        try:
            self.model.to(device)
        except Exception:
            pass

        self.sequence_length = sequence_length
        self.frame_digits = frame_digits
        self.crop_padding = crop_padding  # fraction of box size to pad when cropping
        self.buffer = defaultdict(
            lambda: deque(maxlen=1000)
        )  # id -> deque of (frame_str, kp_flat, score)
        self.next_frame_id = 1

    def _format_frame_str(self, frame_id: int) -> str:
        return str(frame_id).zfill(self.frame_digits)

    def _crop_with_padding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        # ensure all are integers for slicing
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(W, int(x + w + pad_x))
        y2 = min(H, int(y + h + pad_y))

        crop = frame[y1:y2, x1:x2].copy()
        return crop, x1, y1

    def _run_pose_on_crop(
        self, crop: np.ndarray
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Run YOLOv8-pose on the crop. Returns list of (x,y,conf) in CROP coordinates
        for the detected person with highest confidence. If no person detected -> None.
        """
        if crop.size == 0:
            return None
        # ultralytics model expects either file path or numpy array (BGR) directly
        try:
            results = self.model(
                crop, imgsz=640, conf=0.05, verbose=False
            )  # returns a Results object list
        except Exception as e:
            raise RuntimeError(f"YOLO model call failed: {e}")

        if len(results) == 0:
            return None
        r = results[0]

        # Different ultralytics versions expose keypoints differently.
        # We'll try to find keypoints in a few likely places.
        kps = None
        # 1) results[0].keypoints (common)
        if hasattr(r, "keypoints") and r.keypoints is not None:
            try:
                # r.keypoints could be a Tensor-like object; convert to numpy
                kps_np = (
                    np.array(r.keypoints.cpu())
                    if hasattr(r.keypoints, "cpu")
                    else np.array(r.keypoints)
                )
                # shape might be (num_instances, 17, 3)
                if kps_np.ndim == 3:
                    # choose the instance with max mean confidence
                    mean_conf = kps_np[:, :, 2].mean(axis=1)
                    idx = int(np.argmax(mean_conf))
                    kps = kps_np[idx].tolist()
            except Exception:
                pass

        # 2) results[0].keypoints.xy or results[0].keypoints.data (alternate)
        if kps is None:
            # try `r.keypoints.xy` and confidences if present
            try:
                if hasattr(r, "keypoints") and hasattr(r.keypoints, "xy"):
                    xy = np.array(r.keypoints.xy)
                    conf = (
                        np.array(r.keypoints.conf)
                        if hasattr(r.keypoints, "conf")
                        else None
                    )
                    if xy.ndim == 3:
                        # shape (num_instances, 17, 2)
                        if conf is None:
                            # fallback confs to zeros
                            conf = np.zeros((xy.shape[0], xy.shape[1]))
                        mean_conf = conf.mean(axis=1)
                        idx = int(np.argmax(mean_conf))
                        kps = []
                        for j in range(xy.shape[1]):
                            kps.append(
                                (
                                    float(xy[idx, j, 0]),
                                    float(xy[idx, j, 1]),
                                    float(conf[idx, j]),
                                )
                            )
            except Exception:
                pass

        # 3) try parsing r.boxes.keypoints (older/newer variants)
        if kps is None:
            try:
                if (
                    hasattr(r, "boxes")
                    and r.boxes is not None
                    and hasattr(r.boxes, "keypoints")
                ):
                    kps_np = (
                        np.array(r.boxes.keypoints.cpu())
                        if hasattr(r.boxes.keypoints, "cpu")
                        else np.array(r.boxes.keypoints)
                    )
                    if kps_np.ndim == 3:
                        mean_conf = kps_np[:, :, 2].mean(axis=1)
                        idx = int(np.argmax(mean_conf))
                        kps = kps_np[idx].tolist()
            except Exception:
                pass

        # 4) If still None, give up (no detection)
        if kps is None:
            return None

        # kps is list of 17 (x,y,conf) in CROP coordinates
        # ensure length 17
        if len(kps) != 17:
            # If model returns different number, try to adapt: if >17, take first 17; if <17, pad.
            if len(kps) > 17:
                kps = kps[:17]
            else:
                # pad with zeros
                while len(kps) < 17:
                    kps.append((0.0, 0.0, 0.0))
        return kps

    def _flatten_keypoints(
        self, kps_crop: List[Tuple[float, float, float]], offset_x: int, offset_y: int
    ) -> Tuple[List[float], float]:
        """
        Convert list of (x,y,conf) in crop coords -> flattened [x_abs,y_abs,conf]*17
        and produce a person score (mean confidence).
        """
        flattened = []
        scores = []
        for x, y, c in kps_crop:
            x_abs = float(x + offset_x)
            y_abs = float(y + offset_y)
            flattened.extend([x_abs, y_abs, float(c)])
            scores.append(float(c))
        # person score: mean of keypoint confidences (fallback to 0)
        person_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return flattened, person_score

    def update(
        self,
        frame: np.ndarray,
        bbox_map: Dict[Any, Tuple[int, int, int, int]],
        frame_id: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], bool]:
        """
        Process a single frame.
        - frame: HxWx3 BGR image (numpy)
        - bbox_map: dict mapping person_id -> (x, y, w, h) top-left pixel coords
        - frame_id: optional integer frame number; if None, use internal incrementing id

        Returns:
            output_buffer: dict {person_id_str: {frame_str: {"keypoints": [...], "score": float}}}
                for all persons who JUST reached sequence_length (possibly multiple ids).
            multiple_flag: True if >1 person reached sequence_length at the same time.
        """
        if frame_id is None:
            frame_id = self.next_frame_id
        frame_str = self._format_frame_str(frame_id)

        finished = {}  # person_id -> dict of frames to return

        # For every person bbox in current frame:
        for pid, bbox in bbox_map.items():
            # ensure pid is string key in JSON
            pid_str = str(pid)

            # crop with padding and get crop origin offsets
            crop, off_x, off_y = self._crop_with_padding(frame, bbox)
            # print(f"Processing person {pid} with crop shape {crop.shape}")

            # get keypoints for the person in crop coords
            kps_crop = self._run_pose_on_crop(crop)
            # print(pid, kps_crop)

            if kps_crop is None:
                # detection failed: store zero keypoints and 0 score (you may decide another fallback)
                kp_flat = [0.0] * (17 * 3)
                person_score = 0.0
            else:
                kp_flat, person_score = self._flatten_keypoints(kps_crop, off_x, off_y)

            # push into buffer: (frame_str, kp_flat, score)
            self.buffer[pid_str].append((frame_str, kp_flat, person_score))

            # check if this person's buffer length >= sequence_length
            if len(self.buffer[pid_str]) >= self.sequence_length:

                # Build candidate sequence
                seq_dict = {}
                entries = list(self.buffer[pid_str])[: self.sequence_length]

                all_values = []
                for fstr, kpf, sc in entries:
                    seq_dict[fstr] = {"keypoints": kpf, "score": sc}
                    all_values.extend(kpf)

                # Count how many values are NULL (== 0.0)
                zero_count = sum(1 for v in all_values if v == 0.0)

                # YOUR threshold (set XYZ to whatever you want)
                XYZ = 200

                # Remove entries from buffer no matter what
                remaining = list(self.buffer[pid_str])[self.sequence_length:]
                self.buffer[pid_str] = deque(remaining, maxlen=1000)

                # Only return sequence if zero_count is acceptable
                if zero_count <= XYZ:
                    finished[pid_str] = seq_dict
                else:
                    # discard: do NOT include in finished
                    print(f"Person {pid_str} discarded due to high null count: {zero_count}")


        # increment frame counter for next call
        self.next_frame_id = frame_id + 1

        # Format finished as final output dict
        output_buffer = (
            finished  # keys are person ids as strings; values are frame->dicts
        )
        multiple_flag = len(output_buffer) > 1

        return output_buffer, multiple_flag

    def dump_buffer_json(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns current buffer in the same JSON-like layout (for inspection/debugging).
        """
        out = {}
        for pid, dq in self.buffer.items():
            out[pid] = {}
            for fstr, kpf, sc in dq:
                out[pid][fstr] = {"keypoints": kpf, "score": sc}
        return out


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Example: simulate frames and bbox_map
    manager = FrameBufferManager(
        pose_model_path=model_path,
        sequence_length=24,
        frame_digits=4,
        device="cpu",
    )
    # Load a sample frame (replace with your actual frames in a loop)
    frame = cv2.imread(
        "C:\\Users\\LENOVO\\Downloads\\Compressed\\visiongaurd\\20.jpg"
    )  # BGR image
    # Suppose you have tracking ids and bboxes:
    # bbox_map = {
    #     0: (100, 50, 80, 160),   # (x,y,w,h) top-left
    #     1: (300, 40, 90, 170),
    # }

    bbox_map = {
        0: (175.90618896484375, 283.386474609375, 59.803466796875, 140.6126708984375),
        1: (22.92767333984375, 247.980224609375, 55.08995819091797, 135.06640625),
        2: (336.8275146484375, 20.076690673828125, 44.910888671875, 122.70515441894531),
        3: (
            269.208740234375,
            222.44500732421875,
            54.98626708984375,
            134.27813720703125,
        ),
        4: (
            459.22979736328125,
            239.7570343017578,
            64.21868896484375,
            123.30662536621094,
        ),
        5: (235.919921875, 138.51385498046875, 42.9990234375, 115.30282592773438),
        6: (
            302.77349853515625,
            117.91400146484375,
            46.9053955078125,
            122.28732299804688,
        ),
        7: (95.42851257324219, 94.6858901977539, 44.63526916503906, 118.70928192138672),
        8: (
            131.28701782226562,
            134.75360107421875,
            45.83489990234375,
            117.2784423828125,
        ),
        9: (189.84017944335938, 82.3543701171875, 48.1597900390625, 165.18399047851562),
        10: (
            71.75164794921875,
            6.6855316162109375,
            34.08937072753906,
            105.83248901367188,
        ),
        11: (
            164.82196044921875,
            160.82415771484375,
            57.8238525390625,
            148.7620849609375,
        ),
    }

    out, multi = manager.update(frame, bbox_map)  # call this per frame
    if out:
        # out is dict with person ids who just reached sequence_length
        print("Sequences ready:", json.dumps(out, indent=2))
        print("Multiple:", multi)

    # Inspect internal buffer anytime
    # print("Current buffer:", json.dumps(manager.dump_buffer_json(), indent=2))
