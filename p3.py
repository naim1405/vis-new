import collections
import numpy as np
import cv2


# builds input json for the model
def block3(
    seq_len=30,
    use_mediapipe=True,
    mediapipe_static_image_mode=False,
    mediapipe_model_complexity=1,
    mediapipe_enable_seg=False,
    num_joints=33,
    coco17_map=None,
    normalize_mode="bbox",
    missing_keypoint_strategy="last",
    require_all_nonzero=False,
):
    # Internal state
    buffers = {}  # track_id -> deque(maxlen=seq_len) of frames (each is np.array(num_joints,3))
    last_seen = {}  # track_id -> last known frame (np.array)
    import_time = None

    # Setup Mediapipe
    if use_mediapipe:
        try:
            import mediapipe as mp

            mp_pose = mp.solutions.pose
            pose_args = {
                "static_image_mode": mediapipe_static_image_mode,
                "model_complexity": mediapipe_model_complexity,
                "enable_segmentation": mediapipe_enable_seg,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            }
            mp_pipe = mp_pose.Pose(**pose_args)
        except Exception as e:
            raise ImportError("🚀 MediaPipe not available. %s" % e)
    else:
        mp_pipe = None

    # Helpers
    def _ensure_buffer(tid):
        if tid not in buffers:
            buffers[tid] = collections.deque(maxlen=seq_len)

    def _bbox_to_crop(frame, bbox):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        # print(f"🚀 _bbox_to_crop - input bbox (x,y,w,h): {bbox}")
        # print(f"🚀 _bbox_to_crop - frame shape (H,W): ({H},{W})")
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))
        # print(
        #     f"🚀 _bbox_to_crop - calculated coords (x1,y1,x2,y2): ({x1},{y1},{x2},{y2})"
        # )
        crop = frame[y1:y2, x1:x2].copy()
        # print(f"🚀 _bbox_to_crop - crop shape: {crop.shape}")
        return crop, (x1, y1, x2, y2)

    def _mediapipe_keypoints_from_crop(crop, bbox):
        # Returns keypoints as np.array(num_joints, 3) in pixel coordinates relative to the original frame.

        if mp_pipe is None:
            return None

        # print(f"🚀 _mediapipe_keypoints - crop shape: {crop.shape}, bbox: {bbox}")
        if crop.size == 0:
            # print(f"🚀 ERROR: crop is empty! Cannot process.")
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = mp_pipe.process(img)
        if not results.pose_landmarks:
            return None

        # mp_pose has 33 landmarks
        lm = results.pose_landmarks.landmark
        h_crop, w_crop = crop.shape[:2]

        kps = np.zeros((len(lm), 3), dtype=np.float32)
        for i, l in enumerate(lm):
            kps[i, 0] = l.x * w_crop
            kps[i, 1] = l.y * h_crop
            kps[i, 2] = (
                l.visibility
                if hasattr(l, "visibility")
                else l.z
                if hasattr(l, "z")
                else 1.0
            )

        x1, y1, _, _ = bbox
        kps[:, 0] += x1
        kps[:, 1] += y1
        return kps  # shape (33,3)

    def _normalize_keypoints(kps, bbox, mode="bbox"):
        """
        kps: np.array(num_joints,3) in original frame pixel coords
        bbox: (x, y, w, h)
        mode:
          - 'bbox': normalize relative to bbox center & scale -> [-1,1]
          - 'frame': normalize to [0,1] w.r.t full frame (requires frame size)
        Returns np.array(num_joints,3) with x,y normalized and score as-is.
        """
        x, y, w, h = bbox
        if mode == "bbox":
            cx = x + w / 2.0
            cy = y + h / 2.0
            sx = w / 2.0
            sy = h / 2.0
            # protect divide by zero
            sx = max(sx, 1.0)
            sy = max(sy, 1.0)
            kps_norm = np.zeros_like(kps)
            kps_norm[:, 0] = (kps[:, 0] - cx) / sx  # in [-inf,inf], mostly [-1,1]
            kps_norm[:, 1] = (kps[:, 1] - cy) / sy
            kps_norm[:, 2] = kps[:, 2]
            return kps_norm
        else:
            # 'frame' mode (requires global frame dims)
            # if user wants frame normalization, they can call convert later. We keep bbox mode default.
            raise NotImplementedError(
                "Only 'bbox' normalization implemented by default."
            )

    # Core frame processor
    def process_frame(id_bbox_map, frame):
        """
        id_bbox_map: dict(track_id -> (x,y,w,h))
        frame: full RGB/BGR image (numpy array, BGR expected)
        Returns:
                    ready_sequences: dict(track_id -> np.array shape (seq_len, num_joints, 3))
                        (only tracks that have just reached seq_len are returned; buffers slide as deque)
                        If require_all_nonzero=True, returns only when all seq_len frames
                        in the window are non-zero (missing frames are zero-filled).
        """
        ready = {}
        frame_h, frame_w = frame.shape[:2]

        for tid, bbox in id_bbox_map.items():
            _ensure_buffer(tid)
            # print("🚀 buffers : ", len(buffers))
            crop, orig_bbox = _bbox_to_crop(frame, bbox)  # orig_bbox = (x1,y1,x2,y2)
            x1, y1, x2, y2 = orig_bbox
            w = x2 - x1
            h = y2 - y1
            bbox_xywh = (x1, y1, w, h)

            # get keypoints in frame coords
            if use_mediapipe:
                kps = _mediapipe_keypoints_from_crop(crop, (x1, y1, x2, y2))
            else:
                kps = None  # alpha pose or other adapter should be used
            # print("🚀 kps : ", (kps))

            if kps is None:
                # missing detection: fallback to last seen or zeros
                if tid in last_seen and missing_keypoint_strategy == "last":
                    kps_use = last_seen[tid].copy()
                else:
                    # zeros with low confidence
                    kps_use = np.zeros((num_joints, 3), dtype=np.float32)
            else:
                # If mediapipe returns 33 joints but user wants different num_joints:
                if coco17_map is not None:
                    # map from mediapipe indices -> desired 17 joints
                    mapped = np.zeros((len(coco17_map), 3), dtype=np.float32)
                    for i, mp_idx in enumerate(coco17_map):
                        mapped[i] = kps[mp_idx]
                    kps_use = mapped
                else:
                    # resize/pad if needed
                    if kps.shape[0] != num_joints:
                        # if mediapipe returns 33 but num_joints requested smaller, pad/truncate
                        if kps.shape[0] > num_joints:
                            kps_use = kps[:num_joints]
                        else:
                            pad = np.zeros(
                                (num_joints - kps.shape[0], 3), dtype=np.float32
                            )
                            kps_use = np.vstack([kps, pad])
                            # print("🚀 if else: ")
                    else:
                        kps_use = kps
                        # print("🚀 else: ", len(kps_use))

            # normalize relative to bbox center (recommended)
            # print("🚀 kps_use : ", len(kps_use))
            kps_norm = _normalize_keypoints(kps_use, bbox_xywh, mode=normalize_mode)
            # print("🚀 kps_norm : ", len(kps_norm))

            # append to buffer
            buffers[tid].append(kps_norm.astype(np.float32))
            last_seen[tid] = kps_norm.copy()

            # if buffer is full, return sequence
            # print("🚀 len(buffers[tid])  : ", len(buffers[tid]), " ", tid)
            # print("🚀 seq_len : ", seq_len)
            # When buffer reaches configured length, attempt to emit a sequence
            if len(buffers[tid]) == seq_len:
                # print("🚀 seq_len : ", seq_len)
                seq_arr = np.stack(buffers[tid], axis=0)  # (seq_len, num_joints, 3)
                # Option: you may want to copy then pop left to create sliding windows,
                # here we perform sliding by popping left once (so next will overlap)
                # keep last seq_len-1 frames to form sliding window; pop left once
                # but since deque has maxlen, to slide we pop left here so older frame removed
                emit_ok = True
                if require_all_nonzero:
                    try:
                        # Frame considered non-zero if any element differs from 0
                        frame_has_any = np.any(seq_arr != 0, axis=(1, 2))
                        emit_ok = bool(np.all(frame_has_any))
                    except Exception:
                        emit_ok = False

                if emit_ok:
                    try:
                        # produce a copy to avoid mutation
                        ready[tid] = seq_arr.copy()

                        # slide window: remove oldest frame so next fill creates a sliding window
                        buffers[tid].popleft()
                    except Exception:
                        # fallback: clear buffer
                        buffers[tid].clear()

        # print("🚀 ready : ", len(ready))
        return ready

    def get_buffer():
        """Return shallow copy state of buffers for debugging: {tid: len(buffer)}"""
        return {tid: len(buff) for tid, buff in buffers.items()}

    def close():
        if mp_pipe is not None:
            mp_pipe.close()

    # Return the API
    return {"process_frame": process_frame, "get_buffer": get_buffer, "close": close}
