# create block3 object with default params
b3 = block3(seq_len=30)

process_frame = b3["process_frame"]
get_buffer = b3["get_buffer"]
close_block3 = b3["close"]

# main video loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 1) run your detection (block1) -> detections = [ [x,y,w,h], conf ]
    detections = process_frame_with_yolo(frame)  # your function

    # 2) run DeepSORT (block2) -> id_bbox_map = {id: (x,y,w,h), ...}
    id_bbox_map = tracking(detections, frame)  # your function; must return map

    # 3) feed to block3
    ready_sequences = process_frame(
        id_bbox_map, frame
    )  # dict: id -> np.array(seq_len, num_joints, 3)

    # 4) for each ready sequence, call STG-NF inference (block4)
    for tid, seq in ready_sequences.items():
        # seq is np.array shape (30, num_joints, 3), with x,y normalized relative to bbox center
        # convert to torch tensor and to device; reshape as your model expects
        tensor = torch.tensor(seq).float().to(device)  # shape (30, J, 3)
        # if your model expects (batch, timesteps, joints*channels) convert accordingly
        # run inference...
