"""
Utility functions for pose preprocessing - copied from STG-NF codebase
Makes inference_only folder fully independent and portable
"""

import numpy as np


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False,
                         global_pose_data=[], dataset="PoseLift"):  
    """
    Generate an array of segmented sequences from AlphaPose tracking dict
    
    Args:
        clip_dict: Dict with structure {person_id: {frame_num: {keypoints: [...], scores: ...}}}
        start_ofst: Starting offset for segments
        seg_stride: Stride between segments
        seg_len: Length of each segment (default 12 frames)
        scene_id: Scene identifier
        clip_id: Clip identifier
        ret_keys: Return person keys mapping
        global_pose_data: List to store global pose data
        dataset: Dataset name (default "PoseLift")
    
    Returns:
        pose_segs_data_np: Array of shape [N, seg_len, 17, 3] (sequences, frames, keypoints, x/y/conf)
        pose_segs_meta: List of metadata [scene_id, clip_id, person_id, start_frame]
        person_keys: Dict mapping person keys to frame numbers
        global_pose_data_np: Concatenated global pose data
        global_pose_data: List of global pose data
        score_segs_data_np: Array of scores
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)
        
        if dataset == "UBnormal":
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys
        
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = split_pose_to_segments(
            sing_pose_np,
            sing_pose_meta,
            sing_pose_keys,
            start_ofst, 
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            single_score_np=sing_scores_np,
            dataset=dataset
        )

        if dataset == 'PoseLift':  
            curr_pose_score_np = np.zeros(curr_pose_score_np.shape)       
        
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        
        pose_segs_meta += curr_pose_segs_meta
    
    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    
    global_pose_data_np = np.concatenate(global_pose_data, axis=0) if len(global_pose_data) > 0 else np.empty(0)
    del pose_segs_data
    
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np
    else:
        return pose_segs_data_np, pose_segs_meta, global_pose_data_np, global_pose_data, score_segs_data_np


def single_pose_dict2np(person_dict, idx):
    """
    Convert single person's pose dict to numpy array
    
    Args:
        person_dict: Full dict with all persons
        idx: Person ID to extract
    
    Returns:
        sing_pose_np: Array of shape [T, 17, 3] (frames, keypoints, x/y/conf)
        sing_pose_meta: [person_id, first_frame]
        single_person_dict_keys: Sorted frame numbers
        sing_scores_np: Array of scores
    """
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # [index, first_frame]
    
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        
        # Handle both 'scores' (array) and 'score' (single value)
        scores = single_person[key].get('scores')
        if scores is None:
            scores = single_person[key].get('score', 0.0)
        sing_scores_np.append(scores)
    
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.array(sing_scores_np)
    
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Check if a segment has continuous frames
    
    Args:
        sorted_seg_keys: Sorted list of available frame numbers
        start_key: Starting frame number
        seg_len: Required segment length
        missing_th: Number of frames allowed to be missing (default 2)
    
    Returns:
        bool: True if segment is continuous enough
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id='', single_score_np=None, dataset="PoseLift"):
    """
    Split single person's pose sequence into overlapping segments
    
    Args:
        single_pose_np: Array of shape [T, 17, 3]
        single_pose_meta: [person_id, first_frame]
        single_pose_keys: List of frame numbers
        start_ofst: Starting offset
        seg_dist: Stride between segments
        seg_len: Length of each segment
        scene_id: Scene identifier
        clip_id: Clip identifier
        single_score_np: Scores array
        dataset: Dataset name
    
    Returns:
        pose_segs_np: Array of shape [N, seg_len, 17, 3]
        pose_segs_meta: List of metadata
        pose_score_np: Array of scores
    """
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])
    
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            
            if single_score_np is not None and len(single_score_np.shape) > 0:
                curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            else:
                curr_score = np.zeros((1, seg_len))
            
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            
            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    
    return pose_segs_np, pose_segs_meta, pose_score_np


def normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    
    Args:
        pose_data: Array of shape [N, T, V, F] (batch, frames, keypoints, features)
                   e.g., (64, 12, 18, 3) where features are (x, y, confidence)
        vid_res: Video resolution [width, height] (default [856, 480])
        symm_range: Whether to shift data to [-1, 1] range (default False)
    
    Returns:
        pose_data_zero_mean: Normalized pose data with same shape as input
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', False)
    scale = kwargs.get('scale', True)
    scale_proportional = kwargs.get('scale_proportional', True)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    
    if symm_range:  # Shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    pose_data_zero_mean = pose_data_centered
    
    # Normalize by mean and std
    pose_data_zero_mean[..., :2] = (
        pose_data_centered[..., :2] - pose_data_centered[..., :2].mean(axis=(1, 2))[:, None, None, :]
    ) / pose_data_centered[..., 1].std(axis=(1, 2))[:, None, None, None]
    
    return pose_data_zero_mean
