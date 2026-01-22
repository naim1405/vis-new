# STG-NF Model Integration - Issues and Solutions

## Summary
Successfully integrated the STG-NF anomaly detection model into the vis-new video pipeline. The model now processes pose sequences from FrameBufferManager and returns anomaly classifications.

---

## Critical Issues Encountered

### Issue 1: Model Architecture Mismatch
**Problem:** The trained checkpoint had a completely different architecture than the inference code expected.

**Root Cause:** 
- The model was trained with **debug configuration** (`model_hidden_dim=0`) instead of production settings
- Training used minimal architecture for fast experimentation, not optimal performance

**What the code expected:**
```python
hidden_channels: 512       # Production: Large model capacity
L: 3                       # 3 levels of flow layers
flow_permutation: "invconv" # Invertible convolution
learn_top: True            # Learnable prior
edge_importance: True      # Edge weight learning
```

**What the checkpoint actually had:**
```python
hidden_channels: 0         # Debug: Minimal architecture (single block per flow step)
L: 1                       # Only 1 level
flow_permutation: "permute" # Simple permutation (no invconv layers)
learn_top: False           # No learnable prior
edge_importance: False     # No edge weights
```

**Error symptoms:**
- Hundreds of "Missing key(s) in state_dict" errors for invconv layers, learn_top layers
- "size mismatch" errors showing checkpoint had [2, 2, 13, 1] but code expected [512, 512, 7, 1]

**Solution:** Updated `json_inference.py` model configuration to match training args from `Nov25_0118/args.json`:
```python
model_config = {
    "pose_shape": (2, 24, 18),
    "hidden_channels": 0,          # Changed from 512
    "K": 8,
    "L": 1,                        # Changed from 3
    "flow_permutation": "permute", # Changed from "invconv"
    "learn_top": False,            # Changed from True
    "edge_importance": False,      # Changed from True
    "R": 3.0,
    "temporal_kernel_size": None,  # Auto-calculates to 13
}
```

---

### Issue 2: Incorrect Segment Length
**Problem:** Inference code used 12-frame segments, but model was trained on 24-frame segments.

**Root Cause:** Default `seg_len=12` in inference code, but training used `seg_len=24` from args.json.

**Impact:**
- Model input shape: `[N, 2, T, 18]` where T must be 24
- Temporal kernel size auto-calculated as `T // 2 + 1 = 13` (not 7)

**Solution:** Changed segment length in `predict_from_dict()`:
```python
# Before
seg_len=12,
seg_stride=1,

# After  
seg_len=24,        # Match training
seg_stride=6,      # Match training stride
```

---

### Issue 3: Wrong Keypoint Count
**Problem:** FrameBufferManager outputs 17 keypoints (COCO17 format), but model expects 18 (COCO18).

**Solution:** Added padding to convert COCO17 → COCO18:
```python
poses = np.transpose(poses, (0, 3, 1, 2))  # [N, 2, 24, 17]
poses = np.pad(poses, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant')  # [N, 2, 24, 18]
```

---

### Issue 4: Incorrect Shape Transformations
**Problem:** ValueError showing shape mismatch `(18,17,3,12)` vs expected `(3,)` during normalization.

**Root Cause:** 
- `gen_clip_seg_data_np()` outputs `[N, 24, 17, 3]` (batch, frames, keypoints, x/y/conf)
- Code was incorrectly transposing before normalization: `.transpose((0, 2, 3, 1))` 
- This created wrong shape `[N, 17, 3, 24]` instead of expected `[N, 24, 17, 3]`

**Solution:** Removed unnecessary transpose before normalization:
```python
# Before (WRONG)
segs_data_np = normalize_pose(
    segs_data_np.transpose((0, 2, 3, 1)), ...  # Wrong transpose
).transpose((0, 3, 1, 2))

# After (CORRECT)
segs_data_np = normalize_pose(
    segs_data_np, ...  # Already correct shape [N, 24, 17, 3]
)
```

---

### Issue 5: ActNorm Initialization Error
**Problem:** `ValueError: In Eval mode, but ActNorm not inited` when running inference.

**Root Cause:**
- ActNorm layers store an `inited` flag (Python attribute, not in state_dict)
- After `load_state_dict()`, this flag remains `False` even though parameters are loaded
- When model is in eval mode, ActNorm checks this flag and raises error

**Solution:** Manually mark all ActNorm layers as initialized after loading:
```python
self.model.load_state_dict(checkpoint["state_dict"])

# Mark all ActNorm layers as initialized
for module in self.model.modules():
    if hasattr(module, 'inited'):
        module.inited = True

self.model.to(self.device)
self.model.eval()
```

---

## Production vs Debug Model

### Current Model (Debug/Experimental)
- **hidden_channels: 0** → Creates single-block architecture per flow step
- **Pros:** Fast training, small checkpoint (185KB), quick iteration
- **Cons:** 
  - Reduced model capacity
  - May have lower accuracy on complex anomalies
  - Less feature learning capability

### Production Model (Recommended for Deployment)
- **hidden_channels: 512** → Creates two-block architecture per flow step
- **L: 3** → More flow levels for better expressiveness
- **edge_importance: True** → Learns importance weights for skeleton edges
- **learn_top: True** → Learnable prior for better distribution modeling
- **Pros:**
  - Higher model capacity
  - Better anomaly detection accuracy
  - More robust feature learning
- **Cons:**
  - Larger checkpoint size (~50-100MB)
  - Slower inference (~2-3x)
  - Requires retraining

### Recommendation
**Current model works but is suboptimal.** For production deployment with high accuracy requirements:
1. Retrain model with production config (hidden_channels=512, L=3, edge_importance=True)
2. Use all available training data
3. Train for more epochs (current model only trained 8 epochs)
4. Update inference code to match new architecture

---

## Files Modified

### `/home/ezio/Documents/work/vis-new/inference_only/json_inference.py`
**Changes:**
1. Line 40-56: Updated model_config to match training checkpoint
   - `hidden_channels: 0` (was 512)
   - `L: 1` (was 3)
   - `flow_permutation: "permute"` (was "invconv")
   - `learn_top: False` (was True)
   - `edge_importance: False` (was True)
   - `pose_shape: (2, 24, 18)` (was (2, 12, 18))

2. Line 60-65: Added ActNorm initialization after loading checkpoint

3. Line 106-107: Changed segment length from 12 to 24 frames

4. Line 119-131: Fixed shape transformations
   - Removed incorrect transpose before normalization
   - Added COCO17→COCO18 padding with np.pad()
   - Updated all shape comments

---

## Current Pipeline Flow

```
Video Frame
    ↓
YOLOv8 Detection (p1.py)
    ↓
Deep SORT Tracking (p2.py)  
    ↓
FrameBufferManager (p3_k.py)
    ↓ Outputs: {person_id: {frame: {keypoints: [51], score: float}}}
    ↓
JSONAnomalyDetector.predict_from_dict()
    ↓
    1. gen_clip_seg_data_np() → [N, 24, 17, 3]
    2. normalize_pose() → Normalize coordinates
    3. Reshape to [N, 2, 24, 18] (pad 17→18 keypoints)
    4. Model inference → Anomaly scores
    ↓
Results: [{person_id, score, is_abnormal, confidence, ...}]
```

---

## Verification

### Test Result
```
Processing pose data
Scene: 01, Clip: 0222
Persons detected: 1
======================================================================

Extracted 1 sequences of 24 frames
🚀 result: [{
    'sequence_id': 0,
    'person_id': 1, 
    'start_frame': 1,
    'end_frame': 12,
    'score': -1.87,
    'is_abnormal': True,
    'classification': 'Abnormal',
    'confidence': 'Medium'
}]
```

**Status:** ✅ Model successfully loaded and running inference

---

## Next Steps (Optional Improvements)

1. **Retrain with production config** for better accuracy
2. **Tune anomaly threshold** (current: 0.0) based on validation data
3. **Add confidence calibration** using validation set scores
4. **Implement temporal smoothing** across consecutive sequences
5. **Add visualization** of detected anomalies on video frames
6. **Profile inference speed** and optimize if needed

---

## Configuration Reference

### Training Configuration (from Nov25_0118/args.json)
```json
{
    "seg_len": 24,
    "seg_stride": 6,
    "K": 8,
    "L": 1,
    "R": 3.0,
    "temporal_kernel": null,
    "edge_importance": false,
    "flow_permutation": "permute",
    "adj_strategy": "uniform",
    "max_hops": 8,
    "model_hidden_dim": 0
}
```

### Current Inference Configuration
```python
{
    "pose_shape": (2, 24, 18),
    "hidden_channels": 0,
    "K": 8,
    "L": 1,
    "actnorm_scale": 1.0,
    "flow_permutation": "permute",
    "flow_coupling": "affine",
    "LU_decomposed": True,
    "learn_top": False,
    "R": 3.0,
    "edge_importance": False,
    "temporal_kernel_size": None,
    "strategy": "uniform",
    "max_hops": 8
}
```

Both configurations now match exactly, enabling successful model loading and inference.
