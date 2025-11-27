# STG-NF Inference - Minimal Setup

This folder contains **ONLY** the files needed to use a trained STG-NF model for inference.

## 📁 Folder Structure

```
inference_only/
├── README.md                    # This file
├── inference.py                 # Main inference script with examples
├── requirements.txt             # Python dependencies
└── models/
    └── STG_NF/
        ├── model_pose.py        # Main model architecture
        ├── modules_pose.py      # Building blocks (flows, convolutions)
        ├── stgcn.py            # Spatio-temporal graph convolutions
        ├── graph.py            # Skeleton graph structure
        └── utils.py            # Helper functions
```

**Total: Only 5 model files + 1 inference script!**

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Run Examples

```bash
python inference.py
```

This will run 5 different examples showing how to use the model.

---

## 💻 Basic Usage

```python
from inference import AnomalyDetector
import numpy as np

# 1. Load model
detector = AnomalyDetector(
    checkpoint_path="path/to/your/model.pth",
    threshold=0.0  # Adjust based on your EER threshold
)

# 2. Prepare input (from pose estimation system)
pose_sequence = np.load("pose_data.npy")  # Shape: [2, 12, 18]

# 3. Predict
score = detector.predict(pose_sequence)

# 4. Check result
if score < 0:
    print("⚠️  ABNORMAL behavior detected!")
else:
    print("✓ Normal behavior")
```

---

## 📥 Input Format

Your input must be a numpy array or torch tensor with shape: **[2, 12, 18]**

```python
Shape breakdown:
- 2:  Two channels [x_coordinates, y_coordinates]
- 12: Twelve consecutive video frames (~0.4 seconds at 30fps)
- 18: Eighteen body keypoints (COCO format)

Keypoint order (COCO18):
0: Nose          9: Right Knee      
1: Neck         10: Right Ankle     
2: Right Shoulder   11: Left Hip        
3: Right Elbow      12: Left Knee       
4: Right Wrist      13: Left Ankle      
5: Left Shoulder    14: Right Eye       
6: Left Elbow       15: Left Eye        
7: Left Wrist       16: Right Ear       
8: Right Hip        17: Left Ear        
```

**Where to get this data:**
- Use a pose estimation system like:
  - OpenPose
  - AlphaPose
  - MediaPipe
  - YOLO-Pose
  - MMPose

---

## 📤 Output Format

### Simple output:
```python
score = detector.predict(poses)
# Returns: 2.45 (float)
# Higher = more normal, Lower = more abnormal
```

### Detailed output:
```python
result = detector.predict(poses, return_details=True)
# Returns: {
#   'score': 2.45,
#   'is_abnormal': False,
#   'classification': 'Normal',
#   'confidence': 'High',
#   'distance_from_threshold': 2.45
# }
```

---

## 📊 Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| **> 2.0** | Very Normal | No action needed |
| **0 to 2.0** | Probably Normal | Monitor |
| **-0.5 to 0** | Borderline | Review manually |
| **< -0.5** | Likely Abnormal | Alert |
| **< -2.0** | Very Abnormal | Immediate attention |

---

## 🔧 Configuration

### Adjust Detection Threshold

```python
detector = AnomalyDetector(
    checkpoint_path="model.pth",
    threshold=0.0  # Change this value
)
```

**How to find the right threshold:**
1. Use the **EER threshold** from training output
2. Test on validation data and tune
3. Higher threshold = more sensitive (more false alarms)
4. Lower threshold = less sensitive (may miss anomalies)

### Model Architecture Settings

If you get errors loading the model, you may need to adjust the configuration in `inference.py`:

```python
model_config = {
    'pose_shape': (2, 12, 18),      # Must match training
    'hidden_channels': 512,          # Must match training
    'K': 8,                          # Must match training
    'L': 3,                          # Must match training
    # ...
}
```

Check your training logs or saved args to find these values.

---

## 📝 Examples in inference.py

1. **Basic Usage** - Simple prediction with dummy data
2. **Detailed Results** - Get confidence scores and classifications
3. **Batch Processing** - Process multiple sequences efficiently
4. **Load from File** - Load pose data from .npy files
5. **Video Stream** - Sliding window for real-time processing

---

## ⚠️ Important Notes

### 1. Input Normalization
Poses should be normalized (same as training):
- Centered around the person
- Scaled to standard size
- Coordinates typically in [0, 1] range

### 2. GPU vs CPU
```python
# Auto-detect (uses GPU if available)
detector = AnomalyDetector("model.pth", device=None)

# Force CPU
detector = AnomalyDetector("model.pth", device='cpu')

# Force GPU
detector = AnomalyDetector("model.pth", device='cuda')
```

### 3. Batch Processing
For better performance, process multiple sequences at once:
```python
# Slow - one at a time
for pose in poses:
    score = detector.predict(pose)

# Fast - batch processing
scores = detector.predict(poses)  # poses shape: [N, 2, 12, 18]
```

---

## 🐛 Troubleshooting

### Error: "Expected input shape [N, 2, 12, 18]"
- Check your input dimensions
- Make sure you have exactly 12 frames and 18 keypoints
- Transpose if needed: `poses.transpose(2, 0, 1)`

### Error: "Could not find checkpoint file"
- Update the `checkpoint_path` to point to your .pth file
- Use absolute path if relative path doesn't work

### Error: "RuntimeError: Error(s) in loading state_dict"
- Model configuration doesn't match checkpoint
- Check `model_config` values in `inference.py`
- Compare with your training configuration

### Low accuracy / Strange scores
- Input might not be normalized properly
- Threshold might need adjustment
- Model might have been trained on different data format

---

## 📚 What's NOT Included

This folder does NOT contain:
- ❌ Training code
- ❌ Dataset loading code
- ❌ Data preprocessing utilities
- ❌ Evaluation scripts
- ❌ Pose estimation code (you need to provide poses)

This is **inference only** - just load model and predict!

---

## 🔗 Pipeline Overview

```
Your Video → Pose Estimation → [2,12,18] → Model → Anomaly Score
   📹            🤖              Array      🧠        ✓/⚠️
            (OpenPose, etc)                       (This code)
```

**You are responsible for:**
1. Getting pose coordinates from videos (use OpenPose, AlphaPose, etc.)
2. Formatting them as [2, 12, 18] arrays

**This code handles:**
1. Loading the trained model
2. Running inference
3. Returning anomaly scores

---

## 📞 Support

If you need to modify the model or retrain:
- Go back to the full `STG-NF/` folder
- This folder is ONLY for using pre-trained models

---

## ✅ Summary

**Minimum setup:**
- 5 model files
- 1 inference script
- PyTorch + NumPy

**What you need to provide:**
- Trained .pth model file
- Pose sequences in [2, 12, 18] format

**What you get:**
- Anomaly scores for each sequence
- Simple, clean API
- No training code complexity!
