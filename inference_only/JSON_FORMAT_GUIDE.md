# JSON Input Format Guide

## 📥 AlphaPose JSON Format

The model works directly with **AlphaPose tracked person JSON files**.

### File Structure

```json
{
  "person_id": {
    "frame_number": {
      "keypoints": [x1, y1, conf1, x2, y2, conf2, ..., x17, y17, conf17],
      "scores": [...]
    },
    "frame_number": { ... },
    ...
  },
  "person_id": { ... }
}
```

**Example:** `01_0222_alphapose_tracked_person.json`

```json
{
  "3": {
    "7": {
      "keypoints": [517.5, 1282.0, 0.0047607, 455.0, 1282.0, 0.0098495, ...],
      "scores": null
    },
    "8": {
      "keypoints": [548.0, 1279.0, 0.00244331, 548.0, 1279.0, 0.00544357, ...],
      "scores": null
    },
    ...
  }
}
```

### Format Details

- **Person ID** (string): Unique ID for each tracked person (e.g., "3", "4", "5")
- **Frame Number** (string): Video frame number (e.g., "7", "8", "9")
- **Keypoints** (array): 51 numbers = 17 keypoints × 3 values (x, y, confidence)
  - **17 keypoints** in COCO format (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
  - **3 values per keypoint**: X coordinate, Y coordinate, confidence score
- **Scores** (optional): Additional scores from pose estimator

### Keypoint Order (17 keypoints - COCO format)

```
0: Nose       5: Left Shoulder    10: Right Knee
1: Left Eye   6: Right Elbow      11: Left Knee
2: Right Eye  7: Left Elbow       12: Right Ankle
3: Left Ear   8: Right Wrist      13: Left Ankle
4: Right Ear  9: Left Wrist       14: Right Hip
                                  15: Left Hip
                                  16: Neck
```

---

## 🚀 Usage

### Basic Usage

```python
from json_inference import JSONAnomalyDetector

# Load model
detector = JSONAnomalyDetector(
    checkpoint_path="path/to/model.pth",
    threshold=0.0  # Adjust based on EER from training
)

# Process JSON file
results = detector.predict_from_json("video_poses.json")

# Print results
detector.print_results(results)
```

### Output Format

```python
[
  {
    'sequence_id': 0,
    'person_id': 3,
    'start_frame': 7,
    'end_frame': 18,
    'score': 2.45,
    'is_abnormal': False,
    'classification': 'Normal',
    'confidence': 'High',
    'scene_id': '01',
    'clip_id': '0222'
  },
  ...
]
```

---

## 📊 Complete Example

```python
from json_inference import JSONAnomalyDetector

# Initialize detector
detector = JSONAnomalyDetector(
    checkpoint_path="models/stg_nf_best.pth",
    threshold=0.0
)

# Process file
json_file = "01_0222_alphapose_tracked_person.json"
results = detector.predict_from_json(json_file)

# Print results
detector.print_results(results)

# Output:
# ======================================================================
# DETECTION RESULTS
# ======================================================================
#
# 👤 Person 3: 85 sequences analyzed
# ----------------------------------------------------------------------
#    Normal: 80 sequences
#    Abnormal: 5 sequences
#
#    ⚠️  Abnormal sequences detected:
#       Frames 45-56: score=-1.234, conf=High
#       Frames 78-89: score=-0.567, conf=Medium
# ...
```

---

## 🔧 What Happens Internally

```
1. Load JSON file
   ├─ Parse person IDs
   └─ Parse frame data

2. Extract pose sequences
   ├─ Group frames by person
   ├─ Create 12-frame sliding windows
   └─ Extract x,y coordinates from keypoints

3. Preprocessing
   ├─ Normalize coordinates
   ├─ Convert 17 → 18 keypoints (add neck)
   └─ Format: [N, 2, 12, 18]

4. Model inference
   ├─ Pass through STG-NF
   └─ Get anomaly scores

5. Results
   ├─ Score > threshold → Normal
   └─ Score < threshold → Abnormal
```

---

## 📝 Requirements

### Minimum Data Requirements

For the model to work, each person needs:
- ✅ **At least 12 consecutive frames** of pose data
- ✅ **17 keypoints** per frame (COCO format)
- ✅ **Valid x,y coordinates** for each keypoint

### File Naming Convention

Expected format: `{scene_id}_{clip_id}_alphapose_tracked_person.json`

Examples:
- `01_0222_alphapose_tracked_person.json`
- `05_1234_alphapose_tracked_person.json`

---

## 🎯 Batch Processing

```python
import os
from json_inference import JSONAnomalyDetector

detector = JSONAnomalyDetector("model.pth")

# Process directory
json_dir = "path/to/json/files/"
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

all_results = {}
for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)
    results = detector.predict_from_json(json_path)
    all_results[json_file] = results
    
    # Count abnormal sequences
    abnormal = sum(1 for r in results if r['is_abnormal'])
    print(f"{json_file}: {abnormal} abnormal sequences detected")
```

---

## 💾 Save Results

```python
import json

detector = JSONAnomalyDetector("model.pth")
results = detector.predict_from_json("video.json")

# Save to JSON
with open("results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save summary to text
with open("summary.txt", 'w') as f:
    for r in results:
        status = "ABNORMAL" if r['is_abnormal'] else "NORMAL"
        f.write(f"Person {r['person_id']}, "
                f"Frames {r['start_frame']}-{r['end_frame']}: {status}\n")
```

---

## ⚠️ Common Issues

### 1. "No valid sequences found"
**Cause:** Not enough consecutive frames (need ≥12)  
**Solution:** Ensure pose tracker captures at least 12 consecutive frames per person

### 2. "Keypoints length: 51"
**Cause:** This is correct! 17 keypoints × 3 values = 51  
**Solution:** No action needed, this is the expected format

### 3. Missing person IDs
**Cause:** AlphaPose didn't track anyone in those frames  
**Solution:** Check video quality, lighting, and occlusions

### 4. All sequences marked as abnormal
**Cause:** Threshold might be too high  
**Solution:** Adjust threshold based on EER from training

---

## 🔍 Understanding the Output

### Score Interpretation

| Score Range | Meaning | Example Behavior |
|-------------|---------|------------------|
| **> +2.0** | Very Normal | Walking, standing normally |
| **0 to +2.0** | Probably Normal | Normal with slight variations |
| **-0.5 to 0** | Borderline | Uncertain cases |
| **< -0.5** | Likely Abnormal | Unusual movements |
| **< -2.0** | Very Abnormal | Clear anomaly (fighting, falling) |

### Confidence Levels

- **High**: Score is >2.0 away from threshold (very confident)
- **Medium**: Score is 0.5-2.0 away from threshold (moderately confident)
- **Low**: Score is <0.5 away from threshold (borderline case)

---

## 📚 Files in This Directory

```
inference_only/
├── json_inference.py          # Main inference script (USE THIS!)
├── inference.py               # Alternative (manual data format)
├── JSON_FORMAT_GUIDE.md       # This file
├── INPUT_OUTPUT_EXPLAINED.md  # Detailed format specs
├── README.md                  # General documentation
└── models/                    # Model architecture files
    └── STG_NF/
```

---

## ✅ Quick Start Checklist

- [ ] Have trained `.pth` model file
- [ ] Have AlphaPose JSON files with format: `{scene}_{clip}_alphapose_tracked_person.json`
- [ ] Each JSON has person IDs with at least 12 consecutive frames
- [ ] Know your threshold value (from training EER)
- [ ] Run: `python json_inference.py`

---

## 🎬 Complete Pipeline

```
Your Video
    ↓
AlphaPose Tracking
    ↓
JSON file (person poses per frame)
    ↓
json_inference.py  ← YOU ARE HERE
    ↓
Anomaly Detection Results
```

**You need to provide:**
- ✅ AlphaPose JSON files (already have this format!)
- ✅ Trained model (.pth file)

**The script handles:**
- ✅ Loading JSON
- ✅ Extracting sequences
- ✅ Preprocessing
- ✅ Running inference
- ✅ Generating results

---

## 💡 Tips

1. **Use stride=1 for inference** (capture all frames, no skipping)
2. **Adjust threshold** based on validation performance
3. **Process videos in batches** for efficiency
4. **Check confidence scores** - low confidence may need manual review
5. **Save results** for later analysis or visualization

---

## 🆘 Help

If you encounter issues:
1. Check JSON format matches the expected structure
2. Verify model checkpoint path is correct
3. Ensure at least 12 consecutive frames per person
4. Check device (CPU vs GPU) availability
5. Review threshold setting

For detailed format specifications, see `INPUT_OUTPUT_EXPLAINED.md`
