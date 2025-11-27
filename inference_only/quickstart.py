"""
QUICK START - Copy and paste this code!
"""

from inference import AnomalyDetector
import numpy as np

# ============================================================================
# METHOD 1: Simplest usage
# ============================================================================

# Load model
detector = AnomalyDetector(
    checkpoint_path="path/to/your/model.pth",  # UPDATE THIS!
    threshold=0.0
)

# Prepare input [2, 12, 18]
poses = np.random.randn(2, 12, 18)

# Predict
score = detector.predict(poses)
print(f"Score: {score:.2f} - {'ABNORMAL' if score < 0 else 'NORMAL'}")


# ============================================================================
# METHOD 2: With detailed results
# ============================================================================

result = detector.predict(poses, return_details=True)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")


# ============================================================================
# METHOD 3: Batch processing
# ============================================================================

batch = np.random.randn(10, 2, 12, 18)  # 10 sequences
scores = detector.predict(batch)
print(f"Processed {len(scores)} sequences")


# ============================================================================
# METHOD 4: Real video processing
# ============================================================================

def process_video(video_path, detector):
    """
    Process a video file using pose estimation + anomaly detection
    """
    # You need to implement pose estimation first!
    # Use OpenPose, AlphaPose, MediaPipe, etc.
    
    # Pseudo-code:
    # for frame in video:
    #     pose = extract_pose(frame)  # Get [2, 18] from pose estimator
    #     buffer.append(pose)
    #     
    #     if len(buffer) == 12:
    #         sequence = np.stack(buffer, axis=1)  # [2, 12, 18]
    #         score = detector.predict(sequence)
    #         
    #         if score < 0:
    #             alert_anomaly(frame)
    #         
    #         buffer.pop(0)  # Slide window
    pass
