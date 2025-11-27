"""
MINIMAL INFERENCE SCRIPT - STG-NF Anomaly Detection

This folder contains ONLY the files needed to use a trained model.
No training code, no dataset code - just inference!

Usage:
    python inference.py --checkpoint path/to/model.pth --input path/to/pose_data.npy
"""

import torch
import numpy as np
from models.STG_NF.model_pose import STG_NF


class AnomalyDetector:
    """
    Simple wrapper for STG-NF model inference
    """
    
    def __init__(self, checkpoint_path, threshold=0.0, device=None):
        """
        Initialize the anomaly detector
        
        Args:
            checkpoint_path: Path to .pth model file
            threshold: Score threshold for anomaly detection
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.threshold = threshold
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Model configuration - MUST match your training settings
        # If you get errors, check your training logs for these values
        model_config = {
            'pose_shape': (2, 12, 18),      # (channels, time_frames, keypoints)
            'hidden_channels': 512,          # Hidden layer size
            'K': 8,                          # Flow steps per level
            'L': 3,                          # Number of levels
            'actnorm_scale': 1.0,
            'flow_permutation': 'invconv',
            'flow_coupling': 'affine',
            'LU_decomposed': True,
            'learn_top': True,
            'R': 0,
            'edge_importance': True,
            'temporal_kernel_size': None,
            'strategy': 'uniform',
            'max_hops': 8,
            'device': self.device
        }
        
        # Create model
        self.model = STG_NF(**model_config)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    
    def predict(self, poses, return_details=False):
        """
        Predict anomaly scores for pose sequences
        
        Args:
            poses: numpy array or torch tensor
                   Shape: [2, 12, 18] for single sequence
                          [N, 2, 12, 18] for batch
                   
                   Format:
                   - 2: [x_coords, y_coords]
                   - 12: time frames (12 consecutive frames)
                   - 18: body keypoints (COCO format)
            
            return_details: If True, return detailed results
        
        Returns:
            If single input:
                score (float): Anomaly score (higher = more normal)
                OR
                dict with score, classification, confidence
            
            If batch input:
                numpy array of scores
                OR
                list of dicts
        """
        # Convert to tensor if needed
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).float()
        
        # Handle single vs batch
        single_input = (poses.dim() == 3)
        if single_input:
            poses = poses.unsqueeze(0)
        
        poses = poses.to(self.device)
        
        # Validate input shape
        if poses.shape[1] != 2 or poses.shape[2] != 12 or poses.shape[3] != 18:
            raise ValueError(
                f"Expected input shape [N, 2, 12, 18], got {list(poses.shape)}"
            )
        
        # Run inference
        with torch.no_grad():
            z, nll = self.model(
                poses,
                label=torch.ones(poses.shape[0]).to(self.device),
                score=torch.ones(poses.shape[0]).to(self.device)
            )
            scores = -nll.cpu().numpy()
        
        # Return results
        if not return_details:
            return scores[0] if single_input else scores
        
        # Detailed results
        results = []
        for score in scores:
            is_abnormal = score < self.threshold
            distance = abs(score - self.threshold)
            
            if distance > 2.0:
                confidence = "High"
            elif distance > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            results.append({
                'score': float(score),
                'is_abnormal': bool(is_abnormal),
                'classification': 'Abnormal' if is_abnormal else 'Normal',
                'confidence': confidence,
                'distance_from_threshold': float(distance)
            })
        
        return results[0] if single_input else results


def example_1_basic_usage():
    """
    Example 1: Basic usage with dummy data
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # 1. Initialize detector
    detector = AnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0
    )
    
    # 2. Create dummy pose data
    # In practice, this comes from a pose estimation system (OpenPose, AlphaPose, etc.)
    pose_sequence = np.random.randn(2, 12, 18)
    
    # 3. Get prediction
    score = detector.predict(pose_sequence)
    
    print(f"\nInput shape: {pose_sequence.shape}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Classification: {'⚠️  ABNORMAL' if score < 0 else '✓ NORMAL'}")


def example_2_detailed_results():
    """
    Example 2: Get detailed results with confidence
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Detailed Results")
    print("="*70)
    
    detector = AnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0
    )
    
    pose_sequence = np.random.randn(2, 12, 18) * 0.1 + 0.5
    
    # Get detailed results
    result = detector.predict(pose_sequence, return_details=True)
    
    print("\nDetailed Results:")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Classification: {result['classification']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Distance from threshold: {result['distance_from_threshold']:.4f}")


def example_3_batch_processing():
    """
    Example 3: Process multiple sequences at once
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing")
    print("="*70)
    
    detector = AnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0
    )
    
    # Create batch of 5 sequences
    batch_poses = np.random.randn(5, 2, 12, 18) * 0.1 + 0.5
    
    # Process all at once (faster than one-by-one)
    scores = detector.predict(batch_poses)
    
    print(f"\nProcessed {len(scores)} sequences:")
    for i, score in enumerate(scores):
        status = "ABNORMAL" if score < 0 else "NORMAL"
        print(f"  Sequence {i+1}: {status:8s} (score={score:.4f})")
    
    # Or get detailed results for batch
    print("\nDetailed batch results:")
    results = detector.predict(batch_poses, return_details=True)
    for i, r in enumerate(results):
        print(f"  Sequence {i+1}: {r['classification']:8s} "
              f"(score={r['score']:.3f}, conf={r['confidence']})")


def example_4_load_from_file():
    """
    Example 4: Load pose data from file
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Load from File")
    print("="*70)
    
    detector = AnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0
    )
    
    # Example: Load from numpy file
    # pose_data = np.load("my_pose_sequence.npy")
    
    # For demo, create and save
    demo_data = np.random.randn(2, 12, 18)
    np.save("demo_pose.npy", demo_data)
    
    # Load and predict
    loaded_data = np.load("demo_pose.npy")
    score = detector.predict(loaded_data)
    
    print(f"\nLoaded data shape: {loaded_data.shape}")
    print(f"Score: {score:.4f}")
    
    # Cleanup
    import os
    os.remove("demo_pose.npy")


def example_5_video_stream():
    """
    Example 5: Simulated video stream processing
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Video Stream Processing")
    print("="*70)
    
    detector = AnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0
    )
    
    # Simulate a sliding window over video frames
    print("\nSimulating video stream (sliding window):")
    
    pose_buffer = []
    
    for frame_idx in range(20):  # Simulate 20 frames
        # Get pose for current frame (from pose estimation system)
        current_pose = np.random.randn(2, 18)  # [x,y coords, 18 keypoints]
        
        pose_buffer.append(current_pose)
        
        # Once we have 12 frames, analyze
        if len(pose_buffer) == 12:
            # Stack into [2, 12, 18]
            pose_sequence = np.stack(pose_buffer, axis=1)
            
            # Predict
            score = detector.predict(pose_sequence)
            
            if score < 0:
                print(f"  Frame {frame_idx:3d}: ⚠️  ANOMALY DETECTED (score={score:.3f})")
            else:
                print(f"  Frame {frame_idx:3d}: ✓ Normal (score={score:.3f})")
            
            # Slide window: remove oldest frame
            pose_buffer.pop(0)


if __name__ == '__main__':
    print("="*70)
    print("STG-NF ANOMALY DETECTION - INFERENCE EXAMPLES")
    print("="*70)
    
    # Run all examples
    try:
        example_1_basic_usage()
        example_2_detailed_results()
        example_3_batch_processing()
        example_4_load_from_file()
        example_5_video_stream()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find checkpoint file")
        print(f"   {e}")
        print("\n   Please update the checkpoint_path in the examples to point to your .pth file")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
