"""
CLEAN INFERENCE - Works directly with AlphaPose JSON files

This script takes the JSON files from AlphaPose tracking and runs anomaly detection.
No need to convert data formats - uses existing preprocessing functions!
"""

import json
import os
import numpy as np
import torch
from models.STG_NF.model_pose import STG_NF
from .utils import gen_clip_seg_data_np, normalize_pose


class JSONAnomalyDetector:
    def __init__(self, checkpoint_path, threshold=0.0, device=None):
        """
        Args:
            checkpoint_path: Path to trained .pth model
            threshold: Anomaly threshold (from training EER)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.threshold = threshold

        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Model configuration (must match training args.json)
        model_config = {
            "pose_shape": (2, 24, 18),  # seg_len=24
            "hidden_channels": 0,  # model_hidden_dim=0 → single block architecture
            "K": 8,
            "L": 1,
            "actnorm_scale": 1.0,
            "flow_permutation": "permute",
            "flow_coupling": "affine",
            "LU_decomposed": True,
            "learn_top": False,
            "R": 3.0,
            "edge_importance": False,
            "temporal_kernel_size": None,  # Auto: T//2+1 = 13
            "strategy": "uniform",
            "max_hops": 8,
            "device": self.device,
        }

        # Create and load model
        self.model = STG_NF(**model_config)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        # Mark all ActNorm layers as initialized (required after loading checkpoint)
        for module in self.model.modules():
            if hasattr(module, 'inited'):
                module.inited = True
        
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
        print(f"✓ Using device: {self.device}")

    def predict_from_dict(
        self, clip_dict, scene_id="00", clip_id="0000", return_details=False
    ):
        """
        Run anomaly detection on a Python dict with AlphaPose structure

        Args:
            clip_dict: Python dict with AlphaPose tracking data
            scene_id: Scene identifier (default: '00')
            clip_id: Clip identifier (default: '0000')
            return_details: If True, return detailed results for each sequence

        Returns:
            results: List of detection results for each person/sequence

        Dict Format Expected:
        {
          "person_id": {
            "frame_num": {
              "keypoints": [x1, y1, conf1, x2, y2, conf2, ...],  # 51 values (17 keypoints × 3)
              "scores": [...]
            },
            ...
          },
          ...
        }
        """
        print(f"\n" + "=" * 70)
        print(f"Processing pose data")
        print(f"Scene: {scene_id}, Clip: {clip_id}")
        print(f"Persons detected: {len(clip_dict)}")
        print("=" * 70)

        # Process JSON to get pose segments
        segs_data_np, segs_meta, person_keys, _, _, segs_score_np = (
            gen_clip_seg_data_np(
                clip_dict,
                start_ofst=0,
                seg_stride=6,  # Stride=6 to match training
                seg_len=24,  # Model trained with 24 frames
                scene_id=scene_id,
                clip_id=clip_id,
                ret_keys=True,
                dataset="PoseLift",
            )
        )

        if segs_data_np.shape[0] == 0:
            print("⚠️  No valid sequences found (need at least 24 consecutive frames)")
            return []

        print(f"\nExtracted {segs_data_np.shape[0]} sequences of 24 frames")

        # Normalize poses (input: [N, T, V, F], output: same shape)
        # segs_data_np is [N, 24, 17, 3] from gen_clip_seg_data_np
        segs_data_np = normalize_pose(
            segs_data_np, scale=True, scale_proportional=True
        )

        # Convert to model input format [N, 2, 24, 18]
        # Currently: [N, 24, 17, 3] (batch, frames, keypoints, x/y/conf)
        # Need: [N, 2, 24, 18] (batch, x/y, frames, keypoints+1)

        poses = segs_data_np[:, :, :, :2]  # Remove confidence, keep only x,y → [N, 24, 17, 2]
        poses = np.transpose(poses, (0, 3, 1, 2))  # [N, 2, 24, 17]
        
        # Pad from 17 to 18 keypoints (COCO17 → COCO18)
        poses = np.pad(poses, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant')  # [N, 2, 24, 18]

        # Run inference
        poses_tensor = torch.from_numpy(poses).float().to(self.device)

        with torch.no_grad():
            _, nll = self.model(
                poses_tensor,
                label=torch.ones(poses_tensor.shape[0]).to(self.device),
                score=torch.ones(poses_tensor.shape[0]).to(self.device),
            )
            scores = nll.cpu().numpy()

        # Compile results
        results = []
        for i, (score, meta) in enumerate(zip(scores, segs_meta)):
            scene, clip, person_id, start_frame = meta
            is_abnormal = score < self.threshold

            # # Calculate confidence
            distance = abs(score - self.threshold)
            if distance < -3.0:
                confidence = "High"
            elif distance < - 2.0 and distance >-2.9:
                confidence = "Medium"
            else:
                confidence = "Low"

            result = {
                "sequence_id": i,
                "person_id": int(person_id),
                "start_frame": int(start_frame),
                "end_frame": int(start_frame) + 11,
                "score": float(score),
                "is_abnormal": bool(is_abnormal),
                "classification": "Abnormal" if is_abnormal else "Normal",
                "confidence": confidence,
                "scene_id": scene,
                "clip_id": clip,
            }

            results.append(result)

        return results

    def print_results(self, results):
        """Pretty print detection results"""
        if not results:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("DETECTION RESULTS")
        print("=" * 70)

        # Group by person
        persons = {}
        for r in results:
            pid = r["person_id"]
            if pid not in persons:
                persons[pid] = []
            persons[pid].append(r)

        for person_id, person_results in persons.items():
            print(f"\n👤 Person {person_id}: {len(person_results)} sequences analyzed")
            print("-" * 70)

            abnormal_count = sum(1 for r in person_results if r["is_abnormal"])
            normal_count = len(person_results) - abnormal_count

            print(f"   Normal: {normal_count} sequences")
            print(f"   Abnormal: {abnormal_count} sequences")

            # Show abnormal sequences
            if abnormal_count > 0:
                print(f"\n   ⚠️  Abnormal sequences detected:")
                for r in person_results:
                    if r["is_abnormal"]:
                        print(
                            f"      Frames {r['start_frame']}-{r['end_frame']}: "
                            f"score={r['score']:.3f}, conf={r['confidence']}"
                        )

        print("=" * 70)
        print(
            f"SUMMARY: {len(results)} total sequences, "
            f"{sum(1 for r in results if r['is_abnormal'])} abnormal"
        )
        print("=" * 70)

    def predict_from_json(self, json_path, return_details=False):
        """
        Run anomaly detection on a JSON file (convenience wrapper)

        Args:
            json_path: Path to AlphaPose tracked_person.json file
            return_details: If True, return detailed results for each sequence

        Returns:
            results: List of detection results for each person/sequence
        """
        # Load JSON
        with open(json_path, "r") as f:
            clip_dict = json.load(f)

        # Extract scene and clip ID from filename
        filename = os.path.basename(json_path)
        parts = filename.replace("_alphapose_tracked_person.json", "").split("_")
        scene_id = parts[0]
        clip_id = parts[1] if len(parts) > 1 else "0000"

        # Call predict_from_dict
        return self.predict_from_dict(clip_dict, scene_id, clip_id, return_details)


def example_with_dict():
    """
    Example: Process a Python dict directly
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Direct Dict Detection")
    print("=" * 70)

    # Initialize detector
    detector = JSONAnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0,  # Adjust based on your EER threshold from training
    )

    # Your pose dict (same structure as JSON)
    pose_dict = {
        "3": {
            "7": {"keypoints": [517.5, 1282.0, 0.0047607, ...], "scores": None},
            "8": {"keypoints": [548.0, 1279.0, 0.00244331, ...], "scores": None},
            # ... more frames
        },
        # ... more persons
    }

    # Process dict directly
    results = detector.predict_from_dict(pose_dict, scene_id="01", clip_id="0222")

    # Print results
    detector.print_results(results)

    return results


def example_single_file():
    """
    Example: Process a single JSON file
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Single File Detection")
    print("=" * 70)

    # Initialize detector
    detector = JSONAnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0,  # Adjust based on your EER threshold from training
    )

    # Process JSON file
    json_path = "/home/ezio/Documents/work/cuda-test/VisionGuard/Datasets/PoseLift/Json_files-20251124T091336Z-1-001/Json_files/data/PoseLift/pose/test/01_0222_alphapose_tracked_person.json"

    results = detector.predict_from_json(json_path)

    # Print results
    detector.print_results(results)

    return results


def example_single_file():
    """
    Example: Process a single JSON file
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Single File Detection")
    print("=" * 70)

    # Initialize detector
    detector = JSONAnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0,  # Adjust based on your EER threshold from training
    )

    # Process JSON file
    json_path = "/home/ezio/Documents/work/cuda-test/VisionGuard/Datasets/PoseLift/Json_files-20251124T091336Z-1-001/Json_files/data/PoseLift/pose/test/01_0222_alphapose_tracked_person.json"

    results = detector.predict_from_json(json_path)

    # Print results
    detector.print_results(results)

    return results


def example_batch_files():
    """
    Example: Process multiple JSON files
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Batch Processing")
    print("=" * 70)

    detector = JSONAnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0,
    )

    # Directory with JSON files
    json_dir = "/home/ezio/Documents/work/cuda-test/VisionGuard/Datasets/PoseLift/Json_files-20251124T091336Z-1-001/Json_files/data/PoseLift/pose/test/"

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")][
        :3
    ]  # Process first 3 files

    all_results = {}
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        results = detector.predict_from_json(json_path)
        all_results[json_file] = results

    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for filename, results in all_results.items():
        abnormal = sum(1 for r in results if r["is_abnormal"])
        total = len(results)
        status = "⚠️" if abnormal > 0 else "✓"
        print(f"{status} {filename}: {abnormal}/{total} abnormal sequences")


def example_save_results():
    """
    Example: Save results to file
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Save Results")
    print("=" * 70)

    detector = JSONAnomalyDetector(
        checkpoint_path="../STG-NF/data/exp_dir/PoseLift/Nov25_0212/stg_nf_best.pth",
        threshold=0.0,
    )

    json_path = "/home/ezio/Documents/work/cuda-test/VisionGuard/Datasets/PoseLift/Json_files-20251124T091336Z-1-001/Json_files/data/PoseLift/pose/test/01_0222_alphapose_tracked_person.json"

    results = detector.predict_from_json(json_path)

    # Save to JSON
    output_file = "detection_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Save summary to text
    summary_file = "detection_summary.txt"
    with open(summary_file, "w") as f:
        f.write("ANOMALY DETECTION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            status = "ABNORMAL" if r["is_abnormal"] else "NORMAL"
            f.write(
                f"Person {r['person_id']}, Frames {r['start_frame']}-{r['end_frame']}: "
                f"{status} (score={r['score']:.3f}, conf={r['confidence']})\n"
            )

    print(f"✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    print("=" * 70)
    print("STG-NF ANOMALY DETECTION - JSON INFERENCE")
    print("=" * 70)

    try:
        # Run examples

        # Example 1: Use Python dict directly (RECOMMENDED)
        # example_with_dict()

        # Example 2: Load from JSON file
        example_single_file()

        # Uncomment to run other examples:
        # example_batch_files()
        # example_save_results()

    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found")
        print(f"   {e}")
        print("\nPlease update the paths in the script:")
        print("  - checkpoint_path: Path to your .pth model file")
        print("  - json_path: Path to your JSON file")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
