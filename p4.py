"""
Block 4: ST-GCN Model Inference for Normality Scoring
Load stg_nf_latest.pth and perform inference on pose sequences
"""

import torch
import numpy as np
from models.STG_NF.model_pose import STG_NF


model_Path = "models/stg_nf_latest.pth"


class STGCNInference:
    def __init__(self, model_path=model_Path, device=None):
        """
        Initialize the ST-GCN inference handler for normality scoring.

        Args:
            model_path: Path to the .pth model file
            device: torch device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """
        Load the ST-GCN model from .pth file.
        """
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # print(f"🚀 Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                pass
                # print(f"🚀 Checkpoint keys: {checkpoint.keys()}")

            # Model configuration - inferred from checkpoint structure
            # Checkpoint analysis:
            # - Has actnorm (not invconv)
            # - Layer 0 has no residual, layers 1-7 have residual in block.0
            # - Only block.0 exists (single block), but expects block.1 -> R=2
            # - Channels are 2 (from weight shapes)
            model_config = {
                "pose_shape": (2, 24, 18),  # (2 for x/y, 24 frames, 18 keypoints)
                "hidden_channels": 2,  # Hidden channels (from weight: torch.Size([2, 2, 13, 1]))
                "K": 8,  # Flow steps (8 layers: 0-7)
                "L": 1,  # Number of levels
                "actnorm_scale": 1.0,
                "flow_permutation": "shuffle",  # Uses actnorm, not invconv
                "flow_coupling": "affine",  # Affine keeps channel size
                "LU_decomposed": False,
                "learn_top": False,  # No learn_top_fn in checkpoint
                "R": 2,  # 2 blocks per layer (need block.0 and block.1)
                "edge_importance": False,
                "temporal_kernel_size": 13,  # From tcn layer kernel size
                "strategy": "uniform",
                "max_hops": 1,
                "device": self.device,
            }

            # print(
            #     f"🚀 Model config: pose_shape={model_config['pose_shape']}, "
            #     f"hidden_channels={model_config['hidden_channels']}, K={model_config['K']}, "
            #     f"coupling={model_config['flow_coupling']}"
            # )

            # Create model
            self.model = STG_NF(**model_config)

            # Load trained weights from checkpoint
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Use strict=False to allow partial loading (some keys may not match)
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
                if missing_keys:
                    print(
                        f"⚠️  Missing keys (will be randomly initialized): {len(missing_keys)}"
                    )
                if unexpected_keys:
                    print(
                        f"⚠️  Unexpected keys (will be ignored): {len(unexpected_keys)}"
                    )
                print(
                    f"✓ Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)"
                )
            else:
                raise ValueError("Checkpoint format not recognized")

            # Move model to device FIRST
            self.model.to(self.device)

            # Initialize ActNorm layers before setting to eval mode
            # Create dummy input to initialize
            self.model.train()  # Temporarily set to train mode for initialization
            dummy_input = torch.randn(1, 2, 24, 18).to(self.device)
            dummy_label = torch.ones(1).to(self.device)
            dummy_score = torch.ones(1).to(self.device)
            try:
                with torch.no_grad():
                    _ = self.model(dummy_input, label=dummy_label, score=dummy_score)
                print("✓ ActNorm layers initialized")
            except Exception as e:
                print(f"⚠️  Warning during ActNorm initialization: {e}")

            self.model.eval()  # Now set to evaluation mode

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, input_data):
        """
        Perform inference and return normality score.

        Args:
            input_data: Preprocessed input data
                       Can be:
                       - numpy array: (seq_len, num_joints, 3) for single sample
                       - torch tensor: (seq_len, num_joints, 3) for single sample
                       - dict: {track_id: np.array(seq_len, num_joints, 3)}
                       Format: (frames, keypoints, [x, y, confidence])

        Returns:
            normality_score: Float or dict of floats representing normality score(s)
                           Higher score = more normal behavior
        """
        with torch.no_grad():
            # Handle different input types
            if isinstance(input_data, dict):
                # Input is {track_id: sequence}
                results = {}
                for track_id, sequence in input_data.items():
                    score = self._predict_single(sequence)
                    results[track_id] = score
                return results
            else:
                # Input is single sequence (numpy array or tensor)
                return self._predict_single(input_data)

    def _predict_single(self, input_data):
        """
        Predict for a single sequence.

        Args:
            input_data: numpy array or tensor (seq_len, num_joints, 3)

        Returns:
            normality_score: Float
        """
        # Convert to tensor if numpy array
        # print("🚀 input_data : ", input_data)
        print("🚀 model input_data : ", input_data.shape)
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()

        # Add batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # (1, seq_len, num_joints, 3)

        # print(f"🚀 Input tensor shape: {input_tensor.shape}")

        # Transform input from (batch, seq_len, num_joints, 3) to (batch, 2, 24, 18)
        # The model expects: [batch, 2 (x/y), 24 frames, 18 keypoints]
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]

        # Select last 24 frames and first 18 keypoints (COCO format)
        if seq_len < 24:
            print(
                f"⚠️  Warning: Only {seq_len} frames available, need 24. Padding with zeros."
            )
            # Pad with zeros to reach 24 frames
            padding = torch.zeros(batch_size, 24 - seq_len, input_tensor.shape[2], 3)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        else:
            # Take last 24 frames for temporal continuity
            input_tensor = input_tensor[:, -24:, :, :]

        # Take first 18 keypoints (or pad if less)
        if input_tensor.shape[2] < 18:
            print(
                f"⚠️  Warning: Only {input_tensor.shape[2]} keypoints, need 18. Padding."
            )
            padding = torch.zeros(batch_size, 24, 18 - input_tensor.shape[2], 3)
            input_tensor = torch.cat([input_tensor, padding], dim=2)
        else:
            input_tensor = input_tensor[:, :, :18, :]

        # Reshape to (batch, 2, 24, 18) - separate x and y coordinates
        # input_tensor is now (batch, 24, 18, 3) where 3 is [x, y, confidence]
        x_coords = input_tensor[:, :, :, 0]  # (batch, 24, 18)
        y_coords = input_tensor[:, :, :, 1]  # (batch, 24, 18)

        # Stack to (batch, 2, 24, 18)
        model_input = torch.stack([x_coords, y_coords], dim=1)

        # print(f"🚀 Model input shape: {model_input.shape}")

        # Move to device
        model_input = model_input.to(self.device)

        # Forward pass
        z, nll = self.model(
            model_input,
            label=torch.ones(batch_size).to(self.device),
            score=torch.ones(batch_size).to(self.device),
        )

        # Convert to normality score (negative log-likelihood, higher = more normal)
        normality_score = -nll.detach().cpu().numpy()

        # Return scalar for single sample
        return float(normality_score.squeeze())


# Global model instance (initialized once)
_model_instance = None


def get_model_instance(model_path=model_Path, device=None):
    """
    Get or create the global model instance (singleton pattern).

    Args:
        model_path: Path to model file
        device: torch device

    Returns:
        STGCNInference instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = STGCNInference(model_path, device)
    return _model_instance


def predict_normality(input_data, model_path=model_Path):
    """
    Convenience function for normality prediction.
    Uses singleton model instance for efficiency.

    Args:
        input_data: Preprocessed pose sequence (seq_len, num_joints, 3)
        model_path: Path to model file

    Returns:
        normality_score: Float representing normality score
    """
    model = get_model_instance(model_path)
    return model.predict(input_data)


# Example usage
if __name__ == "__main__":
    # Initialize model once
    model_handler = STGCNInference(model_Path)

    # Example: Create dummy input (replace with actual data from p3.py)
    seq_len, num_joints = 30, 33
    dummy_input = np.random.randn(seq_len, num_joints, 3).astype(np.float32)

    # Get normality score
    score = model_handler.predict(dummy_input)
    print(f"Normality score: {score}")

    # Or use convenience function
    score2 = predict_normality(dummy_input)
    print(f"Normality score (convenience): {score2}")
