"""
Block 4: ST-GCN Model Inference for Normality Scoring
Load stg_nf_latest.pth and perform inference on pose sequences
"""

import torch
import numpy as np


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

            # Check if it's a state_dict or full model
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # If model class is needed, import it here
                # from stgcn_model import STGCN
                # self.model = STGCN(...)
                # self.model.load_state_dict(checkpoint['model_state_dict'])
                raise NotImplementedError(
                    "Model class definition needed. Please provide STGCN architecture."
                )
            else:
                # Assume full model is saved
                self.model = checkpoint

            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"ST-GCN model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, input_data):
        """
        Perform inference and return normality score.

        Args:
            input_data: Preprocessed input data (numpy array or tensor)
                       Expected shape: (seq_len, num_joints, 3) for single sample
                                   or (batch, seq_len, num_joints, 3) for batch

        Returns:
            normality_score: Float or array of floats representing normality score(s)
                           Higher score = more normal behavior
        """
        with torch.no_grad():
            # Convert to tensor if numpy array
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = input_data.float()

            # Add batch dimension if needed
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)  # (1, seq_len, num_joints, 3)

            # Move to device
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            output = self.model(input_tensor)

            # Convert to normality score
            normality_score = output.detach().cpu().numpy()

            # If single sample, return scalar
            if normality_score.shape[0] == 1:
                normality_score = float(normality_score.squeeze())

            return normality_score


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
