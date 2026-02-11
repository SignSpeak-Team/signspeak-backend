"""
MSG3D Predictor Wrapper.
Handles model loading and inference for the MSG3D LSE model.
"""

import pickle
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from .msg3d_model import MSG3D
from config import (
    MSG3D_MODEL_PATH,
    MSG3D_LABELS_PATH,
    MSG3D_NUM_FEATURES,
    MSG3D_CHANNELS
)


class MSG3DPredictor:
    """Predictor class for MSG3D LSE model."""
    
    def __init__(self):
        print("[MSG3D] Initializing predictor...")
        self.device = torch.device("cpu")  # Use CPU for inference in production
        
        # Initialize model architecture
        self.model = MSG3D(
            num_class=300,
            num_point=MSG3D_NUM_FEATURES,
            num_person=1,
            in_channels=MSG3D_CHANNELS,
            dropout=0.0  # No dropout for inference
        )
        
        # Load weights
        if MSG3D_MODEL_PATH.exists():
            try:
                state_dict = torch.load(MSG3D_MODEL_PATH, map_location=self.device)
                
                # Handle 'model_state_dict' key if saved from train.py checkpoint format
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                    
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("[MSG3D] Model weights loaded successfully.")
            except Exception as e:
                print(f"[MSG3D] Error loading weights: {e}")
        else:
            print(f"[MSG3D] Warning: Model file not found at {MSG3D_MODEL_PATH}")

        # Load labels
        self.labels = {}
        if MSG3D_LABELS_PATH.exists():
            try:
                with open(MSG3D_LABELS_PATH, "rb") as f:
                    self.labels = pickle.load(f)
                
                # Check format and invert if necessary {label: id} -> {id: label}
                if self.labels and isinstance(list(self.labels.keys())[0], str):
                     self.labels = {v: k for k, v in self.labels.items()}
                     
                print(f"[MSG3D] Loaded {len(self.labels)} class labels.")
            except Exception as e:
                print(f"[MSG3D] Error loading labels: {e}")
        else:
            print(f"[MSG3D] Warning: Label file not found at {MSG3D_LABELS_PATH}")

    def predict(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a sequence of frames.
        
        Args:
            sequence: Numpy array of shape (T, 75, 3)
            
        Returns:
            Dictionary with prediction result.
        """
        if not MSG3D_MODEL_PATH.exists():
             return {
                "word": "MODEL_NOT_FOUND",
                "confidence": 0.0,
                "processing_time_ms": 0
            }
            
        with torch.no_grad():
            start_time = time.time()
            
            # Prepare input tensor
            # Input: (T, V, C) -> (75 keypoints * 3 coords)
            # Received shape: (T, 75, 3)
            
            # Convert to tensor
            data = torch.from_numpy(sequence).float()
            
            # Reshape to Model Input: (N, C, T, V, M)
            # data: (T, V, C) -> permute to (C, T, V)
            data = data.permute(2, 0, 1) 
            
            # Add Batch (N=1) and Person (M=1) dimensions
            # (C, T, V) -> (1, C, T, V, 1)
            data = data.unsqueeze(0).unsqueeze(-1)
            
            data = data.to(self.device)
            
            # Inference
            logits = self.model(data)
            probs = torch.softmax(logits, dim=1)
            
            # Get top prediction
            conf, idx = torch.max(probs, dim=1)
            idx = idx.item()
            conf = conf.item() * 100
            
            label = self.labels.get(idx, f"Unknown_{idx}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "word": label,
                "confidence": round(conf, 2),
                "processing_time_ms": round(processing_time, 2)
            }
