import sys
from pathlib import Path
import torch
import numpy as np

# Setup path to import model
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

print(f"Added {current_dir} to sys.path")

try:
    from model.msg3d import MSG3D, count_parameters
    print("Successfully imported MSG3D")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_model():
    print("=" * 60)
    print("MSG3D Model Verification")
    print("=" * 60)
    
    # Initialize model
    try:
        model = MSG3D(num_class=300, num_point=75, num_person=1, in_channels=3)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        sys.exit(1)
        
    # Check parameters
    params = count_parameters(model)
    print(f"Total parameters: {params:,}")
    
    # Create dummy input
    # (N, C, T, V, M)
    N, C, T, V, M = 2, 3, 64, 75, 1
    x = torch.randn(N, C, T, V, M)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
    except RuntimeError as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Check output shape
    expected_shape = (N, 300)
    if y.shape == expected_shape:
        print("✓ Output shape correct")
    else:
        print(f"✗ Output shape mismatch: expected {expected_shape}, got {y.shape}")
        sys.exit(1)
        
    print("\nVERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    test_model()
