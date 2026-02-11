"""
Export trained MSG3D model to production format (.pt).

Convierte el checkpoint de entrenamiento a un state_dict limpio
para usar en producción con el msg3d_predictor.

Uso:
    python export_model.py --checkpoint best.pth --output msg3d_lse.pt
"""

import argparse
from pathlib import Path
import torch
import pickle

from model.msg3d import MSG3D


def export_model(checkpoint_path, output_path, labels_path):
    """
    Exporta el modelo entrenado a formato de producción.
    
    Args:
        checkpoint_path: Path to training checkpoint (.pth)
        output_path: Path to save production model (.pt)
        labels_path: Path to label_names.pkl from dataset
    """
    print("=" * 60)
    print("MSG3D Model Export")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    print("Creating model...")
    model = MSG3D(
        num_class=300,
        num_point=75,
        num_person=1,
        in_channels=3
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 0):.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Save production model (state_dict only)
    print(f"\nSaving model to: {output_path}")
    torch.save(model.state_dict(), output_path)
    
    # Copy label names to output directory
    output_dir = output_path.parent
    labels_output = output_dir / "msg3d_labels.pkl"
    
    if labels_path.exists():
        import shutil
        shutil.copy(labels_path, labels_output)
        print(f"Copied labels to: {labels_output}")
        
        # Print class count
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        print(f"  Classes: {len(labels)}")
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nProduction files:")
    print(f"  Model:  {output_path}")
    print(f"  Labels: {labels_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export MSG3D model')
    
    parser.add_argument('--checkpoint', type=str, default='best.pth',
                        help='Checkpoint filename (default: best.pth)')
    parser.add_argument('--output', type=str, default='msg3d_lse.pt',
                        help='Output model filename (default: msg3d_lse.pt)')
    
    args = parser.parse_args()
    
    # Paths
    project_dir = Path(__file__).resolve().parents[3]  # dev/
    training_dir = project_dir / "training_output" / "msg3d"
    models_dir = project_dir.parent / "models"  # services/vision_service/models/
    models_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = training_dir / args.checkpoint
    output_path = models_dir / args.output
    labels_path = project_dir / "datasets_processed" / "msg3d" / "label_names.pkl"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    export_model(checkpoint_path, output_path, labels_path)
