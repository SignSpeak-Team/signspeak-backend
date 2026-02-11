"""
MSG3D Training Script para LSE (Lengua de Señas Española).

Entrena el modelo MSG3D con el dataset procesado (.npy files).
Guarda checkpoints, logs de TensorBoard, y el mejor modelo.

Uso:
    python train.py --epochs 100 --batch-size 32 --lr 0.001
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.msg3d import MSG3D, count_parameters


# ────────────────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────────────────

class MSG3DDataset(Dataset):
    """PyTorch Dataset para datos MSG3D (.npy)."""
    
    def __init__(self, data_path, labels_path):
        """
        Args:
            data_path: Path to .npy file with shape (N, C, T, V, M)
            labels_path: Path to .npy file with class labels (N,)
        """
        self.data = np.load(data_path, mmap_mode='r')  # Memory-mapped
        self.labels = np.load(labels_path)
        
        assert len(self.data) == len(self.labels), \
            f"Data and labels mismatch: {len(self.data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convertir a tensores
        data = torch.from_numpy(self.data[idx].copy()).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


# ────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Calcula y guarda el promedio y valor actual."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Entrena una época."""
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        
        losses.update(loss.item(), data.size(0))
        accs.update(acc.item(), data.size(0))
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(loader)}] '
                  f'Loss: {losses.avg:.4f} Acc: {accs.avg:.4f}', flush=True)
    
    return losses.avg, accs.avg


def validate(model, loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Metrics
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean()
            
            losses.update(loss.item(), data.size(0))
            accs.update(acc.item(), data.size(0))
    
    return losses.avg, accs.avg


# ────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ────────────────────────────────────────────────────────────────────────────

def main(args):
    # Paths
    project_dir = Path(__file__).resolve().parents[3]  # dev/
    data_dir = project_dir / "datasets_processed" / "msg3d"
    output_dir = project_dir / "training_output" / "msg3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"MSG3D Training - LSE 300 Signs")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = MSG3DDataset(
        data_dir / "train_data.npy",
        data_dir / "train_labels.npy"
    )
    val_dataset = MSG3DDataset(
        data_dir / "val_data.npy",
        data_dir / "val_labels.npy"
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Data loaders
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers if use_cuda else 0,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers if use_cuda else 0,
        pin_memory=use_cuda
    )
    
    # Model
    print("\nInitializing model...")
    model = MSG3D(
        num_class=300,
        num_point=75,
        num_person=1,
        in_channels=3,
        base_channels=args.base_channels,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Loss function (con class weights si existen)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'=' * 60}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(checkpoint, output_dir / 'best.pth')
            print(f"  [NEW BEST] Val Acc: {val_acc:.4f}\n")
        
        # Save periodic checkpoints
        if epoch % args.save_freq == 0:
            torch.save(checkpoint, output_dir / f'epoch_{epoch}.pth')
    
    # Training complete
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}")
    print(f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Checkpoints saved to: {output_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSG3D Training')
    
    # Data
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Model
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base number of channels (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    main(args)
