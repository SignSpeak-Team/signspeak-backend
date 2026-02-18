"""
MSG3D Training Script para LSE (Lengua de Señas Española).

Entrena el modelo MSG3D con el dataset procesado (.npy files).
Guarda checkpoints, logs de TensorBoard, y el mejor modelo.

Optimizaciones Phase 4:
- Data Augmentation (On-the-fly)
- Label Smoothing CrossEntropy
- Warmup Learing Rate Scheduler
- 120 Epochs

Uso:
    python train.py --epochs 120 --batch-size 32 --lr 0.05
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # For data_augmentation

from model.msg3d import MSG3D, count_parameters
try:
    from data_augmentation import augment_msg3d
except ImportError:
    print("Warning: data_augmentation module not found. proceeding without augmentation.")
    augment_msg3d = None


# ────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION (LABEL SMOOTHING)
# ────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ────────────────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────────────────

class MSG3DDataset(Dataset):
    """PyTorch Dataset para datos MSG3D (.npy) con Augmentation opcional."""
    
    def __init__(self, data_path, labels_path, augment=False):
        """
        Args:
            data_path: Path to .npy file with shape (N, C, T, V, M)
            labels_path: Path to .npy file with class labels (N,)
            augment: Boolean true for training set
        """
        self.data = np.load(data_path, mmap_mode='r')  # Memory-mapped
        self.labels = np.load(labels_path)
        self.augment = augment
        
        assert len(self.data) == len(self.labels), \
            f"Data and labels mismatch: {len(self.data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Copiar datos para evitar modificar mmap
        data_numpy = self.data[idx].copy() # (C, T, V, M)
        
        # Data Augmentation
        if self.augment and augment_msg3d is not None:
            data_numpy = augment_msg3d(data_numpy)
            
        # Convertir a tensores
        data = torch.from_numpy(data_numpy).float()
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
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc: {accs.val:.4f} ({accs.avg:.4f})', flush=True)
    
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


def get_scheduler(optimizer, n_epochs, warmup_epochs=10):
    """Creates a Warmup + CosineAnnealing scheduler."""
    # Warmup scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    # Cosine annealing after warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(n_epochs - warmup_epochs)
    )
    # Chain them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    return scheduler

# ────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ────────────────────────────────────────────────────────────────────────────

def main(args):
    # Paths
    project_dir = Path(__file__).resolve().parents[3]  # dev/
    data_dir = project_dir / "datasets_processed" / "msg3d"
    output_dir = project_dir / "training_output" / "msg3d_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"MSG3D Optimized Training - Phase 5")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Augmentation: Enabled")
    
    # Datasets
    print("\nLoading datasets...")
    # Usar dataset pre-aumentado para evitar cuello de botella en CPU
    train_dataset = MSG3DDataset(
        data_dir / "train_data_aug.npy",
        data_dir / "train_labels_aug.npy",
        augment=False # Desactivar, ya está aumentado
    )
    val_dataset = MSG3DDataset(
        data_dir / "val_data.npy",
        data_dir / "val_labels.npy",
        augment=False
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Data loaders
    # Windows-specific optimization: Persistent workers prevents killing processes each epoch
    use_cuda = device.type == 'cuda'
    num_workers = 4 if use_cuda else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0), # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None # Prefetch batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
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
    
    # Load pretrained weights if available (Warm start)
    # previous_best = project_dir / "training_output" / "msg3d" / "best.pth"
    # if previous_best.exists():
    #     print(f"  Loading pretrained weights from {previous_best}")
    #     checkpoint = torch.load(previous_best, map_location='cpu') # Load to cpu first
    #     model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Loss function (Label Smoothing)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Optimizer (SGD con momentum es mejor para GCN a veces, pero Adam es standard)
    # Cambiamos a SGD con Nesterov para mejor convergencia final
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler (Warmup + Cosine)
    scheduler = get_scheduler(optimizer, args.epochs, warmup_epochs=10)
    
    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting optimized training for {args.epochs} epochs")
    print(f"{'=' * 60}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{args.epochs}] LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step LR
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s\n")
        
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
    parser = argparse.ArgumentParser(description='MSG3D Training Optimized')
    
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
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs (default: 120)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (SGD default: 0.05)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    main(args)
