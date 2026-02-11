"""
MSG3D Model Architecture for Sign Language Recognition.

Multi-Scale Graph 3D Convolutional Network (MSG3D).
Adaptado para MediaPipe skeleton (75 keypoints, 3 canales).

Referencia: https://github.com/kenziyuliu/MS-G3D
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure graph module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.mediapipe_graph import get_adjacency_matrix


class GraphConvolution(nn.Module):
    """
    Graph Convolution: aplica convolución espacial sobre el grafo de joints.
    """
    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        
        self.num_subsets = A.shape[0]
        
        # Una conv 1x1 por subset del grafo
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_subsets)
        ])
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Registrar la adjacency matrix como buffer (no-trainable)
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        
        # Residual connection
        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.res_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.res_conv = None
        
    def forward(self, x):
        """
        x: (N, C, T, V)
        """
        res = x
        
        # Agregar contribución de cada subset
        y = None
        for k in range(self.num_subsets):
            # Graph convolution: x @ A[k] (multiplicación por adjacency)
            # x: (N, C, T, V), A[k]: (V, V) -> x_a: (N, C, T, V)
            x_a = torch.einsum('nctv,vw->nctw', x, self.A[k])
            # 1x1 conv
            x_a = self.convs[k](x_a)
            y = x_a if y is None else y + x_a
        
        y = self.bn(y)
        
        # Residual
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(res)
            y = y + res
        
        return F.relu(y)


class TemporalConv(nn.Module):
    """
    Temporal convolution sobre la dimensión de frames.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, residual=True):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(kernel_size, 1),
                      padding=(padding, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        )
        
        self.residual = residual
        if residual and (in_channels != out_channels or stride != 1):
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_conv = None
    
    def forward(self, x):
        res = x
        y = self.conv(x)
        
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(res)
            y = y + res
        
        return F.relu(y)


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution block.
    GCN (spatial) + TCN (temporal) + residual.
    """
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.25):
        super().__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.dropout(x)
        return x


class MSG3D(nn.Module):
    """
    MSG3D model para reconocimiento de señas.
    
    Input shape: (N, C, T, V, M)
      N = batch size
      C = 3 (x, y, z)
      T = frames (64)
      V = joints (75)
      M = persons (1)
    """
    def __init__(
        self,
        num_class=300,
        num_point=75,
        num_person=1,
        in_channels=3,
        base_channels=64,
        dropout=0.25
    ):
        super().__init__()
        
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        
        # Create adjacency matrix: 3 subsets [identity, spatial, spatial]
        A_spatial = get_adjacency_matrix()
        A = np.stack([np.eye(num_point, dtype=np.float32), A_spatial, A_spatial], axis=0)
        
        # Input normalization: BatchNorm sobre (C*V) features
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        
        # 6-layer GCN backbone (adecuado para CPU y 6K training samples)
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, c1, A, stride=1, dropout=dropout),  # 3 -> 64
            STGCNBlock(c1, c1, A, stride=1, dropout=dropout),            # 64 -> 64
            STGCNBlock(c1, c2, A, stride=1, dropout=dropout),            # 64 -> 128
            STGCNBlock(c2, c2, A, stride=1, dropout=dropout),            # 128 -> 128
            STGCNBlock(c2, c3, A, stride=1, dropout=dropout),            # 128 -> 256
            STGCNBlock(c3, c3, A, stride=1, dropout=dropout),            # 256 -> 256
        ])
        
        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_class)
        self.dropout_fc = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M)
        Returns:
            logits: (N, num_class)
        """
        N, C, T, V, M = x.shape
        
        # Merge person dimension: (N, C, T, V, M) -> (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Input normalization
        # Reshape: (N*M, C, T, V) -> (N*M, C*V, T)  para BN1d
        # Requiere permute para agrupar C y V
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        # Reshape back: (N*M, C*V, T) -> (N*M, C, T, V)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        # GCN backbone
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling: (N*M, C_out, T, V) -> (N*M, C_out, 1, 1)
        x = self.gap(x)
        x = x.view(N, M, -1).mean(dim=1)  # Average over persons -> (N, C_out)
        
        # Classification
        x = self.dropout_fc(x)
        logits = self.fc(x)
        
        return logits


def count_parameters(model):
    """Cuenta parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("MSG3D Model Self-Test")
    print("=" * 60)
    
    model = MSG3D(num_class=300, num_point=75, num_person=1, in_channels=3)
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 64, 75, 1)
    y = model(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 300), f"Expected (2, 300), got {y.shape}"
    print("PASSED!")
