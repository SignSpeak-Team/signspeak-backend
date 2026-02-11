"""
MediaPipe skeleton graph definition for MSG3D.

75 keypoints:
  - [0-32]   Pose (33 landmarks)
  - [33-53]  Left Hand (21 landmarks)
  - [54-74]  Right Hand (21 landmarks)

Referencias MediaPipe:
  - Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
  - Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
"""

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# POSE LANDMARKS (0-32) - 33 landmarks
# ────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose landmark indices:
# 0: nose, 1-2: eyes, 3-4: ears, 5-6: mouth corners
# 7-10: shoulders & elbows, 11-16: wrists & hips, 17-22: knees & ankles
# 23-28: feet, 29-32: hands (connections to hand keypoints)

POSE_EDGES = [
    # Face
    (0, 1), (1, 2), (2, 3),  # nose -> left eye -> left ear
    (0, 4), (4, 5), (5, 6),  # nose -> right eye -> right ear
    (0, 7),                  # nose -> left mouth
    (0, 8),                  # nose -> right mouth
    
    # Torso
    (11, 12),                # left shoulder -> right shoulder
    (11, 23), (23, 24),      # left shoulder -> left hip -> right hip
    (12, 24),                # right shoulder -> right hip
    
    # Left arm
    (11, 13), (13, 15),      # left shoulder -> elbow -> wrist
    
    # Right arm
    (12, 14), (14, 16),      # right shoulder -> elbow -> wrist
    
    # Left leg
    (23, 25), (25, 27),      # left hip -> knee -> ankle
    (27, 29), (29, 31),      # ankle -> heel -> toe
    
    # Right leg
    (24, 26), (26, 28),      # right hip -> knee -> ankle
    (28, 30), (30, 32),      # ankle -> heel -> toe
    
    # Wrist to hand center connections (will connect to hand keypoints)
    (15, 33),                # left wrist -> left hand center (landmark 33)
    (16, 54),                # right wrist -> right hand center (landmark 54)
]


# ────────────────────────────────────────────────────────────────────────────
# HAND LANDMARKS (21 per hand)
# ────────────────────────────────────────────────────────────────────────────
# MediaPipe Hand landmark indices (relative to hand):
# 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky

def hand_edges(offset):
    """Generate edges for a hand with given offset.
    
    Args:
        offset: 33 for left hand, 54 for right hand
    """
    return [
        # Palm
        (offset + 0, offset + 1),   # wrist -> thumb base
        (offset + 0, offset + 5),   # index base
        (offset + 0, offset + 9),   # middle base
        (offset + 0, offset + 13),  # ring base
        (offset + 0, offset + 17),  # pinky base
        (offset + 5, offset + 9),   # index -> middle
        (offset + 9, offset + 13),  # middle -> ring
        (offset + 13, offset + 17), # ring -> pinky
        
        # Thumb
        (offset + 1, offset + 2),
        (offset + 2, offset + 3),
        (offset + 3, offset + 4),
        
        # Index
        (offset + 5, offset + 6),
        (offset + 6, offset + 7),
        (offset + 7, offset + 8),
        
        # Middle
        (offset + 9, offset + 10),
        (offset + 10, offset + 11),
        (offset + 11, offset + 12),
        
        # Ring
        (offset + 13, offset + 14),
        (offset + 14, offset + 15),
        (offset + 15, offset + 16),
        
        # Pinky
        (offset + 17, offset + 18),
        (offset + 18, offset + 19),
        (offset + 19, offset + 20),
    ]


LEFT_HAND_EDGES = hand_edges(33)
RIGHT_HAND_EDGES = hand_edges(54)


# ────────────────────────────────────────────────────────────────────────────
# FULL SKELETON GRAPH
# ────────────────────────────────────────────────────────────────────────────

ALL_EDGES = POSE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES

NUM_NODES = 75
NUM_EDGES = len(ALL_EDGES)


def get_adjacency_matrix(strategy='spatial'):
    """
    Construye la matriz de adyacencia del esqueleto MediaPipe.
    
    Args:
        strategy: 'spatial' (conexiones anatómicas) o 'uniform' (identity)
    
    Returns:
        A: Matriz de adyacencia normalizada (75, 75)
    """
    if strategy == 'uniform':
        return np.eye(NUM_NODES, dtype=np.float32)
    
    # Crear matriz de adyacencia binaria
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    
    # Agregar edges bidireccionales
    for i, j in ALL_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    
    # Agregar self-loops
    A += np.eye(NUM_NODES, dtype=np.float32)
    
    # Normalización: D^(-1/2) * A * D^(-1/2)
    # Donde D es la matriz diagonal de grados
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    
    A_normalized = D_mat @ A @ D_mat
    
    return A_normalized.astype(np.float32)
