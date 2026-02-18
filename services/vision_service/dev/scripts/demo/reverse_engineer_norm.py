"""
Reverse-engineer normalization: Check values of specific landmarks in training data.
"""
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path(r"C:\Users\alan1\PycharmProjects\signSpeak\services\vision_service\dev\datasets_processed\msl150_subset")
val_data = np.load(DATA_DIR / "val_data.npy")

print("=" * 70)
print("REVERSE ENGINEERING NORMALIZATION")
print("=" * 70)

# Shape: (N, 64, 226)
# Format: Pose(99) + LH(63) + RH(63) + pad(1)

# Pose landmarks 0-32 (33 landmarks * 3 coords) = indices 0-98
# Key landmarks indices (x, y, z):
# 0: Nose (0, 1, 2)
# 11: Left Shoulder (33, 34, 35)
# 12: Right Shoulder (36, 37, 38)
# 23: Left Hip (69, 70, 71)
# 24: Right Hip (72, 73, 74)

def get_landmark(frame, index):
    base = index * 3
    return frame[base:base+3]

print(f"Analyzing {len(val_data)} samples...")

nose_vals = []
l_shoulder_vals = []
r_shoulder_vals = []
l_hip_vals = []
r_hip_vals = []

for i in range(100): # Check first 100 samples
    frame = val_data[i][0] # First frame of sequence
    
    nose_vals.append(get_landmark(frame, 0))
    l_shoulder_vals.append(get_landmark(frame, 11))
    r_shoulder_vals.append(get_landmark(frame, 12))
    l_hip_vals.append(get_landmark(frame, 23))
    r_hip_vals.append(get_landmark(frame, 24))

nose_mean = np.mean(nose_vals, axis=0)
ls_mean = np.mean(l_shoulder_vals, axis=0)
rs_mean = np.mean(r_shoulder_vals, axis=0)
lh_mean = np.mean(l_hip_vals, axis=0)
rh_mean = np.mean(r_hip_vals, axis=0)

print(f"\nMean Coordinates (x, y, z):")
print(f"  Nose:           {nose_mean}")
print(f"  Left Shoulder:  {ls_mean}")
print(f"  Right Shoulder: {rs_mean}")
print(f"  Left Hip:       {lh_mean}")
print(f"  Right Hip:      {rh_mean}")

# Check center point
hip_center = (lh_mean + rh_mean) / 2
shoulder_center = (ls_mean + rs_mean) / 2

print(f"\nPotential Centers:")
print(f"  Hip Center:      {hip_center}")
print(f"  Shoulder Center: {shoulder_center}")

# Check scaling (distance between shoulders)
shoulder_dist = np.linalg.norm(ls_mean - rs_mean)
print(f"\nScale Reference:")
print(f"  Shoulder Distance: {shoulder_dist:.4f}")

# Check if maybe Nose is 0?
if np.allclose(nose_mean, 0, atol=0.1):
    print("\n✅ Evidence suggests: Centered on NOSE")
elif np.allclose(hip_center, 0, atol=0.1):
    print("\n✅ Evidence suggests: Centered on HIP CENTER")
elif np.allclose(shoulder_center, 0, atol=0.1):
    print("\n✅ Evidence suggests: Centered on SHOULDER CENTER")
else:
    print("\n❓ Centering unclear. Maybe raw normalized coordinates?")
    
# Range check
all_vals = val_data[:100, 0, :99].flatten()
print(f"\nData Range (Pose): [{all_vals.min():.4f}, {all_vals.max():.4f}]")
