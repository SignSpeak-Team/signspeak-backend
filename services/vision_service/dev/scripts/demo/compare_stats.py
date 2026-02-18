"""
Compara estadísticas de Training Data vs Video del Usuario vs Live Camera
"""
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "datasets_processed" / "msl150_subset"
VIDEO_PATH = PROJECT_DIR / "datasets_raw" / "videos" / "letras" / "cases" / "CASO_0_blur (1).mp4"

# 1. Load Training Data Stats
print("=" * 70)
print("COMPARACIÓN DE DISTRIBUCIONES")
print("=" * 70)

val_data = np.load(DATA_DIR / "val_data.npy") # (N, 64, 226)

# Flatten spatial dims for stats
train_flat = val_data.reshape(-1, 226)
train_mean = np.mean(train_flat, axis=0)
train_std = np.std(train_flat, axis=0)

print(f"📊 Training Data (Validation):")
print(f"   Global Mean: {np.mean(train_flat):.4f}")
print(f"   Global Std:  {np.std(train_flat):.4f}")
print(f"   Shape: {val_data.shape}")

# 2. Extract User Video Stats
print(f"\n📊 User Video (CASO_0):")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

def extract_features(results):
    features = []
    # Usar orden CORRECTO: Pose(99) -> LH(63) -> RH(63) -> Pad(1)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z]) # 33*3=99
    else:
        features.extend([0.0] * 99)
        
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
        
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
        
    if len(features) == 225:
        features.append(0.0)
    return np.array(features)

cap = cv2.VideoCapture(str(VIDEO_PATH))
video_features = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    feats = extract_features(results)
    video_features.append(feats)

cap.release()
holistic.close()

video_data = np.array(video_features)
print(f"   Shape: {video_data.shape}")
print(f"   Global Mean: {np.mean(video_data):.4f}")
print(f"   Global Std:  {np.std(video_data):.4f}")

# 3. Compare Specific Keypoints
# Nose (0,1,2), L Shoulder (33,34,35), R Hip (72,73,74)
def print_kp_stats(name, idx, data_mean, video_mean):
    print(f"\n   {name}:")
    print(f"     Train Mean: {data_mean[idx:idx+3]}")
    print(f"     Video Mean: {video_mean[idx:idx+3]}")
    diff = video_mean[idx:idx+3] - data_mean[idx:idx+3]
    print(f"     Diff:       {diff}")

video_mean = np.mean(video_data, axis=0)

print_kp_stats("Nose (x,y,z)", 0, train_mean, video_mean)
print_kp_stats("L Shoulder", 33, train_mean, video_mean) # 11*3
print_kp_stats("R Shoulder", 36, train_mean, video_mean) # 12*3

# Check if simple shift helps
nose_diff = video_mean[0:2] - train_mean[0:2]
print(f"\n💡 Desplazamiento promedio (Nose X,Y): {nose_diff}")
print(f"   Si es grande (>0.1), el sujeto está en otra posición")

print(f"\n{'=' * 70}")
