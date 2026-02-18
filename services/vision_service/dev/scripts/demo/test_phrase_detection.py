"""
Demo de detección de FRASES completas (múltiples señas en un video).
"""
import torch
import numpy as np
import pickle
import cv2
import mediapipe as mp
from pathlib import Path
import sys

# Setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.training.train_msl150_lstm import HolisticLSTM

# Paths
MODEL_PATH = PROJECT_DIR / "training_output" / "msl150_subset_lstm" / "best.pth"
LABELS_PATH = PROJECT_DIR / "datasets_processed" / "msl150_subset" / "label_names.pkl"
VIDEOS_DIR = PROJECT_DIR / "datasets_raw" / "videos" / "letras" / "cases"

print("=" * 70)
print("DETECCIÓN DE FRASES COMPLETAS - MSL-150")
print("=" * 70)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_holistic_features(results):
    """Extrae 226 features de MediaPipe Holistic en el ORDEN CORRECTO del dataset MSL-150."""
    features = []
    
    # ORDEN IMPORTANTE: DEBE COINCIDIR CON EL DATASET
    # MSL-150 format: Pose (99) + Left Hand (63) + Right Hand (63) + padding (1) = 226
    
    # 1. Pose landmarks (33 × 3 = 99) - solo x,y,z SIN visibility
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 99)
    
    # 2. Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # 3. Right hand (21 × 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # 4. Padding to reach 226 (99 + 63 + 63 + 1 = 226)
    if len(features) == 225:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)

def extract_video_features(video_path):
    """Extrae features de todo el video."""
    cap = cv2.VideoCapture(str(video_path))
    
    features_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for speed
        target_width = 640
        scale = target_width / rgb_frame.shape[1]
        if scale < 1:
            rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
            
        results = holistic.process(rgb_frame)
        features = extract_holistic_features(results)
        features_list.append(features)
    
    cap.release()
    return np.array(features_list) if features_list else None

def segment_signs(features, min_gap=10, min_length=15):
    """Segmenta el video detectando pausas entre señas."""
    # Calcular movimiento (diferencia entre frames)
    movement = np.linalg.norm(np.diff(features, axis=0), axis=1)
    
    # Frames "quietos" = percentil 25 más bajo
    threshold = np.percentile(movement, 25)
    is_moving = movement > threshold
    
    # Encontrar segmentos
    segments = []
    in_motion = False
    start = 0
    gap_count = 0
    
    for i, moving in enumerate(is_moving):
        if moving:
            if not in_motion:
                start = i
                in_motion = True
            gap_count = 0
        else:
            gap_count += 1
            if in_motion and gap_count >= min_gap:
                if i - start >= min_length:
                    segments.append((start, i - gap_count))
                in_motion = False
    
    # Último segmento
    if in_motion and len(is_moving) - start >= min_length:
        segments.append((start, len(is_moving)))
    
    return segments

def predict_segment(features, model, device):
    """Predice la clase de un segmento."""
    T = features.shape[0]
    if T < 64:
        padded = np.zeros((64, 226), dtype=np.float32)
        for i in range(64):
            padded[i] = features[i % T]
    else:
        start = (T - 64) // 2
        padded = features[start:start+64]
    
    x = torch.from_numpy(padded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top5_probs, top5_ids = torch.topk(probs, 5, dim=1)
    
    return top5_ids[0].cpu().numpy(), top5_probs[0].cpu().numpy()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = HolisticLSTM(
    input_size=226,
    hidden_size=512,
    num_layers=3,
    num_classes=150,
    dropout=0.4,
    bidirectional=True
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Modelo cargado (Val Acc: {checkpoint['val_acc']*100:.2f}%)")

# Load labels
with open(LABELS_PATH, 'rb') as f:
    id_to_label = pickle.load(f)

print(f"✓ {len(id_to_label)} clases cargadas\n")

# Process videos
videos = list(VIDEOS_DIR.glob("*.mp4"))
print(f"{'=' * 70}")
print("PREDICCIONES DE FRASES")
print("=" * 70)

for video_path in videos:
    print(f"\n📹 {video_path.name}")
    
    # Extract features
    features = extract_video_features(video_path)
    if features is None:
        print("  ⚠️  Error extrayendo features")
        continue
    
    print(f"  Total frames: {len(features)}")
    
    # Segment
    segments = segment_signs(features)
    print(f"  Señas detectadas: {len(segments)}")
    
    # Predict each segment
    phrase = []
    for i, (start, end) in enumerate(segments, 1):
        segment_features = features[start:end]
        top5_ids, top5_probs = predict_segment(segment_features, model, device)
        
        word = id_to_label[top5_ids[0]]
        conf = top5_probs[0]
        
        phrase.append(word)
        print(f"\n  Seña {i} (frames {start}-{end}, duración: {end-start} frames):")
        print(f"    🥇 '{word}' - {conf*100:.2f}%")
        print(f"    🥈 '{id_to_label[top5_ids[1]]}' - {top5_probs[1]*100:.2f}%")
        print(f"    🥉 '{id_to_label[top5_ids[2]]}' - {top5_probs[2]*100:.2f}%")
    
    # Remove consecutive duplicates
    filtered_phrase = []
    prev = None
    for word in phrase:
        if word != prev:
            filtered_phrase.append(word)
            prev = word
    
    print(f"\n  ✅ FRASE DETECTADA: {' → '.join(filtered_phrase)}")

print(f"\n{'=' * 70}")
print("✓ Análisis completado")
print("=" * 70)

holistic.close()
