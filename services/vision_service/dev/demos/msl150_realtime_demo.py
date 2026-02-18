"""
Demo en tiempo real con cámara para modelo MSL-150.
Detecta señas en vivo y muestra predicciones.

Controles:
- Presiona 'q' para salir
- Presiona 'c' para limpiar buffer de señas
- Presiona ESPACIO para forzar predicción
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import pickle
from pathlib import Path
from collections import deque
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.training.train_msl150_lstm import HolisticLSTM

# Configuración
MODEL_PATH = PROJECT_DIR / "training_output" / "msl150_subset_lstm" / "best.pth"
LABELS_PATH = PROJECT_DIR / "datasets_processed" / "msl150_subset" / "label_names.pkl"

BUFFER_SIZE = 64  # Frames para predicción
CONFIDENCE_THRESHOLD = 0.5
DISPLAY_TOP_N = 5

print("=" * 70)
print("DEMO EN TIEMPO REAL - MSL-150")
print("=" * 70)
print("\nControles:")
print("  'q' - Salir")
print("  'c' - Limpiar buffer")
print("  ESPACIO - Forzar predicción")
print("=" * 70)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

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

print(f"✓ {len(id_to_label)} clases cargadas")

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

def predict(buffer, model, device):
    """Predice usando el buffer actual."""
    if len(buffer) < 10:  # Mínimo 10 frames
        return None, None
    
    # Convert buffer to array
    features = np.array(list(buffer))  # (T, 226)
    
    # Pad/truncate to 64
    T = features.shape[0]
    if T < BUFFER_SIZE:
        padded = np.zeros((BUFFER_SIZE, 226), dtype=np.float32)
        for i in range(BUFFER_SIZE):
            padded[i] = features[i % T]
    else:
        start = (T - BUFFER_SIZE) // 2
        padded = features[start:start+BUFFER_SIZE]
    
    # Predict
    x = torch.from_numpy(padded).unsqueeze(0).to(device)
    # print(f"Input shape: {x.shape}")  # Debug shape
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_ids = torch.topk(probs, DISPLAY_TOP_N, dim=1)
    
    return topk_ids[0].cpu().numpy(), topk_probs[0].cpu().numpy()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: No se pudo abrir la cámara")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"\n✓ Cámara iniciada")
print(f"\nPresiona 'q' para salir\n")

# Buffer for frames
frame_buffer = deque(maxlen=BUFFER_SIZE)
last_prediction = None
last_confidence = 0.0
detected_words = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(rgb_frame)
        
        # Extract features
        features = extract_holistic_features(results)
        frame_buffer.append(features)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Predict automáticamente cuando buffer está lleno
        if len(frame_buffer) >= BUFFER_SIZE:
            topk_ids, topk_probs = predict(frame_buffer, model, device)
            
            if topk_ids is not None:
                word = id_to_label[topk_ids[0]]
                conf = topk_probs[0]
                
                # Actualizar última predicción siempre para mostrar en UI
                last_prediction = word
                last_confidence = conf
                
                # Solo agregar a frase si supera threshold y es diferente
                if conf >= CONFIDENCE_THRESHOLD:
                    if word != (detected_words[-1] if detected_words else None):
                        detected_words.append(word)
                        print(f"🎯 Detectado: '{word}' ({conf*100:.1f}%)")
                        
                        # Mantener solo últimas 10 palabras
                        if len(detected_words) > 10:
                            detected_words.pop(0)
        
        # Draw UI
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "MSL-150 Demo - Tiempo Real", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Buffer status
        buffer_pct = (len(frame_buffer) / BUFFER_SIZE) * 100
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{BUFFER_SIZE} ({buffer_pct:.0f}%)", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Last prediction
        if last_prediction:
            color = (0, 255, 0) if last_confidence >= 0.7 else (0, 255, 255)
            cv2.putText(frame, f"Ultima: '{last_prediction}' ({last_confidence*100:.1f}%)", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Detected phrase
        if detected_words:
            phrase = " -> ".join(detected_words[-5:])  # Últimas 5
            cv2.putText(frame, f"Frase: {phrase}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Top predictions (if available)
        if last_prediction:
            topk_ids, topk_probs = predict(frame_buffer, model, device)
            if topk_ids is not None:
                y_offset = 160
                cv2.putText(frame, "Top-5:", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                
                for i, (class_id, prob) in enumerate(zip(topk_ids, topk_probs)):
                    word = id_to_label[class_id]
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                    text = f"  {i+1}. {word}: {prob*100:.1f}%"
                    cv2.putText(frame, text, (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 18
        
        # Instructions
        cv2.putText(frame, "Controles: 'q'=salir  'c'=limpiar  ESPACIO=predecir", 
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('MSL-150 Real-Time Demo', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_buffer.clear()
            detected_words.clear()
            last_prediction = None
            print("🔄 Buffer limpiado")
        elif key == ord(' '):
            if len(frame_buffer) >= 10:
                topk_ids, topk_probs = predict(frame_buffer, model, device)
                if topk_ids is not None:
                    word = id_to_label[topk_ids[0]]
                    conf = topk_probs[0]
                    print(f"⚡ Predicción forzada: '{word}' ({conf*100:.1f}%)")

except KeyboardInterrupt:
    print("\n\n🛑 Interrumpido por usuario")

finally:
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("\n✓ Demo finalizado")
    
    if detected_words:
        print(f"\n📝 Frase completa detectada:")
        print(f"   {' → '.join(detected_words)}")
