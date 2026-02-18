import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
from pathlib import Path
import os
from collections import deque
import time

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. SETUP PATHS
# Use relative paths from project root (CWD)
MODEL_PATH = Path("services/vision_service/models/best_model.h5")
LABELS_PATH = Path("services/vision_service/models/best_model_labels.pkl")
VIDEO_PATH = Path(r"services\vision_service\dev\datasets_raw\videos\letras\cases\CASO_0_blur (1).mp4")

# 2. CONFIG
BUFFER_SIZE = 30  # El modelo espera 30 frames
DISPLAY_TOP_N = 3
CONFIDENCE_THRESHOLD = 0.5

# 3. PATCHED LSTM
class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, time_major=False, **kwargs):
        super().__init__(**kwargs)
        self.time_major = time_major

    def get_config(self):
        config = super().get_config()
        config['time_major'] = self.time_major
        return config

print(f"🔄 Cargando modelo Keras desde: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'LSTM': PatchedLSTM})
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

print(f"🔄 Cargando etiquetas desde: {LABELS_PATH}")
try:
    with open(LABELS_PATH, 'rb') as f:
        id_to_label = pickle.load(f)
    print(f"✅ {len(id_to_label)} etiquetas cargadas")
except Exception as e:
    print(f"❌ Error cargando etiquetas: {e}")
    exit(1)

# 4. MEDIA PIPE SETUP
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_holistic_features(results):
    """Extrae 226 features en el orden Pose -> LH -> RH (Correcto de MSL-150)."""
    features = []
    
    # 1. Pose landmarks (33 × 3 = 99)
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
    
    # 4. Padding (1) -> Total 226
    if len(features) == 225:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)

# 5. MAIN LOOP
def run_video_test():
    if not VIDEO_PATH.exists():
        print(f"❌ Video no encontrado en: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"▶️ Reproduciendo video: {VIDEO_PATH.name} ({width}x{height} @ {fps:.1f} fps)")

    frame_buffer = deque(maxlen=BUFFER_SIZE)
    current_prediction = "..."
    current_conf = 0.0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                # Reiniciar video para verlo en bucle
                print("🔁 Reiniciando video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_buffer.clear()
                continue
            
            # Process frame
            # 1. Resize for speed (MediaPipe is slow on HD)
            target_width = 640
            scale = target_width / frame.shape[1]
            if scale < 1:
                frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                frame_small = frame
                
            image = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on the small image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Resize back to original for display (optional, or just show small)
            # Let's show the small one for speed check
            display_frame = image
            
            # Extract features
            features = extract_holistic_features(results)
            frame_buffer.append(features)
            
            # Predict
            if len(frame_buffer) == BUFFER_SIZE:
                # Prepare input (1, 30, 226)
                input_data = np.array(frame_buffer)
                input_data = np.expand_dims(input_data, axis=0)
                
                probs = model.predict(input_data, verbose=0)[0] # (150,)
                
                top_idx = np.argsort(probs)[::-1][:DISPLAY_TOP_N]
                
                # Update current prediction if confidence is high
                if probs[top_idx[0]] > CONFIDENCE_THRESHOLD:
                    current_prediction = id_to_label[top_idx[0]]
                    current_conf = probs[top_idx[0]]
                
                # Display predictions on screen
                y_pos = 50
                for i in top_idx:
                    label = id_to_label[i]
                    conf = probs[i]
                    color = (0, 255, 0) if i == top_idx[0] else (0, 150, 255)
                    text = f"{label}: {conf*100:.1f}%"
                    cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 25
            else:
                cv2.putText(display_frame, f"Buffering... {len(frame_buffer)}/{BUFFER_SIZE}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show "Best Guess" permanently
            cv2.putText(display_frame, f"DETECTADO: {current_prediction} ({current_conf*100:.1f}%)", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('SignSpeak Video Test', display_frame)
            
            # Esperar un poco para simular tiempo real si es muy rápido
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_test()
