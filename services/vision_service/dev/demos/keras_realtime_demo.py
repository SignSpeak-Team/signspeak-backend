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
MODEL_PATH = Path("services/vision_service/models/LSTM_64_128.h5")
# Updated to use the correct labels for this model
LABELS_PATH = Path("services/vision_service/models/best_model_labels.pkl")

# 2. CONFIG
BUFFER_SIZE = 30  # El modelo espera 30 frames
DISPLAY_TOP_N = 3
CONFIDENCE_THRESHOLD = 0.6  # Ajustable

# 3. PATCHED LSTM (Para cargar el modelo antiguo)
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
def run_demo():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    last_pred_time = 0
    current_prediction = "..."
    current_conf = 0.0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Extract features
            features = extract_holistic_features(results)
            frame_buffer.append(features)
            
            # Predict
            if len(frame_buffer) == BUFFER_SIZE:
                # Prepare input (1, 30, 226)
                input_data = np.array(frame_buffer)
                input_data = np.expand_dims(input_data, axis=0)
                
                start = time.time()
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
                    cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_pos += 30
            else:
                cv2.putText(image, f"Buffering... {len(frame_buffer)}/{BUFFER_SIZE}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show "Best Guess" permanently
            cv2.putText(image, f"DETECTADO: {current_prediction} ({current_conf*100:.1f}%)", 
                       (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('SignSpeak Keras Demo', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
