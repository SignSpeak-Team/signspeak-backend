import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
from pathlib import Path
import os
from collections import deque
import collections

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. SETUP PATHS
PROJECT_DIR = Path.cwd()
# Ajustar rutas según estructura del proyecto
MODEL_PATH = PROJECT_DIR / "services" / "vision_service" / "models" / "LSTM_64_128.h5"
LABELS_PATH = PROJECT_DIR / "services" / "vision_service" / "models" / "best_model_labels.pkl"
VIDEO_PATH = PROJECT_DIR / "services" / "vision_service" / "dev" / "datasets_raw" / "videos" / "letras" / "cases" / "CASO_5_blur.mp4"

# 2. CONFIG FROM USER CODE
SKIP_FRAMES = 2
THRESHOLD = 0.8
SMOOTHING_WINDOW = 10
BUFFER_SIZE = 30 # Model expects 30 frames

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

# 5. MAIN LOGIC (Based on User Code)
def run_phrase_test():
    if not VIDEO_PATH.exists():
        print(f"❌ Video no encontrado en: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    # Resize logic setup
    target_width = 640

    # User Logic Variables
    sequence = deque(maxlen=30)
    predictions = deque(maxlen=SMOOTHING_WINDOW) 
    sentence = []
    
    frame_count = 0
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("🔁 Reiniciando video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sequence.clear()
                predictions.clear()
                sentence = [] # Reset sentence on loop? Or keep? Let's reset for clarity
                frame_count = 0
                continue
            
            frame_count += 1
            
            # Resize
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
            
            # Draw
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Extract
            keypoints = extract_holistic_features(results)
            sequence.append(keypoints)
            
            # Prediction Logic (Every SKIP_FRAMES)
            if frame_count % SKIP_FRAMES == 0 and len(sequence) == 30:
                # Prepare input
                input_data = np.array(sequence) # (30, 226)
                input_data = np.expand_dims(input_data, axis=0) # (1, 30, 226)
                
                res = model.predict(input_data, verbose=0)[0]
                
                pred_id = int(np.argmax(res))
                score = float(res[pred_id])
                
                predictions.append(pred_id)
                
                # Temporal Smoothing
                if len(predictions) == SMOOTHING_WINDOW:
                    # Find most common prediction in last 10
                    counter = collections.Counter(predictions)
                    most_common_pred, count = counter.most_common(1)[0]
                    
                    # Logic: If moda == current AND score > threshold
                    if most_common_pred == pred_id and score > THRESHOLD:
                        inferred_word = id_to_label[pred_id]
                        
                        # Add to sentence if unique
                        if not sentence or inferred_word != sentence[-1]:
                            sentence.append(inferred_word)
                            print(f"➕ Añadido: {inferred_word} (Conf: {score:.2f})")
            
            # Visualization
            # Show last 5 words
            sentence_vis = sentence[-5:]
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence_vis), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show Raw Prediction status
            if len(predictions) > 0:
                last_pred_id = predictions[-1]
                last_word = id_to_label[last_pred_id]
                cv2.putText(image, f"Raw: {last_word}", (10, height - 20) if 'height' in locals() else (10, 340), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow('SignSpeak Phrase Test', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_phrase_test()
