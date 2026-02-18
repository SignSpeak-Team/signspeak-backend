import cv2
import numpy as np
import torch
import mediapipe as mp
import pickle
from pathlib import Path
import os
from collections import deque
import sys

# 1. SETUP PATHS
# Use relative paths from project root (CWD)
PROJECT_DIR = Path.cwd()
MODEL_PATH = PROJECT_DIR / "services" / "vision_service" / "dev" / "training_output" / "msl150_subset_lstm" / "best.pth"
LABELS_PATH = PROJECT_DIR / "services" / "vision_service" / "dev" / "datasets_processed" / "msl150_subset" / "label_names.pkl"
VIDEO_PATH = PROJECT_DIR / "services" / "vision_service" / "dev" / "datasets_raw" / "videos" / "letras" / "cases" / "CASO_0_blur (1).mp4"

# Add project root to path to import model class
sys.path.insert(0, str(PROJECT_DIR))
try:
    from services.vision_service.dev.scripts.training.train_msl150_lstm import HolisticLSTM
except ImportError:
    # Try different path structure if running from elsewhere
    try:
        from scripts.training.train_msl150_lstm import HolisticLSTM
    except:
        print("⚠️ Could not import HolisticLSTM. Defining it inline.")
        import torch.nn as nn
        class HolisticLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
                super(HolisticLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
                output_dim = hidden_size * 2 if bidirectional else hidden_size
                self.fc = nn.Linear(output_dim, num_classes)
            def forward(self, x):
                h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

# 2. CONFIG
BUFFER_SIZE = 64  # PyTorch model expects 64 frames
DISPLAY_TOP_N = 3
CONFIDENCE_THRESHOLD = 0.5

# 3. LOAD MODEL & LABELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔄 Cargando modelo PyTorch desde: {MODEL_PATH} en {device}")

try:
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
    print(f"✅ Modelo cargado (Val Acc: {checkpoint['val_acc']*100:.2f}%)")
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
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                # Reiniciar video
                print("🔁 Reiniciando video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_buffer.clear()
                continue
            
            # Process frame
            # 1. Resize for speed
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
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            display_frame = image
            
            # Extract features
            features = extract_holistic_features(results)
            frame_buffer.append(features)
            
            # Predict
            if len(frame_buffer) == BUFFER_SIZE:
                # Prepare input (1, 64, 226)
                input_data = np.array(frame_buffer)
                input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    topk_probs, topk_ids = torch.topk(probs, DISPLAY_TOP_N, dim=1)
                
                # Move to cpu
                top_probs = topk_probs[0].cpu().numpy()
                top_ids = topk_ids[0].cpu().numpy()
                
                # Update current prediction if confidence is high
                if top_probs[0] > CONFIDENCE_THRESHOLD:
                    current_prediction = id_to_label[top_ids[0]]
                    current_conf = top_probs[0]
                
                # Display predictions on screen
                y_pos = 50
                for i in range(DISPLAY_TOP_N):
                    label = id_to_label[top_ids[i]]
                    conf = top_probs[i]
                    color = (0, 255, 0) if i == 0 else (0, 150, 255)
                    text = f"{label}: {conf*100:.1f}%"
                    cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 25
            else:
                cv2.putText(display_frame, f"Buffering... {len(frame_buffer)}/{BUFFER_SIZE}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show "Best Guess" permanently
            cv2.putText(display_frame, f"DETECTADO: {current_prediction} ({current_conf*100:.1f}%)", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('SignSpeak PyTorch Video Test', display_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_test()
