import cv2
import numpy as np
import pickle
from collections import deque
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Rutas
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent  # vision_service/
DATA_PATH = BASE_PATH / "datasets" / "processed"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Descargar modelo si no existe
import urllib.request
import os

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo descargado")

# ============================================================
# CARGAR MODELOS
# ============================================================

# Modelo estatico (21 letras)
print("Cargando modelo estatico...")
static_model = keras.models.load_model(f"{DATA_PATH}/sign_model.keras")
with open(f"{DATA_PATH}/label_encoder.pkl", "rb") as f:
    static_labels = pickle.load(f)
static_idx_to_letter = {v: k for k, v in static_labels.items()}
print(f"  Letras estaticas: {list(static_labels.keys())}")

# Modelo dinamico LSTM (6 letras)
print("Cargando modelo LSTM...")
lstm_model = keras.models.load_model(f"{DATA_PATH}/lstm_letters.keras")
with open(f"{DATA_PATH}/lstm_label_encoder.pkl", "rb") as f:
    lstm_labels = pickle.load(f)
lstm_idx_to_letter = {v: k for k, v in lstm_labels.items()}
DYNAMIC_LETTERS = set(lstm_labels.keys())
print(f"  Letras dinamicas: {list(lstm_labels.keys())}")

# Configurar MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ============================================================
# CONFIGURACION LSTM
# ============================================================
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Colores
GREEN = (0, 255, 0)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)


def extract_landmarks(hand_landmarks):
    """Extrae landmarks normalizados."""
    wrist = hand_landmarks[0]
    vector = []
    for lm in hand_landmarks:
        vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return vector


def draw_landmarks(image, hand_landmarks):
    """Dibuja landmarks en la imagen."""
    h, w, _ = image.shape
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 5, GREEN, -1)
    
    # Conectar puntos principales
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                   (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20),(0,17)]
    for start, end in connections:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(image, (x1,y1), (x2,y2), WHITE, 2)


def detect_movement(buffer):
    """Detecta si hay movimiento significativo en el buffer."""
    if len(buffer) < 10:
        return False
    
    first = np.array(buffer[0])
    last = np.array(buffer[-1])
    movement = np.linalg.norm(last - first)
    return movement > 0.15  # Umbral de movimiento


print("\n" + "="*50)
print("SIGNSPEAK LSM - DEMO COMPLETO")
print("="*50)
print("Letras estaticas: A-Z (sin J,K,Ñ,Q,X,Z)")
print("Letras dinamicas: J, K, Ñ, Q, X, Z")
print("="*50)

print("\nIniciando camara...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Presiona Q para salir\n")

# Estado
current_letter = ""
current_confidence = 0
is_dynamic = False
lstm_prediction_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertir a MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detectar manos
    result = detector.detect(mp_image)
    
    hand_detected = False
    
    if result.hand_landmarks:
        hand_detected = True
        hand_landmarks = result.hand_landmarks[0]
        
        # Dibujar
        draw_landmarks(frame, hand_landmarks)
        
        # Extraer landmarks
        landmarks = extract_landmarks(hand_landmarks)
        
        # Agregar al buffer para LSTM
        frame_buffer.append(landmarks)
        
        # Prediccion estatica (siempre)
        static_pred = static_model.predict(np.array([landmarks]), verbose=0)
        static_class = np.argmax(static_pred)
        static_conf = static_pred[0][static_class] * 100
        static_letter = static_idx_to_letter[static_class]
        
        # Prediccion dinamica (si hay suficientes frames y movimiento)
        if len(frame_buffer) >= SEQUENCE_LENGTH and lstm_prediction_cooldown <= 0:
            if detect_movement(frame_buffer):
                # Predecir con LSTM
                sequence = np.array(list(frame_buffer))
                lstm_pred = lstm_model.predict(np.array([sequence]), verbose=0)
                lstm_class = np.argmax(lstm_pred)
                lstm_conf = lstm_pred[0][lstm_class] * 100
                lstm_letter = lstm_idx_to_letter[lstm_class]
                
                # Si LSTM tiene alta confianza, usar esa prediccion
                if lstm_conf > 60:  # Umbral mas permisivo
                    current_letter = lstm_letter
                    current_confidence = lstm_conf
                    is_dynamic = True
                    lstm_prediction_cooldown = 5  # Cooldown mas corto (solo 5 frames)
                    # NO limpiamos buffer para permitir deteccion continua
        
        # Si no es dinamica, usar estatica
        if not is_dynamic or lstm_prediction_cooldown <= 0:
            current_letter = static_letter
            current_confidence = static_conf
            is_dynamic = False
        
        lstm_prediction_cooldown -= 1
    else:
        # Sin mano, limpiar buffer
        frame_buffer.clear()
        current_letter = ""
        current_confidence = 0
        is_dynamic = False
    
    # ============================================================
    # UI
    # ============================================================
    
    # Header
    cv2.rectangle(frame, (0, 0), (640, 70), BLACK, -1)
    header_color = ORANGE if is_dynamic else GREEN
    cv2.rectangle(frame, (0, 68), (640, 70), header_color, -1)
    
    cv2.putText(frame, "SignSpeak LSM", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    
    mode_text = "DINAMICO (LSTM)" if is_dynamic else "ESTATICO (Dense)"
    mode_color = ORANGE if is_dynamic else GREEN
    cv2.putText(frame, mode_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
    
    # Letra predicha
    if hand_detected and current_letter:
        box_color = ORANGE if is_dynamic else GREEN
        cv2.rectangle(frame, (480, 10), (630, 60), box_color, -1)
        cv2.putText(frame, current_letter, (530, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 3)
        cv2.putText(frame, f"{current_confidence:.0f}%", (490, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
    else:
        cv2.rectangle(frame, (480, 10), (630, 60), (50, 50, 50), -1)
        cv2.putText(frame, "---", (530, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    # Barra de buffer LSTM
    buffer_progress = len(frame_buffer) / SEQUENCE_LENGTH
    cv2.rectangle(frame, (10, 75), (200, 85), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 75), (int(10 + 190 * buffer_progress), 85), ORANGE, -1)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", (210, 83),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Footer
    cv2.rectangle(frame, (0, 450), (640, 480), BLACK, -1)
    status = "Mano detectada" if hand_detected else "Muestra tu mano"
    color = GREEN if hand_detected else (100, 100, 100)
    cv2.putText(frame, status, (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, "Q = Salir", (560, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    cv2.imshow("SignSpeak - LSM", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Demo terminada")