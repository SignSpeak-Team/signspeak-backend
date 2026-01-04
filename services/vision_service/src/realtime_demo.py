import cv2
import numpy as np
import pickle
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Rutas
DATA_PATH = "../datasets/processed"
MODEL_PATH = "hand_landmarker.task"

# Descargar modelo si no existe
import urllib.request
import os

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo descargado")

# Cargar modelo de clasificacion
print("Cargando modelo de clasificacion...")
sign_model = keras.models.load_model(f"{DATA_PATH}/sign_model.keras")

# Cargar mapeo
with open(f"{DATA_PATH}/label_encoder.pkl", "rb") as f:
    label_map = pickle.load(f)
idx_to_letter = {v: k for k, v in label_map.items()}

# Configurar MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# Colores
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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

print("Iniciando camara...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nSignSpeak LSM - Presiona Q para salir\n")

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
    
    letter = ""
    confidence = 0
    hand_detected = False
    
    if result.hand_landmarks:
        hand_detected = True
        hand_landmarks = result.hand_landmarks[0]
        
        # Dibujar
        draw_landmarks(frame, hand_landmarks)
        
        # Extraer y predecir
        landmarks = extract_landmarks(hand_landmarks)
        pred = sign_model.predict(np.array([landmarks]), verbose=0)
        pred_class = np.argmax(pred)
        confidence = pred[0][pred_class] * 100
        letter = idx_to_letter[pred_class]
    
    # UI
    cv2.rectangle(frame, (0, 0), (640, 70), BLACK, -1)
    cv2.rectangle(frame, (0, 68), (640, 70), GREEN, -1)
    
    cv2.putText(frame, "SignSpeak LSM", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(frame, "Traductor de Lenguaje de Senas", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    if hand_detected:
        cv2.rectangle(frame, (480, 10), (630, 60), GREEN, -1)
        cv2.putText(frame, letter, (530, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 3)
        cv2.putText(frame, f"{confidence:.0f}%", (490, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
    else:
        cv2.rectangle(frame, (480, 10), (630, 60), (50, 50, 50), -1)
        cv2.putText(frame, "---", (530, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
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