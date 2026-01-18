"""
Demo de detección de palabras en tiempo real con WordBuffer.
Detecta 249 palabras del vocabulario LSM y acumula frases.
"""

import cv2
import numpy as np
import pickle
from collections import deque
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import (
    WORDS_MODEL_PATH, WORDS_LABEL_ENCODER_PATH,
    HAND_LANDMARKER_PATH, SEQUENCE_LENGTH
)
from core.word_buffer import WordBuffer

# Cargar modelo de palabras
print("Cargando modelo de 249 palabras...")
words_model = keras.models.load_model(str(WORDS_MODEL_PATH))
with open(WORDS_LABEL_ENCODER_PATH, "rb") as f:
    words_labels = pickle.load(f)
words_idx_to_word = {v: k for k, v in words_labels.items()}
print(f"✓ Modelo cargado: {len(words_labels)} palabras")

# Configurar MediaPipe
base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Buffers
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
word_buffer = WordBuffer(cooldown_seconds=2.0, min_confidence=70.0)

# Colores
GREEN = (0, 255, 0)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


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
    
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                   (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20),(0,17)]
    for start, end in connections:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(image, (x1,y1), (x2,y2), WHITE, 2)


print("\n" + "="*50)
print("SIGNSPEAK - DEMO PALABRAS (249)")
print("="*50)
print("Detecta palabras y acumula frases con WordBuffer")
print("Cooldown: 2s | Min confianza: 70%")
print("="*50)
print("\nControles:")
print("  Q = Salir")
print("  C = Limpiar frase")
print("  R = Reset buffer")

print("\nIniciando camara...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Estado
current_word = ""
current_confidence = 0
prediction_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    result = detector.detect(mp_image)
    hand_detected = False
    
    if result.hand_landmarks:
        hand_detected = True
        hand_landmarks = result.hand_landmarks[0]
        draw_landmarks(frame, hand_landmarks)
        
        landmarks = extract_landmarks(hand_landmarks)
        frame_buffer.append(landmarks)
        
        # Predecir cuando el buffer está lleno
        if len(frame_buffer) >= SEQUENCE_LENGTH and prediction_cooldown <= 0:
            sequence = np.array(list(frame_buffer))
            prediction = words_model.predict(np.array([sequence]), verbose=0)
            
            class_idx = np.argmax(prediction)
            confidence = float(prediction[0][class_idx] * 100)
            word = words_idx_to_word.get(class_idx, "???")
            
            current_word = word
            current_confidence = confidence
            
            # Intentar agregar al buffer (aplica filtros)
            was_accepted = word_buffer.add_detection(word, confidence)
            
            if was_accepted:
                print(f"✓ Aceptada: '{word}' ({confidence:.1f}%)")
            
            prediction_cooldown = 5  # Frames de espera
        
        if prediction_cooldown > 0:
            prediction_cooldown -= 1
    else:
        frame_buffer.clear()
        current_word = ""
        current_confidence = 0
    
    # ============================================================
    # UI
    # ============================================================
    
    # Header
    cv2.rectangle(frame, (0, 0), (640, 70), BLACK, -1)
    cv2.rectangle(frame, (0, 68), (640, 70), PURPLE, -1)
    cv2.putText(frame, "SignSpeak - PALABRAS", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(frame, f"Vocabulario: {len(words_labels)} palabras", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURPLE, 1)
    
    # Palabra actual
    if hand_detected and current_word:
        cv2.rectangle(frame, (400, 10), (630, 60), PURPLE, -1)
        # Truncar palabra si es muy larga
        display_word = current_word[:10] if len(current_word) > 10 else current_word
        cv2.putText(frame, display_word, (410, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"{current_confidence:.0f}%", (410, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    
    # Buffer progress
    buffer_progress = len(frame_buffer) / SEQUENCE_LENGTH
    cv2.rectangle(frame, (10, 75), (200, 85), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 75), (int(10 + 190 * buffer_progress), 85), PURPLE, -1)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", (210, 83),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # FRASE ACUMULADA (zona principal)
    cv2.rectangle(frame, (10, 380), (630, 445), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 380), (630, 382), PURPLE, -1)
    cv2.putText(frame, "FRASE:", (20, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURPLE, 1)
    
    phrase = word_buffer.get_phrase()
    if phrase:
        # Dividir frase si es muy larga
        words = phrase.split()
        line1 = " ".join(words[:5])
        line2 = " ".join(words[5:10]) if len(words) > 5 else ""
        cv2.putText(frame, line1, (20, 425), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        if line2:
            cv2.putText(frame, line2, (20, 440), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    else:
        cv2.putText(frame, "(haz una seña para empezar)", (20, 425), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Stats
    stats = word_buffer.get_statistics()
    cv2.putText(frame, f"Aceptadas: {stats['total_accepted']}", (450, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
    cv2.putText(frame, f"Filtradas: {stats['rejected_by_cooldown'] + stats['rejected_by_confidence']}", (450, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Footer
    cv2.rectangle(frame, (0, 450), (640, 480), BLACK, -1)
    status = "Mano detectada" if hand_detected else "Muestra tu mano"
    color = GREEN if hand_detected else (100, 100, 100)
    cv2.putText(frame, status, (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, "Q=Salir  C=Limpiar  R=Reset", (400, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    cv2.imshow("SignSpeak - Palabras", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer.clear()
        print("Frase limpiada")
    elif key == ord('r'):
        frame_buffer.clear()
        word_buffer.clear()
        current_word = ""
        current_confidence = 0
        print("Buffer reseteado")

cap.release()
cv2.destroyAllWindows()

# Mostrar resumen final
print("\n" + "="*50)
print("RESUMEN")
print("="*50)
stats = word_buffer.get_statistics()
print(f"Frase final: {word_buffer.get_phrase()}")
print(f"Total recibidas: {stats['total_received']}")
print(f"Total aceptadas: {stats['total_accepted']}")
print(f"Tasa aceptación: {stats['acceptance_rate']:.1f}%")
print("="*50)
