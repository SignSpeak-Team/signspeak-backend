"""
Script mejorado para capturar secuencias de gestos para entrenamiento LSTM.

Uso:
    python capture_sequences.py

Controles:
    - ESPACIO: Iniciar grabacion (con countdown)
    - D: Eliminar ultima secuencia
    - N: Nueva palabra
    - L: Listar palabras guardadas
    - Q: Salir
"""

import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuracion
SEQUENCE_LENGTH = 30  # Frames por secuencia
COUNTDOWN_SECONDS = 3  # Segundos antes de grabar
DATASETS_PATH = Path(__file__).parent.parent.parent / "datasets" / "sequences"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Colores
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
YELLOW = (0, 255, 255)


def download_model():
    """Descarga el modelo de MediaPipe si no existe."""
    if not MODEL_PATH.exists():
        print("Descargando modelo de MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(MODEL_PATH))
        print("Modelo descargado")


class SequenceCapture:
    def __init__(self):
        download_model()
        
        # MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Estado
        self.gesture_name = ""
        self.gesture_path = None
        self.sequences_captured = 0
        self.target_sequences = 50
        self.sequence_counter = 0
        
        # Crear directorio base
        DATASETS_PATH.mkdir(parents=True, exist_ok=True)
    
    def list_gestures(self):
        """Lista todos los gestos guardados."""
        gestures = []
        if DATASETS_PATH.exists():
            for folder in DATASETS_PATH.iterdir():
                if folder.is_dir():
                    count = len(list(folder.glob("*.npy")))
                    gestures.append((folder.name, count))
        return sorted(gestures)
    
    def set_gesture(self, name: str, target: int = 50):
        """Configura el gesto actual."""
        self.gesture_name = name.lower().replace(" ", "_")
        self.target_sequences = target
        self.gesture_path = DATASETS_PATH / self.gesture_name
        self.gesture_path.mkdir(parents=True, exist_ok=True)
        
        existing = list(self.gesture_path.glob("*.npy"))
        self.sequence_counter = len(existing)
        self.sequences_captured = 0
        
        print(f"\n[INFO] Gesto: {self.gesture_name}")
        print(f"[INFO] Existentes: {self.sequence_counter}")
        print(f"[INFO] Meta: {self.target_sequences}")
    
    def extract_landmarks(self, hand_landmarks) -> list:
        """Extrae landmarks normalizados."""
        wrist = hand_landmarks[0]
        vector = []
        for lm in hand_landmarks:
            vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return vector
    
    def save_sequence(self, sequence: list):
        """Guarda una secuencia."""
        sequence_array = np.array(sequence)
        filename = f"seq_{self.sequence_counter:04d}.npy"
        filepath = self.gesture_path / filename
        np.save(filepath, sequence_array)
        
        self.sequence_counter += 1
        self.sequences_captured += 1
        print(f"[OK] Guardado: {filename} | Shape: {sequence_array.shape}")
    
    def delete_last(self):
        """Elimina la ultima secuencia guardada."""
        if self.sequence_counter > 0:
            self.sequence_counter -= 1
            filename = f"seq_{self.sequence_counter:04d}.npy"
            filepath = self.gesture_path / filename
            if filepath.exists():
                os.remove(filepath)
                if self.sequences_captured > 0:
                    self.sequences_captured -= 1
                print(f"[DELETED] {filename}")
                return True
        return False
    
    def draw_landmarks(self, image, hand_landmarks):
        """Dibuja landmarks en la imagen."""
        h, w, _ = image.shape
        
        # Conexiones de la mano
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]
        
        # Dibujar conexiones
        for start, end in connections:
            x1 = int(hand_landmarks[start].x * w)
            y1 = int(hand_landmarks[start].y * h)
            x2 = int(hand_landmarks[end].x * w)
            y2 = int(hand_landmarks[end].y * h)
            cv2.line(image, (x1,y1), (x2,y2), WHITE, 2)
        
        # Dibujar puntos
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 5, GREEN, -1)
    
    def draw_progress_bar(self, image, current, total, y_pos):
        """Dibuja barra de progreso."""
        bar_width = 200
        bar_height = 20
        x_start = 430
        
        # Fondo
        cv2.rectangle(image, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), GRAY, -1)
        
        # Progreso
        progress = int((current / max(total, 1)) * bar_width)
        cv2.rectangle(image, (x_start, y_pos), (x_start + progress, y_pos + bar_height), GREEN, -1)
        
        # Texto
        cv2.putText(image, f"{current}/{total}", (x_start + bar_width + 10, y_pos + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    def run(self):
        """Ejecuta la captura interactiva."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("[ERROR] No se puede abrir la camara")
            return
        
        # Pedir nombre del gesto al inicio
        gesture = input("\nNombre del gesto (ej: hola, gracias): ").strip()
        if not gesture:
            print("Nombre invalido")
            return
        
        target = input("Numero de secuencias a capturar (default 50): ").strip()
        target = int(target) if target.isdigit() else 50
        
        self.set_gesture(gesture, target)
        
        print("\n" + "="*50)
        print("CONTROLES")
        print("="*50)
        print("ESPACIO = Grabar secuencia")
        print("D       = Eliminar ultima")
        print("N       = Nueva palabra")
        print("L       = Listar palabras")
        print("Q       = Salir")
        print("="*50 + "\n")
        
        current_sequence = []
        is_recording = False
        countdown = 0
        countdown_start = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar manos
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)
            
            hand_detected = False
            hand_landmarks = None
            
            if result.hand_landmarks:
                hand_detected = True
                hand_landmarks = result.hand_landmarks[0]
                self.draw_landmarks(frame, hand_landmarks)
            
            # Panel superior
            cv2.rectangle(frame, (0, 0), (640, 80), BLACK, -1)
            
            # Titulo y gesto
            cv2.putText(frame, "SignSpeak - Captura", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            cv2.putText(frame, f"Gesto: {self.gesture_name}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)
            
            # Barra de progreso
            total_needed = self.target_sequences
            current_total = self.sequence_counter
            self.draw_progress_bar(frame, current_total, total_needed, 30)
            
            # Countdown
            if countdown > 0:
                elapsed = cv2.getTickCount() / cv2.getTickFrequency() - countdown_start
                remaining = COUNTDOWN_SECONDS - int(elapsed)
                
                if remaining > 0:
                    # Mostrar countdown grande
                    cv2.putText(frame, str(remaining), (290, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, YELLOW, 10)
                else:
                    # Iniciar grabacion
                    countdown = 0
                    is_recording = True
                    current_sequence = []
            
            # Grabando
            if is_recording:
                cv2.rectangle(frame, (0, 65), (640, 80), RED, -1)
                cv2.putText(frame, f"GRABANDO: {len(current_sequence)}/{SEQUENCE_LENGTH}", (220, 77),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                
                if hand_detected:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    current_sequence.append(landmarks)
                    
                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        self.save_sequence(current_sequence)
                        current_sequence = []
                        is_recording = False
                else:
                    # Mano perdida
                    cv2.putText(frame, "MANO PERDIDA!", (230, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
            else:
                # Estado listo
                status = "Mano detectada - ESPACIO para grabar" if hand_detected else "Muestra tu mano"
                color = GREEN if hand_detected else GRAY
                cv2.putText(frame, status, (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Panel inferior
            cv2.rectangle(frame, (0, 450), (640, 480), BLACK, -1)
            cv2.putText(frame, "ESPACIO=Grabar  D=Borrar  N=Nueva  L=Lista  Q=Salir", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
            
            cv2.imshow("SignSpeak - Captura de Secuencias", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' ') and hand_detected and not is_recording and countdown == 0:
                # Iniciar countdown
                countdown = COUNTDOWN_SECONDS
                countdown_start = cv2.getTickCount() / cv2.getTickFrequency()
            
            elif key == ord('d'):
                self.delete_last()
            
            elif key == ord('n'):
                # Nueva palabra
                cap.release()
                cv2.destroyAllWindows()
                
                gesture = input("\nNuevo gesto: ").strip()
                if gesture:
                    target = input("Secuencias (default 50): ").strip()
                    target = int(target) if target.isdigit() else 50
                    self.set_gesture(gesture, target)
                
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            elif key == ord('l'):
                # Listar gestos
                gestures = self.list_gestures()
                print("\n" + "="*40)
                print("GESTOS GUARDADOS")
                print("="*40)
                if gestures:
                    for name, count in gestures:
                        print(f"  {name}: {count} secuencias")
                else:
                    print("  (ninguno)")
                print("="*40 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n[DONE] Sesion terminada")
        gestures = self.list_gestures()
        if gestures:
            print("\nResumen de gestos:")
            for name, count in gestures:
                print(f"  - {name}: {count} secuencias")


def main():
    print("\n" + "="*50)
    print("SIGNSPEAK - CAPTURA DE SECUENCIAS")
    print("="*50)
    
    capture = SequenceCapture()
    capture.run()


if __name__ == "__main__":
    main()
