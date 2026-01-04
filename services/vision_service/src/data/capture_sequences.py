"""
Script para capturar secuencias de gestos para entrenamiento LSTM.

Uso:
    python capture_sequences.py --gesture hola --sequences 50

Controles:
    - ESPACIO: Iniciar grabación de secuencia
    - Q: Salir
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime

# Configuración
SEQUENCE_LENGTH = 30  # Número de frames por secuencia
DATASETS_PATH = Path(__file__).parent.parent.parent / "datasets" / "sequences"


class SequenceCapture:
    def __init__(self, gesture_name: str, target_sequences: int = 50):
        self.gesture_name = gesture_name
        self.target_sequences = target_sequences
        self.sequences_captured = 0
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Crear directorio para el gesto
        self.gesture_path = DATASETS_PATH / gesture_name
        self.gesture_path.mkdir(parents=True, exist_ok=True)
        
        # Contar secuencias existentes
        existing = list(self.gesture_path.glob("*.npy"))
        self.sequence_counter = len(existing)
        print(f"📁 Directorio: {self.gesture_path}")
        print(f"📊 Secuencias existentes: {self.sequence_counter}")
        
    def extract_landmarks(self, hand_landmarks) -> list:
        """Extrae y normaliza landmarks de la mano."""
        wrist = hand_landmarks.landmark[0]
        
        normalized = []
        for lm in hand_landmarks.landmark:
            normalized.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])
        return normalized
    
    def save_sequence(self, sequence: list):
        """Guarda una secuencia como archivo .npy"""
        sequence_array = np.array(sequence)
        filename = f"seq_{self.sequence_counter:04d}.npy"
        filepath = self.gesture_path / filename
        np.save(filepath, sequence_array)
        
        self.sequence_counter += 1
        self.sequences_captured += 1
        print(f"✅ Guardado: {filename} | Shape: {sequence_array.shape}")
        
    def run(self):
        """Ejecuta la captura interactiva."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            return
        
        print("\n" + "="*50)
        print(f"🎯 Capturando gesto: '{self.gesture_name}'")
        print(f"📌 Meta: {self.target_sequences} secuencias")
        print("="*50)
        print("\n⌨️  Controles:")
        print("   ESPACIO = Iniciar grabación")
        print("   Q = Salir")
        print("\n")
        
        current_sequence = []
        is_recording = False
        
        while self.sequences_captured < self.target_sequences:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Estado visual
            status_color = (0, 0, 255) if is_recording else (200, 200, 200)
            status_text = f"GRABANDO: {len(current_sequence)}/{SEQUENCE_LENGTH}" if is_recording else "Listo (ESPACIO para grabar)"
            
            # Dibujar UI
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesto: {self.gesture_name}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Capturadas: {self.sequences_captured}/{self.target_sequences}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            cv2.putText(frame, status_text, (300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Procesar mano
            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    
                    # Dibujar mano
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Si está grabando, agregar frame
                    if is_recording:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        current_sequence.append(landmarks)
                        
                        # Verificar si completó la secuencia
                        if len(current_sequence) >= SEQUENCE_LENGTH:
                            self.save_sequence(current_sequence)
                            current_sequence = []
                            is_recording = False
            
            # Si no hay mano y está grabando, cancelar
            if is_recording and not hand_detected:
                cv2.putText(frame, "⚠️ Mano perdida!", (200, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Captura de Secuencias - SignSpeak", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not is_recording and hand_detected:
                is_recording = True
                current_sequence = []
                print(f"🔴 Grabando secuencia {self.sequences_captured + 1}...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print(f"✅ Captura completada!")
        print(f"📊 Total secuencias: {self.sequences_captured}")
        print(f"📁 Guardadas en: {self.gesture_path}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Capturar secuencias de gestos")
    parser.add_argument("--gesture", "-g", type=str, required=True,
                       help="Nombre del gesto (ej: hola, gracias, por_favor)")
    parser.add_argument("--sequences", "-s", type=int, default=50,
                       help="Número de secuencias a capturar (default: 50)")
    
    args = parser.parse_args()
    
    capture = SequenceCapture(
        gesture_name=args.gesture,
        target_sequences=args.sequences
    )
    capture.run()


if __name__ == "__main__":
    main()
