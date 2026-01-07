"""
Script para procesar videos y extraer secuencias de landmarks.

Uso:
    python process_videos.py

Estructura esperada:
    datasets/videos/
    ├── J/
    │   ├── video1.mp4
    │   └── video2.mp4
    ├── hola/
    │   └── video1.mp4
    ...

Salida:
    datasets/sequences/
    ├── J/
    │   ├── seq_0000.npy
    │   └── seq_0001.npy
    ...
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
OVERLAP = 15  # Frames de overlap entre secuencias (para mas datos)

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent  # vision_service/
VIDEOS_PATH = BASE_PATH / "datasets" / "videos" / "letras"
SEQUENCES_PATH = BASE_PATH / "datasets" / "sequences"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Extensiones de video soportadas
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def download_model():
    """Descarga el modelo de MediaPipe si no existe."""
    if not MODEL_PATH.exists():
        print("[INFO] Descargando modelo de MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(MODEL_PATH))
        print("[OK] Modelo descargado")


class VideoProcessor:
    def __init__(self):
        download_model()
        
        # MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Estadisticas
        self.total_videos = 0
        self.total_sequences = 0
        self.failed_videos = 0
    
    def extract_landmarks(self, hand_landmarks) -> list:
        """Extrae landmarks normalizados."""
        wrist = hand_landmarks[0]
        vector = []
        for lm in hand_landmarks:
            vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return vector
    
    def process_frame(self, frame) -> list | None:
        """Procesa un frame y extrae landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        
        if result.hand_landmarks:
            return self.extract_landmarks(result.hand_landmarks[0])
        return None
    
    def process_video(self, video_path: Path, output_dir: Path) -> int:
        """
        Procesa un video y genera secuencias de landmarks.
        
        Args:
            video_path: Ruta al video
            output_dir: Directorio donde guardar las secuencias
            
        Returns:
            Numero de secuencias generadas
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"[ERROR] No se puede abrir: {video_path.name}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Procesando: {video_path.name} ({total_frames} frames, {fps:.1f} fps)")
        
        # Extraer landmarks de todos los frames
        all_landmarks = []
        frames_with_hand = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.process_frame(frame)
            if landmarks:
                all_landmarks.append(landmarks)
                frames_with_hand += 1
            else:
                # Sin mano - agregar None o repetir ultimo
                if all_landmarks:
                    all_landmarks.append(all_landmarks[-1])  # Repetir ultimo
                else:
                    all_landmarks.append([0.0] * 63)  # Ceros
        
        cap.release()
        
        if frames_with_hand < SEQUENCE_LENGTH // 2:
            print(f"    [SKIP] Muy pocos frames con mano ({frames_with_hand})")
            return 0
        
        # Crear secuencias con overlap
        sequences_created = 0
        existing = len(list(output_dir.glob("*.npy")))
        
        for start in range(0, len(all_landmarks) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH - OVERLAP):
            sequence = all_landmarks[start:start + SEQUENCE_LENGTH]
            
            if len(sequence) == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence)
                filename = f"seq_{existing + sequences_created:04d}.npy"
                np.save(output_dir / filename, sequence_array)
                sequences_created += 1
        
        print(f"    [OK] {sequences_created} secuencias ({frames_with_hand}/{total_frames} frames con mano)")
        return sequences_created
    
    def scan_structure(self) -> dict:
        """Escanea la estructura de carpetas de videos."""
        structure = {}
        
        if not VIDEOS_PATH.exists():
            return structure
        
        for item in VIDEOS_PATH.iterdir():
            if item.is_dir():
                # Es una carpeta (gesto/palabra)
                videos = [f for f in item.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
                if videos:
                    structure[item.name] = videos
            elif item.suffix.lower() in VIDEO_EXTENSIONS:
                # Video en raiz - usar nombre como gesto
                gesture_name = item.stem
                if gesture_name not in structure:
                    structure[gesture_name] = []
                structure[gesture_name].append(item)
        
        return structure
    
    def process_all(self):
        """Procesa todos los videos encontrados."""
        print("\n" + "="*60)
        print("PROCESADOR DE VIDEOS - SignSpeak")
        print("="*60)
        print(f"[INFO] Buscando videos en: {VIDEOS_PATH}")
        
        # Crear directorio de salida
        SEQUENCES_PATH.mkdir(parents=True, exist_ok=True)
        
        # Escanear estructura
        structure = self.scan_structure()
        
        if not structure:
            print("\n[ERROR] No se encontraron videos")
            print(f"Coloca los videos en: {VIDEOS_PATH}")
            print("Estructura esperada:")
            print("  videos/")
            print("  ├── J/")
            print("  │   └── video.mp4")
            print("  ├── hola/")
            print("  │   └── video.mp4")
            return
        
        print(f"\n[INFO] Encontrados {len(structure)} gestos/palabras:")
        for name, videos in sorted(structure.items()):
            print(f"  - {name}: {len(videos)} videos")
        
        print("\n" + "-"*60)
        
        # Procesar cada gesto
        for gesture_name, videos in sorted(structure.items()):
            print(f"\n[{gesture_name.upper()}]")
            
            # Crear directorio de salida
            output_dir = SEQUENCES_PATH / gesture_name.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            gesture_sequences = 0
            
            for video_path in videos:
                self.total_videos += 1
                sequences = self.process_video(video_path, output_dir)
                
                if sequences > 0:
                    gesture_sequences += sequences
                    self.total_sequences += sequences
                else:
                    self.failed_videos += 1
            
            print(f"  Total: {gesture_sequences} secuencias para '{gesture_name}'")
        
        # Resumen final
        print("\n" + "="*60)
        print("RESUMEN")
        print("="*60)
        print(f"Videos procesados: {self.total_videos}")
        print(f"Videos fallidos: {self.failed_videos}")
        print(f"Secuencias totales: {self.total_sequences}")
        print(f"Guardadas en: {SEQUENCES_PATH}")
        print("="*60)


def main():
    processor = VideoProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()
