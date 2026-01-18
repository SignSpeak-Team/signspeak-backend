"""
Script para probar el modelo de palabras contra un video.
Procesa el video, detecta palabras y genera un reporte.

Uso:
    python test_words_from_video.py <path_to_video>
"""

import cv2
import numpy as np
import pickle
import sys
import time
from collections import deque, Counter
from pathlib import Path
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Add src to path (dev/demos -> dev -> vision_service -> src)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import (
    WORDS_MODEL_PATH, WORDS_LABEL_ENCODER_PATH,
    HAND_LANDMARKER_PATH, SEQUENCE_LENGTH
)

# Cargar modelo
print("="*60)
print("PRUEBA DE MODELO DE PALABRAS CONTRA VIDEO")
print("="*60)

print("\nCargando modelo de 249 palabras...")
words_model = keras.models.load_model(str(WORDS_MODEL_PATH))
with open(WORDS_LABEL_ENCODER_PATH, "rb") as f:
    words_labels = pickle.load(f)
words_idx_to_word = {v: k for k, v in words_labels.items()}
print(f"✓ Modelo cargado: {len(words_labels)} palabras")

# Lista de estados de México para filtrar
ESTADOS_MEXICO = {
    "aguascalientes", "baja california", "baja california sur", "campeche",
    "chiapas", "chihuahua", "ciudad de mexico", "coahuila", "colima",
    "durango", "guanajuato", "guerrero", "hidalgo", "jalisco", "mexico",
    "michoacan", "morelos", "nayarit", "nuevo leon", "oaxaca", "puebla",
    "queretaro", "quintana roo", "san luis potosi", "sinaloa", "sonora",
    "tabasco", "tamaulipas", "tlaxcala", "veracruz", "yucatan", "zacatecas"
}

# Configurar MediaPipe
print("Configurando MediaPipe...")
base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe listo")


def extract_landmarks(hand_landmarks):
    """Extrae landmarks normalizados."""
    wrist = hand_landmarks[0]
    vector = []
    for lm in hand_landmarks:
        vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return vector


def process_video(video_path: str, skip_frames: int = 2):
    """
    Procesa un video y detecta palabras.
    
    Args:
        video_path: Ruta al video
        skip_frames: Procesar cada N frames (para acelerar)
    
    Returns:
        Lista de detecciones (timestamp, palabra, confianza)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Frames totales: {total_frames}")
    print(f"   Duración: {duration/60:.1f} minutos")
    print(f"   Procesando cada {skip_frames} frames...")
    
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    detections = []
    
    frame_count = 0
    hands_detected_count = 0
    predictions_made = 0
    last_prediction_frame = -SEQUENCE_LENGTH  # Para evitar predicciones consecutivas
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames para acelerar
        if frame_count % skip_frames != 0:
            continue
        
        # Progreso
        if frame_count % (int(total_frames/20) or 1) == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (frame_count/total_frames)) - elapsed if frame_count > 0 else 0
            print(f"   Progreso: {progress:.1f}% | ETA: {eta:.0f}s | Manos: {hands_detected_count} | Predicciones: {predictions_made}")
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detectar manos
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            hands_detected_count += 1
            hand_landmarks = result.hand_landmarks[0]
            landmarks = extract_landmarks(hand_landmarks)
            frame_buffer.append(landmarks)
            
            # Predecir cuando el buffer está lleno y han pasado suficientes frames
            if len(frame_buffer) >= SEQUENCE_LENGTH and (frame_count - last_prediction_frame) >= SEQUENCE_LENGTH:
                sequence = np.array(list(frame_buffer))
                prediction = words_model.predict(np.array([sequence]), verbose=0)
                
                class_idx = np.argmax(prediction)
                confidence = float(prediction[0][class_idx] * 100)
                word = words_idx_to_word.get(class_idx, "???")
                
                timestamp = frame_count / fps
                detections.append({
                    "timestamp": timestamp,
                    "frame": frame_count,
                    "word": word,
                    "confidence": confidence
                })
                
                predictions_made += 1
                last_prediction_frame = frame_count
                
                # Solo mostrar predicciones con confianza MUY alta
                if confidence > 85:
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    print(f"   [{mins:02d}:{secs:02d}] {word} ({confidence:.1f}%)")
        else:
            # Sin mano, limpiar buffer
            if len(frame_buffer) > 0:
                frame_buffer.clear()
    
    cap.release()
    
    elapsed_total = time.time() - start_time
    print(f"\n✓ Procesamiento completado en {elapsed_total:.1f}s")
    print(f"   Frames procesados: {frame_count // skip_frames}")
    print(f"   Manos detectadas: {hands_detected_count}")
    print(f"   Predicciones realizadas: {predictions_made}")
    
    return detections


def generate_report(detections: list, min_confidence: float = 50.0):
    """Genera un reporte de las detecciones."""
    
    print("\n" + "="*60)
    print("REPORTE DE DETECCIONES")
    print("="*60)
    
    if not detections:
        print("No se detectaron palabras.")
        return
    
    # Filtrar por confianza
    filtered = [d for d in detections if d["confidence"] >= min_confidence]
    print(f"\nDetecciones con confianza >= {min_confidence}%: {len(filtered)}/{len(detections)}")
    
    # Contar palabras
    word_counts = Counter(d["word"] for d in filtered)
    
    print(f"\n📊 TOP 20 PALABRAS DETECTADAS:")
    print("-"*40)
    for word, count in word_counts.most_common(20):
        avg_conf = np.mean([d["confidence"] for d in filtered if d["word"] == word])
        is_estado = "🇲🇽" if word.lower() in ESTADOS_MEXICO else "  "
        print(f"{is_estado} {word:20} | x{count:3} | conf: {avg_conf:.1f}%")
    
    # Estadísticas de estados de México
    estados_detected = {d["word"].lower() for d in filtered if d["word"].lower() in ESTADOS_MEXICO}
    
    print(f"\n🇲🇽 ESTADOS DE MÉXICO DETECTADOS: {len(estados_detected)}/32")
    print("-"*40)
    for estado in sorted(estados_detected):
        estado_dets = [d for d in filtered if d["word"].lower() == estado]
        count = len(estado_dets)
        avg_conf = np.mean([d["confidence"] for d in estado_dets])
        max_conf = max(d["confidence"] for d in estado_dets)
        print(f"  ✓ {estado:20} | x{count:3} | avg: {avg_conf:.1f}% | max: {max_conf:.1f}%")
    
    # Estados no detectados
    estados_missing = ESTADOS_MEXICO - estados_detected
    if estados_missing:
        print(f"\n❌ ESTADOS NO DETECTADOS: {len(estados_missing)}")
        for estado in sorted(estados_missing):
            print(f"     {estado}")
    
    # Estadísticas generales
    all_confidences = [d["confidence"] for d in filtered]
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   Confianza promedio: {np.mean(all_confidences):.1f}%")
    print(f"   Confianza máxima: {max(all_confidences):.1f}%")
    print(f"   Confianza mínima: {min(all_confidences):.1f}%")
    
    return {
        "total_detections": len(detections),
        "filtered_detections": len(filtered),
        "estados_detected": len(estados_detected),
        "word_counts": dict(word_counts)
    }


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_words_from_video.py <path_to_video>")
        print("\nEjemplo:")
        print("  python test_words_from_video.py C:/Videos/estados_mexico.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Error: El archivo no existe: {video_path}")
        sys.exit(1)
    
    # Procesar video
    detections = process_video(video_path, skip_frames=2)
    
    # Generar reporte (filtrar a 85%)
    report = generate_report(detections, min_confidence=85.0)
    
    print("\n" + "="*60)
    print("FIN DEL ANÁLISIS")
    print("="*60)


if __name__ == "__main__":
    main()
