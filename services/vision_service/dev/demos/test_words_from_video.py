"""
Demo de reconocimiento de palabras usando videos del dataset.
Permite probar el modelo sin necesidad de hacer señas.

Uso:
    python test_words_from_video.py
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys

# Añadir path
sys.path.insert(0, str(Path(__file__).parent / "services" / "vision_service" / "src"))

from core.predictor import get_predictor

# Configuración
DATASET_PATH = Path("services/vision_service/dev/datasets_raw/videos/palabras")
SEQUENCE_LENGTH = 15


def extract_landmarks_from_video(video_path, detector):
    """Extrae landmarks de un video."""
    cap = cv2.VideoCapture(str(video_path))
    landmarks_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detectar manos
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist = hand[0]
            
            # Normalizar respecto a muñeca
            vector = []
            for lm in hand:
                vector.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
            
            landmarks_list.append(np.array(vector))
    
    cap.release()
    return landmarks_list


def test_word_recognition():
    """Prueba el reconocimiento de palabras con videos del dataset."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    print("=" * 70)
    print("🧪 TEST DE RECONOCIMIENTO DE PALABRAS")
    print("=" * 70)
    
    # Cargar predictor
    print("\n📥 Cargando predictor...")
    predictor = get_predictor()
    print("✓ Predictor cargado")
    
    # Configurar MediaPipe
    base_options = python.BaseOptions(
        model_asset_path="services/vision_service/models/hand_landmarker.task"
    )
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Encontrar palabras disponibles
    word_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir() and f.name.isdigit()])
    
    print(f"\n📂 Palabras disponibles: {len(word_folders)}")
    print("\nSelecciona una palabra para probar:")
    print("-" * 70)
    
    # Mostrar primeras 20 palabras
    for i, folder in enumerate(word_folders[:20], 1):
        # Leer nombre de la palabra del Excel o del folder
        samples = list(folder.glob("*/"))
        if samples:
            print(f"  {i}. ID {folder.name} ({len(samples)} muestras)")
    
    print(f"\n  ... y {len(word_folders) - 20} más")
    print("-" * 70)
    
    # Seleccionar palabra (por ahora usar la primera)
    selected_folder = word_folders[0]
    samples = sorted([s for s in selected_folder.iterdir() if s.is_dir()])
    
    if not samples:
        print("❌ No se encontraron muestras")
        return
    
    # Usar primera muestra
    sample_folder = samples[0]
    frames = sorted(sample_folder.glob("*.jpg"))
    
    print(f"\n🎬 Procesando: {selected_folder.name}/{sample_folder.name}")
    print(f"   Frames: {len(frames)}")
    
    # Extraer landmarks de cada frame
    print("\n🔍 Extrayendo landmarks...")
    landmarks_sequence = []
    
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist = hand[0]
            
            vector = []
            for lm in hand:
                vector.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
            
            landmarks_sequence.append(np.array(vector))
    
    print(f"   ✓ {len(landmarks_sequence)} landmarks extraídos")
    
    # Hacer predicción
    print("\n🤖 Haciendo predicción...")
    
    # Resetear buffer
    predictor.reset_buffer("words")
    
    # Alimentar frames uno por uno
    results = []
    for i, landmarks in enumerate(landmarks_sequence):
        result = predictor.predict_word(landmarks)
        if result:
            results.append(result)
    
    if results:
        # Tomar última predicción (más confiable)
        final_result = results[-1]
        
        print("\n" + "=" * 70)
        print("✅ RESULTADO")
        print("=" * 70)
        print(f"   Palabra predicha: {final_result['word']}")
        print(f"   Confianza: {final_result['confidence']:.2f}%")
        print(f"   Frames procesados: {len(landmarks_sequence)}")
        print(f"   Predicciones generadas: {len(results)}")
        print("=" * 70)
    else:
        print("❌ No se pudo generar predicción (menos de 15 frames)")
    
    detector.close()


if __name__ == "__main__":
    try:
        test_word_recognition()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
