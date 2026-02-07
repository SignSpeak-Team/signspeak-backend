"""
Script de diagnóstico STANDALONE para analizar predicciones.
No depende de prometheus ni otras dependencias del API.
"""

import sys
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np


def load_holistic_model():
    """Carga el modelo holístico y label encoder."""
    from tensorflow import keras
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, InputLayer
    from tensorflow.keras.models import Sequential
    
    model_path = Path(__file__).parent / "models" / "best_model.h5"
    encoder_path = Path(__file__).parent / "models" / "holistic_label_encoder.pkl"
    
    print(f"   Cargando modelo: {model_path.name}")
    
    # Constantes del modelo
    HOLISTIC_SEQUENCE_LENGTH = 30
    HOLISTIC_NUM_FEATURES = 226
    
    try:
        model = keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"   ! Carga estándar falló, reconstruyendo arquitectura...")
        # Reconstruir arquitectura del modelo
        model = Sequential([
            InputLayer(input_shape=(HOLISTIC_SEQUENCE_LENGTH, HOLISTIC_NUM_FEATURES)),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(150, activation="softmax"),
        ])
        model.load_weights(str(model_path))
    
    print(f"   Cargando encoder: {encoder_path.name}")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Invertir el encoder para obtener palabras desde índices
    idx_to_word = {v: k for k, v in label_encoder.items()}
    
    return model, label_encoder, idx_to_word


def extract_holistic_features(frame, holistic):
    """Extrae características holísticas de un frame."""
    import mediapipe as mp
    
    # Convertir a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    
    features = []
    
    # Pose (33 landmarks * 3 coords = 99)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 99)
    
    # Left hand (21 landmarks * 3 coords = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand (21 landmarks * 3 coords = 63)  
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Total: 99 + 63 + 63 = 225 (pero el modelo usa 226?)
    # Agregar un feature extra de presencia general
    has_hands = 1.0 if (results.left_hand_landmarks or results.right_hand_landmarks) else 0.0
    features.append(has_hands)
    
    return np.array(features, dtype=np.float32)


def resample_sequence(sequence, target_length=30):
    """Resamplea una secuencia al tamaño objetivo."""
    current_length = len(sequence)
    if current_length == target_length:
        return np.array(sequence)
    
    indices = np.linspace(0, current_length - 1, target_length).astype(int)
    return np.array([sequence[i] for i in indices])


def predict(model, idx_to_word, features):
    """Realiza predicción y retorna palabra con confianza."""
    # Agregar dimensión de batch
    x = np.expand_dims(features, axis=0)
    
    # Predecir
    probs = model.predict(x, verbose=0)[0]
    
    # Obtener top predicciones
    top_indices = np.argsort(probs)[::-1][:5]
    top_preds = []
    for idx in top_indices:
        word = idx_to_word.get(idx, f"UNKNOWN_{idx}")
        conf = float(probs[idx]) * 100
        top_preds.append({"word": word, "confidence": conf})
    
    return {
        "word": top_preds[0]["word"],
        "confidence": top_preds[0]["confidence"],
        "top_predictions": top_preds
    }


def diagnose_video(video_path: str):
    """Diagnóstica un video mostrando todas las predicciones."""
    import mediapipe as mp
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Video no encontrado: {video_path}")
        return
    
    print(f"\n{'='*70}")
    print("🔍 DIAGNÓSTICO DE PREDICCIONES POR VENTANA")
    print(f"{'='*70}")
    print(f"📹 Video: {video_file.name}")
    print(f"📊 Tamaño: {video_file.stat().st_size / 1024:.1f} KB\n")
    
    # Cargar modelo
    print("⏳ Cargando modelo...")
    model, label_encoder, idx_to_word = load_holistic_model()
    print(f"   Vocabulario: {len(idx_to_word)} palabras\n")
    
    # Abrir video
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Info del video:")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - Frames: {total_frames}")
    print(f"   - Duración: {duration:.1f}s\n")
    
    # Inicializar MediaPipe
    print("⏳ Inicializando MediaPipe...")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Extraer TODOS los features del video
    print("⏳ Extrayendo features de todo el video...")
    all_features = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        features = extract_holistic_features(frame, holistic)
        all_features.append(features)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"   Procesando frame {frame_idx}/{total_frames}...")
    
    cap.release()
    holistic.close()
    print(f"✓ Extracción completa: {len(all_features)} frames\n")
    
    # Aplicar sliding window
    window_size_sec = 2.0
    stride_sec = 0.75
    window_frames = int(window_size_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    print(f"📊 Configuración sliding window:")
    print(f"   - window_size: {window_size_sec}s ({window_frames} frames)")
    print(f"   - stride: {stride_sec}s ({stride_frames} frames)")
    print(f"   - target_frames: 30\n")
    
    # Procesar ventanas
    print(f"{'='*70}")
    print("📋 PREDICCIONES POR VENTANA")
    print(f"{'='*70}\n")
    
    predictions = []
    unique_words = set()
    
    window_num = 0
    for start in range(0, len(all_features), stride_frames):
        end = min(start + window_frames, len(all_features))
        
        # Saltar ventanas muy cortas
        if (end - start) < (window_frames * 0.75):
            continue
        
        window_num += 1
        
        # Obtener features de la ventana y resamplear a 30 frames
        window_features = all_features[start:end]
        resampled = resample_sequence(window_features, 30)
        
        # Predecir
        result = predict(model, idx_to_word, resampled)
        
        start_time = start / fps
        end_time = end / fps
        
        predictions.append({
            "window": window_num,
            "start": start_time,
            "end": end_time,
            "word": result["word"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"]
        })
        
        unique_words.add(result["word"])
        
        # Mostrar
        conf_bar = "█" * int(result["confidence"] / 10)
        print(f"Ventana {window_num:2d}: [{start_time:5.1f}s - {end_time:5.1f}s] "
              f"→ '{result['word']:15s}' {conf_bar} {result['confidence']:.1f}%")
        
        # Top 3
        top3 = result["top_predictions"][:3]
        top_str = ", ".join([f"{p['word']}:{p['confidence']:.0f}%" for p in top3])
        print(f"           Top 3: {top_str}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("📈 RESUMEN")
    print(f"{'='*70}")
    print(f"\n• Total ventanas: {len(predictions)}")
    print(f"• Palabras únicas: {len(unique_words)}")
    print(f"• Palabras: {sorted(unique_words)}")
    
    # Distribución
    word_counts = {}
    for p in predictions:
        word_counts[p["word"]] = word_counts.get(p["word"], 0) + 1
    
    print(f"\n• Distribución:")
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        print(f"   '{word}': {count} ventanas ({pct:.0f}%)")
    
    # Diagnóstico
    print(f"\n{'='*70}")
    print("🔎 DIAGNÓSTICO FINAL")
    print(f"{'='*70}")
    
    if len(unique_words) == 1:
        print("\n⚠️  PROBLEMA: El modelo predice la MISMA palabra en todas las ventanas.")
        print("   Esto puede significar:")
        print("   1. El video realmente solo contiene una seña")
        print("   2. Hay una seña dominante que el modelo reconoce")
        print("   3. Las señas en el video no están en el vocabulario del modelo")
    else:
        print(f"\n✅ El modelo detecta {len(unique_words)} palabras diferentes.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose_video(sys.argv[1])
    else:
        print("Uso: python debug_predictions.py <ruta_video>")
        print("\nEjemplo:")
        print("  python debug_predictions.py dev\\datasets_raw\\videos\\letras\\cases\\CASO_4_blur.mp4")
