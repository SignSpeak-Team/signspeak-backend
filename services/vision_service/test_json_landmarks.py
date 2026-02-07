"""
Prueba directa del modelo con landmarks pre-extraídos.
Compara prediccion de landmarks JSON vs extracción de video en vivo.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential


def load_model():
    """Carga el modelo holístico."""
    HOLISTIC_SEQUENCE_LENGTH = 30
    HOLISTIC_NUM_FEATURES = 226
    
    model_path = Path(__file__).parent / "models" / "best_model.h5"
    encoder_path = Path(__file__).parent / "models" / "holistic_label_encoder.pkl"
    
    print("📥 Cargando modelo...")
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
    
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    idx_to_word = {v: k for k, v in label_encoder.items()}
    
    print(f"   ✓ Modelo cargado ({len(idx_to_word)} palabras)")
    return model, idx_to_word


def resample_sequence(sequence, target_length=30):
    """Resamplea a 30 frames."""
    current = len(sequence)
    if current == target_length:
        return np.array(sequence)
    indices = np.linspace(0, current - 1, target_length).astype(int)
    return np.array([sequence[i] for i in indices])


def predict_sequence(model, idx_to_word, features):
    """Predice una secuencia de 30 frames."""
    x = np.expand_dims(features, axis=0)
    probs = model.predict(x, verbose=0)[0]
    
    top_indices = np.argsort(probs)[::-1][:5]
    return [{
        "word": idx_to_word.get(idx, f"UNK_{idx}"),
        "confidence": float(probs[idx]) * 100
    } for idx in top_indices]


def test_with_json_landmarks(json_path: str, model, idx_to_word):
    """Prueba con landmarks pre-extraídos de un JSON."""
    print(f"\n{'='*60}")
    print("TEST CON LANDMARKS PRE-EXTRAÍDOS (JSON)")
    print(f"{'='*60}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    landmarks = data['landmarks']
    print(f"📁 Archivo: {Path(json_path).name}")
    print(f"📊 Frames: {len(landmarks)}")
    print(f"📊 Features por frame: {len(landmarks[0])}")
    
    # Aplicar sliding window
    window_size = 30
    stride = 15  # 50% overlap
    fps_estimate = 30
    
    print(f"\n⏳ Procesando con sliding window...")
    print(f"   Window: {window_size} frames, Stride: {stride} frames\n")
    
    predictions = []
    unique_words = set()
    
    for start in range(0, len(landmarks), stride):
        end = min(start + window_size, len(landmarks))
        
        if (end - start) < (window_size * 0.75):
            continue
        
        window = landmarks[start:end]
        resampled = resample_sequence(window, 30)
        
        top5 = predict_sequence(model, idx_to_word, resampled)
        predictions.append({
            "start_frame": start,
            "end_frame": end,
            "word": top5[0]["word"],
            "confidence": top5[0]["confidence"],
            "top3": top5[:3]
        })
        
        unique_words.add(top5[0]["word"])
        
        # Mostrar
        conf_bar = "█" * int(top5[0]["confidence"] / 10)
        print(f"Frames [{start:3d}-{end:3d}] → '{top5[0]['word']:15s}' {conf_bar} {top5[0]['confidence']:.1f}%")
        top3_str = ", ".join([f"{p['word']}:{p['confidence']:.0f}%" for p in top5[:3]])
        print(f"           Top 3: {top3_str}")
    
    # Resumen
    print(f"\n{'='*60}")
    print("📈 RESUMEN - LANDMARKS JSON")
    print(f"{'='*60}")
    print(f"• Ventanas analizadas: {len(predictions)}")
    print(f"• Palabras únicas: {len(unique_words)}")
    print(f"• Palabras: {sorted(unique_words)}")
    
    word_counts = {}
    for p in predictions:
        word_counts[p["word"]] = word_counts.get(p["word"], 0) + 1
    
    print(f"\n• Distribución:")
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        print(f"   '{word}': {count} ({pct:.0f}%)")
    
    return predictions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_json_landmarks.py <archivo_landmarks.json>")
        print("\nEjemplo:")
        print("  python test_json_landmarks.py dev\\datasets_raw\\videos\\letras\\cases\\CASO_4_blur_landmarks.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"❌ Archivo no encontrado: {json_path}")
        sys.exit(1)
    
    model, idx_to_word = load_model()
    test_with_json_landmarks(json_path, model, idx_to_word)
    
    print(f"\n{'='*60}")
    print("✅ Test completado")
    print(f"{'='*60}\n")
