"""
Prueba con el modelo de 249 palabras.
Este modelo usa solo landmarks de una mano (63 features) vs holístico (226).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_words_model():
    """Carga el modelo de 249 palabras."""
    from tensorflow import keras
    
    model_path = Path(__file__).parent / "models" / "words_model.keras"
    encoder_path = Path(__file__).parent / "models" / "words_label_encoder.pkl"
    
    print("📥 Cargando modelo de 249 palabras...")
    model = keras.models.load_model(str(model_path), compile=False)
    
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    idx_to_word = {v: k for k, v in label_encoder.items()}
    
    print(f"   ✓ Modelo cargado ({len(idx_to_word)} palabras)")
    print(f"   Input shape: {model.input_shape}")
    return model, idx_to_word


def extract_right_hand_from_holistic(holistic_features):
    """
    Extrae los landmarks de la mano derecha del formato holístico.
    Holístico: [pose(99) + left_hand(63) + right_hand(63) + extra(1)] = 226
    La mano derecha está en posición 99+63 = 162 hasta 162+63 = 225
    """
    if len(holistic_features) != 226:
        print(f"   Warning: Expected 226 features, got {len(holistic_features)}")
    
    # Posiciones: pose=0-98, left_hand=99-161, right_hand=162-224
    right_hand = holistic_features[162:225]
    return right_hand


def resample_sequence(sequence, target_length):
    """Resamplea a N frames."""
    current = len(sequence)
    if current == target_length:
        return np.array(sequence)
    if current == 0:
        return np.zeros((target_length, len(sequence[0]) if sequence else 63))
    indices = np.linspace(0, current - 1, target_length).astype(int)
    return np.array([sequence[i] for i in indices])


def predict_sequence(model, idx_to_word, features):
    """Predice una secuencia."""
    x = np.expand_dims(features, axis=0)
    probs = model.predict(x, verbose=0)[0]
    
    top_indices = np.argsort(probs)[::-1][:5]
    return [{
        "word": idx_to_word.get(idx, f"UNK_{idx}"),
        "confidence": float(probs[idx]) * 100
    } for idx in top_indices]


def test_with_json_landmarks(json_path: str, model, idx_to_word):
    """Prueba con landmarks pre-extraídos."""
    print(f"\n{'='*60}")
    print("TEST CON MODELO DE 249 PALABRAS")
    print(f"{'='*60}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    landmarks = data['landmarks']
    print(f"📁 Archivo: {Path(json_path).name}")
    print(f"📊 Frames originales: {len(landmarks)}")
    print(f"📊 Features por frame: {len(landmarks[0])}")
    
    # Extraer solo mano derecha de cada frame
    print("\n⏳ Extrayendo landmarks de mano derecha...")
    right_hand_landmarks = []
    for frame in landmarks:
        right_hand = extract_right_hand_from_holistic(frame)
        right_hand_landmarks.append(right_hand)
    
    print(f"   ✓ Extraídos: {len(right_hand_landmarks)} frames de 63 features cada uno")
    
    # Aplicar sliding window (15 frames para el modelo de palabras)
    window_size = 15  # El modelo usa 15 frames
    stride = 7  # ~50% overlap
    
    print(f"\n⏳ Procesando con sliding window...")
    print(f"   Window: {window_size} frames, Stride: {stride} frames\n")
    
    predictions = []
    unique_words = set()
    
    for start in range(0, len(right_hand_landmarks), stride):
        end = min(start + window_size, len(right_hand_landmarks))
        
        if (end - start) < (window_size * 0.75):
            continue
        
        window = right_hand_landmarks[start:end]
        resampled = resample_sequence(window, window_size)
        
        # Verificar que no sea todos ceros (mano no detectada)
        if np.sum(np.abs(resampled)) < 0.01:
            print(f"Frames [{start:3d}-{end:3d}] → [SIN MANO DETECTADA]")
            continue
        
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
    print("📈 RESUMEN - MODELO 249 PALABRAS")
    print(f"{'='*60}")
    print(f"• Ventanas analizadas: {len(predictions)}")
    print(f"• Palabras únicas: {len(unique_words)}")
    print(f"• Palabras: {sorted(unique_words)}")
    
    if predictions:
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
        print("Uso: python test_249_model.py <archivo_landmarks.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"❌ Archivo no encontrado: {json_path}")
        sys.exit(1)
    
    model, idx_to_word = load_words_model()
    test_with_json_landmarks(json_path, model, idx_to_word)
    
    print(f"\n{'='*60}")
    print("✅ Test completado")
    print(f"{'='*60}\n")
