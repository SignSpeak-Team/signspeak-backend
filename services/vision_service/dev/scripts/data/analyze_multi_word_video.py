"""
Analiza video con múltiples palabras usando ventana deslizante.

El modelo holístico necesita 30 frames para hacer una predicción.
Este script divide el video en ventanas de 30 frames para detectar todas las palabras.

Uso:
    python analyze_multi_word_video.py --input "CASO_4_blur_landmarks.json" --url "..."
"""

import json
import requests
import time
from pathlib import Path
import argparse
from collections import deque


def clear_holistic_buffer(base_url: str):
    """Limpia el buffer holístico."""
    try:
        response = requests.post(f"{base_url}/predict/holistic/clear")
        return response.status_code == 200
    except:
        return False


def send_frame(base_url: str, landmarks: list) -> dict:
    """Envía un frame al endpoint holístico."""
    payload = {"landmarks": landmarks}
    
    try:
        response = requests.post(
            f"{base_url}/predict/holistic",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "status": response.status_code, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_window(base_url: str, window_frames: list, window_start: int) -> dict:
    """Analiza una ventana de 30 frames."""
    # Limpiar buffer
    clear_holistic_buffer(base_url)
    
    # Enviar frames de la ventana
    last_result = None
    for frame_landmarks in window_frames:
        result = send_frame(base_url, frame_landmarks)
        if result["success"]:
            last_result = result["data"]
        time.sleep(0.05)  # Delay mínimo
    
    return {
        "window_start": window_start,
        "window_end": window_start + len(window_frames) - 1,
        "result": last_result
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analiza video con múltiples palabras usando ventana deslizante"
    )
    parser.add_argument("--input", type=str, required=True, help="Archivo JSON con landmarks")
    parser.add_argument("--url", type=str, required=True, help="URL base del API")
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Tamaño de ventana en frames (default: 30)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Salto entre ventanas en frames (default: 20, overlap=10)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=70.0,
        help="Confianza mínima para aceptar predicción (default: 70.0)"
    )
    
    args = parser.parse_args()
    
    # Leer landmarks
    input_path = Path(args.input)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_frames = data['landmarks']
    
    print(f"\n{'='*70}")
    print("ANÁLISIS DE VIDEO CON MÚLTIPLES PALABRAS")
    print(f"{'='*70}")
    print(f"Archivo: {input_path.name}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Ventana: {args.window_size} frames")
    print(f"Stride: {args.stride} frames (overlap: {args.window_size - args.stride})")
    print(f"Confianza mínima: {args.min_confidence}%")
    print(f"API: {args.url}")
    print(f"{'='*70}\n")
    
    # Generar ventanas
    windows = []
    for start in range(0, len(all_frames) - args.window_size + 1, args.stride):
        end = start + args.window_size
        windows.append((start, end))
    
    print(f"Ventanas generadas: {len(windows)}")
    print(f"Analizando...\n")
    
    # Analizar cada ventana
    detections = []
    for i, (start, end) in enumerate(windows):
        print(f"[Ventana {i+1}/{len(windows)}] Frames {start}-{end-1}...", end=" ")
        
        window_frames = all_frames[start:end]
        result = analyze_window(args.url, window_frames, start)
        
        if result["result"] and result["result"].get("accepted"):
            word = result["result"]["word"]
            confidence = result["result"]["confidence"]
            
            if confidence >= args.min_confidence:
                print(f"✓ {word} ({confidence:.1f}%)")
                detections.append({
                    "window": (start, end-1),
                    "word": word,
                    "confidence": confidence,
                    "time_sec": (start / 30, end / 30)  # Asumiendo 30 FPS
                })
            else:
                print(f"⊗ {word} ({confidence:.1f}% - baja confianza)")
        else:
            print("✗ Sin detección")
        
        time.sleep(0.2)
    
    # Resumen
    print(f"\n{'='*70}")
    print("PALABRAS DETECTADAS")
    print(f"{'='*70}")
    
    if detections:
        # Agrupar palabras consecutivas iguales
        unique_words = []
        last_word = None
        
        for det in detections:
            if det["word"] != last_word:
                unique_words.append(det)
                last_word = det["word"]
        
        print(f"\nTotal detecciones: {len(detections)}")
        print(f"Palabras únicas (sin repeticiones): {len(unique_words)}\n")
        
        for i, det in enumerate(unique_words, 1):
            start_time, end_time = det["time_sec"]
            print(f"{i}. \"{det['word']}\" - {det['confidence']:.1f}% confianza")
            print(f"   Frames: {det['window'][0]}-{det['window'][1]} | Tiempo: {start_time:.1f}s - {end_time:.1f}s\n")
        
        # Generar frase
        phrase = " ".join([det["word"] for det in unique_words])
        print(f"{'='*70}")
        print(f"FRASE DETECTADA: \"{phrase}\"")
        print(f"{'='*70}")
    else:
        print("No se detectaron palabras con confianza suficiente.")
    
    print()


if __name__ == "__main__":
    main()
