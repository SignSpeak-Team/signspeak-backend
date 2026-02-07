"""
Script para enviar secuencia completa al endpoint holístico y obtener predicción.

Envía los 30 frames uno por uno al endpoint /predict/holistic y muestra el resultado.

Uso:
    python test_holistic_prediction.py --input "CASO_0_blur (1)_landmarks.json" --url "http://localhost:8000"
"""

import json
import requests
import time
from pathlib import Path
import argparse


def clear_holistic_buffer(base_url: str):
    """Limpia el buffer holístico antes de enviar nueva secuencia."""
    try:
        response = requests.post(f"{base_url}/predict/holistic/clear")
        if response.status_code == 200:
            print("[OK] Buffer holístico limpiado")
        else:
            print(f"[WARN] No se pudo limpiar buffer: {response.status_code}")
    except Exception as e:
        print(f"[WARN] Error al limpiar buffer: {e}")


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
            return {
                "success": False,
                "status": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Envía secuencia completa al endpoint holístico"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Archivo JSON con landmarks extraídos"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL base del API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay entre frames en segundos (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Leer landmarks
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Archivo no encontrado: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    landmarks_sequence = data['landmarks']
    
    print(f"\n{'='*60}")
    print("TEST DE PREDICCIÓN HOLÍSTICA")
    print(f"{'='*60}")
    print(f"Archivo: {input_path.name}")
    print(f"Total frames: {len(landmarks_sequence)}")
    print(f"Features por frame: {len(landmarks_sequence[0])}")
    print(f"API URL: {args.url}")
    print(f"{'='*60}\n")
    
    # Limpiar buffer
    clear_holistic_buffer(args.url)
    print()
    
    # Enviar frames
    print("Enviando frames...")
    last_result = None
    
    for i, frame_landmarks in enumerate(landmarks_sequence[:30]):  # Solo primeros 30
        print(f"  Frame {i+1}/30...", end=" ")
        
        result = send_frame(args.url, frame_landmarks)
        
        if result["success"]:
            print("✓")
            last_result = result["data"]
        else:
            print(f"✗ Error: {result.get('error', result.get('status'))}")
            if i == 0:  # Si falla el primero, abortar
                print("\n[ERROR] No se pudo conectar al API. Asegúrate de que esté corriendo.")
                return
        
        time.sleep(args.delay)
    
    # Mostrar resultado final
    print(f"\n{'='*60}")
    print("RESULTADO DE LA PREDICCIÓN")
    print(f"{'='*60}")
    
    if last_result:
        print(json.dumps(last_result, indent=2, ensure_ascii=False))
        
        # Resumen
        if last_result.get("accepted"):
            print(f"\n{'='*60}")
            print(f"✅ PREDICCIÓN ACEPTADA")
            print(f"{'='*60}")
            print(f"Palabra: {last_result.get('word', 'N/A')}")
            print(f"Confianza: {last_result.get('confidence', 0):.1f}%")
            print(f"Tiempo: {last_result.get('processing_time_ms', 0):.1f}ms")
            print(f"{'='*60}")
        else:
            print(f"\n⏳ Buffer aún llenándose o confianza insuficiente")
    else:
        print("[ERROR] No se obtuvo respuesta del servidor")
    
    print()


if __name__ == "__main__":
    main()
