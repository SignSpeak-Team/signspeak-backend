"""
Convierte el JSON de secuencia completa a formato individual de frames para Postman.

Uso:
    python prepare_postman_json.py --input "CASO_0_blur (1)_landmarks.json" --frame 0
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extrae un frame específico del JSON para Postman"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Archivo JSON con la secuencia completa"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Número de frame a extraer (0-29)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo de salida (opcional)"
    )
    
    args = parser.parse_args()
    
    # Leer JSON completo
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Archivo no encontrado: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validar frame
    if args.frame < 0 or args.frame >= len(data['landmarks']):
        print(f"[ERROR] Frame {args.frame} fuera de rango (0-{len(data['landmarks'])-1})")
        return
    
    # Extraer frame específico
    frame_landmarks = data['landmarks'][args.frame]
    
    # Crear JSON para Postman
    postman_json = {
        "landmarks": frame_landmarks
    }
    
    # Determinar archivo de salida
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_frame{args.frame}.json"
    
    # Guardar
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(postman_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FRAME EXTRAÍDO PARA POSTMAN")
    print(f"{'='*60}")
    print(f"Frame: {args.frame}/{len(data['landmarks'])-1}")
    print(f"Features: {len(frame_landmarks)} valores")
    print(f"Archivo: {output_path}")
    print(f"{'='*60}\n")
    
    # Mostrar formato
    print("FORMATO PARA POSTMAN:")
    print("-" * 60)
    print("Endpoint: POST http://localhost:8000/predict/holistic")
    print("\nBody (raw JSON):")
    print(json.dumps(postman_json, indent=2)[:500] + "...")
    print(f"\n[TIP] Envía múltiples frames (0-29) para llenar el buffer")
    print(f"[TIP] El modelo necesita 30 frames en el buffer para predecir")
    print(f"{'='*60}\n")
    
    # Generar script de múltiples frames
    print("[OPCIÓN AVANZADA] Generar JSONs para todos los frames:")
    print(f"  for i in {{0..29}}; do")
    print(f"    python prepare_postman_json.py --input \"{input_path.name}\" --frame $i")
    print(f"  done")


if __name__ == "__main__":
    main()
