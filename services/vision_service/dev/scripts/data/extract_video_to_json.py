"""
Script simple para extraer landmarks holísticos de un video y generar JSON para Postman.

Uso:
    python extract_video_to_json.py --video "path/to/video.mp4"
"""

import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path

SEQUENCE_LENGTH = 30
POSE_INDICES = list(range(0, 25))  # Upper body landmarks


class HolisticExtractor:
    """Extract holistic landmarks: pose + left hand + right hand."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Extract 226 features from RGB frame."""
        results = self.holistic.process(frame_rgb)
        
        if not self._has_valid_landmarks(results):
            return None
        
        features = []
        
        # Pose landmarks (75 features: 25 landmarks × 3)
        if results.pose_landmarks:
            pose_ref = results.pose_landmarks.landmark[0]  # Nose as reference
            for idx in POSE_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                features.extend([
                    lm.x - pose_ref.x,
                    lm.y - pose_ref.y,
                    lm.z - pose_ref.z
                ])
        else:
            features.extend([0.0] * (len(POSE_INDICES) * 3))
        
        # Left hand (63 features)
        features.extend(self._extract_hand(results.left_hand_landmarks))
        
        # Right hand (63 features)
        features.extend(self._extract_hand(results.right_hand_landmarks))
        
        # Pad to exactly 226
        while len(features) < 226:
            features.append(0.0)
        
        return np.array(features[:226], dtype=np.float32)
    
    def _extract_hand(self, hand_landmarks) -> list:
        """Extract normalized hand landmarks."""
        if not hand_landmarks:
            return [0.0] * 63
        
        wrist = hand_landmarks.landmark[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return features
    
    def _has_valid_landmarks(self, results) -> bool:
        """Check if we have at least pose or one hand."""
        return (
            results.pose_landmarks is not None
            or results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )
    
    def close(self):
        """Release resources."""
        self.holistic.close()


def extract_landmarks_from_video(video_path: Path) -> dict:
    """
    Extrae landmarks de un video y devuelve en formato JSON para Postman.
    
    Returns:
        dict con estructura: {"landmarks": [[226 valores], [226 valores], ...]}
    """
    print(f"\n{'='*60}")
    print("EXTRACTOR DE LANDMARKS HOLÍSTICOS")
    print(f"{'='*60}")
    print(f"[INFO] Video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir el video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] FPS: {fps:.1f}")
    print(f"{'='*60}\n")
    
    # Inicializar extractor
    extractor = HolisticExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    all_landmarks = []
    frames_with_detection = 0
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extraer landmarks
            landmarks = extractor.extract(frame_rgb)
            
            if landmarks is not None:
                all_landmarks.append(landmarks.tolist())
                frames_with_detection += 1
                
                if frame_count % 10 == 0:
                    print(f"  Procesado frame {frame_count}/{total_frames} - Detecciones: {frames_with_detection}")
            else:
                # Sin detección - repetir último o ceros
                if all_landmarks:
                    all_landmarks.append(all_landmarks[-1])
                else:
                    all_landmarks.append([0.0] * 226)
        
    finally:
        cap.release()
        extractor.close()
    
    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    print(f"Total frames procesados: {frame_count}")
    print(f"Frames con detección: {frames_with_detection}")
    print(f"Tasa de detección: {frames_with_detection/frame_count*100:.1f}%")
    print(f"Total landmarks extraídos: {len(all_landmarks)}")
    print(f"{'='*60}\n")
    
    # Para videos con múltiples palabras, guardar TODOS los frames
    # Para modelo de predicción simple, tomar solo primeros 30
    full_sequence = all_landmarks  # Todos los frames
    model_sequence = all_landmarks[:SEQUENCE_LENGTH] if len(all_landmarks) >= SEQUENCE_LENGTH else all_landmarks
    
    # Rellenar si hay menos de 30 frames
    if len(model_sequence) < SEQUENCE_LENGTH:
        model_sequence = model_sequence + [[0.0] * 226] * (SEQUENCE_LENGTH - len(model_sequence))
    
    return {
        "landmarks": full_sequence,  # TODOS los frames para análisis multi-palabra
        "landmarks_30": model_sequence,  # Primeros 30 para predicción simple
        "metadata": {
            "total_frames": frame_count,
            "frames_with_detection": frames_with_detection,
            "detection_rate": f"{frames_with_detection/frame_count*100:.1f}%",
            "total_landmarks_saved": len(full_sequence),
            "sequence_length_for_model": len(model_sequence),
            "features_per_frame": 226
        }
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extrae landmarks holísticos de un video y genera JSON para Postman"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Ruta al video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del archivo JSON de salida (opcional)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"[ERROR] Video no encontrado: {video_path}")
        return
    
    # Extraer landmarks
    result = extract_landmarks_from_video(video_path)
    
    if result is None:
        print("[ERROR] No se pudieron extraer landmarks")
        return
    
    # Determinar archivo de salida
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = video_path.parent / f"{video_path.stem}_landmarks.json"
    
    # Guardar JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"[OK] JSON guardado en: {output_file}")
    print(f"\n{'='*60}")
    print("FORMATO PARA POSTMAN")
    print(f"{'='*60}")
    print("Endpoint: POST http://localhost:8000/predict/holistic")
    print("\nBody (raw JSON):")
    print(json.dumps({"landmarks": result["landmarks"]}, indent=2)[:500] + "...")
    print(f"\n[INFO] Archivo completo en: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

