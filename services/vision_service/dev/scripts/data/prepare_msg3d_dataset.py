"""
Prepara el dataset MSG3D: convierte .pkl (objetos MediaPipe) a .npy (tensores numéricos).

Formato de salida para MSG3D GCN:
  data:   (N, C, T, V, M) = (samples, 3, 64, 75, 1)
  labels: (N,) = class IDs

Uso:
  python prepare_msg3d_dataset.py
"""

import os
import sys
import csv
import pickle
import time
import numpy as np
from pathlib import Path

# Fix Windows cp1252 encoding for print statements
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing mediapipe.framework modules.
    
    The .pkl files contain serialized mediapipe.framework.formats.landmark_pb2
    objects (NormalizedLandmarkList, LandmarkList) which can't be deserialized
    on systems where mediapipe.framework is not available. This unpickler
    creates stub classes for those objects, but the 'pose' and 'hands' keys
    use the modern mediapipe.tasks API which loads fine.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Create a stub class for unresolvable mediapipe protobuf types
            return type(f'{module}.{name}', (), {})

# ─── Configuración ───────────────────────────────────────────────────────────

# Paths relativos desde este script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # dev/

RAW_DIR = PROJECT_DIR / "datasets_raw" / "msg3d"
PKL_DIR = RAW_DIR / "MEDIAPIPE"
OUTPUT_DIR = PROJECT_DIR / "datasets_processed" / "msg3d"

# Archivos de anotaciones
TRAIN_LABELS = RAW_DIR / "train_labels.csv"
VAL_LABELS = RAW_DIR / "val_labels.csv"
TEST_LABELS = RAW_DIR / "test_labels.csv"
ANNOTATIONS = RAW_DIR / "videos_ref_annotations.csv"

# Parámetros del dataset
MAX_FRAMES = 64        # Longitud fija de secuencia (pad/truncate)
NUM_POSE = 33          # Landmarks de pose
NUM_HAND = 21          # Landmarks por mano
NUM_JOINTS = NUM_POSE + NUM_HAND * 2  # 75 total
NUM_CHANNELS = 3       # x, y, z
NUM_CLASSES = 300


# ─── Funciones de extracción ─────────────────────────────────────────────────

def extract_frame_keypoints(frame: dict) -> np.ndarray:
    """
    Extrae 75 keypoints (x,y,z) de un frame.
    
    Orden: [33 pose] + [21 left hand] + [21 right hand]
    Si no hay manos detectadas, se rellenan con ceros.
    
    Returns: (75, 3) array
    """
    keypoints = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    
    # --- Pose (33 landmarks) ---
    pose_result = frame.get('pose')
    if pose_result and hasattr(pose_result, 'pose_landmarks'):
        landmarks_list = pose_result.pose_landmarks
        if landmarks_list and len(landmarks_list) > 0:
            for i, lm in enumerate(landmarks_list[0]):
                if i < NUM_POSE:
                    keypoints[i] = [lm.x, lm.y, lm.z]
    
    # --- Manos (21 + 21 landmarks) ---
    hands_result = frame.get('hands')
    if hands_result and hasattr(hands_result, 'hand_landmarks'):
        hand_landmarks = hands_result.hand_landmarks
        handedness = hands_result.handedness if hasattr(hands_result, 'handedness') else []
        
        left_offset = NUM_POSE           # 33
        right_offset = NUM_POSE + NUM_HAND  # 54
        
        for hand_idx in range(len(hand_landmarks)):
            if hand_idx >= len(handedness) or not handedness[hand_idx]:
                continue
            
            # Determinar si es izquierda o derecha
            hand_label = handedness[hand_idx][0].category_name.lower()
            offset = left_offset if hand_label == 'left' else right_offset
            
            for j, lm in enumerate(hand_landmarks[hand_idx]):
                if j < NUM_HAND:
                    keypoints[offset + j] = [lm.x, lm.y, lm.z]
    
    return keypoints


def process_pkl_file(pkl_path: str) -> np.ndarray:
    """
    Carga un .pkl y extrae la secuencia de keypoints.
    
    Returns: (T, 75, 3) array con T = número de frames originales
    """
    with open(pkl_path, 'rb') as f:
        frames = SafeUnpickler(f).load()
    
    sequence = np.zeros((len(frames), NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    
    for t, frame in enumerate(frames):
        sequence[t] = extract_frame_keypoints(frame)
    
    return sequence


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normaliza la secuencia centrando en la nariz (landmark 0 de pose).
    """
    # Usar el primer frame con datos para encontrar el centro
    for t in range(len(sequence)):
        nose = sequence[t, 0]  # Landmark 0 = nariz
        if np.any(nose != 0):
            center = nose.copy()
            break
    else:
        return sequence  # Sin datos válidos
    
    # Centrar toda la secuencia en la nariz
    normalized = sequence.copy()
    mask = np.any(sequence != 0, axis=-1, keepdims=True)  # (T, V, 1)
    normalized = np.where(mask, sequence - center, 0.0)
    
    return normalized


def pad_or_truncate(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ajusta la secuencia a target_len frames.
    - Si es más corta: repite (loop) hasta llenar
    - Si es más larga: trunca desde el centro
    """
    T = sequence.shape[0]
    
    if T == target_len:
        return sequence
    elif T > target_len:
        # Truncar desde el centro (mantener la parte más relevante)
        start = (T - target_len) // 2
        return sequence[start:start + target_len]
    else:
        # Repetir en loop hasta llenar
        result = np.zeros((target_len, *sequence.shape[1:]), dtype=np.float32)
        for i in range(target_len):
            result[i] = sequence[i % T]
        return result


def reshape_to_msg3d(sequence: np.ndarray) -> np.ndarray:
    """
    Convierte de (T, V, C) a formato MSG3D (C, T, V, M).
    M=1 (una persona).
    """
    # (T, V, C) → (C, T, V)
    data = sequence.transpose(2, 0, 1)
    # (C, T, V) → (C, T, V, 1)
    data = np.expand_dims(data, axis=-1)
    return data


# ─── Carga de labels ─────────────────────────────────────────────────────────

def load_labels_csv(csv_path: str) -> dict:
    """
    Lee un CSV de labels (sin header): filename_id,class_id
    Returns: {filename_id: class_id}
    """
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                labels[row[0].strip()] = int(row[1].strip())
    return labels


def load_class_names(annotations_path: str) -> dict:
    """
    Lee videos_ref_annotations.csv: FILENAME,CLASS_ID,LABEL
    Returns: {class_id: label_name}
    """
    class_names = {}
    with open(annotations_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                class_id = int(row[1].strip())
                label = row[2].strip()
                class_names[class_id] = label
    return class_names


# ─── Procesamiento principal ─────────────────────────────────────────────────

def process_split(split_name: str, labels_csv: Path, pkl_dir: Path) -> tuple:
    """
    Procesa un split completo (train/val/test).
    
    Returns: (data_array, labels_array, stats_dict)
    """
    labels_map = load_labels_csv(str(labels_csv))
    
    print(f"\n{'='*60}")
    print(f"  Procesando split: {split_name} ({len(labels_map)} muestras)")
    print(f"{'='*60}")
    
    data_list = []
    labels_list = []
    skipped = 0
    errors = 0
    frame_counts = []
    
    for idx, (filename_id, class_id) in enumerate(labels_map.items()):
        pkl_path = pkl_dir / f"{filename_id}.pkl"
        
        if not pkl_path.exists():
            skipped += 1
            continue
        
        try:
            # 1. Extraer keypoints del .pkl
            sequence = process_pkl_file(str(pkl_path))
            frame_counts.append(len(sequence))
            
            # 2. Normalizar
            sequence = normalize_sequence(sequence)
            
            # 3. Pad/truncate a longitud fija
            sequence = pad_or_truncate(sequence, MAX_FRAMES)
            
            # 4. Reshape a formato MSG3D (C, T, V, M)
            data = reshape_to_msg3d(sequence)
            
            data_list.append(data)
            labels_list.append(class_id)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] Error en {filename_id}: {e}")
        
        # Progreso
        if (idx + 1) % 500 == 0 or idx == len(labels_map) - 1:
            print(f"  Progreso: {idx + 1}/{len(labels_map)}"
                  f" (ok: {len(data_list)}, skip: {skipped}, err: {errors})",
                  flush=True)
    
    data_array = np.array(data_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int64)
    
    stats = {
        'total': len(labels_map),
        'processed': len(data_list),
        'skipped': skipped,
        'errors': errors,
        'frame_counts': frame_counts,
    }
    
    if frame_counts:
        print(f"  [OK] {split_name}: {data_array.shape} | "
              f"frames min={min(frame_counts)}, "
              f"max={max(frame_counts)}, "
              f"mean={np.mean(frame_counts):.1f}")
    
    return data_array, labels_array, stats


def main():
    print("=" * 60)
    print("  MSG3D Dataset Preparation")
    print(f"  Input:  {PKL_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Config: {MAX_FRAMES} frames x {NUM_JOINTS} joints x {NUM_CHANNELS} channels")
    print("=" * 60)
    
    # Verificar que existen los archivos necesarios
    for path, name in [(PKL_DIR, "PKL directory"),
                       (TRAIN_LABELS, "train_labels.csv"),
                       (VAL_LABELS, "val_labels.csv"),
                       (TEST_LABELS, "test_labels.csv"),
                       (ANNOTATIONS, "videos_ref_annotations.csv")]:
        if not path.exists():
            print(f"[ERROR] No encontrado: {name} -> {path}")
            sys.exit(1)
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    all_stats = {}
    
    # Procesar cada split
    splits = [
        ("train", TRAIN_LABELS),
        ("val", VAL_LABELS),
        ("test", TEST_LABELS),
    ]
    
    for split_name, labels_path in splits:
        data, labels, stats = process_split(split_name, labels_path, PKL_DIR)
        all_stats[split_name] = stats
        
        # Guardar
        np.save(OUTPUT_DIR / f"{split_name}_data.npy", data)
        np.save(OUTPUT_DIR / f"{split_name}_labels.npy", labels)
        
        size_mb = data.nbytes / (1024 * 1024)
        print(f"  Guardado: {split_name}_data.npy ({size_mb:.1f} MB)")
    
    # Guardar mapeo de nombres de clases
    class_names = load_class_names(str(ANNOTATIONS))
    with open(OUTPUT_DIR / "label_names.pkl", 'wb') as f:
        pickle.dump(class_names, f)
    print(f"\n  Guardado: label_names.pkl ({len(class_names)} clases)")
    
    elapsed = time.time() - start_time
    
    # Generar reporte
    report_lines = [
        "MSG3D Dataset Processing Report",
        "=" * 40,
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tiempo total: {elapsed:.1f}s",
        f"",
        f"Configuración:",
        f"  MAX_FRAMES: {MAX_FRAMES}",
        f"  NUM_JOINTS: {NUM_JOINTS}  (33 pose + 21 left + 21 right)",
        f"  NUM_CHANNELS: {NUM_CHANNELS}  (x, y, z)",
        f"  NUM_CLASSES: {NUM_CLASSES}",
        f"  Formato salida: (N, C, T, V, M) = (N, {NUM_CHANNELS}, {MAX_FRAMES}, {NUM_JOINTS}, 1)",
        f"",
    ]
    
    for split_name in ["train", "val", "test"]:
        s = all_stats[split_name]
        fc = s['frame_counts']
        report_lines.extend([
            f"{split_name.upper()}:",
            f"  Total en CSV: {s['total']}",
            f"  Procesados:   {s['processed']}",
            f"  Saltados:     {s['skipped']}  (archivo no encontrado)",
            f"  Errores:      {s['errors']}",
            f"  Frames: min={min(fc) if fc else 0}, max={max(fc) if fc else 0}, "
            f"mean={np.mean(fc):.1f}" if fc else "",
            f"",
        ])
    
    # Clases
    report_lines.append(f"Clases ({len(class_names)}):")
    for cid in sorted(class_names.keys()):
        report_lines.append(f"  {cid:3d}: {class_names[cid]}")
    
    report_text = "\n".join(report_lines)
    with open(OUTPUT_DIR / "processing_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n{'='*60}")
    print(f"  COMPLETADO en {elapsed:.1f}s")
    print(f"  Archivos en: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
