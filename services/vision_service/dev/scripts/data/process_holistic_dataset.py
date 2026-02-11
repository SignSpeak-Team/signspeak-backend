"""
Procesador de dataset MSLwords1 con MediaPipe Holistic.
Extrae 144 features por frame (manos + brazos) y genera secuencias normalizadas .npy.

Uso:
    python process_holistic_dataset.py              # 249 clases
    python process_holistic_dataset.py --subset 20  # Primeras 20 clases
"""

import argparse
import pickle
import sys
import io
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Forzar UTF-8 en stdout (Windows usa cp1252 por defecto)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# --- Rutas ---
BASE_PATH = Path(__file__).parent.parent.parent
DATASET_PATH = BASE_PATH / "datasets_raw" / "MSLwords1"
OUTPUT_PATH = BASE_PATH / "datasets_processed" / "holistic_palabras"
EXCEL_PATH = DATASET_PATH / "clases.xlsx"

# --- Constantes ---
TARGET_SIZE = (640, 480)
MIN_VALID_FRAMES = 3
POSE_ARM_INDICES = [11, 12, 13, 14, 15, 16]
NUM_HAND_FEATURES = 21 * 3
NUM_FEATURES = (NUM_HAND_FEATURES * 2) + (len(POSE_ARM_INDICES) * 3)  # 144


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize a 640x480 con padding + gamma correction + CLAHE."""
    h, w = image.shape[:2]
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
    x_off = (TARGET_SIZE[0] - new_w) // 2
    y_off = (TARGET_SIZE[1] - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Gamma correction adaptativo para imagenes oscuras
    mean = canvas.mean()
    if mean < 50:
        gamma = 0.4
    elif mean < 80:
        gamma = 0.6
    else:
        gamma = 1.0

    if gamma < 1.0:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        canvas = cv2.LUT(canvas, table)

    # CLAHE sobre canal L
    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_and_scale(raw_features: list[float]) -> list[float]:
    """Escala features a rango [-1, 1]."""
    arr = np.array(raw_features)
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr = arr / max_val
    return arr.tolist()


def extract_hand_features(hand_landmarks) -> list[float]:
    """Extrae 63 features de una mano, centrados a la muñeca y escalados."""
    if not hand_landmarks:
        return [0.0] * NUM_HAND_FEATURES

    wrist = hand_landmarks.landmark[0]
    raw = []
    for lm in hand_landmarks.landmark:
        raw.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return normalize_and_scale(raw)


def extract_pose_arm_features(pose_landmarks) -> list[float]:
    """Extrae 18 features de brazos (hombros, codos, muñecas)."""
    if not pose_landmarks:
        return [0.0] * (len(POSE_ARM_INDICES) * 3)

    pose = pose_landmarks.landmark
    ref_x = (pose[11].x + pose[12].x) / 2
    ref_y = (pose[11].y + pose[12].y) / 2
    ref_z = (pose[11].z + pose[12].z) / 2

    raw = []
    for idx in POSE_ARM_INDICES:
        lm = pose[idx]
        raw.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
    return normalize_and_scale(raw)


def extract_landmarks(holistic, image_rgb: np.ndarray) -> np.ndarray | None:
    """Extrae 144 features. Retorna None si no detecta manos."""
    results = holistic.process(image_rgb)

    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return None

    features = (
        extract_hand_features(results.left_hand_landmarks)
        + extract_hand_features(results.right_hand_landmarks)
        + extract_pose_arm_features(results.pose_landmarks)
    )
    return np.array(features, dtype=np.float32)


def load_class_mapping() -> dict[int, str]:
    """Carga mapeo ID -> palabra desde clases.xlsx."""
    df = pd.read_excel(EXCEL_PATH)
    mapping = {}
    for _, row in df.iterrows():
        word_id = int(row.iloc[0])
        word = str(row.iloc[1]).strip().lower().replace(" ", "_")
        word = word.replace("(", "").replace(")", "").replace("/", "_")
        mapping[word_id] = word
    return mapping


def pad_or_truncate(frames: list[np.ndarray], target_len: int) -> np.ndarray:
    """Normaliza secuencia: trunca uniformemente o rellena con zeros."""
    arr = np.array(frames)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        indices = np.linspace(0, len(arr) - 1, target_len, dtype=int)
        return arr[indices]
    padded = np.zeros((target_len, NUM_FEATURES), dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


class HolisticDatasetProcessor:
    def __init__(self, subset: int | None = None):
        self.id_to_word = load_class_mapping()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.3,
        )
        self.subset = subset
        self.sequence_lengths: list[int] = []
        self.stats = {
            "words": 0, "samples_ok": 0, "samples_fail": 0,
            "frames_discarded": 0, "sequences": 0, "detail": [],
        }

    def _process_sample(self, sample_path: Path) -> list[np.ndarray] | None:
        frames = sorted(sample_path.glob("*.jpg"))
        valid_landmarks = []

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                self.stats["frames_discarded"] += 1
                continue

            enhanced = preprocess_image(image)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            landmarks = extract_landmarks(self.holistic, rgb)

            if landmarks is not None:
                valid_landmarks.append(landmarks)
            else:
                self.stats["frames_discarded"] += 1

        if len(valid_landmarks) < MIN_VALID_FRAMES:
            return None
        return valid_landmarks

    def _process_word(self, word_id: int, word_folder: Path) -> tuple[str, list]:
        word_name = self.id_to_word.get(word_id, f"unknown_{word_id}")
        samples = sorted([s for s in word_folder.iterdir() if s.is_dir()])
        raw_sequences = []

        for sample_path in samples:
            result = self._process_sample(sample_path)
            if result:
                raw_sequences.append(result)
                self.sequence_lengths.append(len(result))
                self.stats["samples_ok"] += 1
            else:
                self.stats["samples_fail"] += 1

        return word_name, raw_sequences

    def _compute_sequence_length(self) -> int:
        if not self.sequence_lengths:
            return 15

        p90 = int(np.percentile(self.sequence_lengths, 90))
        print(f"\n[STATS] Longitudes de secuencia:")
        print(f"   Rango: {min(self.sequence_lengths)}-{max(self.sequence_lengths)}")
        print(f"   Media: {np.mean(self.sequence_lengths):.1f} | Mediana: {int(np.median(self.sequence_lengths))}")
        print(f"   -> Seleccionado (P90): {p90}")
        return p90

    def process_all(self):
        print("\n" + "=" * 70)
        print(" PROCESADOR HOLISTICO - MSLwords1")
        print("=" * 70)

        word_folders = sorted([
            f for f in DATASET_PATH.iterdir() if f.is_dir() and f.name.isdigit()
        ])
        if self.subset:
            word_folders = word_folders[:self.subset]
            print(f"[*] Modo subset: {self.subset} clases")

        total = len(word_folders)
        print(f"[i] Dataset: {DATASET_PATH}")
        print(f"[i] Clases: {total} | Features: {NUM_FEATURES}\n")

        start = datetime.now()
        all_data = []

        for i, folder in enumerate(word_folders, 1):
            word_id = int(folder.name)
            word_name, raw_seqs = self._process_word(word_id, folder)
            all_data.append((word_name, raw_seqs))
            self.stats["words"] += 1
            self.stats["detail"].append((word_id, word_name, len(raw_seqs)))

            elapsed = (datetime.now() - start).total_seconds()
            eta = (elapsed / i) * (total - i)
            pct = int(i / total * 30)
            bar = "#" * pct + "." * (30 - pct)
            icon = "OK" if raw_seqs else "SKIP"
            print(f"   [{bar}] {i:3d}/{total} | {icon:4s} {word_id:03d}: {word_name:20s} | {len(raw_seqs):2d} seq | ETA: {int(eta)}s")

        # Normalizar y guardar
        seq_len = self._compute_sequence_length()
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        words_with_data = []
        for word_name, raw_seqs in all_data:
            if not raw_seqs:
                continue
            words_with_data.append(word_name)
            out_dir = OUTPUT_PATH / "sequences" / word_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for j, seq in enumerate(raw_seqs):
                normalized = pad_or_truncate(seq, seq_len)
                np.save(out_dir / f"seq_{j:04d}.npy", normalized)
                self.stats["sequences"] += 1

        label_map = {w: i for i, w in enumerate(sorted(set(words_with_data)))}
        with open(OUTPUT_PATH / "holistic_label_encoder.pkl", "wb") as f:
            pickle.dump(label_map, f)

        config = {"num_features": NUM_FEATURES, "sequence_length": seq_len, "num_classes": len(label_map)}
        with open(OUTPUT_PATH / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        elapsed_total = (datetime.now() - start).total_seconds()
        self._print_summary(seq_len, len(label_map), elapsed_total)
        self._save_report(seq_len)

    def _print_summary(self, seq_len: int, num_classes: int, elapsed: float):
        s = self.stats
        print("\n" + "=" * 70)
        print(" COMPLETADO")
        print("=" * 70)
        print(f"   Palabras: {s['words']} | OK: {s['samples_ok']} | Fallidas: {s['samples_fail']}")
        print(f"   Frames descartados: {s['frames_discarded']}")
        print(f"   Secuencias: {s['sequences']} | Features: {NUM_FEATURES} | Seq length: {seq_len}")
        print(f"   Clases con datos: {num_classes}")
        print(f"   Tiempo: {int(elapsed)}s ({elapsed / 60:.1f} min)")
        print(f"   Salida: {OUTPUT_PATH}")
        print("=" * 70)

    def _save_report(self, seq_len: int):
        report_path = OUTPUT_PATH / "processing_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Procesamiento Holistico MSLwords1 - {datetime.now():%Y-%m-%d %H:%M}\n")
            f.write(f"Features: {NUM_FEATURES} | Seq length: {seq_len}\n\n")
            for word_id, word_name, n_seqs in self.stats["detail"]:
                f.write(f"  {word_id:03d}: {word_name:25s} -> {n_seqs} seq\n")

    def close(self):
        self.holistic.close()


def main():
    parser = argparse.ArgumentParser(description="Procesar MSLwords1 con MediaPipe Holistic")
    parser.add_argument("--subset", type=int, default=None, help="Solo primeras N clases")
    args = parser.parse_args()

    processor = HolisticDatasetProcessor(subset=args.subset)
    try:
        processor.process_all()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
