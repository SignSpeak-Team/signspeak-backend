import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent  # vision_service/
DATASET_PATH = BASE_PATH / "datasets" / "lsm_zenodo" / "lsm-abc-A"
OUTPUT_PATH = BASE_PATH / "datasets" / "processed"

# Configuracion
BATCH_SIZE = 1000  # Guardar progreso cada N imagenes


class LandmarkExtractor:
    """Extrae landmarks de imagenes usando MediaPipe."""

    def __init__(self):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # Imagenes estaticas, no video
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

        # Estadisticas
        self.processed = 0
        self.failed = 0

    def extract_from_image(self, image_path: Path) -> list | None:
        """
        Extrae landmarks de una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Lista de 63 valores (21 puntos x 3 coords) o None si falla
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Convertir BGR -> RGB (MediaPipe usa RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detectar manos
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                return None

            # Tomar primera mano detectada
            hand_landmarks = results.multi_hand_landmarks[0]

            # Punto de referencia: muneca (landmark 0)
            wrist = hand_landmarks.landmark[0]

            # Extraer y normalizar landmarks
            vector = []
            for lm in hand_landmarks.landmark:
                # Posicion relativa a la muneca
                vector.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            return vector  # 63 valores

        except Exception:
            return None

    def process_split(self, split: str = "train") -> tuple:
        """
        Procesa todas las imagenes de un split (train o test).

        Args:
            split: 'train' o 'test'

        Returns:
            (X, y) donde X son landmarks y y son etiquetas
        """
        split_path = DATASET_PATH / split

        if not split_path.exists():
            print(f"[ERROR] No existe: {split_path}")
            return None, None

        # Obtener carpetas de letras
        letter_folders = sorted([f for f in split_path.iterdir() if f.is_dir()])

        print(f"\n[INFO] Procesando {split.upper()}")
        print(f"[INFO] Letras encontradas: {len(letter_folders)}")
        print(f"[INFO] {[f.name for f in letter_folders]}")

        # Crear mapeo letra -> numero
        label_map = {folder.name: idx for idx, folder in enumerate(letter_folders)}

        X_all = []  # Landmarks
        y_all = []  # Etiquetas

        for folder in letter_folders:
            letter = folder.name
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

            print(f"\n[{letter}] {len(images)} imagenes...")

            letter_landmarks = []
            letter_labels = []

            for i, img_path in enumerate(images):
                # Mostrar progreso cada 500 imagenes
                if i > 0 and i % 500 == 0:
                    print(f"   Procesadas: {i}/{len(images)}")

                landmarks = self.extract_from_image(img_path)

                if landmarks is not None:
                    letter_landmarks.append(landmarks)
                    letter_labels.append(label_map[letter])
                    self.processed += 1
                else:
                    self.failed += 1

            X_all.extend(letter_landmarks)
            y_all.extend(letter_labels)

            print(
                f"   OK: {len(letter_landmarks)} | Fallidas: {len(images) - len(letter_landmarks)}"
            )

        return np.array(X_all), np.array(y_all), label_map

    def close(self):
        """Libera recursos de MediaPipe."""
        self.hands.close()


def main():
    print("=" * 60)
    print("EXTRACTOR DE LANDMARKS - SignSpeak")
    print("=" * 60)
    print(f"[INFO] Dataset: {DATASET_PATH}")
    print(f"[INFO] Output: {OUTPUT_PATH}")

    # Crear carpeta de salida si no existe
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor()

    try:
        # Procesar TRAIN
        X_train, y_train, label_map = extractor.process_split("train")

        if X_train is not None:
            print(f"\n[TRAIN] Shape X: {X_train.shape}")
            print(f"[TRAIN] Shape y: {y_train.shape}")

            # Guardar
            np.savez_compressed(
                OUTPUT_PATH / "landmarks_train.npz", X=X_train, y=y_train
            )
            print("[OK] Guardado: landmarks_train.npz")

        # Procesar TEST
        X_test, y_test, _ = extractor.process_split("test")

        if X_test is not None:
            print(f"\n[TEST] Shape X: {X_test.shape}")
            print(f"[TEST] Shape y: {y_test.shape}")

            np.savez_compressed(OUTPUT_PATH / "landmarks_test.npz", X=X_test, y=y_test)
            print("[OK] Guardado: landmarks_test.npz")

        # Guardar mapeo de etiquetas
        with open(OUTPUT_PATH / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_map, f)
        print("[OK] Guardado: label_encoder.pkl")

        # Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN")
        print("=" * 60)
        print(f"Procesadas: {extractor.processed}")
        print(f"Fallidas: {extractor.failed}")
        print(
            f"Tasa exito: {extractor.processed / (extractor.processed + extractor.failed) * 100:.1f}%"
        )

    finally:
        extractor.close()

    print("\n[DONE] Extraccion completada!")


if __name__ == "__main__":
    main()
