import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from datetime import datetime
import urllib.request

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent  # vision_service/dev
DATASET_PATH = BASE_PATH / "datasets_raw" / "videos" / "palabras"
OUTPUT_PATH = BASE_PATH / "datasets_processed" / "palabras"
EXCEL_PATH = DATASET_PATH / "classes.xlsx"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Configuración
MIN_FRAMES = 5  # Mínimo de frames con mano detectada
SEQUENCE_LENGTH = 15  # Normalizar todas las secuencias a esta longitud


def download_model():
    """Descarga el modelo de MediaPipe si no existe."""
    if not MODEL_PATH.exists():
        print("[INFO] Descargando modelo de MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(MODEL_PATH))
        print("[OK] Modelo descargado")


class WordsDatasetProcessor:
    """Procesa el dataset de 249 palabras LSM."""
    
    def __init__(self):
        # Cargar mapeo de Excel
        print("[INFO] Cargando mapeo de palabras...")
        df = pd.read_excel(EXCEL_PATH)
        
        # Crear mapeo ID -> palabra (limpiar nombres)
        self.id_to_word = {}
        for _, row in df.iterrows():
            word_id = int(row.iloc[0])
            word = str(row.iloc[1]).strip().lower().replace(" ", "_")
            # Limpiar caracteres especiales
            word = word.replace("(", "").replace(")", "").replace("/", "_")
            self.id_to_word[word_id] = word
        
        print(f"  Palabras cargadas: {len(self.id_to_word)}")
        
        # Descargar modelo si no existe
        download_model()
        
        # Inicializar MediaPipe Vision Tasks API
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2  # Algunas señas usan 2 manos
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Estadísticas
        self.stats = {
            "total_words": 0,
            "total_samples": 0,
            "total_sequences": 0,
            "failed_samples": 0,
            "words_processed": []
        }
    
    def extract_landmarks(self, image_path: Path) -> list | None:
        """Extrae landmarks de una imagen usando Vision Tasks API."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convertir BGR -> RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detectar manos
            result = self.detector.detect(mp_image)
            
            if not result.hand_landmarks:
                return None
            
            # Tomar primera mano
            hand = result.hand_landmarks[0]
            wrist = hand[0]
            
            # Normalizar respecto a muñeca
            vector = []
            for lm in hand:
                vector.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
            
            return vector  # 63 valores
            
        except Exception:
            return None
    
    def normalize_sequence(self, frames: list) -> np.ndarray | None:
        """Normaliza una secuencia a longitud fija."""
        if len(frames) < MIN_FRAMES:
            return None
        
        frames_array = np.array(frames)
        current_len = len(frames_array)
        
        if current_len == SEQUENCE_LENGTH:
            return frames_array
        
        # Interpolar a longitud deseada
        indices = np.linspace(0, current_len - 1, SEQUENCE_LENGTH)
        normalized = np.zeros((SEQUENCE_LENGTH, 63))
        
        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(lower + 1, current_len - 1)
            weight = idx - lower
            normalized[i] = frames_array[lower] * (1 - weight) + frames_array[upper] * weight
        
        return normalized
    
    def process_sample(self, sample_path: Path) -> np.ndarray | None:
        """Procesa una muestra (carpeta con frames)."""
        frames = sorted(sample_path.glob("*.jpg"))
        
        if not frames:
            return None
        
        landmarks_list = []
        for frame_path in frames:
            landmarks = self.extract_landmarks(frame_path)
            if landmarks:
                landmarks_list.append(landmarks)
        
        if len(landmarks_list) < MIN_FRAMES:
            return None
        
        # Normalizar secuencia
        return self.normalize_sequence(landmarks_list)
    
    def process_word(self, word_id: int, word_folder: Path) -> int:
        """Procesa todas las muestras de una palabra."""
        word_name = self.id_to_word.get(word_id, f"unknown_{word_id}")
        
        # Crear carpeta de salida
        output_dir = OUTPUT_PATH / "sequences" / word_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener muestras (subcarpetas)
        samples = sorted([s for s in word_folder.iterdir() if s.is_dir()])
        
        sequences_created = 0
        existing = len(list(output_dir.glob("*.npy")))
        
        for sample_path in samples:
            sequence = self.process_sample(sample_path)
            
            if sequence is not None:
                filename = f"seq_{existing + sequences_created:04d}.npy"
                np.save(output_dir / filename, sequence)
                sequences_created += 1
                self.stats["total_samples"] += 1
            else:
                self.stats["failed_samples"] += 1
        
        return sequences_created
    
    def process_all(self):
        """Procesa todo el dataset."""
        print("\n" + "=" * 70)
        print("🤟 PROCESADOR DE DATASET - 249 PALABRAS LSM")
        print("=" * 70)
        print(f"📂 Dataset:  {DATASET_PATH}")
        print(f"📁 Output:   {OUTPUT_PATH}")
        print(f"🎞️  Frames por secuencia: {SEQUENCE_LENGTH}")
        print(f"📊 Mínimo frames válidos: {MIN_FRAMES}")
        
        # Crear carpeta de salida
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # Obtener carpetas de palabras
        word_folders = sorted([
            f for f in DATASET_PATH.iterdir() 
            if f.is_dir() and f.name.isdigit()
        ])
        
        total_words = len(word_folders)
        print(f"\n📝 Palabras encontradas: {total_words}")
        
        # Mostrar primeras palabras del mapeo
        print("\n" + "-" * 70)
        print("📋 MAPEO ID → PALABRA (primeras 15):")
        print("-" * 70)
        for i, (word_id, word_name) in enumerate(sorted(self.id_to_word.items())[:15]):
            print(f"   {word_id:03d} → {word_name}")
        print(f"   ... y {len(self.id_to_word) - 15} más")
        
        print("\n" + "-" * 70)
        print("🔄 PROCESANDO...")
        print("-" * 70)
        
        start_time = datetime.now()
        
        for i, word_folder in enumerate(word_folders, 1):
            word_id = int(word_folder.name)
            word_name = self.id_to_word.get(word_id, f"unknown_{word_id}")
            
            # Contar muestras antes de procesar
            samples = [s for s in word_folder.iterdir() if s.is_dir()]
            
            sequences = self.process_word(word_id, word_folder)
            self.stats["total_sequences"] += sequences
            self.stats["total_words"] += 1
            self.stats["words_processed"].append((word_id, word_name, sequences))
            
            # Calcular progreso y tiempo estimado
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / i
            remaining = avg_time * (total_words - i)
            
            # Barra de progreso simple
            progress = int((i / total_words) * 30)
            bar = "█" * progress + "░" * (30 - progress)
            
            # Mostrar progreso
            status = "✅" if sequences > 0 else "⚠️"
            print(f"   [{bar}] {i:3d}/{total_words} | {status} {word_id:03d}: {word_name:20s} | {sequences:2d} seq | ETA: {int(remaining)}s")
        
        # Guardar label encoder
        label_map = {word: idx for idx, word in enumerate(sorted(set(self.id_to_word.values())))}
        with open(OUTPUT_PATH / "words_label_encoder.pkl", "wb") as f:
            pickle.dump(label_map, f)
        
        # Generar reporte
        self._generate_report()
        
        # Resumen final
        elapsed_total = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"   📊 Palabras procesadas:  {self.stats['total_words']}")
        print(f"   ✅ Muestras exitosas:    {self.stats['total_samples']}")
        print(f"   ❌ Muestras fallidas:    {self.stats['failed_samples']}")
        print(f"   📁 Secuencias totales:   {self.stats['total_sequences']}")
        print(f"   ⏱️  Tiempo total:         {int(elapsed_total)}s ({elapsed_total/60:.1f} min)")
        print(f"\n   📂 Archivos guardados en:")
        print(f"      {OUTPUT_PATH}")
        print("=" * 70)
    
    def _generate_report(self):
        """Genera reporte de procesamiento."""
        report_path = OUTPUT_PATH / "processing_report.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE PROCESAMIENTO - DATASET 249 PALABRAS\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total palabras: {self.stats['total_words']}\n")
            f.write(f"Total secuencias: {self.stats['total_sequences']}\n")
            f.write(f"Muestras fallidas: {self.stats['failed_samples']}\n\n")
            
            f.write("DETALLE POR PALABRA:\n")
            f.write("-" * 40 + "\n")
            
            for word_id, word_name, sequences in self.stats["words_processed"]:
                f.write(f"  {word_id:03d}: {word_name:20s} -> {sequences} secuencias\n")
        
        print(f"\n[INFO] Reporte guardado: {report_path}")
    
    def close(self):
        """Libera recursos."""
        pass  # Vision Tasks API no requiere close explícito


def main():
    processor = WordsDatasetProcessor()
    
    try:
        processor.process_all()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
