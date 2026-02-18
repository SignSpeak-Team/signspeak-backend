"""
Script de diagnóstico para comparar features de entrenamiento vs tiempo real.
"""
import numpy as np
import pickle
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "datasets_processed" / "msl150_subset"
LABELS_PATH = DATA_DIR / "label_names.pkl"

print("=" * 70)
print("DIAGNÓSTICO: Features Training vs Inference")
print("=" * 70)

# Load training data
train_data = np.load(DATA_DIR / "train_data.npy")
train_labels = np.load(DATA_DIR / "train_labels.npy")

with open(LABELS_PATH, 'rb') as f:
    id_to_label = pickle.load(f)

print(f"\n📊 Dataset de entrenamiento:")
print(f"  Shape: {train_data.shape}")
print(f"  Clases: {len(id_to_label)}")

# Analizar features por clase
print(f"\n🔍 Analizando 'ojo' y 'huesos':")

# Encontrar IDs
ojo_id = None
huesos_id = None
for class_id, name in id_to_label.items():
    if name == "ojo":
        ojo_id = class_id
    elif name == "huesos":
        huesos_id = class_id

if ojo_id is not None:
    ojo_samples = train_data[train_labels == ojo_id]
    print(f"\n  'ojo' (class {ojo_id}):")
    print(f"    Muestras: {len(ojo_samples)}")
    print(f"    Mean de features: {ojo_samples.mean():.6f}")
    print(f"    Std de features: {ojo_samples.std():.6f}")
    print(f"    Min: {ojo_samples.min():.6f}, Max: {ojo_samples.max():.6f}")

if huesos_id is not None:
    huesos_samples = train_data[train_labels == huesos_id]
    print(f"\n  'huesos' (class {huesos_id}):")
    print(f"    Muestras: {len(huesos_samples)}")
    print(f"    Mean de features: {huesos_samples.mean():.6f}")
    print(f"    Std de features: {huesos_samples.std():.6f}")
    print(f"    Min: {huesos_samples.min():.6f}, Max: {huesos_samples.max():.6f}")

# Distribución de clases
print(f"\n📈 Distribución de clases (top 10):")
from collections import Counter
class_counts = Counter(train_labels)
for class_id, count in class_counts.most_common(10):
    print(f"  {id_to_label[class_id]:20s}: {count} muestras")

# Estadísticas globales
print(f"\n📊 Estadísticas globales del dataset:")
print(f"  Mean global: {train_data.mean():.6f}")
print(f"  Std global: {train_data.std():.6f}")
print(f"  Min global: {train_data.min():.6f}")
print(f"  Max global: {train_data.max():.6f}")

print(f"\n💡 Análisis:")
print(f"  - Si 'ojo' y 'huesos' tienen muchas más muestras, hay desbalance")
print(f"  - Si los valores están en rango [0, 1], están normalizados")
print(f"  - Features en tiempo real DEBEN tener misma escala")

print(f"\n{'=' * 70}")
