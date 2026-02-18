"""
Verifica que el modelo realmente predice bien en el validation set.
Carga directamente los .npy y hace predicciones.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
import random
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.training.train_msl150_lstm import HolisticLSTM

# Paths
MODEL_PATH = PROJECT_DIR / "training_output" / "msl150_subset_lstm" / "best.pth"
DATA_DIR = PROJECT_DIR / "datasets_processed" / "msl150_subset"

print("=" * 70)
print("VERIFICACIÓN: ¿Realmente el modelo tiene 100% accuracy?")
print("=" * 70)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HolisticLSTM(
    input_size=226,
    hidden_size=512,
    num_layers=3,
    num_classes=150,
    dropout=0.4,
    bidirectional=True
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load validation data
val_data = np.load(DATA_DIR / "val_data.npy")
val_labels = np.load(DATA_DIR / "val_labels.npy")

with open(DATA_DIR / "label_names.pkl", 'rb') as f:
    id_to_label = pickle.load(f)

print(f"\n✓ Modelo cargado")
print(f"✓ Validation set: {val_data.shape}")

# Tomar 20 samples aleatorios
num_samples = 20
random_indices = random.sample(range(len(val_data)), num_samples)

print(f"\n🔍 Probando {num_samples} samples aleatorios del validation set:")
print("=" * 70)

correct = 0
for i, idx in enumerate(random_indices, 1):
    # Get sample
    x = torch.from_numpy(val_data[idx]).unsqueeze(0).to(device)  # (1, 64, 226)
    true_label = val_labels[idx]
    true_word = id_to_label[true_label]
    
    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_prob, pred_id = torch.max(probs, dim=1)
    
    pred_id = pred_id.item()
    pred_prob = pred_prob.item()
    pred_word = id_to_label[pred_id]
    
    is_correct = (pred_id == true_label)
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{i:2d}. {status} True: '{true_word:15s}' | Pred: '{pred_word:15s}' ({pred_prob*100:.1f}%)")

accuracy = (correct / num_samples) * 100
print("=" * 70)
print(f"\n📊 Accuracy en {num_samples} samples: {correct}/{num_samples} = {accuracy:.1f}%")

# Ahora verificar TODA la validation set
print(f"\n🔍 Verificando TODA la validation set ({len(val_data)} samples)...")

all_correct = 0
with torch.no_grad():
    # Por batches para velocidad
    batch_size = 64
    for i in range(0, len(val_data), batch_size):
        batch_x = torch.from_numpy(val_data[i:i+batch_size]).to(device)
        batch_labels = val_labels[i:i+batch_size]
        
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_correct += (preds == batch_labels).sum()

total_accuracy = (all_correct / len(val_data)) * 100
print(f"✓ Accuracy total: {all_correct}/{len(val_data)} = {total_accuracy:.2f}%")

print(f"\n{'=' * 70}")
print("💡 CONCLUSIÓN:")
if total_accuracy >= 99.5:
    print("  ✅ El modelo SÍ funciona perfectamente en validation set")
    print("  ⚠️  El problema está en la extracción de features en tiempo real")
    print("  📋 Siguiente paso: Comparar features del dataset vs tiempo real")
else:
    print(f"  ❌ El modelo NO tiene {checkpoint['val_acc']*100:.1f}% accuracy")
    print("  ⚠️  Hay un problema con el checkpoint o los datos")

print("=" * 70)
