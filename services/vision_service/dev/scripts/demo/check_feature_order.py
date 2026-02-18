"""
Identificar el orden de los bloques de features (Pose vs Manos).
Imprime los valores medios de los índices de inicio de bloque.
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\alan1\PycharmProjects\signSpeak\services\vision_service\dev\datasets_processed\msl150_subset")
val_data = np.load(DATA_DIR / "val_data.npy") # (N, 64, 226)

print("=" * 70)
print("IDENTIFICACIÓN DE ORDEN DE FEATURES")
print("=" * 70)

# Flatten
flat_data = val_data.reshape(-1, 226)
mean_data = np.mean(flat_data, axis=0)

def print_block_start(name, idx):
    vals = mean_data[idx:idx+3]
    print(f"\n📍 Index {idx} ({name} Candidate):")
    print(f"   Values (x,y,z): {vals}")
    
    # Heurísticas
    if 0.4 < vals[0] < 0.6 and 0.2 < vals[1] < 0.4:
        print("   ✅ PARECE NOSE (Cara/Pose)")
    elif 0.4 < vals[0] < 0.6 and 0.5 < vals[1] < 0.9:
        print("   ✅ PARECE WRIST (Mano/Hands en reposo o haciendo seña)")
    elif vals[0] == 0 and vals[1] == 0:
        print("   ⚠️  Ceros (Empty/Padding?)")
    else:
        print("   ❓ Indeterminado")

# Probar los 3 puntos de inicio posibles
# Block sizes: 63 (Hand), 63 (Hand), 99/100 (Pose)

print_block_start("Block 1 Start", 0)
print_block_start("Block 2 Start (if Block 1 is Hand)", 63)
print_block_start("Block 3 Start (if Blocks 1+2 are Hands)", 126)

# También probar si Pose es primero
print_block_start("Block 2 Start (if Block 1 is Pose 99)", 99)

print("\n" + "="*70)
