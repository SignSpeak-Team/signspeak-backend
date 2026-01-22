"""
Análisis y extracción de landmarks del dataset LSM.

Estructura del dataset (226 features por frame):
- [0:63]   = Mano Derecha (21 landmarks × 3 coords)
- [63:126] = Mano Izquierda (21 landmarks × 3 coords)
- [126:226] = Pose (25 landmarks × 4 valores: x,y,z,visibility)
"""
import numpy as np
import os

folder = 'services/vision_service/dev/datasets_raw/numpy/1'
files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])

print("=" * 70)
print("ANÁLISIS DE DATASET LSM - NÚMERO '1'")
print("=" * 70)
print(f"\n📁 Archivos encontrados: {len(files)} frames")

# Cargar todos los frames
sequence = []
for f in files:
    data = np.load(os.path.join(folder, f))
    sequence.append(data)

sequence = np.array(sequence)
print(f"📐 Shape secuencia completa: {sequence.shape}")

# Extraer componentes
right_hand = sequence[:, 0:63]      # Mano derecha
left_hand = sequence[:, 63:126]     # Mano izquierda
pose = sequence[:, 126:226]         # Pose

print("\n" + "-" * 70)
print("COMPONENTES EXTRAÍDOS")
print("-" * 70)
print(f"  ✋ Mano Derecha:   {right_hand.shape} (21 landmarks × 3 coords)")
print(f"  🤚 Mano Izquierda: {left_hand.shape} (21 landmarks × 3 coords)")
print(f"  🧍 Pose:           {pose.shape} (25 landmarks × 4 valores)")

# Verificar si hay datos válidos (no todos ceros)
rh_has_data = np.any(right_hand != 0)
lh_has_data = np.any(left_hand != 0)
pose_has_data = np.any(pose != 0)

print("\n" + "-" * 70)
print("VERIFICACIÓN DE DATOS")
print("-" * 70)
print(f"  ✋ Mano Derecha tiene datos:   {'✅ SÍ' if rh_has_data else '❌ NO (zeros)'}")
print(f"  🤚 Mano Izquierda tiene datos: {'✅ SÍ' if lh_has_data else '❌ NO (zeros)'}")
print(f"  🧍 Pose tiene datos:           {'✅ SÍ' if pose_has_data else '❌ NO (zeros)'}")

# Estadísticas de cada componente
print("\n" + "-" * 70)
print("ESTADÍSTICAS")
print("-" * 70)

for name, data in [("Mano Derecha", right_hand), ("Mano Izquierda", left_hand), ("Pose", pose)]:
    if np.any(data != 0):
        print(f"\n  {name}:")
        print(f"    Rango: [{data.min():.4f}, {data.max():.4f}]")
        print(f"    Media: {data.mean():.4f}")
        print(f"    Std:   {data.std():.4f}")
    else:
        print(f"\n  {name}: Sin datos (zeros)")

# Compatibilidad con tu modelo
print("\n" + "=" * 70)
print("COMPATIBILIDAD CON TU MODELO")
print("=" * 70)
print(f"""
Tu modelo actual espera: 63 features (shape: (N, 63))
Este dataset tiene:
  - Mano Derecha: {right_hand.shape} ✅ COMPATIBLE
  - Mano Izquierda: {left_hand.shape} ✅ COMPATIBLE

Para usar con tu modelo de palabras (LSTM):
  - Necesitas secuencias de 15 frames
  - Cada frame con 63 features
  - Este ejemplo tiene {len(files)} frames

👉 PUEDES extraer `sequence[:, 0:63]` para mano derecha
   o `sequence[:, 63:126]` para mano izquierda
""")

# Mostrar primer frame de mano derecha
print("-" * 70)
print("MUESTRA: Primer frame, Mano Derecha (primeros 15 valores)")
print("-" * 70)
print(right_hand[0, :15])
