"""
Compara el formato de features entre dataset y extracción en tiempo real.
"""
import numpy as np
from pathlib import Path

# Cargar un sample del dataset original (.npy)
DATASET_DIR = Path(r"C:\Users\alan1\PycharmProjects\signSpeak\services\vision_service\dev\datasets_raw\numpy\MSL-150_raw_npy_v1\raw_npy")

# Tomar un frame de "yo" (clase que detecta bien)
yo_dir = DATASET_DIR / "yo" / "1"
frame_file = yo_dir / "0.npy"

if frame_file.exists():
    original_features = np.load(frame_file)
    
    print("=" * 70)
    print("ANÁLISIS: Formato de Features Original vs Demo")
    print("=" * 70)
    
    print(f"\n📄 Archivo original: {frame_file}")
    print(f"   Shape: {original_features.shape}")
    print(f"   Total features: {len(original_features)}")
    
    print(f"\n🔍 Análisis por secciones:")
    print(f"   Features 0-62 (primeros 63): {original_features[:63].mean():.6f}")
    print(f"   Features 63-125 (siguientes 63): {original_features[63:126].mean():.6f}")
    print(f"   Features 126-225 (últimos 100): {original_features[126:226].mean():.6f}")
    
    print(f"\n📊 Primeros 10 valores:")
    for i in range(10):
        print(f"   [{i}]: {original_features[i]:.6f}")
    
    print(f"\n💡 FORMATO DEL DATASET ORIGINAL:")
    print(f"   Según documentación MSL-150:")
    print(f"   - Right hand (21 landmarks × 3): features 0-62")
    print(f"   - Left hand (21 landmarks × 3): features 63-125")
    print(f"   - Pose (25 landmarks × 4): features 126-225")
    
    print(f"\n💡 FORMATO EN LA DEMO:")
    print(f"   Actualmente usamos:")
    print(f"   - Right hand (21 × 3): features 0-62")
    print(f"   - Left hand (21 × 3): features 63-125")
    print(f"   - Pose (25 × 4): features 126-225")
    
    print(f"\n✅ El orden PARECE correcto, pero...")
    print(f"\n⚠️  POSIBLE PROBLEMA:")
    print(f"   MediaPipe Holistic tiene más de 25 landmarks de pose (tiene 33)")
    print(f"   Estamos tomando solo los primeros 25 en la demo")
    print(f"   ¿El dataset también usa solo los primeros 25?")
    
    print(f"\n{'=' * 70}")
else:
    print(f"❌ No se encontró el archivo: {frame_file}")
