"""
Script para explorar la estructura del dataset LSM de Zenodo.

Uso:
    python explore_dataset.py
"""

import pickle
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent.parent / "datasets" / "lsm_zenodo"


def explore_pickle(filename: str):
    """Carga y analiza un archivo pickle."""
    filepath = DATASET_PATH / filename
    
    print("\n" + "=" * 60)
    print(f"[ARCHIVO] {filename}")
    print("=" * 60)
    
    if not filepath.exists():
        print("[ERROR] Archivo no encontrado")
        return None
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    print(f"[TIPO] {type(data).__name__}")
    
    if isinstance(data, dict):
        print(f"[CLAVES] {len(data)}")
        keys_list = list(data.keys())
        print(f"[KEYS] {keys_list[:20]}")
        
        if len(keys_list) > 0:
            first_key = keys_list[0]
            first_value = data[first_key]
            print(f"\n[EJEMPLO] Clave '{first_key}':")
            print(f"   Tipo: {type(first_value).__name__}")
            
            if isinstance(first_value, list):
                print(f"   Elementos: {len(first_value)}")
                if len(first_value) > 0:
                    elem = first_value[0]
                    print(f"   Tipo elemento: {type(elem).__name__}")
                    if hasattr(elem, 'shape'):
                        print(f"   Shape: {elem.shape}")
                    elif isinstance(elem, list):
                        print(f"   Longitud: {len(elem)}")
    
    elif isinstance(data, list):
        print(f"[ELEMENTOS] {len(data)}")
        if len(data) > 0:
            print(f"[PRIMER ELEM] Tipo: {type(data[0]).__name__}")
    
    else:
        print(f"[CONTENIDO] {data}")
    
    return data


def main():
    print("\n" + "EXPLORADOR DE DATASET LSM".center(60))
    print("=" * 60)
    print(f"[DIR] {DATASET_PATH}")
    
    pickle_files = list(DATASET_PATH.glob("*.pickle"))
    print(f"[ARCHIVOS] {len(pickle_files)}")
    
    for pf in pickle_files:
        print(f"   - {pf.name} ({pf.stat().st_size / 1024:.1f} KB)")
    
    for filename in ["ABECEDARIO.pickle", "NUMEROS.pickle", "PALABRAS.pickle"]:
        explore_pickle(filename)
    
    print("\n[OK] Exploracion completada")


if __name__ == "__main__":
    main()
