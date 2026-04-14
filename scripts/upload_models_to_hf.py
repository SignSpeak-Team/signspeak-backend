"""
Upload SignSpeak models to Hugging Face Hub.

Usage:
    python scripts/upload_models_to_hf.py

Requires:
    pip install huggingface_hub
    HF_TOKEN env var set (or pass --token)
"""

import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID = "alanctinaDev/signspeak-models"   # cambia si tu HF user es otro
MODELS_DIR = Path(__file__).parent.parent / "services" / "vision_service" / "models"
HF_TOKEN = os.getenv("HF_TOKEN")

EXPECTED_FILES = [
    "sign_model.keras",
    "label_encoder.pkl",
    "lstm_letters.keras",
    "lstm_label_encoder.pkl",
    "words_model.keras",
    "words_label_encoder.pkl",
    "best_model.h5",
    "best_model_labels.pkl",
    "holistic_label_encoder.pkl",
    "msg3d_lse.pt",
    "msg3d_labels.pkl",
]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub no está instalado.")
        print("   Corre: pip install huggingface_hub")
        sys.exit(1)

    if not HF_TOKEN:
        print("❌ HF_TOKEN no está definido.")
        print("   Corre: $env:HF_TOKEN='hf_tu_token_aqui'")
        sys.exit(1)

    if not MODELS_DIR.exists():
        print(f"❌ No se encontró la carpeta de modelos: {MODELS_DIR}")
        sys.exit(1)

    print(f"📦 Subiendo modelos a: {REPO_ID}")
    print(f"📁 Desde: {MODELS_DIR}")
    print()

    api = HfApi(token=HF_TOKEN)

    # Crear el repo si no existe
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
        print(f"✅ Repo '{REPO_ID}' listo en HF Hub")
    except Exception as e:
        print(f"❌ Error creando repo: {e}")
        sys.exit(1)

    # Verificar que todos los archivos existen
    missing = [f for f in EXPECTED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"⚠️  Archivos no encontrados localmente: {missing}")
        print("   Continuando con los que sí existen...")
    print()

    # Subir cada archivo
    files_to_upload = [f for f in EXPECTED_FILES if (MODELS_DIR / f).exists()]
    for filename in files_to_upload:
        local_path = MODELS_DIR / filename
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"⬆️  Subiendo {filename} ({size_mb:.2f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"   ✅ {filename} subido")
        except Exception as e:
            print(f"   ❌ Error subiendo {filename}: {e}")

    print()
    print("🎉 ¡Listo! Modelos disponibles en:")
    print(f"   https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
