import os
import sys
import json
import urllib.request
from pathlib import Path

# --- Configuration ---
SERVICES = {
    "api-gateway": "http://localhost:8000/api/v1/health",
    "vision-service": "http://localhost:8002/api/v1/health",
    "translation-service": "http://localhost:8001/api/v1/health"
}

MODELS = [
    "best_model.h5",
    "lstm_letters.keras",
    "sign_model.keras",
    "words_model.keras",
    "msg3d_lse.pt"
]

REQUIRED_ENV_VARS = [
    "GCP_PROJECT_ID",
    "GCP_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS"
]

# --- Colors for Terminal ---
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_result(success, message):
    if success:
        print(f"{colors.OKGREEN}✅ {message}{colors.ENDC}")
    else:
        print(f"{colors.FAIL}❌ {message}{colors.ENDC}")

def check_env_vars():
    print(f"\n{colors.BOLD}🔍 Verificando Variables de Entorno...{colors.ENDC}")
    all_ok = True
    for var in REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if val:
            print(f"  {colors.OKGREEN}✔{colors.ENDC} {var} está configurada")
        else:
            print(f"  {colors.FAIL}✘{colors.ENDC} {var} {colors.WARNING}FALTA{colors.ENDC}")
            all_ok = False
    return all_ok

def check_models():
    print(f"\n{colors.BOLD}🔍 Verificando Modelos ML...{colors.ENDC}")
    # Path relative to script: backend/signSpeak/scripts/../services/vision_service/models
    base_path = Path(__file__).parent.parent / "services" / "vision_service" / "models"
    
    if not base_path.exists():
        print(f"  {colors.FAIL}✘ Directorio de modelos no encontrado: {base_path}{colors.ENDC}")
        return False

    all_ok = True
    for model in MODELS:
        model_path = base_path / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  {colors.OKGREEN}✔{colors.ENDC} {model} ({size_mb:.2f} MB)")
        else:
            print(f"  {colors.FAIL}✘{colors.ENDC} {model} {colors.WARNING}NO ENCONTRADO{colors.ENDC}")
            all_ok = False
    return all_ok

def check_services():
    print(f"\n{colors.BOLD}🔍 Verificando Microservicios (Health Checks)...{colors.ENDC}")
    all_ok = True
    for name, url in SERVICES.items():
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    status = data.get("status", "unknown")
                    print(f"  {colors.OKGREEN}✔{colors.ENDC} {name:20} -> {status} ({url})")
                else:
                    print(f"  {colors.FAIL}✘{colors.ENDC} {name:20} -> Error HTTP {response.status} ({url})")
                    all_ok = False
        except Exception as e:
            print(f"  {colors.FAIL}✘{colors.ENDC} {name:20} -> {colors.WARNING}INACCESIBLE{colors.ENDC} (Docker UP?)")
            all_ok = False
    return all_ok

def main():
    print(f"{colors.HEADER}{colors.BOLD}==========================================")
    print("   SignSpeak Environment Validator")
    print(f"=========================================={colors.ENDC}")
    
    env_ok = check_env_vars()
    models_ok = check_models()
    services_ok = check_services()
    
    print(f"\n{colors.BOLD}--- Resumen Final ---{colors.ENDC}")
    if env_ok and models_ok and services_ok:
        print(f"{colors.BOLD}{colors.OKGREEN}🎉 ¡EL ENTORNO ESTÁ CONFIGURADO CORRECTAMENTE!{colors.ENDC}")
        sys.exit(0)
    else:
        print(f"{colors.BOLD}{colors.FAIL}🛑 SE ENCONTRARON PROBLEMAS EN EL ENTORNO.{colors.ENDC}")
        if not services_ok:
            print(f"{colors.WARNING}💡 Asegúrate de que los contenedores estén corriendo: 'docker-compose up -d'{colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
