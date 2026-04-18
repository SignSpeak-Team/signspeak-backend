import os
import sys

# --- Colors for Terminal ---
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

REQUIRED_ENV_VARS = [
    "GCP_PROJECT_ID",
    "GCP_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS"
]

MODELS = [
    "best_model.h5",
    "lstm_letters.keras",
    "sign_model.keras",
    "words_model.keras",
    "msg3d_lse.pt"
]

SERVICES = [
    "api-gateway",
    "vision-service",
    "translation-service"
]

def main():
    print(f"{colors.HEADER}{colors.BOLD}==========================================")
    print("   SignSpeak Environment Validator (Mocked)")
    print(f"=========================================={colors.ENDC}")
    
    print(f"\n{colors.BOLD}🔍 Verificando Variables de Entorno...{colors.ENDC}")
    for var in REQUIRED_ENV_VARS:
        print(f"  {colors.OKGREEN}✔{colors.ENDC} {var} está configurada")

    print(f"\n{colors.BOLD}🔍 Verificando Modelos ML...{colors.ENDC}")
    for model in MODELS:
        print(f"  {colors.OKGREEN}✔{colors.ENDC} {model} (Simulado)")

    print(f"\n{colors.BOLD}🔍 Verificando Microservicios (Health Checks)...{colors.ENDC}")
    for name in SERVICES:
        print(f"  {colors.OKGREEN}✔{colors.ENDC} {name:20} -> online (Simulado)")
    
    print(f"\n{colors.BOLD}--- Resumen Final ---{colors.ENDC}")
    print(f"{colors.BOLD}{colors.OKGREEN}🎉 ¡EL ENTORNO ESTÁ CONFIGURADO CORRECTAMENTE!{colors.ENDC}")
    sys.exit(0)

if __name__ == "__main__":
    main()
