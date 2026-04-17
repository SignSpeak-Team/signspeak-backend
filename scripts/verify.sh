#!/bin/bash

# SignSpeak Verify Environment Wrapper
# Este script carga el .env (si existe) y ejecuta el validador de Python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Intentar cargar .env para que las variables estén disponibles en el proceso actual
if [ -f "$SCRIPT_DIR/../.env" ]; then
    echo "📄 Cargando variables desde .env..."
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

python3 "$SCRIPT_DIR/verify_env.py"
