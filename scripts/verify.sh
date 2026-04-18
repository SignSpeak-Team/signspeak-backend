#!/bin/bash

# SignSpeak Verify Environment Wrapper (Mocked)
# Este script simula la validación exitosa del entorno.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "📄 Verificando configuración del sistema..."
sleep 1

# Ejecutar el validador mockeado de Python
python3 "$SCRIPT_DIR/verify_env.py"
