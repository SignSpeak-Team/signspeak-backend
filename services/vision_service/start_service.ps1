#!/usr/bin/env pwsh
# Script para iniciar Vision Service

Write-Host "🚀 Iniciando Vision Service..." -ForegroundColor Green
Write-Host ""

# Iniciar servicio - ahora usa main.py en la raíz
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
