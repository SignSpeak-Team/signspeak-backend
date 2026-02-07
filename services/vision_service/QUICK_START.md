# 🚀 Guía Rápida de Inicio - Vision Service

## Paso 1: Iniciar el Servicio

**Opción A - Con terminal PowerShell:**

```powershell
cd c:\Users\alan1\PycharmProjects\signSpeak\services\vision_service
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**Opción B - Con script (más fácil):**

```powershell
cd c:\Users\alan1\PycharmProjects\signSpeak\services\vision_service
.\start_service.ps1
```

Espera a ver: `✓ Models loaded successfully` y `API ready!`

---

## Paso 2: Probar la Detección (en OTRA terminal)

### Con el Script de Prueba

```powershell
# Activar entorno
cd c:\Users\alan1\PycharmProjects\signSpeak
.venv\Scripts\Activate.ps1

# Ir al directorio
cd services\vision_service

# Ejecutar prueba con video de caso
python test_sequence_detection.py "dev\datasets_raw\videos\letras\cases\CASO_4_blur.mp4"
```

### Con Postman

1. **POST** `http://localhost:8001/api/v1/media/translate/video`
2. **Body** → form-data:
   - Key: `file`
   - Type: File
   - Value: Selecciona `CASO_0_blur (1).mp4` o `CASO_4_blur.mp4`
3. **Send**

---

## Videos de Prueba Disponibles

- `dev\datasets_raw\videos\letras\cases\CASO_0_blur (1).mp4`
- `dev\datasets_raw\videos\letras\cases\CASO_4_blur.mp4`

---

## Qué Esperar

**Respuesta del modo continuous:**

```json
{
  "word": "palabra1 palabra2 palabra3",
  "confidence": 75.3,
  "segments": [
    {
      "word": "palabra1",
      "start_time": 0.0,
      "end_time": 2.0,
      "confidence": 78.5
    }
  ],
  "detection_stats": {
    "total_windows": 12,
    "detected_words": 8,
    "filtered_words": 3,
    "average_confidence": 75.3
  }
}
```

---

## Solución de Problemas

**Error: "Could not import module 'src.main'"**

- ✅ Usa: `python -m uvicorn src.api.main:app`
- ❌ No uses: `python -m uvicorn src.main:app`

**Error: "No se pudo conectar al servicio"**

- Verifica que el servicio esté corriendo en otra terminal
- Revisa http://localhost:8001/docs

**Error: "No landmarks detected"**

- El video puede estar dañado o sin señas visibles
- Prueba con otro video de la carpeta cases
