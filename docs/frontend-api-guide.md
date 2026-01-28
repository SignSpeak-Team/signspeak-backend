# SignSpeak API - Documentación para Frontend

> **Versión:** 1.0.0  
> **Base URL (Desarrollo):** `http://localhost:8000`  
> **Base URL (Producción):** `https://api.signspeak.com` _(configurar al desplegar)_

---

## 📋 Índice

1. [Configuración Inicial](#configuración-inicial)
2. [Endpoints Disponibles](#endpoints-disponibles)
3. [Modelos de Request/Response](#modelos-de-requestresponse)
4. [Ejemplos de Integración React](#ejemplos-de-integración-react)
5. [MediaPipe Integration](#mediapipe-integration)
6. [Manejo de Errores](#manejo-de-errores)
7. [Flujo Recomendado](#flujo-recomendado)

---

## Configuración Inicial

### Variables de Entorno (React)

```env
# .env.development
REACT_APP_API_URL=http://localhost:8000

# .env.production
REACT_APP_API_URL=https://api.signspeak.com
```

### Cliente API Base

```javascript
// src/services/api.js
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const apiClient = {
  async post(endpoint, data) {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Error en la API");
    }

    return response.json();
  },

  async get(endpoint) {
    const response = await fetch(`${API_URL}${endpoint}`);
    if (!response.ok) throw new Error("Error en la API");
    return response.json();
  },
};
```

---

## Endpoints Disponibles

### Health Check

| Método | Endpoint         | Descripción                               |
| ------ | ---------------- | ----------------------------------------- |
| GET    | `/api/v1/health` | Estado del servicio                       |
| GET    | `/api/v1/status` | Estado detallado con servicios downstream |

---

### Predicción de Letras

#### `POST /api/v1/predict/static`

Predice letras estáticas del alfabeto LSM (A-Z excepto J, K, Q, X, Z, Ñ).

**Request:**

```json
{
  "landmarks": [
    [0.123, 0.456, 0.001], // Landmark 0: WRIST
    [0.234, 0.567, 0.002] // Landmark 1: THUMB_CMC
    // ... 21 landmarks total
  ]
}
```

**Response:**

```json
{
  "letter": "A",
  "confidence": 95.5,
  "type": "static",
  "processing_time_ms": 12.5
}
```

---

#### `POST /api/v1/predict/dynamic`

Predice letras dinámicas que requieren movimiento (J, K, Q, X, Z, Ñ).

**Request:**

```json
{
  "sequence": [
    // Frame 1 (21 landmarks)
    [
      [0.1, 0.2, 0.0],
      [0.2, 0.3, 0.0] /* ... 21 landmarks */
    ],
    // Frame 2
    [
      [0.1, 0.2, 0.0],
      [0.2, 0.3, 0.0] /* ... 21 landmarks */
    ]
    // ... 15 frames total
  ]
}
```

**Response:**

```json
{
  "letter": "Ñ",
  "confidence": 87.3,
  "type": "dynamic",
  "processing_time_ms": 45.2
}
```

---

### Predicción de Palabras

#### `POST /api/v1/predict/words`

Predice palabras del vocabulario LSM (249 palabras).

**Request:**

```json
{
  "sequence": [
    // 15 frames × 21 landmarks × 3 coordenadas
    // Mismo formato que /predict/dynamic
  ]
}
```

**Response:**

```json
{
  "word": "hola",
  "confidence": 92.1,
  "phrase": "hola como estas",
  "accepted": true,
  "processing_time_ms": 38.7
}
```

> **Nota:** `phrase` acumula las palabras detectadas. `accepted` indica si la palabra pasó el filtro de confianza.

---

#### `POST /api/v1/predict/holistic`

Predice vocabulario médico (150 palabras) usando pose completa + manos.

**Request:**

```json
{
  "landmarks": [0.1, 0.2, 0.3 /* ... 226 valores total */]
}
```

**Composición de los 226 valores:**

- Pose: 33 puntos × 4 valores (x, y, z, visibility) = 132
- Mano izquierda: 21 puntos × 3 valores = 63
- Mano derecha: 21 puntos × 3 valores = 63
- **Total: 132 + 63 + 63 = 258** _(se usan 226 después de procesamiento)_

**Response:**

```json
{
  "word": "dolor",
  "confidence": 88.5,
  "phrase": "dolor cabeza",
  "accepted": true,
  "processing_time_ms": 42.1
}
```

---

### Buffer Management

#### `GET /api/v1/predict/words/stats`

Obtiene estadísticas del buffer de predicción.

**Response:**

```json
{
  "buffer_size": 10,
  "phrase_length": 3,
  "current_phrase": "hola como estas"
}
```

---

#### `POST /api/v1/predict/words/clear`

Limpia el buffer y la frase acumulada.

**Response:**

```json
{
  "message": "Buffer cleared"
}
```

---

#### `POST /api/v1/predict/holistic/clear`

Limpia el buffer holístico.

**Response:**

```json
{
  "message": "Holistic buffer cleared"
}
```

---

## Modelos de Request/Response

### Landmarks (21 puntos de la mano)

```
     8   12  16  20
     |   |   |   |
     7   11  15  19
     |   |   |   |
     6   10  14  18
     |   |   |   |
     5---9---13--17
      \         /
       4
        \     /
         3
          \ /
           2
           |
           1
           |
           0 (WRIST)
```

| Índice | Nombre     | Descripción |
| ------ | ---------- | ----------- |
| 0      | WRIST      | Muñeca      |
| 1-4    | THUMB\_\*  | Pulgar      |
| 5-8    | INDEX\_\*  | Índice      |
| 9-12   | MIDDLE\_\* | Medio       |
| 13-16  | RING\_\*   | Anular      |
| 17-20  | PINKY\_\*  | Meñique     |

### Formato de Coordenadas

```javascript
// Cada landmark es un array de 3 valores
const landmark = [x, y, z];

// x: posición horizontal (0.0 = izquierda, 1.0 = derecha)
// y: posición vertical (0.0 = arriba, 1.0 = abajo)
// z: profundidad (valores negativos = más cerca de la cámara)
```

---

## Ejemplos de Integración React

### Hook para Predicción de Letras

```javascript
// src/hooks/useSignPrediction.js
import { useState, useCallback } from "react";
import { apiClient } from "../services/api";

export function useSignPrediction() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predictStatic = useCallback(async (landmarks) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.post("/api/v1/predict/static", {
        landmarks,
      });
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const predictDynamic = useCallback(async (sequence) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.post("/api/v1/predict/dynamic", {
        sequence,
      });
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { prediction, loading, error, predictStatic, predictDynamic };
}
```

### Hook para Predicción de Palabras

```javascript
// src/hooks/useWordPrediction.js
import { useState, useCallback } from "react";
import { apiClient } from "../services/api";

export function useWordPrediction() {
  const [phrase, setPhrase] = useState("");
  const [lastWord, setLastWord] = useState(null);
  const [loading, setLoading] = useState(false);

  const predictWord = useCallback(async (sequence) => {
    setLoading(true);
    try {
      const result = await apiClient.post("/api/v1/predict/words", {
        sequence,
      });
      setLastWord(result);
      if (result.accepted) {
        setPhrase(result.phrase);
      }
      return result;
    } catch (err) {
      console.error("Prediction error:", err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearBuffer = useCallback(async () => {
    await apiClient.post("/api/v1/predict/words/clear");
    setPhrase("");
    setLastWord(null);
  }, []);

  return { phrase, lastWord, loading, predictWord, clearBuffer };
}
```

### Componente de Cámara con MediaPipe

```javascript
// src/components/SignCamera.jsx
import { useEffect, useRef, useState } from "react";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { useSignPrediction } from "../hooks/useSignPrediction";

export function SignCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const { prediction, predictStatic } = useSignPrediction();
  const [lastLandmarks, setLastLandmarks] = useState(null);

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
      if (results.multiHandLandmarks && results.multiHandLandmarks[0]) {
        const landmarks = results.multiHandLandmarks[0].map((lm) => [
          lm.x,
          lm.y,
          lm.z,
        ]);
        setLastLandmarks(landmarks);
      }
    });

    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          await hands.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }

    return () => hands.close();
  }, []);

  // Predict every 500ms when landmarks are available
  useEffect(() => {
    if (!lastLandmarks) return;

    const interval = setInterval(() => {
      predictStatic(lastLandmarks);
    }, 500);

    return () => clearInterval(interval);
  }, [lastLandmarks, predictStatic]);

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas ref={canvasRef} width={640} height={480} />

      {prediction && (
        <div className="prediction-result">
          <h2>{prediction.letter}</h2>
          <p>Confianza: {prediction.confidence.toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
```

---

## MediaPipe Integration

### Instalación

```bash
npm install @mediapipe/hands @mediapipe/camera_utils @mediapipe/drawing_utils
```

### Extracción de Landmarks

```javascript
// utils/landmarkExtractor.js

// Para letras estáticas (21 landmarks)
export function extractHandLandmarks(mediaPipeLandmarks) {
  return mediaPipeLandmarks.map((lm) => [lm.x, lm.y, lm.z]);
}

// Para letras dinámicas/palabras (buffer de 15 frames)
export class FrameBuffer {
  constructor(size = 15) {
    this.size = size;
    this.frames = [];
  }

  add(landmarks) {
    this.frames.push(landmarks);
    if (this.frames.length > this.size) {
      this.frames.shift();
    }
  }

  isFull() {
    return this.frames.length === this.size;
  }

  getSequence() {
    return [...this.frames];
  }

  clear() {
    this.frames = [];
  }
}
```

### Uso del Buffer para Predicción Dinámica

```javascript
// En tu componente
const frameBuffer = useRef(new FrameBuffer(15));

hands.onResults((results) => {
  if (results.multiHandLandmarks?.[0]) {
    const landmarks = extractHandLandmarks(results.multiHandLandmarks[0]);
    frameBuffer.current.add(landmarks);

    if (frameBuffer.current.isFull()) {
      const sequence = frameBuffer.current.getSequence();
      predictDynamic(sequence);
      // O para palabras: predictWord(sequence);
    }
  }
});
```

---

## Manejo de Errores

### Códigos de Error

| Código | Descripción                                  | Acción                            |
| ------ | -------------------------------------------- | --------------------------------- |
| 400    | Request inválido (landmarks mal formateados) | Verificar formato de datos        |
| 422    | Validación fallida (landmarks insuficientes) | Verificar cantidad de landmarks   |
| 502    | Vision Service no disponible                 | Reintentar o notificar al usuario |
| 503    | Servicio no disponible                       | Mostrar mensaje de mantenimiento  |
| 504    | Timeout                                      | Reintentar la petición            |

### Ejemplo de Manejo

```javascript
try {
  const result = await predictStatic(landmarks);
} catch (error) {
  if (error.message.includes("502")) {
    showNotification("El servicio de visión no está disponible");
  } else if (error.message.includes("422")) {
    console.error("Landmarks inválidos:", landmarks.length);
  } else {
    showNotification("Error de conexión");
  }
}
```

---

## Flujo Recomendado

### Para Deletreo (Letras)

```
┌─────────────────────────────────────────────────────┐
│  1. Detectar mano con MediaPipe                     │
│  2. Verificar que hay 21 landmarks                  │
│  3. Si la mano está ESTÁTICA por 500ms:             │
│     → POST /predict/static                          │
│  4. Si la mano tiene MOVIMIENTO:                    │
│     → Acumular 15 frames                           │
│     → POST /predict/dynamic                         │
│  5. Mostrar letra con confidence > 70%              │
│  6. Acumular letras para formar palabra             │
└─────────────────────────────────────────────────────┘
```

### Para Palabras (Vocabulario)

```
┌─────────────────────────────────────────────────────┐
│  1. Detectar mano con MediaPipe                     │
│  2. Acumular 15 frames continuamente                │
│  3. POST /predict/words cada 500ms                  │
│  4. Si `accepted: true`:                            │
│     → Mostrar palabra detectada                     │
│     → Actualizar frase con `phrase`                 │
│  5. Botón "Limpiar" → POST /predict/words/clear     │
└─────────────────────────────────────────────────────┘
```

---

## Referencia Rápida

```javascript
// LETRAS ESTÁTICAS
POST /api/v1/predict/static
Body: { landmarks: [[x,y,z], ...] }  // 21 landmarks
Returns: { letter, confidence, type, processing_time_ms }

// LETRAS DINÁMICAS
POST /api/v1/predict/dynamic
Body: { sequence: [[[x,y,z], ...], ...] }  // 15 frames × 21 landmarks
Returns: { letter, confidence, type, processing_time_ms }

// PALABRAS (249 vocab)
POST /api/v1/predict/words
Body: { sequence: [[[x,y,z], ...], ...] }  // 15 frames × 21 landmarks
Returns: { word, confidence, phrase, accepted, processing_time_ms }

// VOCABULARIO MÉDICO (150 palabras)
POST /api/v1/predict/holistic
Body: { landmarks: [f1, f2, ...] }  // 226 features
Returns: { word, confidence, phrase, accepted, processing_time_ms }

// LIMPIAR BUFFER
POST /api/v1/predict/words/clear
POST /api/v1/predict/holistic/clear

// HEALTH
GET /api/v1/health
GET /api/v1/status
```

---

**Última actualización:** Enero 2026  
**Versión API:** 1.0.0
