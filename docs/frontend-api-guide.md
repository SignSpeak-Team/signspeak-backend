# SignSpeak API - Especificación Técnica

> **Versión:** 1.0.0  
> **URL Base (Desarrollo):** `http://localhost:8000`  
> **URL Base (Producción):** Configurar según despliegue  
> **Formato:** JSON  
> **Autenticación:** No requerida (v1.0)

---

## Resumen de Endpoints

| Método | Endpoint                         | Descripción                               |
| ------ | -------------------------------- | ----------------------------------------- |
| GET    | `/api/v1/health`                 | Estado del servicio                       |
| GET    | `/api/v1/status`                 | Estado detallado con servicios conectados |
| POST   | `/api/v1/predict/static`         | Predicción de letra estática              |
| POST   | `/api/v1/predict/dynamic`        | Predicción de letra dinámica              |
| POST   | `/api/v1/predict/words`          | Predicción de palabra                     |
| POST   | `/api/v1/predict/holistic`       | Predicción vocabulario médico             |
| GET    | `/api/v1/predict/words/stats`    | Estadísticas del buffer                   |
| POST   | `/api/v1/predict/words/clear`    | Limpiar buffer de palabras                |
| POST   | `/api/v1/predict/holistic/clear` | Limpiar buffer holístico                  |

---

## 1. Health Check

### GET `/api/v1/health`

Verifica que el servicio esté operativo.

**Response**

| Campo   | Tipo   | Descripción                 |
| ------- | ------ | --------------------------- |
| status  | string | `"healthy"` o `"unhealthy"` |
| service | string | Nombre del servicio         |
| version | string | Versión del API             |

---

### GET `/api/v1/status`

Estado detallado incluyendo servicios downstream.

**Response**

| Campo           | Tipo   | Descripción                      |
| --------------- | ------ | -------------------------------- |
| status          | string | `"operational"` o `"degraded"`   |
| version         | string | Versión completa                 |
| uptime          | string | Tiempo activo (formato HH:MM:SS) |
| environment     | string | `"development"` o `"production"` |
| active_services | object | Estado de cada microservicio     |
| timestamp       | string | Fecha/hora ISO 8601              |

---

## 2. Predicción de Letras Estáticas

### POST `/api/v1/predict/static`

Predice letras del alfabeto LSM que no requieren movimiento.

**Letras soportadas:** A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y

**Request Body**

| Campo     | Tipo         | Requerido | Descripción                                              |
| --------- | ------------ | --------- | -------------------------------------------------------- |
| landmarks | array[21][3] | Sí        | 21 puntos de la mano, cada uno con coordenadas [x, y, z] |

**Especificación de landmarks:**

- Cantidad exacta: 21 puntos
- Cada punto: array de 3 valores float
- Rango de coordenadas: 0.0 a 1.0 (normalizadas)
- Origen: Coordenadas proporcionadas por MediaPipe Hands

**Response**

| Campo              | Tipo   | Descripción                             |
| ------------------ | ------ | --------------------------------------- |
| letter             | string | Letra predicha (A-Z)                    |
| confidence         | float  | Porcentaje de confianza (0-100)         |
| type               | string | Siempre `"static"`                      |
| processing_time_ms | float  | Tiempo de procesamiento en milisegundos |

---

## 3. Predicción de Letras Dinámicas

### POST `/api/v1/predict/dynamic`

Predice letras que requieren movimiento para su representación.

**Letras soportadas:** J, K, Q, X, Z, Ñ

**Request Body**

| Campo    | Tipo             | Requerido | Descripción            |
| -------- | ---------------- | --------- | ---------------------- |
| sequence | array[15][21][3] | Sí        | Secuencia de 15 frames |

**Especificación de sequence:**

- Frames requeridos: exactamente 15
- Cada frame: 21 landmarks
- Cada landmark: [x, y, z] normalizados
- Captura recomendada: ~30 FPS (0.5 segundos de movimiento)

**Response**

| Campo              | Tipo   | Descripción                     |
| ------------------ | ------ | ------------------------------- |
| letter             | string | Letra predicha                  |
| confidence         | float  | Porcentaje de confianza (0-100) |
| type               | string | Siempre `"dynamic"`             |
| processing_time_ms | float  | Tiempo de procesamiento         |

---

## 4. Predicción de Palabras

### POST `/api/v1/predict/words`

Predice palabras del vocabulario LSM general.

**Vocabulario:** 249 palabras comunes en Lenguaje de Señas Mexicano

**Request Body**

| Campo    | Tipo             | Requerido | Descripción                                        |
| -------- | ---------------- | --------- | -------------------------------------------------- |
| sequence | array[15][21][3] | Sí        | Secuencia de 15 frames (mismo formato que dynamic) |

**Response**

| Campo              | Tipo    | Descripción                                      |
| ------------------ | ------- | ------------------------------------------------ |
| word               | string  | Palabra predicha                                 |
| confidence         | float   | Porcentaje de confianza (0-100)                  |
| phrase             | string  | Frase acumulada de predicciones anteriores       |
| accepted           | boolean | `true` si la palabra pasó el umbral de confianza |
| processing_time_ms | float   | Tiempo de procesamiento                          |

**Notas:**

- El campo `phrase` acumula palabras aceptadas automáticamente
- Usar `/predict/words/clear` para reiniciar la frase

---

## 5. Predicción de Vocabulario Médico

### POST `/api/v1/predict/holistic`

Predice palabras del vocabulario médico usando pose completa del cuerpo y ambas manos.

**Vocabulario:** 150 términos médicos en LSM

**Request Body**

| Campo     | Tipo       | Requerido | Descripción                              |
| --------- | ---------- | --------- | ---------------------------------------- |
| landmarks | array[226] | Sí        | Vector de 226 características holísticas |

**Composición del vector (226 valores):**

| Componente          | Puntos | Valores por punto       | Total                         |
| ------------------- | ------ | ----------------------- | ----------------------------- |
| Pose (torso/brazos) | 33     | 4 (x, y, z, visibility) | 132                           |
| Mano izquierda      | 21     | 3 (x, y, z)             | 63                            |
| Mano derecha        | 21     | 3 (x, y, z)             | 63                            |
| **Total**           |        |                         | **258** → procesado a **226** |

**Response**

| Campo              | Tipo    | Descripción                     |
| ------------------ | ------- | ------------------------------- |
| word               | string  | Palabra médica predicha         |
| confidence         | float   | Porcentaje de confianza (0-100) |
| phrase             | string  | Frase acumulada                 |
| accepted           | boolean | Si la palabra fue aceptada      |
| processing_time_ms | float   | Tiempo de procesamiento         |

---

## 6. Gestión de Buffer

### GET `/api/v1/predict/words/stats`

Obtiene estadísticas del buffer de predicción de palabras.

**Response**

| Campo          | Tipo   | Descripción                        |
| -------------- | ------ | ---------------------------------- |
| buffer_size    | int    | Cantidad de predicciones en buffer |
| phrase_length  | int    | Número de palabras en la frase     |
| current_phrase | string | Frase acumulada actual             |

---

### POST `/api/v1/predict/words/clear`

Limpia el buffer y reinicia la frase acumulada.

**Response**

| Campo   | Tipo   | Descripción        |
| ------- | ------ | ------------------ |
| message | string | `"Buffer cleared"` |

---

### POST `/api/v1/predict/holistic/clear`

Limpia el buffer del modelo holístico.

**Response**

| Campo   | Tipo   | Descripción                 |
| ------- | ------ | --------------------------- |
| message | string | `"Holistic buffer cleared"` |

---

## Estructura de Landmarks (Mano)

Los 21 puntos de la mano siguen el estándar MediaPipe:

| Índice | Nombre     | Ubicación                     |
| ------ | ---------- | ----------------------------- |
| 0      | WRIST      | Muñeca                        |
| 1      | THUMB_CMC  | Base del pulgar               |
| 2      | THUMB_MCP  | Articulación media pulgar     |
| 3      | THUMB_IP   | Articulación superior pulgar  |
| 4      | THUMB_TIP  | Punta del pulgar              |
| 5      | INDEX_MCP  | Base del índice               |
| 6      | INDEX_PIP  | Articulación media índice     |
| 7      | INDEX_DIP  | Articulación superior índice  |
| 8      | INDEX_TIP  | Punta del índice              |
| 9-12   | MIDDLE\_\* | Dedo medio (misma estructura) |
| 13-16  | RING\_\*   | Dedo anular                   |
| 17-20  | PINKY\_\*  | Dedo meñique                  |

---

## Códigos de Error

| Código HTTP | Significado         | Causa común                               |
| ----------- | ------------------- | ----------------------------------------- |
| 400         | Bad Request         | Formato de JSON inválido                  |
| 422         | Validation Error    | Cantidad incorrecta de landmarks o frames |
| 502         | Bad Gateway         | Vision Service no disponible              |
| 503         | Service Unavailable | Servicio en mantenimiento                 |
| 504         | Gateway Timeout     | Timeout en procesamiento ML               |

**Formato de error:**

| Campo  | Tipo   | Descripción           |
| ------ | ------ | --------------------- |
| detail | string | Descripción del error |

---

## Consideraciones de Integración

### Frecuencia de Llamadas

- **Letras estáticas:** Máximo 2 llamadas/segundo recomendado
- **Letras dinámicas/Palabras:** 1 llamada cada 500ms después de acumular 15 frames
- **Health check:** Cada 30 segundos para monitoreo

### Umbral de Confianza

- Recomendado mostrar predicciones con `confidence >= 70`
- Las palabras con `accepted: false` tienen confianza por debajo del umbral interno

### Acumulación de Frames

- Para endpoints que requieren `sequence`, acumular 15 frames consecutivos
- Mantener buffer FIFO (First In, First Out)
- No enviar frames duplicados

---

## Variables de Entorno Recomendadas (Frontend)

| Variable    | Desarrollo              | Producción                  |
| ----------- | ----------------------- | --------------------------- |
| API_URL     | `http://localhost:8000` | URL del servidor desplegado |
| API_TIMEOUT | 30000                   | 30000 (milisegundos)        |

---

**Última actualización:** Enero 2026
