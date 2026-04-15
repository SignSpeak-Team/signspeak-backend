# 📁 Estructura del Proyecto — SignSpeak Backend

Backend de microservicios construido con **Python 3.11 + FastAPI + Docker**.
Desplegado en **AWS ECS Fargate** vía **Terraform**.

```
signSpeak/                               # Raíz del monorepo de backend
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                       # CI: linting, tests unitarios e integración
│   │   ├── cd.yml                       # CD: build ECR → Terraform → ECS Fargate
│   │   └── deploy-hf.yml               # Deploy legacy a Hugging Face Spaces
│   └── PULL_REQUEST_TEMPLATE.md        # Plantilla estándar para Pull Requests
│
├── services/                           # Microservicios Python
│   │
│   ├── api_gateway/                    # 🚪 Puerta de entrada — puerto 7860
│   │   ├── src/
│   │   │   ├── config/                 # Configuración de la aplicación
│   │   │   ├── routes/                 # Endpoints FastAPI
│   │   │   │   ├── health.py           # GET /api/v1/health  y  /status
│   │   │   │   ├── prediction.py       # POST /api/v1/predict/{static|dynamic|words|holistic}
│   │   │   │   └── translate.py        # POST /api/v1/translate/ (legacy)
│   │   │   ├── schemas/                # Modelos Pydantic de request/response
│   │   │   │   ├── prediction.py       # StaticPredictionRequest, LetterPredictionResponse…
│   │   │   │   └── translate.py        # TranslateRequest, TranslateResponse
│   │   │   ├── services/               # Clientes HTTP hacia servicios internos
│   │   │   │   ├── http_client.py      # Cliente base con httpx
│   │   │   │   ├── vision_client.py    # Llamadas al vision-service
│   │   │   │   └── translation_client.py  # Llamadas al translation-service
│   │   │   ├── main.py                 # App FastAPI — CORS, lifespan, routers
│   │   │   └── settings.py             # Configuración vía pydantic-settings / .env
│   │   ├── Dockerfile                  # python:3.11-slim, uvicorn en 7860
│   │   └── requirements.txt            # fastapi, uvicorn, httpx, pydantic-settings
│   │
│   ├── vision_service/                 # 👁️  Modelos ML — puerto 8000
│   │   ├── src/
│   │   │   ├── api/
│   │   │   │   ├── main.py             # App FastAPI del vision-service
│   │   │   │   ├── models/             # Esquemas Pydantic internos
│   │   │   │   └── routes/
│   │   │   │       ├── health.py       # GET /api/v1/health
│   │   │   │       ├── media.py        # POST /api/v1/predict/video (stream)
│   │   │   │       └── prediction.py   # POST /api/v1/predict/{static|dynamic|words|holistic}
│   │   │   ├── core/                   # Lógica ML y procesamiento
│   │   │   │   ├── predictor.py        # Predictor principal (static + dynamic + words)
│   │   │   │   ├── holistic_extractor.py   # Extractor de landmarks pose+manos (holistic)
│   │   │   │   ├── sequence_processor.py   # Preprocesado de secuencias de frames
│   │   │   │   ├── video_processor.py      # Pipeline de procesamiento de video
│   │   │   │   ├── word_buffer.py          # Buffer de palabras para detección continua
│   │   │   │   ├── msg3d_model.py          # Arquitectura MS-G3D (vocabulario médico)
│   │   │   │   ├── msg3d_predictor.py      # Predictor con MS-G3D
│   │   │   │   ├── msg3d_graph.py          # Definición del grafo de la mano
│   │   │   │   └── metrics.py              # Métricas Prometheus
│   │   │   └── config.py               # Rutas a modelos, umbrales y parámetros ML
│   │   ├── models/                     # Pesos de modelos ML (no versionados en git)
│   │   │   ├── sign_model.keras         # Modelo estático — 21 letras
│   │   │   ├── label_encoder.pkl
│   │   │   ├── lstm_letters.keras       # Modelo dinámico — J, K, Q, X, Z, Ñ
│   │   │   ├── lstm_label_encoder.pkl
│   │   │   ├── words_model.keras        # Modelo de palabras — 249 señas
│   │   │   ├── words_label_encoder.pkl
│   │   │   ├── best_model.h5            # Modelo holístico — vocabulario médico
│   │   │   ├── holistic_label_encoder.pkl
│   │   │   ├── msg3d_lse.pt             # Modelo MS-G3D (LSE)
│   │   │   ├── msg3d_labels.pkl
│   │   │   └── hand_landmarker.task     # Modelo MediaPipe Hand Landmarker
│   │   ├── Dockerfile                   # python:3.11-slim + libGL/OpenCV/MediaPipe
│   │   ├── main.py                      # Entry point para uvicorn
│   │   └── requirements.txt             # tensorflow, keras, mediapipe, opencv, fastapi
│   │
│   └── translation_service/             # 🔤 Traducción de secuencias — puerto 8001
│       ├── src/
│       │   ├── routes/
│       │   │   └── translate.py         # POST /translate/
│       │   ├── schemas/                 # Modelos Pydantic
│       │   ├── main.py                  # App FastAPI
│       │   └── settings.py             # Configuración: VISION_SERVICE_URL, PORT
│       ├── Dockerfile                   # python:3.11-slim, uvicorn en 8001
│       └── requirements.txt            # fastapi, uvicorn
│
├── terraform/                          # ☁️  Infraestructura como Código (AWS)
│   ├── main.tf                         # Provider AWS + backend S3 + llamadas a módulos
│   ├── variables.tf                    # Variables de entrada con defaults
│   ├── outputs.tf                      # URLs del ALB, ARNs, cluster name…
│   ├── terraform.tfvars.example        # Plantilla de valores (no commitear .tfvars real)
│   ├── modules/
│   │   ├── vpc/                        # VPC, subnets pub/priv, IGW, NAT GW
│   │   ├── ecr/                        # Repositorios Docker privados + lifecycle policy
│   │   ├── iam/                        # Execution role + Task role + políticas SSM/S3
│   │   ├── alb/                        # Application Load Balancer, SG, listeners
│   │   └── ecs/                        # Cluster Fargate, task defs, services, auto scaling
│   └── scripts/
│       ├── build_and_push.sh           # Build de las 3 imágenes y push a ECR
│       └── deploy.sh                   # Wrapper: build → push → terraform apply
│
├── tests/                              # Tests automatizados
│   ├── unit/
│   │   ├── api_gateway/                # Tests unitarios del api-gateway
│   │   ├── translation_service/        # Tests unitarios del translation-service
│   │   └── vision_service/             # Tests unitarios del vision-service
│   ├── integration/
│   │   ├── conftest.py                 # Fixtures compartidos (cliente HTTP de prueba)
│   │   ├── test_api_gateway.py         # Integración: api-gateway ↔ servicios internos
│   │   ├── test_vision_service.py      # Integración: vision-service con modelos reales
│   │   ├── test_media_pipeline.py      # Pipeline de video end-to-end
│   │   └── test_database.py            # Integración con persistencia (si aplica)
│   └── e2e/
│       └── test_full_pipeline.py       # Pipeline completo: landmarks → letra/palabra
│
├── .env.example                        # Variables de entorno de ejemplo para todos los servicios
├── .gitignore                          # Excluye .env, .terraform, modelos grandes, __pycache__
├── docker-compose.yml                  # Orquestación local de los 3 servicios
├── pyproject.toml                      # Configuración de ruff, black y mypy
├── pytest.ini                          # Marcadores y configuración de pytest
├── requirements-dev.txt                # Herramientas de desarrollo: pytest, ruff, black, mypy
└── README.md                           # Documentación principal del proyecto
```

---

## 🧩 Descripción de microservicios

| Servicio                | Puerto | Responsabilidad                                               | Tecnología destacada                 |
| ----------------------- | ------ | ------------------------------------------------------------- | ------------------------------------ |
| **api_gateway**         | 7860   | Punto de entrada único — enruta requests al servicio correcto | FastAPI, httpx, pydantic-settings    |
| **vision_service**      | 8000   | Carga los modelos ML y hace la predicción de señas            | TensorFlow, Keras, MediaPipe, MS-G3D |
| **translation_service** | 8001   | Traduce secuencias de señas a texto/frases                    | FastAPI, lógica de buffer            |

## ☁️ Arquitectura AWS

```
Internet
    │
    ▼
[ALB]  puerto 80/443
    │
    └──► ECS Service: api-gateway  (Fargate, 0.25 vCPU / 512 MB)
              │
              ├──► ECS Service: vision-service       (Fargate, 1 vCPU / 2 GB)
              └──► ECS Service: translation-service  (Fargate, 0.25 vCPU / 512 MB)

[ECR]  → repositorios privados de imágenes Docker
[VPC]  → subnets públicas (ALB) + privadas (ECS) + NAT Gateway
[IAM]  → Execution Role + Task Role con mínimo privilegio
[CW]   → CloudWatch Logs — un log group por servicio
```

## 🔗 Flujo de una predicción

```
App móvil
    │  POST /api/v1/predict/static { landmarks: [[x,y,z] × 21] }
    ▼
api_gateway          → valida el request con Pydantic
    │  POST /api/v1/predict/static (interno)
    ▼
vision_service       → carga sign_model.keras → predice letra
    │  { letter: "A", confidence: 92.5, type: "static" }
    ▼
api_gateway          → reenvía la respuesta al cliente
    │
    ▼
App móvil            → muestra resultado en translation-result.tsx
```
