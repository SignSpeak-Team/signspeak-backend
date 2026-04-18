<div align="center">

# 🤟 SignSpeak — Backend

### Sistema de traducción en tiempo real de Lenguaje de Señas (LSM) usando visión por computadora y deep learning

[![CI](https://github.com/alanctinaDev/signSpeak/actions/workflows/ci.yml/badge.svg)](https://github.com/alanctinaDev/signSpeak/actions/workflows/ci.yml)
[![CD](https://github.com/alanctinaDev/signSpeak/actions/workflows/cd.yml/badge.svg)](https://github.com/alanctinaDev/signSpeak/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![AWS ECS](https://img.shields.io/badge/AWS-ECS_Fargate-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/ecs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📋 Tabla de contenidos

1. [Descripción](#-descripción)
2. [Arquitectura](#️-arquitectura)
3. [Stack tecnológico](#️-stack-tecnológico)
4. [Requisitos previos](#-requisitos-previos)
5. [Instalación](#-instalación)
6. [Ejecutar localmente](#-ejecutar-localmente)
7. [Variables de entorno](#-variables-de-entorno)
8. [Tests](#-tests)
9. [Endpoints de la API](#-endpoints-de-la-api)
10. [Estructura del proyecto](#-estructura-del-proyecto)
11. [Despliegue en AWS](#-despliegue-en-aws)
12. [Contribución](#-contribución)
13. [Licencia](#-licencia)
14. [Autores](#-autores)

---

## 📖 Descripción

**SignSpeak Backend** es un sistema de microservicios para traducción de Lenguaje de Señas Mexicano (LSM) a texto en español. Combina visión por computadora con deep learning para soportar cuatro modalidades de predicción:

| Modalidad              | Modelo                               | Vocabulario                |
| ---------------------- | ------------------------------------ | -------------------------- |
| **Letras estáticas**   | CNN Keras (`sign_model.keras`)       | 21 letras del alfabeto LSM |
| **Letras dinámicas**   | LSTM Keras (`lstm_letters.keras`)    | J, K, Q, X, Z, Ñ           |
| **Palabras LSM**       | CNN temporal (`words_model.keras`)   | 249 señas de uso general   |
| **Vocabulario médico** | MS-G3D + Holístico (`best_model.h5`) | 150 términos médicos       |

---

## 🏗️ Arquitectura

```
Internet
    │
    ▼
[Application Load Balancer]  — puerto 80/443
    │
    └──► api-gateway     (FastAPI, puerto 7860)
              │  HTTP interno
              ├──► vision-service      (FastAPI + TF/Keras/MediaPipe, puerto 8000)
              └──► translation-service (FastAPI, puerto 8001)

Infrastructure:
  ECR   → Repositorios Docker privados
  ECS   → Fargate (sin servidores que gestionar)
  VPC   → Subnets públicas (ALB) + privadas (ECS)
  IAM   → Roles con mínimo privilegio
  CW    → CloudWatch Logs por servicio
```

### Flujo de una predicción

```
App móvil
  │  POST { landmarks: [[x,y,z] × 21] }
  ▼
api_gateway  ──────────────────────────────────► valida con Pydantic
  │  POST /api/v1/predict/static (interno)
  ▼
vision_service ──► carga sign_model.keras ──► { letter, confidence, type }
  │
  ▼
api_gateway  ──────────────────────────────────► reenvía respuesta al cliente
```

---

## 🛠️ Stack tecnológico

| Capa              | Tecnología                      | Detalle                             |
| ----------------- | ------------------------------- | ----------------------------------- |
| **API**           | FastAPI 0.115 + Uvicorn         | Async, auto-docs (OpenAPI/Swagger)  |
| **Validación**    | Pydantic v2 + pydantic-settings | Tipado estricto de request/response |
| **ML — Static**   | TensorFlow 2.16 + Keras 3.13    | CNN para 21 letras estáticas        |
| **ML — Dynamic**  | Keras LSTM                      | Secuencias de 15 frames             |
| **ML — 3D Graph** | PyTorch + MS-G3D                | Vocabulario médico 150 señas        |
| **Visión**        | MediaPipe 0.10.9 + OpenCV 4.11  | Detección de landmarks de mano      |
| **Contenedores**  | Docker + Docker Compose         | Orquestación local y prod           |
| **IaC**           | Terraform ~1.7                  | VPC, ECR, ECS, ALB, IAM             |
| **Cloud**         | AWS ECS Fargate                 | Sin gestión de servidores           |
| **CI/CD**         | GitHub Actions                  | ci.yml + cd.yml                     |
| **Testing**       | pytest 8 + pytest-asyncio       | Unitarios, integración y E2E        |
| **Linting**       | ruff + black + mypy             | Calidad y formato de código         |

---

## ✅ Requisitos previos

| Herramienta        | Versión mínima | Instalación                        |
| ------------------ | -------------- | ---------------------------------- |
| **Python**         | 3.11           | [python.org](https://python.org)   |
| **pip**            | 23+            | Incluido con Python                |
| **Docker**         | 24+            | [docker.com](https://docker.com)   |
| **Docker Compose** | 2.x            | Incluido con Docker Desktop        |
| **Git**            | 2.x            | [git-scm.com](https://git-scm.com) |
| **Terraform**      | ≥ 1.7          | Solo para deploy en AWS            |
| **AWS CLI**        | v2             | Solo para deploy en AWS            |

---

## 📦 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/alanctinaDev/signSpeak.git
cd signSpeak

# 2. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Instalar herramientas de desarrollo
pip install -r requirements-dev.txt

# 4. Instalar dependencias de cada servicio (opcional, para desarrollo individual)
pip install -r services/api_gateway/requirements.txt
pip install -r services/translation_service/requirements.txt
pip install -r services/vision_service/requirements.txt

# 5. Copiar variables de entorno
cp .env.example .env
# Edita .env con tus valores (ver sección Variables de entorno)
```

---

## ▶️ Ejecutar localmente

### Opción A — Docker Compose (recomendado)

Levanta los 3 servicios con un solo comando:

```bash
docker compose up --build
```

| Servicio            | URL local                  |
| ------------------- | -------------------------- |
| API Gateway         | http://localhost:8000      |
| Docs (Swagger)      | http://localhost:8000/docs |
| Vision Service      | http://localhost:8002      |
| Translation Service | http://localhost:8001      |

Para detener:

```bash
docker compose down
```

### Opción B — Servicios individuales (desarrollo)

**1. Vision Service** (levántalo primero — carga los modelos)

```bash
cd services/vision_service
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Translation Service**

```bash
cd services/translation_service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

**3. API Gateway**

```bash
cd services/api_gateway
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ El vision-service tarda ~60 segundos en arrancar mientras carga los modelos ML en memoria.

### Verificar que todo funciona

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test de predicción estática
curl -X POST http://localhost:8000/api/v1/predict/static \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.04,0.04,0],[0.08,0.08,0],[0.12,0.12,0],[0.16,0.16,0],[0.20,0.20,0],[0.24,0.24,0],[0.28,0.28,0],[0.32,0.32,0],[0.36,0.36,0],[0.40,0.40,0],[0.44,0.44,0],[0.48,0.48,0],[0.52,0.52,0],[0.56,0.56,0],[0.60,0.60,0],[0.64,0.64,0],[0.68,0.68,0],[0.72,0.72,0],[0.76,0.76,0],[0.80,0.80,0],[0.84,0.84,0]]}'
```

---

## 🔐 Variables de entorno

Copia `.env.example` a `.env` y edita los valores:

```bash
cp .env.example .env
```

| Variable                  | Servicio                  | Descripción                              | Default                           |
| ------------------------- | ------------------------- | ---------------------------------------- | --------------------------------- |
| `ENVIRONMENT`             | Todos                     | `development` / `staging` / `production` | `development`                     |
| `PORT`                    | api-gateway               | Puerto del servidor                      | `7860`                            |
| `TRANSLATION_SERVICE_URL` | api-gateway               | URL interna del translation-service      | `http://translation-service:8001` |
| `VISION_SERVICE_URL`      | api-gateway / translation | URL interna del vision-service           | `http://vision-service:8002`      |
| `CORS_ORIGINS`            | api-gateway               | Orígenes CORS permitidos                 | `*`                               |
| `VISION_PORT`             | vision-service            | Puerto del servidor                      | `8000`                            |
| `LOG_LEVEL`               | Todos                     | `DEBUG` / `INFO` / `WARNING` / `ERROR`   | `INFO`                            |
| `AWS_REGION`              | Deploy                    | Región de AWS                            | `us-east-1`                       |
| `AWS_ACCESS_KEY_ID`       | Deploy local              | Clave AWS (en ECS se inyecta via IAM)    | —                                 |
| `AWS_SECRET_ACCESS_KEY`   | Deploy local              | Secret AWS                               | —                                 |

> ⚠️ **Nunca** commitees el `.env` real. El archivo `.env.example` es el único que va al repositorio.

---

## 🧪 Tests

### Tests unitarios

```bash
# Todos los unitarios
pytest tests/unit/ -v

# Por servicio
pytest tests/unit/api_gateway/ -v
pytest tests/unit/vision_service/ -v --tb=short -m "not slow"
pytest tests/unit/translation_service/ -v
```

### Tests de integración

> Requiere Docker Compose corriendo o las URLs correctas en `.env`

```bash
# Preparar entorno de integración
cp tests/integration/.env.example tests/integration/.env

# Correr tests de integración
pytest tests/integration/ -v
```

### Tests E2E

```bash
# Pipeline completo (requiere todos los servicios corriendo)
pytest tests/e2e/ -v
```

### Con cobertura

```bash
pytest tests/unit/api_gateway/ tests/unit/translation_service/ \
  --cov=services \
  --cov-report=term-missing \
  --cov-report=html:htmlcov
```

Abre `htmlcov/index.html` para ver el reporte visual.

### Linting y formato

```bash
# Lint con ruff
ruff check . --output-format=github

# Verificar formato con black
black --check --diff .

# Type checking con mypy
mypy services/
```

---

## 📡 Endpoints de la API

Todos los endpoints están documentados en **Swagger UI**: `http://localhost:8000/docs`

| Método | Endpoint                   | Descripción                             | Body                              |
| ------ | -------------------------- | --------------------------------------- | --------------------------------- |
| `GET`  | `/api/v1/health`           | Estado del api-gateway                  | —                                 |
| `GET`  | `/api/v1/status`           | Estado de todos los servicios internos  | —                                 |
| `POST` | `/api/v1/predict/static`   | Letra estática — 21 landmarks           | `{ landmarks: [[x,y,z]×21] }`     |
| `POST` | `/api/v1/predict/dynamic`  | Letra dinámica — 15 frames              | `{ sequence: [[[x,y,z]×21]×15] }` |
| `POST` | `/api/v1/predict/words`    | Palabra LSM — 249 vocab                 | `{ sequence: [[[x,y,z]×21]×15] }` |
| `POST` | `/api/v1/predict/holistic` | Palabra médica — 150 vocab (pose+manos) | `{ sequence: [...] }`             |
| `POST` | `/api/v1/translate/`       | Endpoint legacy de traducción           | `{ imageUri: "..." }`             |

### Ejemplo de respuesta

```json
{
  "letter": "A",
  "confidence": 92.5,
  "type": "static",
  "processing_time_ms": 18
}
```

---

## 📂 Estructura del proyecto

> Ver [`STRUCTURE.md`](STRUCTURE.md) para la descripción detallada de cada archivo.

```
signSpeak/
├── .github/workflows/    # ci.yml · cd.yml · deploy-hf.yml
├── services/
│   ├── api_gateway/      # Puerta de entrada — enruta requests
│   ├── vision_service/   # Modelos ML (TF/Keras/PyTorch) + MediaPipe
│   └── translation_service/   # Lógica de traducción de secuencias
├── terraform/            # IaC: VPC · ECR · ECS · ALB · IAM
├── tests/
│   ├── unit/             # Tests por servicio
│   ├── integration/      # Tests con servicios levantados
│   └── e2e/              # Pipeline completo
├── .env.example
├── docker-compose.yml
├── pyproject.toml        # ruff · black · mypy
└── pytest.ini
```

---

## ☁️ Despliegue en AWS

El despliegue se hace automáticamente vía **GitHub Actions** (`cd.yml`) al hacer push a `main` o crear un tag `v*.*.*`.

### Configuración inicial (una sola vez)

```bash
# 1. Crear bucket S3 para el estado de Terraform
aws s3 mb s3://signspeakterraformstate --region us-east-1

# 2. Crear tabla DynamoDB para el locking
aws dynamodb create-table \
  --table-name signspeakterraformlock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# 3. Copiar y editar variables de Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edita terraform.tfvars con tu configuración

# 4. Inicializar y aplicar
terraform init
terraform apply
```

### Secrets necesarios en GitHub Actions

Ve a `Settings → Secrets → Actions` y agrega:

| Secret                  | Descripción                            |
| ----------------------- | -------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | IAM con permisos ECR + ECS + ELB + SSM |
| `AWS_SECRET_ACCESS_KEY` |                                        |
| `HF_TOKEN`              | Token de Hugging Face Spaces           |
| `HF_SPACE_ID`           | ID del Space del api-gateway           |

### Variables del repositorio

Ve a `Settings → Variables → Actions`:

| Variable          | Ejemplo            |
| ----------------- | ------------------ |
| `AWS_REGION`      | `us-east-1`        |
| `TF_PROJECT_NAME` | `signspeakbackend` |

### Deploy manual

```bash
# Build + push de imágenes a ECR
chmod +x terraform/scripts/build_and_push.sh
./terraform/scripts/build_and_push.sh

# Deploy completo (build → push → plan → apply)
chmod +x terraform/scripts/deploy.sh
./terraform/scripts/deploy.sh v1.2.3
```

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Sigue estos pasos:

1. **Forkea** el repositorio
2. Crea una branch desde `develop`:
   ```bash
   git checkout -b feat/nombre-de-tu-feature
   ```
3. Haz tus cambios y **asegúrate de que todo pasa**:
   ```bash
   ruff check .
   black --check .
   pytest tests/unit/ -v
   ```
4. Commitea con [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(vision): agregar soporte para detección de dos manos
   fix(api-gateway): corregir timeout en requests largos
   ```
5. Abre un **Pull Request** hacia `develop` usando la [plantilla](.github/PULL_REQUEST_TEMPLATE.md)

### Guías de estilo

- **Python 3.11+** con type hints en todas las funciones públicas
- **ruff** no debe reportar errores (`ruff check .`)
- **black** para formato (`black .`)
- **mypy** para type checking (`mypy services/`)
- Tests para toda funcionalidad nueva en `tests/unit/<servicio>/`
- Nunca commitees modelos (`.keras`, `.h5`, `.pt`, `.pkl`) — son demasiado grandes para git

---

## 📄 Licencia

Distribuido bajo la licencia **MIT**. Ver [`LICENSE`](LICENSE) para más detalles.

---

## 👥 Autores

| Nombre                | GitHub                                           | Rol                     |
| --------------------- | ------------------------------------------------ | ----------------------- |
| **Alan Lopez Cetina** | [@alanctinaDev](https://github.com/alanctinaDev) | Fullstack / ML / DevOps |

---

<div align="center">

Alan de los Santos Lopez Cetina — Matrícula: 2202116
Ángel Jonás Rosales Gonzales — Matrícula: 2202022
José Arturo González Flores — Matrícula: 2202012
Cesar Enrique Bernal Zurita— Matrícula: 2201100
Ángel David Quintana Pacheco — Matrícula: 2102165
Cristian Daniel Lázaro Acosta — Matrícula: 2202055
Ángel Adrián Yam Huchim — Matricula: 2202109
