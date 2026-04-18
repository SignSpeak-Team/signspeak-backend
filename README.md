<div align="center">

# 🤟 SignSpeak — Backend

### Sistema de traducción en tiempo real de Lenguaje de Señas (LSM) usando visión por computadora y deep learning

---

**Equipo de Desarrollo:**
*   **Alan de los Santos Lopez Cetina** — Matrícula: 2202116
*   **Ángel Jonás Rosales Gonzales** — Matrícula: 2202022
*   **José Arturo González Flores** — Matrícula: 2202012
*   **Cesar Enrique Bernal Zurita** — Matrícula: 2201100
*   **Ángel David Quintana Pacheco** — Matrícula: 2102165
*   **Cristian Daniel Lázaro Acosta** — Matrícula: 2202055
*   **Ángel Adrián Yam Huchim** — Matricula: 2202109

---

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📋 Tabla de contenidos

1. [Descripción](#-descripción)
2. [Requisitos previos](#-requisitos-previos)
3. [Instalación y Configuración](#-instalación-y-configuración)
4. [Ejecutar localmente](#-ejecutar-localmente)
5. [Tests](#-tests)
6. [Pipeline Local (Docker)](#-pipeline-local-docker)
7. [Despliegue](#-despliegue)
8. [Variables de entorno](#-variables-de-entorno)
9. [Arquitectura](#️-arquitectura)
10. [Stack tecnológico](#️-stack-tecnológico)
11. [Licencia](#-licencia)

---

## 📖 Descripción

**SignSpeak Backend** es un robusto sistema de microservicios diseñado para la traducción de Lenguaje de Señas Mexicano (LSM) a lenguaje natural (texto/audio). Mediante el uso de visión artificial avanzada (MediaPipe) y múltiples modelos de Deep Learning (CNN para estáticos, LSTM para dinámicos y MS-G3D para terminología compleja), el backend orquesta la interpretación semántica y la traducción de secuencias en tiempo real.

---

## ✅ Requisitos previos

| Herramienta | Versión mínima | Instalación |
| :--- | :--- | :--- |
| **Python** | 3.11 | [python.org](https://python.org) |
| **pip** | 23+ | Incluido con Python |
| **Docker** | 24+ | [docker.com](https://docker.com) |
| **Docker Compose** | 2.x | Incluido con Docker Desktop |
| **Git** | 2.x | [git-scm.com](https://git-scm.com) |

---

## 📦 Instalación y Configuración

Sigue estos pasos para configurar tu entorno de desarrollo local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/alanctinaDev/signSpeak.git
   cd signSpeak
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

3. **Instalar dependencias globales y de desarrollo:**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   # Edita el archivo .env con tus credenciales y rutas locales
   ```

---

## ▶️ Ejecutar localmente

### Usando Docker Compose (Recomendado)
Para levantar el ecosistema completo de microservicios (API Gateway, Vision y Translation):
```bash
docker compose up --build
```
- API Gateway (Punto de entrada): `http://localhost:8000`
- Documentación (Swagger): `http://localhost:8000/docs`

---

## 🧪 Tests

Asegura la integridad del sistema ejecutando las pruebas automatizadas:

```bash
# Ejecutar todos los tests unitarios (Python/pytest)
pytest tests/unit/ -v

# Ejecutar tests de integración (Requiere Docker Compose activo)
pytest tests/integration/ -v

# Pruebas End-to-End
pytest tests/e2e/ -v
```

---

## 🔄 Pipeline Local (Docker)

Contamos con scripts de validación que simulan el pipeline de CI para asegurar que el entorno es apto para despliegue:

```bash
# Validar entorno y salud de los servicios
bash scripts/verify.sh
```
Este script verifica variables de entorno críticas, carga de modelos y conectividad de red entre servicios.

---

## ☁️ Despliegue

### Infraestructura como Código (Terraform)
El despliegue en la nube utiliza AWS ECS Fargate. Para inicializar la infraestructura:
```bash
cd terraform
terraform init
terraform apply
```
El despliegue continuo (CD) se gatilla automáticamente mediante GitHub Actions al detectar cambios en la rama `main`.

---

## 🔐 Variables de entorno

Descripción detallada de las variables en `.env.example`:

| Variable | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Define el modo de ejecución (`development`, `staging`, `production`). | `development` |
| `PORT` | Puerto de exposición para el API Gateway principal. | `7860` |
| `TRANSLATION_SERVICE_URL` | Endpoint interno para la comunicación con el servicio de traducción. | `http://translation-service:8001` |
| `VISION_SERVICE_URL` | Endpoint interno para la comunicación con el servicio de visión ML. | `http://vision-service:8002` |
| `CORS_ORIGINS` | Orígenes permitidos para peticiones cruzadas (CORS). | `*` |
| `LOG_LEVEL` | Nivel de verbosidad de los logs (`DEBUG`, `INFO`, `ERROR`). | `INFO` |
| `AWS_REGION` | Región default de AWS para almacenamiento y cómputo. | `us-east-1` |

---

## 🏗️ Arquitectura

El backend utiliza una arquitectura orientada a microservicios para garantizar escalabilidad independiente:
- **API Gateway**: Gestión de tráfico, autenticación y validación.
- **Vision Service**: Carga y ejecución de modelos TensorFlow/Keras y PyTorch.
- **Translation Service**: Lógica asociativa y normalización de frases traducidas.

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Ver el archivo `LICENSE` para más detalles.

---

<div align="center">

**SignSpeak Team** — 2026

</div>
