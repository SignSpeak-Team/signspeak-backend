# 🤟 SignSpeak

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

> Sistema de traducción en tiempo real de Lenguaje de Señas (LSM/LSE) a texto en español, usando visión por computadora y modelos de deep learning (MSG3D + MediaPipe).

---

## 🛠️ Stack

| Capa            | Tecnología                                                                 |
| --------------- | -------------------------------------------------------------------------- |
| **Backend**     | FastAPI, Python 3.11, Uvicorn                                              |
| **ML / Visión** | PyTorch, MSG3D, MediaPipe, OpenCV                                          |
| **Modelos**     | Letras estáticas · Letras dinámicas · Palabras LSM (249) · Holístico (150) |
| **DevOps**      | Docker, GitHub Actions                                                     |
| **Deploy**      | Hugging Face Spaces · Vercel                                               |

---

## 🚀 Quick Start

```bash
git clone https://github.com/alanctinaDev/signSpeak.git
cd signSpeak
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r services/vision_service/requirements.txt
pip install -r requirements-dev.txt
```

**1. Vision Service** (puerto 8001)

```powershell
cd services\vision_service
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**2. API Gateway** (puerto 8000)

```powershell
cd services\api_gateway
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva: http://localhost:8000/docs

---

## 📡 Endpoints

| Método | Endpoint                   | Descripción                   |
| ------ | -------------------------- | ----------------------------- |
| `POST` | `/api/v1/predict/static`   | Letra estática (21 landmarks) |
| `POST` | `/api/v1/predict/dynamic`  | Letra dinámica (15 frames)    |
| `POST` | `/api/v1/predict/words`    | Palabra LSM (249 vocab)       |
| `POST` | `/api/v1/predict/holistic` | Palabra médica (150 vocab)    |

---

## 🧪 Tests

```bash
pytest tests/unit/
pytest tests/integration/
```

---

## 👨‍💻 Autor

**Alan Lopez Cetina** — [@alanctinaDev](https://github.com/alanctinaDev)

📄 MIT License
