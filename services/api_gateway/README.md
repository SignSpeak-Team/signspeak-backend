---
title: SignSpeak API Gateway
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SignSpeak API Gateway

REST API for the SignSpeak sign language recognition system.

## Endpoints

- `GET /docs` — Swagger UI
- `GET /health` — Health check
- `POST /predict/lse` — LSE sign prediction (MSG3D)
- `POST /predict/static` — Static alphabet prediction
- `POST /predict/dynamic` — Dynamic alphabet prediction

## Architecture

- **API Gateway** (this Space) — Routes and orchestrates requests
- **Vision Service** — ML model inference (MediaPipe + MSG3D + Keras)
- **Translation Service** — Text post-processing
