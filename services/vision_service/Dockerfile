# Dockerfile para SignSpeak Vision Service
# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Metadatos
LABEL maintainer="SignSpeak Team"
LABEL description="Vision Service para reconocimiento de lenguaje de señas LSM"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# Instalar dependencias del sistema para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar SOLO código fuente (los modelos se descargan de HF Hub en runtime)
COPY src/ /app/src/

# Exponer puerto — Cloud Run inyecta $PORT dinámicamente
EXPOSE 8080

# CMD usa $PORT de Cloud Run (default 8080 para desarrollo local)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
