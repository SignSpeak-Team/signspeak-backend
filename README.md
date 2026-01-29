# 🤟 SignSpeak - Traductor de Lenguaje de Señas Mexicano

Sistema de traducción en tiempo real de Lenguaje de Señas Mexicano (LSM) a texto en español mediante visión por computadora y machine learning.

---

## 📖 Descripción

SignSpeak utiliza MediaPipe para detección de manos y TensorFlow para clasificación de señas, implementado con arquitectura de microservicios para escalabilidad y mantenibilidad.

## 🏗️ Arquitectura

**4 Microservicios principales:**

- **API Gateway** (8000): Punto de entrada, autenticación, rate limiting
- **Translation Service** (8001): Orquestación de traducciones
- **ML Service** (8002): Detección y clasificación de señas
- **Storage Service** (8003): Gestión de archivos multimedia

**Infraestructura:**

- PostgreSQL (metadata), MongoDB (métricas), MinIO (archivos)
- RabbitMQ (comunicación asíncrona entre servicios)
- Docker & Kubernetes (containerización y orquestación)

## 🛠️ Stack Tecnológico

**Backend:** FastAPI, Python 3.13, SQLAlchemy, Pydantic  
**ML:** MediaPipe, TensorFlow/Keras  
**Databases:** PostgreSQL, MongoDB  
**Storage:** MinIO (S3-compatible)  
**DevOps:** Docker, Kubernetes, GitHub Actions  
**Monitoring:** Prometheus, Grafana

---

## 🚀 Quick Start

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/alanctinaDev/signSpeak.git
cd signSpeak

# Configurar variables de entorno
cp .env.example .env

# Levantar servicios
docker-compose up -d

# Verificar estado
docker-compose ps
```

### Endpoints principales

```
API Documentation:  http://localhost:8000/docs
Translation API:    http://localhost:8001/docs
ML Service:         http://localhost:8002/docs
Storage Service:    http://localhost:8003/docs
RabbitMQ UI:        http://localhost:15672
MinIO Console:      http://localhost:9001
```

---

## 📦 Comandos Útiles

### Docker

```bash
# Ver logs de un servicio
docker-compose logs -f <service-name>

# Rebuild servicio específico
docker-compose up -d --build <service-name>

# Detener servicios
docker-compose down

# Limpiar todo (incluye volúmenes)
docker-compose down -v
```

### Testing

```bash
# Ejecutar tests
pytest

# Con coverage
pytest --cov=src --cov-report=html
```

### Kubernetes (Producción)

```bash
# Deploy
kubectl apply -f infrastructure/kubernetes/

# Ver estado
kubectl get pods -n signspeak

# Logs
kubectl logs -f <pod-name> -n signspeak
```

---

## 📁 Estructura

```
signSpeak/
├── services/              # Microservicios
│   ├── api-gateway/
│   ├── translation-service/
│   ├── ml-service/
│   └── storage-service/
├── shared/               # Código compartido
├── infrastructure/       # Configs DevOps
├── docs/                # Documentación
└── docker-compose.yml   # Orquestación local
```

---

## 🔄 Flujo de Traducción

1. Usuario envía video → API Gateway
2. Gateway → Translation Service
3. Translation Service → RabbitMQ (cola de procesamiento)
4. ML Service procesa: detecta manos + clasifica señas
5. Resultado → Translation Service → PostgreSQL
6. Usuario obtiene traducción

---

## 👨‍💻 Autor

**Alan Lopez Cetina**  
GitHub: [@alanctinaDev](https://github.com/alanctinaDev)

---

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

**⭐ Star este proyecto si te resulta útil**
