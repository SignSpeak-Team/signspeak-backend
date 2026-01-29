# 🚀 Quick Start: Railway + Google Cloud

## Resumen Ultra-Rápido

1. **Google Cloud Run** (Vision + Translation) - GRATIS
2. **Railway** (API Gateway) - $1/mes

---

## 📝 Pasos Rápidos

### 1. Google Cloud (10 min)

```powershell
# Instalar gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login
gcloud config set project signspeak-prod

# Deploy Vision
cd services/vision_service
gcloud run deploy vision-service --source . --region us-central1 --allow-unauthenticated

# Deploy Translation
cd ../translation_service
gcloud run deploy translation-service --source . --region us-central1 --allow-unauthenticated --set-env-vars VISION_SERVICE_URL=<tu-url-vision>
```

### 2. Railway (5 min)

1. railway.app → Login con GitHub
2. New Project → Deploy from GitHub
3. Seleccionar repo → Root: `services/api_gateway`
4. Variables:
   ```
   VISION_SERVICE_URL=<tu-url-cloud-run>
   TRANSLATION_SERVICE_URL=<tu-url-cloud-run>
   ```
5. Generate Domain

---

## ✅ Resultado

```
URL: https://signspeak-production.up.railway.app/docs
Costo: $1/mes
Tiempo: 15 minutos
```

---

Ver guía completa en: `guia_railway_cloudrun.md`
