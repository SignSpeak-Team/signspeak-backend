# Roadmap: SignSpeak + LSTM + AWS + DevOps

- [x] **Fase 1: LSTM y Modelos**

  - [x] Entrenar modelo estático (21 letras) <!-- id: 0 -->
  - [x] Entrenar modelo dinámico LSTM (J, K, Ñ, Q, X, Z) <!-- id: 1 -->
  - [x] Generar estadísticas y gráficos de entrenamiento <!-- id: 2 -->
  - [x] Generar diagramas de arquitectura de modelos <!-- id: 3 -->
    - [x] Integrar ambos modelos en `realtime_demo.py` <!-- id: 4 -->
    - [x] Optimizar latencia de detección en demo <!-- id: 5 -->
    - [x] Optimizaciones de código aplicadas (GPU requiere instalación manual de CUDA) <!-- id: 6 -->
    - [x] Implementar Data Augmentation - 99.79% accuracy logrado <!-- id: 7 -->

- [x] **Fase 1.5: Refactorización Arquitectura** <!-- id: 8 -->

  - [x] Crear estructura dev/models/src <!-- id: 9 -->
  - [x] Mover datasets raw a /dev <!-- id: 10 -->
  - [x] Consolidar modelos en /models <!-- id: 11 -->
  - [x] Mover scripts de training a /dev/scripts <!-- id: 12 -->
  - [x] Crear src/config.py centralizado <!-- id: 13 -->
  - [x] Actualizar .gitignore <!-- id: 14 -->
  - [x] Verificar que todo funciona <!-- id: 15 -->

- [x] **Fase 2: Dockerización** <!-- id: 6 -->

  - [x] Crear Dockerfile para `vision_service` <!-- id: 7 -->
  - [x] Integrar vision_service en `docker-compose.yml` <!-- id: 8 -->
  - [x] Verificar funcionamiento en contenedores <!-- id: 9 -->

- [ ] **Fase 3: AWS Setup** ⏸️ PAUSADA (prioridad: entrenar más señas) <!-- id: 10 -->

  - [x] Configurar AWS CLI y credenciales <!-- id: 11 -->
  - [ ] Crear repositorios ECR (Elastic Container Registry) <!-- id: 12 -->
  - [ ] Subir imágenes Docker a ECR <!-- id: 13 -->
  - [ ] Configurar base de datos RDS (PostgreSQL) <!-- id: 14 -->
  - [ ] Configurar bucket S3 para almacenamiento de modelos <!-- id: 15 -->

- [ ] **Fase 4: Despliegue y CI/CD** ⏸️ PAUSADA <!-- id: 16 -->
  - [ ] Desplegar en EC2 (instancia t2.micro Free Tier) <!-- id: 17 -->
  - [ ] Configurar GitHub Actions para CI (tests) <!-- id: 18 -->
  - [ ] Configurar GitHub Actions para CD (deploy automático) <!-- id: 19 -->

---

## 🎯 PRIORIDAD ACTUAL: Expandir Reconocimiento

### Fase 2.5: Entrenar Palabras (249 palabras)

- [x] Obtener dataset de 249 palabras
- [x] Organizar estructura de carpetas (carpeta por palabra)
- [x] Procesar videos → extraer secuencias de landmarks (1876 secuencias)
- [x] Entrenar modelo LSTM para palabras (91.24% accuracy con data augmentation)
- [x] Validar accuracy del modelo
- [x] Integrar nuevo modelo en API ✅ Mergeado a main
- [ ] Implementar Buffer Inteligente de Palabras → PRÓXIMO 📋

### Fase 2.6: Buffer Inteligente + Endpoint API

- [ ] Crear `word_buffer.py` (filtrado de repeticiones)
- [ ] Integrar buffer en `predictor.py`
- [ ] Tests del buffer
- [ ] Endpoint API `/predict/words` (usa buffer)
- [ ] Demo actualizado

### Fase 2.7: Servicio de Frases (Claude API + TTS)

- [ ] Crear `phrase_service/` microservicio
- [ ] Integrar Claude API (Haiku) para conjugación LSM→Español
- [ ] Implementar buffer de palabras detectadas
- [ ] Agregar endpoint `/api/v1/translate`
- [ ] Integrar TTS (Google/Azure/gTTS)
- [ ] Agregar endpoint `/api/v1/speak`
- [ ] Conectar con vision_service

### Arquitectura Final:

```
Cámara → Vision Service → Phrase Service → TTS
           (LSTM)         (Claude API)    (Audio)
              ↓                ↓              ↓
          "palabra"      "frase correcta"   🔊
```
