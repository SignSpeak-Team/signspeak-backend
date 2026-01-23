# SignSpeak - Next Session Plan

## Fecha: 2026-01-23

---

## 1. MediaPipe Holistic Integration

Extraer 226 features (pose + ambas manos) para el modelo médico.

### Tareas

- [ ] Crear `holistic_extractor.py` con `mp.solutions.holistic`
- [ ] Actualizar `realtime_demo.py` para modo holístico
- [ ] Probar predicción `/predict/holistic` con datos reales

### Estructura de 226 features

| Componente | Features |
| ---------- | -------- |
| Pose       | ~100     |
| Left hand  | 63       |
| Right hand | 63       |

---

## 2. LLM Integration (Natural Language)

Usar un LLM para convertir las palabras detectadas en frases naturales.

### Ejemplo

```
Input:  "yo hospital ir dolor cabeza"
LLM:    "Necesito ir al hospital porque me duele la cabeza"
```

### Opciones

- **OpenAI API** (GPT-4o-mini) - Mejor calidad
- **Ollama local** (Llama 3) - Sin costo, offline
- **Gemini API** - Alternativa

### Tareas

- [ ] Elegir proveedor LLM
- [ ] Crear `llm_service.py` para normalizar frases
- [ ] Endpoint `/normalize` o integrar en word buffer

---

## 3. TTS (Text-to-Speech)

Convertir texto a voz natural en español.

### Flujo completo

```
Señas → Palabras → LLM → Frase natural → TTS → Audio
```

### Opciones

- **pyttsx3** - Offline, voces del sistema
- **gTTS** (Google) - Online, mejor calidad
- **ElevenLabs API** - Premium, muy natural

### Tareas

- [ ] Elegir librería TTS
- [ ] Crear `tts_service.py`
- [ ] Integrar en demo para reproducir frases

---

## Prioridad sugerida

1. **MediaPipe Holistic** (completa el modelo médico)
2. **LLM** (transforma palabras a lenguaje natural)
3. **TTS** (output de voz)
