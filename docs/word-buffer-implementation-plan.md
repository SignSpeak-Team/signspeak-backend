# 🧠 Plan de Implementación: Buffer Inteligente de Palabras

**Fecha creación:** 2026-01-15  
**Prioridad:** Alta  
**Tiempo estimado:** 2-3 horas

---

## 🎯 Objetivo

Crear un sistema que filtre repeticiones de palabras detectadas y acumule frases coherentes, evitando que el modelo prediga "hola hola hola buenos buenos días" cuando el usuario solo quiso decir "hola buenos días".

---

## 📋 Checklist de Implementación

### Fase 1: Crear Estructura Base (30 min)

- [ ] Crear archivo `services/vision_service/src/core/word_buffer.py`
- [ ] Definir clase `WordDetection` (dataclass)
- [ ] Definir clase `WordBuffer` (clase principal)
- [ ] Añadir imports necesarios

### Fase 2: Implementar Lógica Core (45 min)

- [ ] Método `__init__()` con configuración
- [ ] Método `add_detection()` con filtros
- [ ] Método `get_phrase()` para obtener frase
- [ ] Método `detect_pause()` para detectar pausas
- [ ] Método `get_statistics()` para métricas
- [ ] Método `clear()` para limpiar buffer

### Fase 3: Integración con Predictor (30 min)

- [ ] Modificar `predictor.py`
- [ ] Añadir `self.word_buffer` en `__init__`
- [ ] Crear método `predict_word_with_buffer()`
- [ ] Preservar método `predict_word()` original

### Fase 4: Testing (45 min)

- [ ] Crear `dev/demos/test_word_buffer.py`
- [ ] Test: Filtrado de repeticiones
- [ ] Test: Cooldown funciona
- [ ] Test: Construcción de frases
- [ ] Test: Detección de pausas
- [ ] Test: Estadísticas correctas

### Fase 5: Documentación (15 min)

- [ ] Docstrings en todos los métodos
- [ ] README en `core/` si es necesario
- [ ] Actualizar `task.md`

---

## 🛠️ Herramientas Necesarias

Todo ya está instalado:

```python
from collections import deque         # Buffer circular
from dataclasses import dataclass     # Estructuras limpias
import time                           # Timestamps
from typing import List, Optional     # Type hints
import logging                        # Debug
import numpy as np                    # Estadísticas
```

---

## 📁 Archivos a Crear/Modificar

```
✨ NUEVO:
services/vision_service/src/core/word_buffer.py (350 líneas aprox)
dev/demos/test_word_buffer.py (100 líneas)

📝 MODIFICAR:
services/vision_service/src/core/predictor.py (+50 líneas)
docs/task.md (actualizar progreso)
```

---

## 💻 Código de Referencia

### Estructura de WordBuffer:

```python
@dataclass
class WordDetection:
    word: str
    confidence: float
    timestamp: float
    frame_count: int = 0

class WordBuffer:
    def __init__(self,
                 cooldown_seconds: float = 2.0,
                 min_confidence: float = 80.0,
                 max_phrase_length: int = 10):
        self.detections: List[WordDetection] = []
        self.recent_buffer = deque(maxlen=5)
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.max_phrase_length = max_phrase_length
        self.last_word: Optional[str] = None
        self.last_timestamp: float = 0.0

    def add_detection(self, word: str, confidence: float) -> bool:
        """Añade detección con filtros"""
        # TODO: Implementar filtros
        pass

    def get_phrase(self) -> str:
        """Retorna frase acumulada"""
        return " ".join(d.word for d in self.detections)

    def detect_pause(self, pause_threshold: float = 3.0) -> bool:
        """Detecta si el usuario pausó"""
        # TODO: Implementar
        pass

    def get_statistics(self) -> Dict:
        """Retorna estadísticas"""
        # TODO: Implementar
        pass

    def clear(self):
        """Limpia buffer"""
        self.detections.clear()
```

---

## 🧪 Tests Esperados

```python
def test_basic():
    buffer = WordBuffer()
    assert buffer.add_detection("hola", 90.0) == True
    assert buffer.get_phrase() == "hola"

def test_repetition_filter():
    buffer = WordBuffer()
    buffer.add_detection("hola", 90)
    assert buffer.add_detection("hola", 92) == False  # Rechaza

def test_phrase_building():
    buffer = WordBuffer(cooldown_seconds=0.5)
    buffer.add_detection("hola", 90)
    time.sleep(0.6)
    buffer.add_detection("buenos", 92)
    assert buffer.get_phrase() == "hola buenos"
```

---

## ✅ Criterios de Éxito

- ✅ Buffer filtra repeticiones (mismo palabra < 2s)
- ✅ Solo acepta palabras con confianza > 80%
- ✅ Construye frases correctamente
- ✅ Detecta pausas (> 3s sin señas)
- ✅ Provee estadísticas útiles
- ✅ Todos los tests pasan

---

## 🔄 Próximos Pasos (Después del Buffer)

1. Endpoint API `/predict/words` (usa el buffer)
2. Servicio de Frases con Claude API
3. TTS para audio
4. Frontend demo

---

## 📝 Notas Importantes

- **No romper funcionalidad existente:** El método `predict_word()` original debe seguir funcionando
- **Logging:** Usar logger para debug, no prints
- **Type hints:** Mantener código tipado
- **Tests:** Escribir tests ANTES de mergear

---

## 🎓 Conceptos Clave

**Cooldown:** Tiempo mínimo entre detecciones de la misma palabra  
**Threshold:** Confianza mínima para aceptar palabra  
**Buffer circular:** deque con maxlen automáticamente descarta viejos  
**Pause detection:** Detecta cuando usuario dejó de hacer señas

---

## 📞 Contacto/Referencias

- Documentación Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- Documentación collections.deque: https://docs.python.org/3/library/collections.html#collections.deque

---

**Última actualización:** 2026-01-15 17:38  
**Siguiente sesión:** 2026-01-16 (mañana)
