"""
Word Buffer - Sistema de filtrado de detecciones de palabras.

Filtra repeticiones y acumula palabras para construir frases coherentes.
Evita predicciones duplicadas como "hola hola hola" cuando el usuario
solo quiso decir "hola".
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class WordDetection:
    """Representa una detección individual de palabra."""
    word: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class WordBuffer:
    """
    Buffer inteligente que filtra repeticiones de palabras.
    
    Attributes:
        cooldown_seconds: Tiempo mínimo entre misma palabra (default: 2.0s)
        min_confidence: Confianza mínima requerida (default: 80.0%)
        max_phrase_length: Máximo de palabras en buffer (default: 20)
        pause_threshold: Segundos sin detección para considerar pausa (default: 3.0s)
    """
    
    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        min_confidence: float = 80.0,
        max_phrase_length: int = 20,
        pause_threshold: float = 3.0
    ):
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.max_phrase_length = max_phrase_length
        self.pause_threshold = pause_threshold
        
        # Storage
        self._detections: List[WordDetection] = []
        self._recent_words: deque = deque(maxlen=5)
        
        # State tracking
        self._last_word: Optional[str] = None
        self._last_timestamp: float = 0.0
        
        # Statistics
        self._total_received: int = 0
        self._total_accepted: int = 0
        self._rejected_by_cooldown: int = 0
        self._rejected_by_confidence: int = 0
    
    def add_detection(self, word: str, confidence: float) -> bool:
        """
        Añade una detección al buffer con filtros aplicados.
        
        Args:
            word: Palabra detectada
            confidence: Confianza de la predicción (0-100)
        
        Returns:
            True si la palabra fue aceptada, False si fue rechazada
        """
        self._total_received += 1
        now = time.time()
        
        # Filtro 1: Confianza mínima
        if confidence < self.min_confidence:
            self._rejected_by_confidence += 1
            logger.debug(f"Rejected '{word}': low confidence ({confidence:.1f}%)")
            return False
        
        # Filtro 2: Cooldown para misma palabra
        if word == self._last_word:
            elapsed = now - self._last_timestamp
            if elapsed < self.cooldown_seconds:
                self._rejected_by_cooldown += 1
                logger.debug(f"Rejected '{word}': cooldown ({elapsed:.1f}s < {self.cooldown_seconds}s)")
                return False
        
        # Aceptar palabra
        detection = WordDetection(word=word, confidence=confidence, timestamp=now)
        self._detections.append(detection)
        self._recent_words.append(word)
        
        # Limitar tamaño del buffer
        if len(self._detections) > self.max_phrase_length:
            self._detections.pop(0)
        
        # Actualizar estado
        self._last_word = word
        self._last_timestamp = now
        self._total_accepted += 1
        
        logger.info(f"Accepted '{word}' ({confidence:.1f}%)")
        return True
    
    def get_phrase(self) -> str:
        """Retorna la frase acumulada de todas las detecciones."""
        return " ".join(d.word for d in self._detections)
    
    def get_words(self) -> List[str]:
        """Retorna lista de palabras detectadas."""
        return [d.word for d in self._detections]
    
    def detect_pause(self) -> bool:
        """
        Detecta si el usuario ha pausado (sin detecciones recientes).
        
        Returns:
            True si ha pasado más tiempo que pause_threshold desde última detección
        """
        if self._last_timestamp == 0.0:
            return False
        
        elapsed = time.time() - self._last_timestamp
        return elapsed > self.pause_threshold
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del buffer."""
        acceptance_rate = 0.0
        if self._total_received > 0:
            acceptance_rate = (self._total_accepted / self._total_received) * 100
        
        return {
            "total_received": self._total_received,
            "total_accepted": self._total_accepted,
            "rejected_by_cooldown": self._rejected_by_cooldown,
            "rejected_by_confidence": self._rejected_by_confidence,
            "acceptance_rate": round(acceptance_rate, 1),
            "current_phrase_length": len(self._detections),
            "current_phrase": self.get_phrase()
        }
    
    def clear(self):
        """Limpia el buffer y reinicia estadísticas."""
        self._detections.clear()
        self._recent_words.clear()
        self._last_word = None
        self._last_timestamp = 0.0
        self._total_received = 0
        self._total_accepted = 0
        self._rejected_by_cooldown = 0
        self._rejected_by_confidence = 0
        logger.info("Buffer cleared")
    
    def __len__(self) -> int:
        """Retorna número de palabras en el buffer."""
        return len(self._detections)
    
    def __repr__(self) -> str:
        return f"WordBuffer(words={len(self)}, phrase='{self.get_phrase()}')"
