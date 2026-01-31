"""Tests unitarios para WordBuffer - Sistema de filtrado de palabras detectadas."""

import time

import pytest

# Necesitamos agregar el path del vision_service
import sys
from pathlib import Path

VISION_SERVICE_PATH = Path(__file__).parent.parent.parent / "services" / "vision_service" / "src"
if str(VISION_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_PATH))

from core.word_buffer import WordBuffer, WordDetection


class TestWordDetection:
    """Tests para dataclass WordDetection."""

    def test_word_detection_creation(self):
        """Test creación básica de WordDetection."""
        detection = WordDetection(word="hola", confidence=95.0)
        
        assert detection.word == "hola"
        assert detection.confidence == 95.0
        assert detection.timestamp > 0

    def test_word_detection_with_timestamp(self):
        """Test WordDetection con timestamp específico."""
        detection = WordDetection(word="test", confidence=80.0, timestamp=12345.0)
        
        assert detection.timestamp == 12345.0


class TestWordBufferBasics:
    """Tests básicos de WordBuffer."""

    def test_buffer_initialization(self):
        """Test inicialización con valores default."""
        buffer = WordBuffer()
        
        assert buffer.cooldown_seconds == 2.0
        assert buffer.min_confidence == 80.0
        assert buffer.max_phrase_length == 20
        assert buffer.pause_threshold == 3.0
        assert len(buffer) == 0

    def test_buffer_custom_params(self):
        """Test inicialización con parámetros personalizados."""
        buffer = WordBuffer(
            cooldown_seconds=1.5,
            min_confidence=90.0,
            max_phrase_length=10,
            pause_threshold=5.0
        )
        
        assert buffer.cooldown_seconds == 1.5
        assert buffer.min_confidence == 90.0
        assert buffer.max_phrase_length == 10
        assert buffer.pause_threshold == 5.0


class TestWordBufferAddDetection:
    """Tests para add_detection con filtros."""

    def test_add_detection_accepted_high_confidence(self):
        """Palabra con alta confianza es aceptada."""
        buffer = WordBuffer(min_confidence=80.0)
        
        result = buffer.add_detection("hola", 95.0)
        
        assert result is True
        assert len(buffer) == 1
        assert buffer.get_phrase() == "hola"

    def test_add_detection_rejected_low_confidence(self):
        """Palabra con baja confianza es rechazada."""
        buffer = WordBuffer(min_confidence=80.0)
        
        result = buffer.add_detection("hola", 70.0)
        
        assert result is False
        assert len(buffer) == 0
        assert buffer.get_phrase() == ""

    def test_add_detection_rejected_cooldown(self):
        """Misma palabra dentro del cooldown es rechazada."""
        buffer = WordBuffer(cooldown_seconds=2.0, min_confidence=50.0)
        
        # Primera palabra - aceptada
        result1 = buffer.add_detection("hola", 90.0)
        # Misma palabra inmediatamente - rechazada por cooldown
        result2 = buffer.add_detection("hola", 90.0)
        
        assert result1 is True
        assert result2 is False
        assert len(buffer) == 1

    def test_add_detection_different_word_no_cooldown(self):
        """Palabra diferente no tiene cooldown."""
        buffer = WordBuffer(cooldown_seconds=2.0, min_confidence=50.0)
        
        result1 = buffer.add_detection("hola", 90.0)
        result2 = buffer.add_detection("mundo", 90.0)  # Diferente palabra
        
        assert result1 is True
        assert result2 is True
        assert len(buffer) == 2
        assert buffer.get_phrase() == "hola mundo"

    def test_add_detection_same_word_after_cooldown(self):
        """Misma palabra después del cooldown es aceptada."""
        buffer = WordBuffer(cooldown_seconds=0.1, min_confidence=50.0)
        
        buffer.add_detection("hola", 90.0)
        time.sleep(0.15)  # Esperar más que el cooldown
        result = buffer.add_detection("hola", 90.0)
        
        assert result is True
        assert len(buffer) == 2
        assert buffer.get_phrase() == "hola hola"

    def test_add_detection_boundary_confidence(self):
        """Test con confianza exactamente en el límite."""
        buffer = WordBuffer(min_confidence=80.0)
        
        # Exactamente 80% - debería ser aceptada (>= no es <)
        result = buffer.add_detection("test", 80.0)
        
        assert result is True


class TestWordBufferPhraseBuilding:
    """Tests para construcción de frases."""

    def test_get_phrase_empty(self):
        """Phrase vacía cuando no hay detecciones."""
        buffer = WordBuffer()
        
        assert buffer.get_phrase() == ""

    def test_get_phrase_single_word(self):
        """Phrase con una sola palabra."""
        buffer = WordBuffer(min_confidence=50.0)
        buffer.add_detection("hola", 90.0)
        
        assert buffer.get_phrase() == "hola"

    def test_get_phrase_multiple_words(self):
        """Phrase con múltiples palabras."""
        buffer = WordBuffer(min_confidence=50.0)
        buffer.add_detection("buenos", 90.0)
        buffer.add_detection("días", 90.0)
        buffer.add_detection("doctor", 90.0)
        
        assert buffer.get_phrase() == "buenos días doctor"

    def test_get_words_returns_list(self):
        """get_words retorna lista de palabras."""
        buffer = WordBuffer(min_confidence=50.0)
        buffer.add_detection("hola", 90.0)
        buffer.add_detection("mundo", 90.0)
        
        words = buffer.get_words()
        
        assert words == ["hola", "mundo"]

    def test_max_phrase_length_respected(self):
        """Buffer no excede max_phrase_length."""
        buffer = WordBuffer(min_confidence=50.0, max_phrase_length=3)
        
        for i in range(5):
            buffer.add_detection(f"word{i}", 90.0)
        
        # Solo las últimas 3 palabras
        assert len(buffer) == 3
        words = buffer.get_words()
        assert words == ["word2", "word3", "word4"]


class TestWordBufferPauseDetection:
    """Tests para detección de pausas."""

    def test_detect_pause_no_detections(self):
        """Sin detecciones, no hay pausa."""
        buffer = WordBuffer()
        
        assert buffer.detect_pause() is False

    def test_detect_pause_recent_detection(self):
        """Detección reciente, no hay pausa."""
        buffer = WordBuffer(pause_threshold=3.0, min_confidence=50.0)
        buffer.add_detection("test", 90.0)
        
        assert buffer.detect_pause() is False

    def test_detect_pause_after_threshold(self):
        """Pausa detectada después del threshold."""
        buffer = WordBuffer(pause_threshold=0.1, min_confidence=50.0)
        buffer.add_detection("test", 90.0)
        
        time.sleep(0.15)  # Esperar más que pause_threshold
        
        assert buffer.detect_pause() is True


class TestWordBufferStatistics:
    """Tests para estadísticas del buffer."""

    def test_statistics_initial(self):
        """Estadísticas iniciales."""
        buffer = WordBuffer()
        stats = buffer.get_statistics()
        
        assert stats["total_received"] == 0
        assert stats["total_accepted"] == 0
        assert stats["rejected_by_cooldown"] == 0
        assert stats["rejected_by_confidence"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["current_phrase_length"] == 0
        assert stats["current_phrase"] == ""

    def test_statistics_after_detections(self):
        """Estadísticas después de varias detecciones."""
        buffer = WordBuffer(min_confidence=80.0, cooldown_seconds=2.0)
        
        # 1 aceptada
        buffer.add_detection("hola", 90.0)
        # 1 rechazada por confianza
        buffer.add_detection("mundo", 70.0)
        # 1 rechazada por cooldown
        buffer.add_detection("hola", 90.0)
        # 1 aceptada (diferente palabra)
        buffer.add_detection("adios", 85.0)
        
        stats = buffer.get_statistics()
        
        assert stats["total_received"] == 4
        assert stats["total_accepted"] == 2
        assert stats["rejected_by_cooldown"] == 1
        assert stats["rejected_by_confidence"] == 1
        assert stats["acceptance_rate"] == 50.0
        assert stats["current_phrase_length"] == 2


class TestWordBufferClear:
    """Tests para limpieza del buffer."""

    def test_clear_resets_everything(self):
        """Clear limpia buffer y estadísticas."""
        buffer = WordBuffer(min_confidence=50.0)
        buffer.add_detection("hola", 90.0)
        buffer.add_detection("mundo", 90.0)
        
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.get_phrase() == ""
        stats = buffer.get_statistics()
        assert stats["total_received"] == 0
        assert stats["total_accepted"] == 0

    def test_clear_allows_new_detections(self):
        """Después de clear, nuevas detecciones funcionan."""
        buffer = WordBuffer(min_confidence=50.0, cooldown_seconds=100.0)
        buffer.add_detection("hola", 90.0)
        
        buffer.clear()
        
        # Misma palabra debería ser aceptada (sin cooldown previo)
        result = buffer.add_detection("hola", 90.0)
        assert result is True


class TestWordBufferRepr:
    """Tests para representación string del buffer."""

    def test_repr_empty(self):
        """Repr con buffer vacío."""
        buffer = WordBuffer()
        
        assert "words=0" in repr(buffer)
        assert "phrase=''" in repr(buffer)

    def test_repr_with_words(self):
        """Repr con palabras."""
        buffer = WordBuffer(min_confidence=50.0)
        buffer.add_detection("hola", 90.0)
        buffer.add_detection("mundo", 90.0)
        
        repr_str = repr(buffer)
        assert "words=2" in repr_str
        assert "hola mundo" in repr_str
