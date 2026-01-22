"""Tests for WordBuffer - filtrado de repeticiones de palabras."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import directly to avoid loading predictor and its heavy dependencies
from core.word_buffer import WordBuffer, WordDetection


def test_basic_detection():
    """Palabra con alta confianza se acepta."""
    buffer = WordBuffer()
    result = buffer.add_detection("hola", 90.0)
    
    assert result == True, "Should accept word with high confidence"
    assert buffer.get_phrase() == "hola"
    assert len(buffer) == 1
    print("✓ test_basic_detection passed")


def test_confidence_filter():
    """Baja confianza se rechaza."""
    buffer = WordBuffer(min_confidence=80.0)
    
    result = buffer.add_detection("hola", 70.0)
    assert result == False, "Should reject word with low confidence"
    assert len(buffer) == 0
    print("✓ test_confidence_filter passed")


def test_cooldown_filter():
    """Repeticiones dentro del cooldown se rechazan."""
    buffer = WordBuffer(cooldown_seconds=1.0)
    
    # Primera detección se acepta
    assert buffer.add_detection("hola", 90.0) == True
    
    # Misma palabra inmediatamente se rechaza
    assert buffer.add_detection("hola", 92.0) == False
    
    # Palabra diferente se acepta
    assert buffer.add_detection("buenos", 88.0) == True
    
    assert buffer.get_phrase() == "hola buenos"
    print("✓ test_cooldown_filter passed")


def test_cooldown_expires():
    """Después del cooldown, misma palabra se acepta."""
    buffer = WordBuffer(cooldown_seconds=0.5)
    
    buffer.add_detection("hola", 90.0)
    
    # Esperar que expire cooldown
    time.sleep(0.6)
    
    result = buffer.add_detection("hola", 92.0)
    assert result == True, "Should accept same word after cooldown"
    assert buffer.get_phrase() == "hola hola"
    print("✓ test_cooldown_expires passed")


def test_phrase_building():
    """Construye frases correctamente."""
    buffer = WordBuffer(cooldown_seconds=0.1)
    
    buffer.add_detection("hola", 90.0)
    time.sleep(0.15)
    buffer.add_detection("buenos", 88.0)
    time.sleep(0.15)
    buffer.add_detection("dias", 85.0)
    
    assert buffer.get_phrase() == "hola buenos dias"
    assert buffer.get_words() == ["hola", "buenos", "dias"]
    print("✓ test_phrase_building passed")


def test_statistics():
    """Estadísticas se calculan correctamente."""
    buffer = WordBuffer(cooldown_seconds=1.0, min_confidence=80.0)
    
    buffer.add_detection("hola", 90.0)      # Accepted
    buffer.add_detection("hola", 92.0)      # Rejected (cooldown)
    buffer.add_detection("test", 70.0)      # Rejected (confidence)
    buffer.add_detection("buenos", 85.0)    # Accepted
    
    stats = buffer.get_statistics()
    
    assert stats["total_received"] == 4
    assert stats["total_accepted"] == 2
    assert stats["rejected_by_cooldown"] == 1
    assert stats["rejected_by_confidence"] == 1
    assert stats["acceptance_rate"] == 50.0
    print("✓ test_statistics passed")


def test_clear():
    """Clear reinicia todo."""
    buffer = WordBuffer()
    buffer.add_detection("hola", 90.0)
    buffer.add_detection("mundo", 88.0)
    
    buffer.clear()
    
    assert len(buffer) == 0
    assert buffer.get_phrase() == ""
    stats = buffer.get_statistics()
    assert stats["total_received"] == 0
    print("✓ test_clear passed")


def test_pause_detection():
    """Detecta pausas correctamente."""
    buffer = WordBuffer(pause_threshold=0.5)
    
    # Sin detecciones, no hay pausa
    assert buffer.detect_pause() == False
    
    buffer.add_detection("hola", 90.0)
    
    # Inmediatamente después, no hay pausa
    assert buffer.detect_pause() == False
    
    # Esperar más que el threshold
    time.sleep(0.6)
    assert buffer.detect_pause() == True
    print("✓ test_pause_detection passed")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running WordBuffer Tests")
    print("=" * 50 + "\n")
    
    test_basic_detection()
    test_confidence_filter()
    test_cooldown_filter()
    test_cooldown_expires()
    test_phrase_building()
    test_statistics()
    test_clear()
    test_pause_detection()
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
