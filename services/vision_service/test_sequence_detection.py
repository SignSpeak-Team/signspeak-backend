"""
Script de prueba para el endpoint de detección de secuencias.

Prueba el nuevo modo continuous con videos de múltiples palabras.
"""

import requests
import sys
from pathlib import Path


def test_continuous_mode(video_path: str, api_url: str = "http://localhost:8001"):
    """
    Prueba el endpoint /media/translate/video en modo continuous.
    
    Args:
        video_path: Ruta al archivo de video
        api_url: URL base del API
    """
    endpoint = f"{api_url}/api/v1/media/translate/video"
    
    # Verificar que el archivo existe
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video no encontrado en {video_path}")
        return
    
    print(f"📹 Probando con video: {video_file.name}")
    print(f"📊 Tamaño: {video_file.stat().st_size / 1024:.1f} KB\n")
    
    # Test 1: Modo continuous (default) con parámetros por defecto
    print("=" * 60)
    print("TEST 1: Modo Continuous (Default)")
    print("=" * 60)
    
    with open(video_file, "rb") as f:
        files = {"file": (video_file.name, f, "video/mp4")}
        response = requests.post(endpoint, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print_results(result, "Continuous (Default)")
    else:
        print(f"❌ Error {response.status_code}: {response.text}\n")
        return
    
    # Test 2: Modo continuous con parámetros personalizados
    print("\n" + "=" * 60)
    print("TEST 2: Modo Continuous (Parámetros Personalizados)")
    print("=" * 60)
    print("Configuración: min_confidence=70, window_size=2.5, stride=1.0\n")
    
    with open(video_file, "rb") as f:
        files = {"file": (video_file.name, f, "video/mp4")}
        data = {
            "mode": "continuous",
            "min_confidence": 70,
            "window_size": 2.5,
            "stride": 1.0
        }
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print_results(result, "Continuous (Custom)")
    else:
        print(f"❌ Error {response.status_code}: {response.text}\n")
        return
    
    # Test 3: Modo holistic (para comparación)
    print("\n" + "=" * 60)
    print("TEST 3: Modo Holistic (Solo 1 Palabra)")
    print("=" * 60)
    
    with open(video_file, "rb") as f:
        files = {"file": (video_file.name, f, "video/mp4")}
        data = {"mode": "holistic"}
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print_results(result, "Holistic")
    else:
        print(f"❌ Error {response.status_code}: {response.text}\n")


def print_results(result: dict, test_name: str):
    """Imprime los resultados de forma legible."""
    print(f"✅ {test_name} - Resultados:\n")
    
    # Frase detectada
    print(f"🗣️  FRASE DETECTADA: '{result['word']}'")
    print(f"📊 Confianza promedio: {result['confidence']:.1f}%")
    print(f"⏱️  Tiempo total: {result['total_time_ms']:.0f}ms")
    print(f"🎞️  Frames procesados: {result['frames_processed']}")
    
    # Palabras individuales (si hay)
    if result.get('segments'):
        print(f"\n📝 Palabras detectadas ({len(result['segments'])}):")
        for i, seg in enumerate(result['segments'], 1):
            print(f"   {i}. '{seg['word']}' "
                  f"[{seg['start_time']:.1f}s - {seg['end_time']:.1f}s] "
                  f"({seg['confidence']:.1f}%)")
    
    # Estadísticas (si hay)
    if result.get('detection_stats'):
        stats = result['detection_stats']
        print(f"\n📈 Estadísticas de Detección:")
        print(f"   • Ventanas procesadas: {stats['total_windows']}")
        print(f"   • Palabras detectadas: {stats['detected_words']}")
        print(f"   • Palabras filtradas: {stats['filtered_words']}")
        print(f"   • Tasa de filtrado: {stats['filter_rate']:.1f}%")
    
    print()


def test_health(api_url: str = "http://localhost:8001"):
    """Verifica que el servicio esté funcionando."""
    try:
        response = requests.get(f"{api_url}/api/v1/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Servicio activo")
            print(f"   Version: {health.get('version', 'N/A')}")
            print(f"   Modelos cargados: {health.get('models_loaded', False)}\n")
            return True
        else:
            print(f"❌ Servicio respondió con código {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servicio")
        print("   Asegúrate de que el servicio esté corriendo en http://localhost:8001\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TEST DE DETECCIÓN DE SECUENCIAS DE PALABRAS")
    print("=" * 60 + "\n")
    
    # Verificar servicio
    if not test_health():
        print("💡 Para iniciar el servicio:")
        print("   cd services/vision_service")
        print("   python -m uvicorn main:app --host 0.0.0.0 --port 8001")
        print("\n   O usa el script:")
        print("   .\\start_service.ps1\n")
        sys.exit(1)
    
    # Solicitar video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("📹 Uso: python test_sequence_detection.py <ruta_al_video>")
        print("\nEjemplo con videos de casos:")
        print("   python test_sequence_detection.py dev\\datasets_raw\\videos\\letras\\cases\\CASO_0_blur(1).mp4")
        print("   python test_sequence_detection.py dev\\datasets_raw\\videos\\letras\\cases\\CASO_4_blur.mp4\n")
        sys.exit(1)
    
    # Ejecutar pruebas
    test_continuous_mode(video_path)
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("=" * 60)
