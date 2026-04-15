import time

import requests

# Configuración
API_URL = "http://localhost:8000/api/v1/predict/static"


def test_static_letter(letter_name, landmarks):
    print(f"\n[TEST] Probando Letra: {letter_name}")
    try:
        payload = {"landmarks": landmarks}
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=10)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            print(
                f"OK - Respuesta: {data['letter']} | Confianza: {data['confidence']}%"
            )
            print(f"Latencia: {latency:.2f}ms")
            return data
        else:
            print(f"ERROR {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"ERROR de conexion: {e}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("DEMO DE VALIDACION DE ALFABETO - SignSpeak")
    print("=" * 50)

    # Ejemplo de landmarks (una mano "plana" desplazada de la camara)
    # MediaPipe entrega valores en [0,1], el modelo espera (LM - Wrist)
    # Simulamos una mano con wrist en (0.5, 0.5)
    wrist = [0.5, 0.5, 0.0]
    hand_sample = []
    for _ in range(21):
        # Todos los puntos un poco desplazados respecto al wrist
        hand_sample.append([wrist[0] + 0.05, wrist[1] + 0.05, wrist[2] + 0.01])

    test_static_letter("Prueba de Escala", hand_sample)

    print("\nINFO: Si el resultado es una letra consistente y la latencia es baja,")
    print("   significa que la normalizacion en el backend esta funcionando.")
    print("=" * 50)
