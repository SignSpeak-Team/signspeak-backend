"""
Probar el modelo best_model.h5 con el ORDEN CORRECTO del vocabulario
(extraído del notebook original)
"""
import numpy as np
import os

# VOCABULARIO EN ORDEN CORRECTO (del notebook)
vocabulary = [
    'hospital', 'si', 'duro', 'lunes', 'perro', 'cansado', 'ayer', 'yo', 'nosotros',
    'beber', 'ambulancia', 'infeccion', 'ojo', 'no', 'pregunta', 'duda', 'bien',
    'mal', 'suave', 'normal', 'frio', 'caliente', 'mejor', 'peor', 'estresado',
    'rapido', 'lento', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado',
    'domingo', 'gato', 'camaron', 'pollo', 'abeja', 'confundido', 'ahora', 'hoy',
    'manana', 'nunca', 'siempre', 'diario', 'mama', 'papa', 'esposo', 'esposa',
    'hijo', 'hija', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
    'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '10', 'como', 'cuantos', 'para_que', 'por_que', 'cocinar',
    'recibir', 'estudiar', 'interpretar', 'ir', 'no_ver', 'dormir', 'pelear',
    'trabajar', 'descansar', 'comer', 'correr', 'caminar', 'jarabe', 'virus',
    'aborto', 'accidente', 'doctor', 'enfermera', 'enfermero', 'paciente',
    'enfermo', 'terapia', 'pastillas', 'inyeccion', 'contagiar', 'revisar',
    'calentura', 'cancer', 'infarto', 'lesion', 'embarazo', 'sangre', 'gripa',
    'garganta', 'tos', 'debil', 'huesos', 'farmacia', 'emergencia', 'inflamacion',
    'analisis', 'coronavirus', 'cita', 'dolor', 'fractura', 'urgencia', 'orina',
    'popo', 'mareo', 'vomito', 'convulciones', 'gases', 'diarrea', 'moco', 'sed',
    'nariz', 'oreja', 'boca', 'cuello', 'hombro', 'espalda', 'brazo', 'codo',
    'muneca', 'mano', 'panza', 'cintura', 'pene', 'vagina', 'piernas', 'rodilla',
    'tobillo', 'pie'
]

print("=" * 70)
print("VERIFICACIÓN CON VOCABULARIO CORRECTO")
print("=" * 70)
print(f"Vocabulario: {len(vocabulary)} palabras")

# Verificar posición del "1"
idx_1 = vocabulary.index('1') if '1' in vocabulary else -1
print(f"Índice de '1' en vocabulario: {idx_1}")

# Cargar datos del número "1"
folder = 'services/vision_service/dev/datasets_raw/numpy/1'
files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])

print(f"\n📁 Cargando datos de: {folder}")

sequence = []
for f in files:
    data = np.load(os.path.join(folder, f))
    sequence.append(data)

sequence = np.array(sequence)
print(f"   Shape: {sequence.shape}")

input_data = np.expand_dims(sequence, axis=0)

# Recrear la arquitectura
print("\n📥 Cargando modelo...")
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, LSTM, Dropout, Dense

model = Sequential([
    InputLayer(input_shape=(30, 226)),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(150, activation='softmax')
])

model.load_weights('services/vision_service/models/best_model.h5')
print("   ✅ Modelo cargado")

# Predicción
print("\n🤖 Haciendo predicción...")
prediction = model.predict(input_data, verbose=0)

# Top 5 predicciones
top_5_idx = np.argsort(prediction[0])[-5:][::-1]

print("\n" + "=" * 70)
print("RESULTADO")
print("=" * 70)
print(f"\n🎯 Datos de entrada: carpeta '1' (debería predecir número 1)")
print(f"\n📊 Top 5 predicciones:")
for i, idx in enumerate(top_5_idx, 1):
    word = vocabulary[idx] if idx < len(vocabulary) else f"IDX_{idx}"
    conf = prediction[0][idx] * 100
    marker = "✅" if word == "1" else "  "
    print(f"   {marker} {i}. '{word}' (índice {idx}) - {conf:.2f}%")

predicted = vocabulary[top_5_idx[0]]
print("\n" + "=" * 70)
if predicted == "1":
    print("✅ ¡VOCABULARIO CORRECTO! El modelo predice correctamente")
else:
    print(f"⚠️  Predijo '{predicted}' en lugar de '1'")
    print("   Puede que los datos no correspondan al número 1")
    print("   o el orden aún no sea correcto")
