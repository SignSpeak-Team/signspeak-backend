import numpy as np
import pickle 
from tensorflow import keras

DATA_PATH = "../../datasets/processed"

print("Cargando modelo...")
model = keras.models.load_model(f"{DATA_PATH}/sign_model.keras")

test_data = np.load(f"{DATA_PATH}/landmarks_test.npz")
X_test = test_data["X"]
y_test = test_data["y"]

print(f"Datos de test cargados: {X_test.shape[0]} muestras")

with open(f"{DATA_PATH}/label_encoder.pkl", "rb") as f:
    label_map = pickle.load(f)

idx_to_letter = {v: k for k, v in label_map.items()}

# Evaluar modelo
print("\nEvaluando...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"RESULTADOS EN DATOS DE TEST")
print(f"{'='*40}")
print(f"Precision: {accuracy * 100:.2f}%")
print(f"Loss: {loss:.4f}")
# Mostrar algunas predicciones de ejemplo
print(f"\n{'='*40}")
print("EJEMPLOS DE PREDICCIONES")
print(f"{'='*40}")
# Tomar 10 muestras aleatorias
indices = np.random.choice(len(X_test), 10, replace=False)
for i in indices:
    pred = model.predict(X_test[i:i+1], verbose=0)
    pred_class = np.argmax(pred)
    real_class = y_test[i]
    
    pred_letter = idx_to_letter[pred_class]
    real_letter = idx_to_letter[real_class]
    confidence = pred[0][pred_class] * 100
    
    status = "OK" if pred_class == real_class else "X"
    print(f"[{status}] Real: {real_letter} | Prediccion: {pred_letter} ({confidence:.1f}%)")