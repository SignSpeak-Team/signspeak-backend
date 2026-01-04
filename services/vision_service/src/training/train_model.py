import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Ruta a los datos
DATA_PATH = "../../datasets/processed"

# Cargar datos
train_data = np.load(f"{DATA_PATH}/landmarks_train.npz")
X_train = train_data["X"]
y_train = train_data["y"]

print(f"Datos cargados: {X_train.shape[0]} muestras")

# Mezclar y dividir datos (IMPORTANTE para evitar overfitting)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,       # 20% para validacion
    random_state=42,     # Reproducibilidad
    stratify=y_train     # Mantener proporcion de clases
)

print(f"Train: {X_train_split.shape[0]} | Val: {X_val.shape[0]}")

# Numero de clases
num_classes = len(set(y_train))
print(f"Clases: {num_classes}")

# Crear modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
print("\nEntrenando modelo...")
history = model.fit(
    X_train_split,
    y_train_split,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)  # Datos de validacion mezclados
)

# Guardar modelo
model.save("../../datasets/processed/sign_model.keras")
print("\nModelo guardado en: datasets/processed/sign_model.keras")