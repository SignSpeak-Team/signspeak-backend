"""
Script para entrenar modelo LSTM con las 249 palabras.

Incluye:
- Carga de secuencias procesadas
- Data Augmentation (5x más datos)
- Entrenamiento LSTM
- Validación y métricas
- Guardado del modelo

Uso:
    python train_words_lstm.py
"""

import numpy as np
import pickle
import sys
from pathlib import Path

# Añadir path para imports
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
import matplotlib.pyplot as plt
from data_augmentation import apply_augmentation

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent  # vision_service/dev
SEQUENCES_PATH = BASE_PATH / "datasets_processed" / "palabras" / "sequences"
LABEL_ENCODER_PATH = BASE_PATH / "datasets_processed" / "palabras" / "words_label_encoder.pkl"
OUTPUT_PATH = Path(__file__).parent.parent.parent.parent / "models"  # vision_service/models

# Configuración
SEQUENCE_LENGTH = 15  # Debe coincidir con process_words_dataset.py
NUM_FEATURES = 63  # 21 landmarks x 3 coords


def load_sequences():
    """Carga todas las secuencias y crea etiquetas."""
    print("\n" + "=" * 70)
    print("🔄 CARGANDO SECUENCIAS")
    print("=" * 70)
    
    # Cargar label encoder
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_map = pickle.load(f)
    
    print(f"📋 Palabras en vocabulary: {len(label_map)}")
    
    # Invertir mapeo para usar con carpetas
    word_to_idx = label_map
    
    X = []
    y = []
    word_counts = {}
    
    # Obtener carpetas de palabras
    word_folders = sorted([f for f in SEQUENCES_PATH.iterdir() if f.is_dir()])
    
    for folder in word_folders:
        word = folder.name
        
        if word not in word_to_idx:
            print(f"  ⚠️ Palabra no encontrada en label encoder: {word}")
            continue
        
        label_idx = word_to_idx[word]
        sequences = list(folder.glob("*.npy"))
        
        loaded = 0
        for seq_file in sequences:
            try:
                seq = np.load(seq_file)
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    X.append(seq)
                    y.append(label_idx)
                    loaded += 1
            except Exception as e:
                pass
        
        word_counts[word] = loaded
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Total secuencias cargadas: {len(X)}")
    print(f"📊 Clases: {len(set(y))}")
    print(f"📊 Shape X: {X.shape}")
    
    # Mostrar algunas palabras con sus conteos
    print("\n📋 Primeras 15 palabras:")
    for i, (word, count) in enumerate(sorted(word_counts.items())[:15]):
        print(f"   {word}: {count} secuencias")
    print(f"   ... y {len(word_counts) - 15} más")
    
    return X, y, word_to_idx


def create_lstm_model(num_classes):
    """Crea el modelo LSTM para palabras."""
    model = Sequential([
        # Capa LSTM 1
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa LSTM 2
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa LSTM 3
        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capas Dense
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Salida
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, output_path):
    """Genera gráficas de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Precisión del Modelo', fontsize=14)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Pérdida del Modelo', fontsize=14)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"   📈 Gráfica guardada: {output_path}")


def main():
    print("\n" + "=" * 70)
    print("🤟 ENTRENAMIENTO LSTM - 249 PALABRAS LSM")
    print("=" * 70)
    
    # Crear carpeta de salida si no existe
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    X, y, label_map = load_sequences()
    
    if len(X) == 0:
        print("❌ [ERROR] No se encontraron secuencias")
        return
    
    # Data Augmentation
    print("\n" + "=" * 70)
    print("🔄 APLICANDO DATA AUGMENTATION")
    print("=" * 70)
    print("Esto generará ~5x más datos para mejorar el modelo.")
    
    X_aug, y_aug = apply_augmentation(X, y, augmentations_per_sample=4)
    
    # Dividir datos
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    print(f"\n📊 División de datos:")
    print(f"   Train: {len(X_train)} secuencias")
    print(f"   Validation: {len(X_val)} secuencias")
    
    # Crear modelo
    num_classes = len(set(y))
    model = create_lstm_model(num_classes)
    
    print("\n" + "=" * 70)
    print("🏗️ ARQUITECTURA DEL MODELO")
    print("=" * 70)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            OUTPUT_PATH / 'words_model_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entrenar
    print("\n" + "=" * 70)
    print("🚀 ENTRENANDO")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar modelo final
    model.save(OUTPUT_PATH / 'words_model.keras')
    print(f"\n✅ Modelo guardado: {OUTPUT_PATH / 'words_model.keras'}")
    
    # Guardar label encoder en la misma carpeta
    with open(OUTPUT_PATH / 'words_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    print(f"✅ Label encoder guardado: {OUTPUT_PATH / 'words_label_encoder.pkl'}")
    
    # Generar gráficas
    print("\n📈 Generando gráficas de entrenamiento...")
    plot_training_history(history, OUTPUT_PATH / 'words_training_history.png')
    
    # Resultados finales
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print("\n" + "=" * 70)
    print("✅ RESULTADOS FINALES")
    print("=" * 70)
    print(f"   📊 Precisión validación: {val_acc * 100:.2f}%")
    print(f"   📉 Loss validación: {val_loss:.4f}")
    print(f"   📁 Modelo: {OUTPUT_PATH / 'words_model.keras'}")
    print(f"   📁 Encoder: {OUTPUT_PATH / 'words_label_encoder.pkl'}")
    print(f"   📈 Gráficas: {OUTPUT_PATH / 'words_training_history.png'}")
    print("=" * 70)
    
    print("\n✅ [DONE] Entrenamiento completado!")


if __name__ == "__main__":
    main()
