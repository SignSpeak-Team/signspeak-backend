"""
Entrenamiento de modelo Bidirectional LSTM para palabras LSM.
Usa secuencias holisticas de 144 features procesadas por process_holistic_dataset.py.

Uso:
    python train_holistic_lstm.py
"""

import io
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Forzar UTF-8 en stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Input
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent))
from data_augmentation import apply_augmentation

# --- Rutas ---
BASE_PATH = Path(__file__).parent.parent.parent
PROCESSED_PATH = BASE_PATH / "datasets_processed" / "holistic_palabras"
SEQUENCES_PATH = PROCESSED_PATH / "sequences"
MODEL_OUTPUT = Path(__file__).parent.parent.parent.parent / "models"


def detect_device():
    """Detecta y reporta GPU disponible."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] Detectada(s): {len(gpus)}")
        for gpu in gpus:
            print(f"   -> {gpu.name}")
        # Permitir crecimiento de memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[CPU] No se detecto GPU, usando CPU")
    return len(gpus) > 0


def load_config() -> dict:
    with open(PROCESSED_PATH / "config.pkl", "rb") as f:
        return pickle.load(f)


def load_sequences(seq_length: int, num_features: int) -> tuple:
    with open(PROCESSED_PATH / "holistic_label_encoder.pkl", "rb") as f:
        label_map = pickle.load(f)

    X, y = [], []
    word_counts = {}

    for folder in sorted(SEQUENCES_PATH.iterdir()):
        if not folder.is_dir() or folder.name not in label_map:
            continue

        label_idx = label_map[folder.name]
        loaded = 0

        for npy_file in folder.glob("*.npy"):
            seq = np.load(npy_file)
            if seq.shape == (seq_length, num_features):
                X.append(seq)
                y.append(label_idx)
                loaded += 1

        word_counts[folder.name] = loaded

    X, y = np.array(X), np.array(y)
    print(f"[DATA] Secuencias: {len(X)} | Clases: {len(set(y))} | Shape: {X.shape}")
    return X, y, label_map


def build_model(seq_length: int, num_features: int, num_classes: int) -> Sequential:
    """Bidirectional LSTM: 256 -> 512 -> 256 + Dense."""
    model = Sequential([
        Input(shape=(seq_length, num_features)),

        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),

        Bidirectional(LSTM(512, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),

        Bidirectional(LSTM(256, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def plot_history(history, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val", linewidth=2)
    ax1.set_title("Precision")
    ax1.set_xlabel("Epoca")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val", linewidth=2)
    ax2.set_title("Perdida")
    ax2.set_xlabel("Epoca")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[PLOT] Grafica: {output_path}")


def main():
    print("\n" + "=" * 70)
    print(" ENTRENAMIENTO BIDIRECTIONAL LSTM - HOLISTICO 144 FEATURES")
    print("=" * 70)

    has_gpu = detect_device()

    config = load_config()
    seq_length = config["sequence_length"]
    num_features = config["num_features"]
    print(f"[CFG] seq_length={seq_length}, features={num_features}")

    X, y, label_map = load_sequences(seq_length, num_features)
    if len(X) == 0:
        print("[ERROR] No se encontraron secuencias")
        return

    # Data augmentation
    print("\n[AUG] Aplicando data augmentation...")
    X_aug, y_aug = apply_augmentation(X, y, augmentations_per_sample=4)

    # Split 70/15/15 estratificado
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y_aug, test_size=0.30, random_state=42, stratify=y_aug
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[SPLIT] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    class_weights = compute_weights(y_train)
    print(f"[WEIGHTS] Calculados para {len(class_weights)} clases")

    num_classes = len(label_map)
    model = build_model(seq_length, num_features, num_classes)

    print("\n[MODEL] Arquitectura:")
    model.summary()

    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            str(MODEL_OUTPUT / "holistic_words_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    print("\n[TRAIN] Entrenando...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_OUTPUT / "holistic_words_model.keras")

    with open(MODEL_OUTPUT / "holistic_words_label_encoder.pkl", "wb") as f:
        pickle.dump(label_map, f)

    plot_history(history, MODEL_OUTPUT / "holistic_training_history.png")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print("\n" + "=" * 70)
    print(" RESULTADOS")
    print("=" * 70)
    print(f"   Val  accuracy: {val_acc * 100:.2f}% | loss: {val_loss:.4f}")
    print(f"   Test accuracy: {test_acc * 100:.2f}% | loss: {test_loss:.4f}")
    print(f"   Modelo: {MODEL_OUTPUT / 'holistic_words_model.keras'}")
    print(f"   Encoder: {MODEL_OUTPUT / 'holistic_words_label_encoder.pkl'}")
    print(f"   Device: {'GPU' if has_gpu else 'CPU'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
