import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from data_augmentation import apply_augmentation

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent  # vision_service/
SEQUENCES_PATH = BASE_PATH / "datasets" / "sequences"
OUTPUT_PATH = BASE_PATH / "datasets" / "processed"

# Configuracion
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63  # 21 landmarks x 3 coords


def load_sequences():
    """Carga todas las secuencias y crea etiquetas."""
    X = []
    y = []
    label_map = {}
    label_idx = 0
    
    print("\n" + "="*60)
    print("CARGANDO SECUENCIAS")
    print("="*60)
    
    # Obtener carpetas (excluyendo 'test')
    folders = sorted([f for f in SEQUENCES_PATH.iterdir() 
                     if f.is_dir() and f.name != 'test'])
    
    for folder in folders:
        letter = folder.name.upper()
        sequences = list(folder.glob("*.npy"))
        
        if not sequences:
            continue
        
        label_map[letter] = label_idx
        
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
        
        print(f"  {letter}: {loaded} secuencias")
        label_idx += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal: {len(X)} secuencias")
    print(f"Clases: {len(label_map)} ({list(label_map.keys())})")
    print(f"Shape X: {X.shape}")
    
    return X, y, label_map


def create_lstm_model(num_classes):
    """Crea el modelo LSTM."""
    model = Sequential([
        # LSTM layers
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("\n" + "="*60)
    print("ENTRENAMIENTO LSTM - LETRAS DINAMICAS")
    print("="*60)
    
    # Cargar datos
    X, y, label_map = load_sequences()
    
    if len(X) == 0:
        print("[ERROR] No se encontraron secuencias")
        return
    
    # NUEVO: Aplicar Data Augmentation
    print(f"\n{'='*60}")
    print("¿APLICAR DATA AUGMENTATION?")
    print(f"{'='*60}")
    print("Esto generará variaciones de tus datos existentes para mejorar el modelo.")
    print("Incrementará el dataset aproximadamente 5x.\n")
    
    use_augmentation = input("¿Aplicar augmentation? (s/n): ").lower() == 's'
    
    if use_augmentation:
        X, y = apply_augmentation(X, y, augmentations_per_sample=4)
    else:
        print("\n[INFO] Continuando sin data augmentation\n")
    
    # Dividir datos
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")
    
    # Crear modelo
    num_classes = len(label_map)
    model = create_lstm_model(num_classes)
    
    print("\n" + "="*60)
    print("ARQUITECTURA DEL MODELO")
    print("="*60)
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        OUTPUT_PATH / 'lstm_letters_best.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Entrenar
    print("\n" + "="*60)
    print("ENTRENANDO")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Guardar modelo final
    model.save(OUTPUT_PATH / 'lstm_letters.keras')
    
    # Guardar label map
    with open(OUTPUT_PATH / 'lstm_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    
    # Generar diagrama del modelo
    print("\n[INFO] Generando diagrama del modelo...")
    try:
        plot_model(
            model, 
            to_file=OUTPUT_PATH / 'lstm_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=150
        )
        print(f"  Diagrama guardado: {OUTPUT_PATH / 'lstm_architecture.png'}")
    except Exception as e:
        print(f"  [WARN] No se pudo generar diagrama: {e}")
    
    # Generar graficas de entrenamiento
    print("[INFO] Generando graficas de entrenamiento...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Grafica de accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Precision del Modelo')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Grafica de loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Perdida del Modelo')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'lstm_training_history.png', dpi=150)
    print(f"  Graficas guardadas: {OUTPUT_PATH / 'lstm_training_history.png'}")
    
    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Precision validacion: {val_acc * 100:.2f}%")
    print(f"Loss validacion: {val_loss:.4f}")
    
    print(f"\nArchivos generados:")
    print(f"  - {OUTPUT_PATH / 'lstm_letters.keras'}")
    print(f"  - {OUTPUT_PATH / 'lstm_label_encoder.pkl'}")
    print(f"  - {OUTPUT_PATH / 'lstm_architecture.png'}")
    print(f"  - {OUTPUT_PATH / 'lstm_training_history.png'}")
    
    print("\n[DONE] Entrenamiento completado")


if __name__ == "__main__":
    main()
