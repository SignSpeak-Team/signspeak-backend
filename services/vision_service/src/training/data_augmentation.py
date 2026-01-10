"""
Data Augmentation para secuencias de lenguaje de señas.
Genera variaciones de secuencias existentes para mejorar el entrenamiento del modelo.
"""

import numpy as np
from typing import List, Tuple


def add_noise(sequence: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """
    Añade ruido gaussiano a las coordenadas de la secuencia.
    
    Args:
        sequence: Array de shape (SEQUENCE_LENGTH, NUM_FEATURES)
        noise_factor: Desviación estándar del ruido (default: 0.01)
    
    Returns:
        Secuencia con ruido añadido
    """
    noise = np.random.normal(0, noise_factor, sequence.shape)
    return sequence + noise


def temporal_scale(sequence: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Escala la secuencia temporalmente (más rápido o más lento).
    
    Args:
        sequence: Array de shape (SEQUENCE_LENGTH, NUM_FEATURES)
        scale_factor: Factor de escala (>1 = más lento, <1 = más rápido)
   
    Returns:
        Secuencia escalada temporalmente
    """
    seq_len = sequence.shape[0]
    new_len = int(seq_len * scale_factor)
    
    # Interpolar para cambiar velocidad
    indices = np.linspace(0, seq_len - 1, new_len)
    scaled = np.array([np.interp(indices, np.arange(seq_len), sequence[:, i]) 
                      for i in range(sequence.shape[1])]).T
    
    # Ajustar al tamaño original (padding o truncate)
    if new_len > seq_len:
        # Más lento: truncar al final
        return scaled[:seq_len]
    else:
        # Más rápido: repetir último frame
        padding = np.repeat(scaled[-1:], seq_len - new_len, axis=0)
        return np.vstack([scaled, padding])


def horizontal_flip(sequence: np.ndarray) -> np.ndarray:
    """
    Invierte horizontalmente la secuencia (espejo).
    Útil para simular mano izquierda vs derecha.
    
    Args:
        sequence: Array de shape (SEQUENCE_LENGTH, NUM_FEATURES)
    
    Returns:
        Secuencia invertida horizontalmente
    """
    flipped = sequence.copy()
    
    # Invertir coordenadas X (cada tercera coordenada, empezando en 0)
    # Features: [x0, y0, z0, x1, y1, z1, ...]
    for i in range(0, sequence.shape[1], 3):
        flipped[:, i] = -flipped[:, i]  # Invertir X
    
    return flipped


def spatial_scale(sequence: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Escala espacialmente la secuencia (mano más grande o más pequeña).
    Simula acercar/alejar la mano de la cámara.
    
    Args:
        sequence: Array de shape (SEQUENCE_LENGTH, NUM_FEATURES)
        scale_factor: Factor de escala espacial (>1 = más grande, <1 = más pequeña)
    
    Returns:
        Secuencia escalada espacialmente
    """
    return sequence * scale_factor


def augment_sequence(sequence: np.ndarray, 
                     augmentation_type: str = 'all',
                     num_augmentations: int = 3) -> List[np.ndarray]:
    """
    Genera múltiples variaciones aumentadas de una secuencia.
    
    Args:
        sequence: Array de shape (SEQUENCE_LENGTH, NUM_FEATURES)
        augmentation_type: Tipo de augmentación ('noise', 'temporal', 'flip', 'spatial', 'all')
        num_augmentations: Número de variaciones a generar
    
    Returns:
        Lista de secuencias aumentadas
    """
    augmented = []
    
    if augmentation_type in ['noise', 'all']:
        # Añadir ruido con diferentes intensidades
        for noise_level in np.linspace(0.005, 0.02, num_augmentations):
            augmented.append(add_noise(sequence, noise_level))
    
    if augmentation_type in ['temporal', 'all']:
        # Escalar temporalmente
        for scale in [0.8, 1.2]:  # Más rápido y más lento
            augmented.append(temporal_scale(sequence, scale))
    
    if augmentation_type in ['flip', 'all']:
        # Invertir horizontalmente
        augmented.append(horizontal_flip(sequence))
    
    if augmentation_type in ['spatial', 'all']:
        # Escalar espacialmente
        for scale in [0.9, 1.1]:  # Más pequeña y más grande
            augmented.append(spatial_scale(sequence, scale))
    
    return augmented


def apply_augmentation(X: np.ndarray, y: np.ndarray, 
                      augmentations_per_sample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica data augmentation a todo el dataset.
    
    Args:
        X: Array de secuencias, shape (N, SEQUENCE_LENGTH, NUM_FEATURES)
        y: Array de etiquetas, shape (N,)
        augmentations_per_sample: Número de variaciones por muestra original
    
    Returns:
        (X_augmented, y_augmented): Dataset aumentado
    """
    print(f"\n{'='*60}")
    print(f"APLICANDO DATA AUGMENTATION")
    print(f"{'='*60}")
    print(f"Dataset original: {len(X)} muestras")
    print(f"Augmentaciones por muestra: {augmentations_per_sample}")
    
    X_aug = [X]  # Incluir datos originales
    y_aug = [y]
    
    for i, (sequence, label) in enumerate(zip(X, y)):
        # Generar variaciones
        augmented = augment_sequence(
            sequence, 
            augmentation_type='all',
            num_augmentations=3
        )
        
        # Tomar solo las primeras N variaciones
        augmented = augmented[:augmentations_per_sample]
        
        # Añadir al dataset
        X_aug.append(np.array(augmented))
        y_aug.append(np.full(len(augmented), label))
        
        if (i + 1) % 20 == 0:
            print(f"  Procesadas: {i + 1}/{len(X)} muestras", end='\r')
    
    X_augmented = np.vstack(X_aug)
    y_augmented = np.concatenate(y_aug)
    
    print(f"\n\nDataset aumentado: {len(X_augmented)} muestras")
    print(f"Factor de aumento: {len(X_augmented) / len(X):.1f}x")
    print(f"{'='*60}\n")
    
    return X_augmented, y_augmented
