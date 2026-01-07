"""
Script para generar estadisticas de modelos entrenados.
Genera un reporte con arquitectura, parametros y metricas.
"""

import numpy as np
import pickle
from pathlib import Path
from tensorflow import keras

# Rutas
BASE_PATH = Path(__file__).parent.parent.parent
MODELS_PATH = BASE_PATH / "datasets" / "processed"
OUTPUT_PATH = BASE_PATH / "datasets" / "reports"


def analyze_model(model_path, name, label_encoder_path=None):
    """Analiza un modelo y retorna estadisticas."""
    print(f"\n{'='*60}")
    print(f"MODELO: {name}")
    print(f"{'='*60}")
    
    model = keras.models.load_model(model_path)
    
    stats = {
        "nombre": name,
        "archivo": model_path.name,
        "capas": len(model.layers),
        "parametros_totales": model.count_params(),
        "parametros_entrenables": sum([np.prod(w.shape) for w in model.trainable_weights]),
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "clases": model.output_shape[-1]
    }
    
    # Cargar label encoder si existe
    if label_encoder_path and label_encoder_path.exists():
        with open(label_encoder_path, 'rb') as f:
            labels = pickle.load(f)
        stats["etiquetas"] = list(labels.keys())
    
    # Mostrar
    print(f"Archivo: {stats['archivo']}")
    print(f"Capas: {stats['capas']}")
    print(f"Parametros: {stats['parametros_totales']:,}")
    print(f"Input: {stats['input_shape']}")
    print(f"Output: {stats['output_shape']}")
    print(f"Clases: {stats['clases']}")
    
    if "etiquetas" in stats:
        print(f"Etiquetas: {stats['etiquetas']}")
    
    print("\nArquitectura:")
    model.summary()
    
    # Generar diagrama de arquitectura
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    diagram_name = model_path.stem + "_architecture.png"
    diagram_path = OUTPUT_PATH / diagram_name
    
    try:
        from keras.utils import plot_model
        plot_model(
            model,
            to_file=diagram_path,
            show_shapes=True,
            show_layer_names=True,
            dpi=150
        )
        print(f"\n[OK] Diagrama guardado: {diagram_path}")
        stats["diagrama"] = diagram_name
    except Exception as e:
        print(f"\n[WARN] No se pudo generar diagrama: {e}")
    
    return stats, model


def generate_report(all_stats):
    """Genera reporte en markdown."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    report = """# Estadisticas de Modelos - SignSpeak

## Resumen

| Modelo | Capas | Parametros | Clases | Input Shape |
|--------|-------|------------|--------|-------------|
"""
    
    for stats in all_stats:
        report += f"| {stats['nombre']} | {stats['capas']} | {stats['parametros_totales']:,} | {stats['clases']} | {stats['input_shape']} |\n"
    
    report += "\n---\n\n"
    
    for stats in all_stats:
        report += f"""## {stats['nombre']}

- **Archivo**: `{stats['archivo']}`
- **Capas**: {stats['capas']}
- **Parametros totales**: {stats['parametros_totales']:,}
- **Input shape**: {stats['input_shape']}
- **Output shape**: {stats['output_shape']}
- **Numero de clases**: {stats['clases']}
"""
        if "etiquetas" in stats:
            report += f"- **Clases**: {', '.join(stats['etiquetas'])}\n"
        
        report += "\n---\n\n"
    
    # Guardar
    report_path = OUTPUT_PATH / "model_statistics.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] Reporte guardado: {report_path}")
    return report_path


def main():
    print("\n" + "="*60)
    print("GENERADOR DE ESTADISTICAS - SignSpeak")
    print("="*60)
    
    all_stats = []
    
    # Modelo 1: Abecedario estatico
    static_model = MODELS_PATH / "sign_model.keras"
    static_labels = MODELS_PATH / "label_encoder.pkl"
    
    if static_model.exists():
        stats, _ = analyze_model(
            static_model, 
            "Abecedario Estatico (Dense)",
            static_labels
        )
        all_stats.append(stats)
    
    # Modelo 2: Letras dinamicas LSTM
    lstm_model = MODELS_PATH / "lstm_letters.keras"
    lstm_labels = MODELS_PATH / "lstm_label_encoder.pkl"
    
    if lstm_model.exists():
        stats, _ = analyze_model(
            lstm_model,
            "Letras Dinamicas (LSTM)",
            lstm_labels
        )
        all_stats.append(stats)
    
    # Generar reporte
    if all_stats:
        generate_report(all_stats)
    else:
        print("[ERROR] No se encontraron modelos")
    
    print("\n[DONE] Analisis completado")


if __name__ == "__main__":
    main()
