"""
Análisis del modelo best_model.h5 con compatibilidad legacy
"""
import sys
import os

# Suprimir warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("ANÁLISIS DEL MODELO best_model.h5")
print("=" * 70)

model_path = 'services/vision_service/models/best_model.h5'
print(f"\n📥 Cargando modelo desde: {model_path}")

# Intentar diferentes métodos de carga
try:
    # Método 1: Keras estándar
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Cargado con keras.models.load_model (compile=False)")
except Exception as e1:
    print(f"⚠️  Método 1 falló: {e1}")
    
    try:
        # Método 2: Custom objects vacío
        model = keras.models.load_model(model_path, compile=False, custom_objects={})
        print("✅ Cargado con custom_objects={}")
    except Exception as e2:
        print(f"⚠️  Método 2 falló: {e2}")
        
        try:
            # Método 3: Usando h5py directamente para inspeccionar
            import h5py
            print("\n📂 Inspeccionando archivo HDF5 directamente...")
            
            with h5py.File(model_path, 'r') as f:
                print("\nGrupos en el archivo:")
                def print_structure(name, obj):
                    print(f"  {name}")
                f.visititems(print_structure)
                
                # Buscar configuración del modelo
                if 'model_config' in f.attrs:
                    import json
                    config = json.loads(f.attrs['model_config'])
                    print("\n📊 CONFIGURACIÓN DEL MODELO:")
                    print(f"  Clase: {config.get('class_name', 'Unknown')}")
                    
                    if 'config' in config:
                        layers = config['config'].get('layers', [])
                        print(f"  Número de capas: {len(layers)}")
                        
                        print("\n📋 CAPAS:")
                        for i, layer in enumerate(layers):
                            layer_class = layer.get('class_name', 'Unknown')
                            layer_config = layer.get('config', {})
                            layer_name = layer_config.get('name', 'unnamed')
                            
                            info = f"{i:2d}. {layer_name:25s} ({layer_class})"
                            
                            # Extraer info relevante
                            if 'units' in layer_config:
                                info += f" - units: {layer_config['units']}"
                            if 'rate' in layer_config:
                                info += f" - dropout: {layer_config['rate']}"
                            if 'batch_input_shape' in layer_config:
                                info += f" - input: {layer_config['batch_input_shape']}"
                                
                            print(info)
                        
                        # Buscar input y output shape
                        first_layer = layers[0] if layers else {}
                        last_layer = layers[-1] if layers else {}
                        
                        input_shape = first_layer.get('config', {}).get('batch_input_shape', 'Unknown')
                        output_units = last_layer.get('config', {}).get('units', 'Unknown')
                        
                        print(f"\n📐 RESUMEN:")
                        print(f"  Input shape: {input_shape}")
                        print(f"  Output units (clases): {output_units}")
                        
            print("\n✅ Análisis completado")
            sys.exit(0)
            
        except Exception as e3:
            print(f"❌ Método 3 falló: {e3}")
            sys.exit(1)

# Si llegamos aquí, el modelo se cargó exitosamente
print("\n" + "=" * 70)
print("📊 ARQUITECTURA DEL MODELO")
print("=" * 70)
model.summary()

print(f"\n📐 Input shape: {model.input_shape}")
print(f"📐 Output shape: {model.output_shape}")
print(f"📐 Número de clases: {model.output_shape[-1]}")
print(f"🧮 Total parámetros: {model.count_params():,}")
