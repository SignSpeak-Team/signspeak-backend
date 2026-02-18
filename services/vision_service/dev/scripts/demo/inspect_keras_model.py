"""
Inspects a Keras model (.h5) to show its architecture and input/output shapes.
"""
import tensorflow as tf
from pathlib import Path
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = r"services\vision_service\models\best_model (1).h5"

print("=" * 70)
print(f"INSPECTING MODEL: {MODEL_PATH}")
print("=" * 70)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_FILE = PROJECT_DIR / "model_summary.txt"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"✅ Model loaded successfully!\n\n")
        
        f.write(f"📊 Input Shape(s):\n")
        for i, input_tensor in enumerate(model.inputs):
            f.write(f"  Input {i}: {input_tensor.shape} (dtype: {input_tensor.dtype.name})\n")
            
        f.write(f"\n📊 Output Shape(s):\n")
        for i, output_tensor in enumerate(model.outputs):
            f.write(f"  Output {i}: {output_tensor.shape} (dtype: {output_tensor.dtype.name})\n")
            
        f.write(f"\n🏗️  Model Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        
        if model.optimizer:
            f.write(f"\n🔧 Optimizer Config:\n")
            try:
                cfg = model.optimizer.get_config()
                for k, v in cfg.items():
                    f.write(f"  {k}: {v}\n")
            except:
                f.write("  Could not retrieve optimizer config.\n")

    print("Check model_summary.txt for details.")
            
except Exception as e:
    print(f"\n❌ Error loading model: {e}")

print("\n" + "="*70)
