import os
import sys

# Suppress everything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
from pathlib import Path

MODEL_PATH = r"services\vision_service\models\LSTM_64_128.h5"


class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, time_major=False, **kwargs):
        # Ignore time_major, it's not supported in new TF
        super().__init__(**kwargs)
        self.time_major = time_major

    def get_config(self):
        config = super().get_config()
        config['time_major'] = self.time_major
        return config

try:
    # Restore stderr slightly earlier to catch import errors if any
    sys.stderr = stderr
    print(f"TF Version: {tf.__version__}")
    
    # Load with custom objects
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'LSTM': PatchedLSTM})
    
    print("---MODEL_INSPECTION_START---")
    print(f"Inputs: {[i.shape for i in model.inputs]}")
    print(f"Outputs: {[o.shape for o in model.outputs]}")
    
    if model.optimizer:
        try:
            cfg = model.optimizer.get_config()
            print(f"Learning Rate: {cfg.get('learning_rate', 'unknown')}")
        except:
            pass
            
    model.summary()
    print("---MODEL_INSPECTION_END---")

except Exception as e:
    sys.stderr = stderr
    print(f"Error: {e}")
