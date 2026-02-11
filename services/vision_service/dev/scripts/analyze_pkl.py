"""Deeper analysis of the inner structure of MSG3D pkl files."""
import pickle
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets_raw', 'msg3d', 'MEDIAPIPE')

# Custom unpickler
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(f'{module}.{name}', (), {'__repr__': lambda self: f'<stub {module}.{name}>'})

files = sorted(os.listdir(DATA_DIR))
f_path = os.path.join(DATA_DIR, files[0])

with open(f_path, 'rb') as f:
    data = SafeUnpickler(f).load()

print(f"File: {files[0]}")
print(f"Total frames: {len(data)}")
print(f"Frame 0 keys: {list(data[0].keys())}")

# Inspect each key in the first frame
for key in data[0]:
    val = data[0][key]
    print(f"\n--- Key: '{key}' ---")
    print(f"  Type: {type(val)}")
    print(f"  Repr: {str(val)[:200]}")
    
    # Try to access common protobuf-like attributes
    if hasattr(val, '__dict__'):
        print(f"  __dict__ keys: {list(val.__dict__.keys())}")
        for k2, v2 in val.__dict__.items():
            print(f"    {k2}: type={type(v2)}, repr={str(v2)[:100]}")
    
    if isinstance(val, dict):
        print(f"  Dict keys: {list(val.keys())}")
        for k2, v2 in val.items():
            print(f"    '{k2}': type={type(v2)}, repr={str(v2)[:150]}")
            if hasattr(v2, '__dict__'):
                print(f"      __dict__: {list(v2.__dict__.keys())}")
    
    if isinstance(val, (list, tuple)):
        print(f"  List length: {len(val)}")
        for i, item in enumerate(val[:3]):
            print(f"    [{i}]: type={type(item)}, repr={str(item)[:150]}")
            if hasattr(item, '__dict__'):
                print(f"      __dict__: {list(item.__dict__.keys())}")

# Check if file names contain label info
# Also look for any metadata files
print(f"\n{'='*60}")
print("Looking for label/metadata files...")
parent_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets_raw', 'msg3d')
for item in os.listdir(parent_dir):
    full = os.path.join(parent_dir, item)
    if os.path.isfile(full):
        print(f"  Found file: {item} ({os.path.getsize(full)} bytes)")
    else:
        print(f"  Found dir: {item}/")

# Check various frame counts across samples
print(f"\n{'='*60}")
print("Sampling frame counts across dataset...")
import random
random.seed(42)
sample_files = random.sample(files, min(20, len(files)))
for fname in sample_files:
    fp = os.path.join(DATA_DIR, fname)
    with open(fp, 'rb') as f:
        d = SafeUnpickler(f).load()
    if isinstance(d, list):
        print(f"  {fname}: {len(d)} frames")
    else:
        print(f"  {fname}: type={type(d)}")
