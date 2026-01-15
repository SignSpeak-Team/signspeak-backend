"""Quick test to verify all models load correctly"""
from services.vision_service.src.core.predictor import get_predictor

print("Loading predictor...")
p = get_predictor()

print("\n✓ All models loaded successfully!")
info = p.get_models_info()

print(f"\nStatic Model: {info['static_model']['count']} letters")
print(f"Dynamic Model: {info['dynamic_model']['count']} letters") 
print(f"Words Model: {info['words_model']['vocabulary_size']} words")
print(f"Words Accuracy: {info['words_model']['accuracy']}%")
