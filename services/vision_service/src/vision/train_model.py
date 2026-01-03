import json
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

DATASET_FILE = "services/vision_service/datasets/gestures.json"
MODEL_FILE = "services/vision_service/src/vision/model.pkl"

def main():
    with open(DATASET_FILE, "r") as f:
        dataset = json.load(f)

    X = []
    y = []

    for sample in dataset:
        X.append(sample["features"])
        y.append(sample["label"])

    X = np.array(X)
    y = np.array(y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print("✅ Modelo entrenado y guardado")

if __name__ == "__main__":
    main()
