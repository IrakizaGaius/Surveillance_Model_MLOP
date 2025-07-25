# retrain.py

import os
import re
from datetime import datetime
from src.model import train_model
from src.preprocessing import load_data_from_directory
from keras.models import save_model as keras_save_model

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_NAME_BASE = "yamnet_sesa_model"
LATEST_MODEL_ALIAS = f"{MODEL_NAME_BASE}_latest.keras"
VERSION_FILE = os.path.join(MODEL_DIR, "latest_version.txt")

def get_next_version():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(VERSION_FILE):
        return 1

    with open(VERSION_FILE, "r") as f:
        last_version = f.read().strip()

    return int(last_version) + 1 if last_version.isdigit() else 1

def save_version_number(version: int):
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))

def main():
    # Get most recent training data folder
    versions = [d for d in os.listdir(DATA_DIR) if re.match(r"train_v\d+", d)]
    if not versions:
        print("[ERROR] No versioned training data found (e.g., 'train_v1')")
        return

    latest_train_dir = os.path.join(DATA_DIR, sorted(versions)[-1])
    print(f"[INFO] Loading training data from: {latest_train_dir}")
    X, y = load_data_from_directory(latest_train_dir)

    if len(X) == 0 or len(y) == 0:
        print("[ERROR] No training data loaded.")
        return

    print(f"[INFO] Training on {len(X)} samples...")
    model, history = train_model(X, y)

    version = get_next_version()
    versioned_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME_BASE}_v{version}.keras")
    alias_model_path = os.path.join(MODEL_DIR, LATEST_MODEL_ALIAS)

    keras_save_model(model, versioned_model_path)
    keras_save_model(model, alias_model_path)
    save_version_number(version)

    accuracy = history.history.get("accuracy", [None])[-1]
    print(f"[INFO] Training complete. Accuracy: {round(float(accuracy), 4) if accuracy else 'N/A'}")
    print(f"[INFO] Saved model: {versioned_model_path}")
    print(f"[INFO] Updated alias: {alias_model_path}")
    print(f"[INFO] Model version: {version}")

if __name__ == "__main__":
    main()
