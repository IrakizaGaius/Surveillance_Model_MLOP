import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Constants
NUM_CLASSES = 4
MODEL_SAVE_PATH = 'models/yamnet_sesa_model.h5'


def build_simple_classifier():
    """Builds a simple classifier on top of YAMNet embeddings (1024-dim input)"""
    model = models.Sequential([
        layers.Input(shape=(1024,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X, y, test_size=0.2, batch_size=16, epochs=15, model_path=MODEL_SAVE_PATH):
    print(f"[INFO] Splitting data: train/test = {1 - test_size}/{test_size}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("[INFO] Building model...")
    model = build_simple_classifier()

    print("[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    return model, history


def load_trained_model(model_path=MODEL_SAVE_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)
