import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUM_CLASSES = 4

def build_simple_classifier():
    model = models.Sequential([
        layers.Input(shape=(1024,)),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    logging.info("Built new Sequential model")
    return model

def train_model(
    X, y,
    model_path: str,
    base_model: models.Sequential | None = None,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 100
):
    if not isinstance(model_path, str):
        raise ValueError("model_path must be a string")
    if not model_path.endswith('.keras'):
        model_path += '.keras'
        logging.info(f"Appended .keras to model_path: {model_path}")

    print(f"[INFO] Splitting data: train/test = {1 - test_size}/{test_size}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("[INFO] Building or loading model...")
    if base_model is None:
        model = build_simple_classifier()
    else:
        model = base_model
        logging.info("Using existing model for fine-tuning")

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        checkpoint
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose="auto"
    )

    model.save(model_path)
    logging.info(f"Model saved to: {model_path}")

    return model, history

def load_trained_model(model_path: str = "None"):
    if not isinstance(model_path, str):
        raise ValueError("model_path must be a string")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = models.load_model(model_path)
    logging.info(f"Loaded model from: {model_path}")
    return model