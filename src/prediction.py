import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers, models
from keras.models import load_model as keras_load_model
import os
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    0: "Casual",
    1: "Gunshot",
    2: "Explosion",
    3: "Siren/Alarm"
}

# Load YAMNet model once
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


def preprocess_audio(file_path: str, target_sr: int = 16000, max_len: int = 10) -> Optional[np.ndarray]:
    """
    Load and preprocess an audio file to a fixed length waveform.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sampling rate.
        max_len: Max length in seconds.

    Returns:
        np.ndarray: Audio waveform of shape (target_sr * max_len,) or None if failed.
    """
    try:
        waveform, sr = librosa.load(file_path, sr=target_sr)
        waveform = waveform[:target_sr * max_len]  # trim or pad
        if len(waveform) < target_sr * max_len:
            pad_width = target_sr * max_len - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        return waveform
    except Exception as e:
        logger.error(f"Error loading audio from {file_path}: {e}")
        return None

def extract_embedding(waveform: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract mean-pooled YAMNet embedding from waveform.

    Args:
        waveform: Audio waveform, shape (num_samples,).

    Returns:
        np.ndarray: Embedding vector of shape (1024,) or None if extraction fails.
    """
    if waveform is None or len(waveform) == 0:
        logger.warning("Empty or None waveform provided for embedding extraction")
        return None

    try:
        waveform = waveform.astype(np.float32)

        # Run the YAMNet model
        scores, embeddings, spectrogram = yamnet_model(waveform)  # type: ignore

        # Mean-pool the embeddings across time (axis=0)
        mean_embedding = tf.reduce_mean(embeddings, axis=0)  # shape: (1024,)

        if mean_embedding.shape[-1] != 1024:
            logger.warning(f"Unexpected embedding shape: {mean_embedding.shape}")

        return mean_embedding.numpy()

    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        return None



def load_model(model_path: str = '../models/yamnet_sesa_model.keras') -> models.Model:
    """
    Load the classification model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = keras_load_model(model_path)
    if not isinstance(model, models.Model):
        raise TypeError(f"Loaded object is not a keras.models.Model, got type: {type(model)}")
    return model


def predict_audio(file_path: str, model: models.Model) -> Optional[Dict[str, Any]]:
    """
    Predict the class of an audio file.

    Args:
        file_path: Path to audio file.
        model: Loaded Keras classification model.

    Returns:
        Dictionary with keys: class_id, label, confidence; or None on failure.
    """
    waveform = preprocess_audio(file_path)
    if waveform is None:
        return None

    embedding = extract_embedding(waveform)
    if embedding is None:
        return None

    embedding = np.expand_dims(embedding, axis=0)  # (1, 1024)

    try:
        predictions = model.predict(embedding)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

    class_id = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_id])

    return {
        "class_id": class_id,
        "label": LABEL_MAP.get(class_id, "Unknown"),
        "confidence": confidence
    }
