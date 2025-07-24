import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import random
import logging
from typing import Optional, Dict, Any

# Set up logger
logger = logging.getLogger(__name__)

# Load YAMNet model from TF Hub (full SavedModel, NOT KerasLayer)
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Label mapping
LABEL_TO_INDEX = {
    "casual": 0,
    "gunshot": 1,
    "explosion": 2,
    "siren": 3 #with Alarms
}

def get_label_from_filename(filename):
    """Extract label string from filename prefix and map to index."""
    label_str = filename.split("_")[0].lower()
    return LABEL_TO_INDEX.get(label_str)

# ---------------------- AUGMENTATION FUNCTIONS ----------------------

def add_noise(waveform, noise_level=0.005):
    noise = np.random.randn(len(waveform))
    return waveform + noise_level * noise

def pitch_shift(waveform, sr, n_steps=2):
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

def time_stretch(waveform, sr=16000, rate=1.1):
    try:
        # Minimum length needed is ~2x frame length (usually ~2048 samples)
        if len(waveform) < 4096:
            raise ValueError("Waveform too short for time stretching")

        return librosa.effects.time_stretch(waveform, rate=rate)
    except Exception as e:
        print(f"[AUGMENT] Skipping time-stretch (reason: {e})")
        return waveform

def apply_augmentation(waveform, sr):
    """Randomly apply one or more augmentations."""
    if random.random() < 0.3:
        waveform = add_noise(waveform)
    if random.random() < 0.3:
        waveform = pitch_shift(waveform, sr, n_steps=random.choice([-2, -1, 1, 2]))
    if random.random() < 0.3:
        waveform = time_stretch(waveform, sr, rate=random.uniform(0.9, 1.1))
    return waveform

# ---------------------- EMBEDDING EXTRACTION ----------------------

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


# ---------------------- FEATURE EXTRACTION ----------------------

def extract_features_from_file(file_path, sr=16000, target_duration=10.0, augment=False):
    try:
        waveform, _ = librosa.load(file_path, sr=sr)

        if len(waveform) < sr:
            print(f"[SKIP] {file_path} is too short")
            return None

        waveform = librosa.util.normalize(waveform)

        energy = np.sum(waveform ** 2) / len(waveform)
        if energy < 1e-6:
            print(f"[SKIP] {file_path} is silent (energy={energy:.2e})")
            return None

        if augment:
            waveform = apply_augmentation(waveform, sr)

        waveform = librosa.util.fix_length(waveform, size=int(sr * target_duration))

        return extract_embedding(waveform)

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return None

# ---------------------- DATA LOADER ----------------------

def load_data_from_directory(data_dir, augment=False, num_augments=2):
    """
    Load .wav files from a directory and convert to features + labels.

    Args:
        data_dir (str): Path to directory with .wav files
        augment (bool): Whether to apply augmentation (for training only)

    Returns:
        X, y: np.ndarray of features and labels
    """
    X, y = [], []

    for file in os.listdir(data_dir):
        if not file.lower().endswith(".wav"):
            continue

        label = get_label_from_filename(file)
        if label is None:
            print(f"[SKIP] Unknown label in filename: {file}")
            continue

        file_path = os.path.join(data_dir, file)

        # Add original
        features = extract_features_from_file(file_path, augment=False)
        if features is not None:
            X.append(features)
            y.append(label)

        # Add augmentations
        if augment:
            for _ in range(num_augments):
                features_aug = extract_features_from_file(file_path, augment=True)
                if features_aug is not None:
                    X.append(features_aug)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Loaded {len(X)} valid samples from '{data_dir}' (augment={augment})")
    return X, y
