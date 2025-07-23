import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet model once
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Map label strings to numeric classes
LABEL_TO_INDEX = {
    "casual": 0,
    "gunshot": 1,
    "explosion": 2,
    "siren": 3
    # Add "alarm" here if you treat it as separate
}

def get_label_from_filename(filename):
    """Extract label string from filename prefix and map to index"""
    label_str = filename.split("_")[0].lower()
    return LABEL_TO_INDEX.get(label_str)

def extract_yamnet_embeddings(waveform):
    """Get YAMNet mean embedding for a waveform"""
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy()

def extract_features_from_file(file_path):
    """Load audio and extract YAMNet embedding"""
    try:
        waveform, sr = librosa.load(file_path, sr=16000, duration=10.0)
        return extract_yamnet_embeddings(waveform)
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return None

def load_data_from_directory(data_dir):
    """Load .wav files from directory and extract embeddings + labels"""
    X, y = [], []
    for file in os.listdir(data_dir):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(data_dir, file)
        label = get_label_from_filename(file)
        if label is None:
            print(f"Skipping unknown label in file: {file}")
            continue

        features = extract_features_from_file(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Loaded {len(X)} samples from {data_dir}")
    return X, y
