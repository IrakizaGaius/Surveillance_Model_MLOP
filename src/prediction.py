import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import os

# Match the label map used during training
LABEL_MAP = {
    0: "Casual",
    1: "Gunshot",
    2: "Explosion",
    3: "Siren/Alarm"
}

YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

# Load YAMNet once globally
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)

def preprocess_audio(file_path, target_sr=16000, max_len=10):
    try:
        waveform, sr = librosa.load(file_path, sr=target_sr)
        waveform = waveform[:target_sr * max_len]  # Trim to 10 seconds
        if len(waveform) < target_sr * max_len:
            pad_width = target_sr * max_len - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        return waveform
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def extract_embedding(waveform):
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
    return tf.reduce_mean(embeddings, axis=0).numpy()  # Shape: (1024,)

def load_model(model_path='models/yamnet_sesa_model.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return keras.models.load_model(model_path)

def predict_audio(file_path, model):
    waveform = preprocess_audio(file_path)
    if waveform is None:
        return None

    embedding = extract_embedding(waveform)  # Shape: (1024,)
    embedding = np.expand_dims(embedding, axis=0)  # Shape: (1, 1024)

    predictions = model.predict(embedding)

    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id]

    return {
        "class_id": int(class_id),
        "label": LABEL_MAP[class_id],
        "confidence": float(confidence)
    }
