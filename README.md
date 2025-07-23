
# **Sound Events for Surveillance Applications (SESA) Dataset**

**Author:** Tito Spadini

A curated audio dataset designed for  **surveillance-based sound event detection (SED)** , focusing on threat detection in security applications. This dataset is ideal for training **machine learning models** to classify hazardous sounds such as gunshots, explosions, and alarms.

## **Dataset Overview**

### **Classes**

The dataset contains **4 distinct classes** of sound events:

* **0 - Casual** (non-threatening sounds, background noise)
* **1 - Gunshot** (firearms discharge)
* **2 - Explosion** (bombs, blasts)
* **3 - Siren** (emergency alarms, police/ambulance sirens)

### **Audio Specifications**

* **Format:** WAV
* **Channels:** Mono
* **Sampling Rate:** 16 kHz
* **Bit Depth:** 16-bit
* **Duration:** Up to **33 seconds** per clip

### **Dataset Split**

* **Training Set:** 480 audio files
* **Test Set:** 105 audio files

## **Intended Use Cases**

This dataset is suitable for:
✔ **Surveillance & Security Systems** – Detecting gunshots, explosions, and alarms in real-time.
✔ **Emergency Response AI** – Automating threat detection for law enforcement.
✔ **Machine Learning Research** – Benchmarking **audio classification** and **anomaly detection** models.

## **Source & Curation**

* Audio files were sourced from **Freesound** and manually verified.
* Balanced distribution between threat/non-threat categories.

## **How to Use**

1. Clone this repository.
2. Load the dataset using Python libraries like `librosa` or `torchaudio`.
3. Train a **CNN, RNN, or Transformer-based model** (e.g., YAMNet, VGGish) for sound classification.

## **License**

[Specify License – e.g., **CC-BY 4.0** or  **MIT** ]
