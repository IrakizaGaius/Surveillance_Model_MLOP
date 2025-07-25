import time
import streamlit as st
import requests
import pandas as pd
import datetime
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Surveillance Audio Classifier", layout="centered")
st.title("🔊 Surveillance Sound Classifier")
st.markdown("Upload a `.wav` file to classify sounds like gunshots, explosions, sirens, or casual background noise.")

# Temporary log storage (in real use, you'd load from file/db)
if "logs" not in st.session_state:
    st.session_state.logs = []

def article_for(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"

# --- Upload and Predict ---
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Predict"):
        with st.spinner("Sending file to model..."):
            files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
            response = requests.post(f"{API_URL}/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                label = result['label']
                confidence = result['confidence']

                # Log the prediction

                st.session_state.logs.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": uploaded_file.name,
                    "label": label,
                    "confidence": confidence,
                    "model_version": result.get("model_version", "N/A")
                })
                article = article_for(result['label'])
                st.markdown(
                    f"""
                    <div style="font-size:1.3em; font-weight:bold; color:#2E86C1;">
                    This audio is {article} <em>{result['label']}</em>
                    </div>
                    <div style="font-size:1em; color:#555;">
                    Confidence: <strong>{result['confidence']:.2f}</strong>
                    </div>
                    """,
                unsafe_allow_html=True
                )
                # --- Optional: Show model version if available
                if result.get("model_version"):
                    st.info(f"Model Version: {result['model_version']}")

            else:
                st.error("❌ Prediction failed. Check backend logs.")

# --- Model Retraining ---
st.markdown("---")
st.subheader("🔁 Model Retraining")

# Upload new training files
uploaded_files = st.file_uploader("📤 Upload training .wav files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    # Create versioned training directory
    base_path = "data"
    existing_versions = [d for d in os.listdir(base_path) if d.startswith("train_v")]
    next_version = (
        max([int(d.split("_v")[-1]) for d in existing_versions], default=0) + 1
        if existing_versions else 1
    )
    version_dir = os.path.join(base_path, f"train_v{next_version}")
    os.makedirs(version_dir, exist_ok=True)

    # Save uploaded files
    for file in uploaded_files:
        file_path = os.path.join(version_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

    st.success(f"✅ Uploaded {len(uploaded_files)} file(s) to `{version_dir}`.")

    # Option to trigger retraining
    if st.button("🚀 Trigger Retraining"):
        with st.spinner("Retraining model..."):
            retrain_response = requests.post(f"{API_URL}/retrain")
            if retrain_response.status_code == 200:
                res = retrain_response.json()
                st.success("✅ Model retrained successfully.")
                st.json(res)
            else:
                st.error("❌ Retraining failed.")
                st.text(retrain_response.text)
else:
    st.info("Upload a batch of `.wav` files before retraining.")


# --- Model Status ---
st.markdown("---")
st.subheader("📈 Model & API Status")

try:
    status = requests.get(f"{API_URL}/status").json()

    uptime_seconds = status.get("uptime_seconds", 0)
    uptime_str = time.strftime('%Hh %Mm %Ss', time.gmtime(uptime_seconds))

    model_path = status.get("model_path", "N/A")
    model_version = model_path.split("_v")[-1].replace(".keras", "") if "_v" in model_path else "N/A"

    st.metric(label="Uptime", value=uptime_str)
    st.text(f"Model Path: {model_path}")
    st.text(f"Model Version: {model_version}")

except Exception as e:
    st.error("❌ Unable to fetch API status.")
    st.exception(e)
# --- Logs Section ---
st.markdown("---")
st.subheader("🧾 Recent Prediction Logs")

if st.session_state.logs:
    df_logs = pd.DataFrame(st.session_state.logs)

    # Optional: Format timestamp if present
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Show most recent first
    st.dataframe(df_logs[::-1], use_container_width=True)
else:
    st.info("No predictions yet.")

