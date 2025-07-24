import streamlit as st
import requests
import pandas as pd
import time
import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Surveillance Audio Classifier", layout="centered")
st.title("🔊 Surveillance Sound Classifier")
st.markdown("Upload a `.wav` file to classify sounds like gunshots, explosions, sirens, or casual background noise.")

# Temporary log storage (in real use, you'd load from file/db)
if "logs" not in st.session_state:
    st.session_state.logs = []

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
                st.success(f"🎯 Prediction: **{label}** with confidence **{confidence:.2f}**")

                # Log this prediction
                st.session_state.logs.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": uploaded_file.name,
                    "label": label,
                    "confidence": confidence
                })

                # --- Chart ---
                st.markdown("#### 🔬 Prediction Confidence")
                chart_df = pd.DataFrame(result["all_predictions"], index=result["labels"])
                st.bar_chart(chart_df)

            else:
                st.error("❌ Prediction failed. Check backend logs.")

# --- Model Retraining ---
st.markdown("---")
st.subheader("🔁 Model Retraining")

if st.button("Trigger Retraining"):
    with st.spinner("Retraining model..."):
        retrain_response = requests.post(f"{API_URL}/retrain")
        if retrain_response.status_code == 200:
            st.success("✅ Model retrained successfully.")
        else:
            st.error("❌ Retraining failed.")

# --- Model Status ---
st.markdown("---")
st.subheader("📈 Model & API Status")

try:
    status = requests.get(f"{API_URL}/status").json()
    st.metric(label="Uptime (seconds)", value=status.get("uptime_seconds", "N/A"))
    st.text(f"Model Path: {status.get('model_path', 'N/A')}")
except:
    st.error("❌ Unable to fetch API status.")

# --- Logs Section ---
st.markdown("---")
st.subheader("🧾 Recent Prediction Logs")

if st.session_state.logs:
    df_logs = pd.DataFrame(st.session_state.logs)
    st.dataframe(df_logs[::-1], use_container_width=True)
else:
    st.info("No predictions yet.")
