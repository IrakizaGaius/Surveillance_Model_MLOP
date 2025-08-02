import time
import streamlit as st
import requests
import pandas as pd
import datetime
import os
import io
import plotly.graph_objects as go

API_URL = "https://surveillancemodelmlop-production.up.railway.app"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max per file
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500MB total for all files
ALLOWED_FILE_TYPE = "audio/wav"

st.set_page_config(page_title="Surveillance Audio Classifier", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

def article_for(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #2E86C1; color: white; border-radius: 8px; }
    .stFileUploader { border: 2px dashed #2E86C1; border-radius: 8px; padding: 10px; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .prediction-title { font-size: 1.5em; font-weight: bold; color: #2E86C1; }
    .confidence-text { font-size: 1.1em; color: #555; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    .disclaimer { font-size: 0.9em; color: #666; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar with app info
with st.sidebar:
    st.header("🔊 Surveillance Sound Classifier")
    st.markdown("""
        Upload `.wav` files to classify sounds like gunshots, explosions, sirens, or background noise.
        Use the tabs below to predict, retrain the model, or view logs and status.
    """)

# Tabs for organized navigation
tab1, tab2, tab3 = st.tabs(["📁 Predict", "🔁 Retrain", "📈 Status & Logs"])

# --- Predict Tab ---
with tab1:
    st.subheader("Classify Audio")
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"], key="predict_uploader")
    
    if uploaded_file:
        with st.container():
            file_bytes = uploaded_file.read()
            file_stream_audio = io.BytesIO(file_bytes)
            file_stream_api = io.BytesIO(file_bytes)
            st.audio(file_stream_audio, format='audio/wav')
            
            if st.button("🔍 Predict", key="predict_button"):
                st.session_state.predict_clicked = True
            
            if st.session_state.predict_clicked:
                with st.spinner("🔄 Sending file to model..."):
                    try:
                        files = {"file": (uploaded_file.name, file_stream_api, "audio/wav")}
                        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                        st.session_state.predict_clicked = False
                        
                        if response.status_code == 200:
                            result = response.json()
                            label = result['label']
                            confidence = result['confidence']
                            article = article_for(label)
                            
                            st.session_state.logs.append({
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "file": uploaded_file.name,
                                "label": label,
                                "confidence": confidence,
                                "model_version": result.get("model_version", "N/A")
                            })
                            
                            # Prediction display
                            st.markdown(f"""
                                <div class="prediction-title">
                                    This audio is {article} <em>{label}</em>
                                </div>
                                <div class="confidence-text">
                                    Confidence: <strong>{confidence:.2f}</strong>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence table with color coding
                            if confidence <= 0.5:
                                # Red (255, 0, 0) for 0.0 to 0.5
                                t = (confidence - 0.0) / (0.5 - 0.0) if confidence > 0 else 0
                                r = 255
                                g = int(0 * t)
                                b = int(0 * t)
                                color = f"rgb({r}, {g}, {b})"
                            elif confidence <= 0.7:
                                # Blue (0, 128, 255) for 0.5 to 0.7
                                t = (confidence - 0.5) / (0.7 - 0.5)
                                r = int(0 * (1 - t))
                                g = int(128 * t)
                                b = int(255 * t)
                                color = f"rgb({r}, {g}, {b})"
                            else:
                                # Green (0, 255, 0) for > 0.7
                                t = min((confidence - 0.7) / (1.0 - 0.7), 1.0)
                                r = int(0 * (1 - t))
                                g = int(255 * t)
                                b = int(0 * (1 - t))
                                color = f"rgb({r}, {g}, {b})"
                            
                            df_conf = pd.DataFrame({
                                "Predicted Class": [label],
                                "Confidence": [f"{confidence:.2f}"]
                            })
                            fig_conf = go.Figure(data=[
                                go.Table(
                                    header=dict(
                                        values=["Predicted Class", "Confidence"],
                                        fill_color='white',
                                        font_color='black',
                                        align='center'
                                    ),
                                    cells=dict(
                                        values=[df_conf["Predicted Class"], df_conf["Confidence"]],
                                        fill_color=[color],
                                        font_color=['white' if confidence <= 0.5 else 'black'],
                                        align='center',
                                        height=30
                                    )
                                )
                            ])
                            fig_conf.update_layout(
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=100
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)
                            
                            # Color disclaimer
                            st.markdown("""
                                <div class="disclaimer">
                                    <strong>Color Coding:</strong><br>
                                    - <span style="color: rgb(255, 0, 0);">Red</span>: Low confidence (0.0–0.5) - The model is uncertain; results may be unreliable.<br>
                                    - <span style="color: rgb(0, 128, 255);">Blue</span>: Moderate confidence (0.5–0.7) - The model has moderate certainty.<br>
                                    - <span style="color: rgb(0, 255, 0);">Green</span>: High confidence (>0.7) - The model is highly certain of the prediction.
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if result.get("model_version"):
                                st.info(f"Model Version: {result['model_version']}")
                        else:
                            st.error(f"❌ Prediction failed (Status Code: {response.status_code})")
                            st.text(response.text)
                    except requests.exceptions.RequestException as e:
                        st.session_state.predict_clicked = False
                        st.error(f"❌ Network error during prediction: {str(e)}")
                    except Exception as e:
                        st.session_state.predict_clicked = False
                        st.error(f"❌ General error during prediction: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

# --- Retrain Tab ---
with tab2:
    st.subheader("Model Retraining")
    
    # File uploader for multiple .wav files
    uploaded_files = st.file_uploader(
        "Upload training .wav files",
        type=["wav"],
        accept_multiple_files=True,
        key="retrain_uploader",
        help="Upload multiple .wav files (max 100MB each, 500MB total) for model retraining."
    )
    
    if uploaded_files:
        with st.container():            
            # Validate files
            total_size = sum(file.size for file in uploaded_files)
            invalid_files = [
                file.name for file in uploaded_files
                if file.size > MAX_FILE_SIZE or file.type != ALLOWED_FILE_TYPE
            ]
            
            if invalid_files:
                st.error(f"❌ Invalid files: {', '.join(invalid_files)}. Each file must be a .wav file and under 100MB.")
            elif total_size > MAX_TOTAL_SIZE:
                st.error(f"❌ Total file size ({total_size // (1024 * 1024)}MB) exceeds limit of {MAX_TOTAL_SIZE // (1024 * 1024)}MB.")
            else:
                st.success(f"✅ Selected {len(uploaded_files)} file(s) for retraining.")
                
                if st.button("🚀 Trigger Retraining", key="retrain_button"):
                    with st.spinner("🧠 Sending files to retrain model..."):
                        try:
                            # Prepare files for multipart/form-data upload
                            files = [
                                ("files", (file.name, file.read(), ALLOWED_FILE_TYPE))
                                for file in uploaded_files
                            ]
                            response = requests.post(f"{API_URL}/retrain", files=files, timeout=120)
                            
                            if response.status_code == 200:
                                st.success("✅ Model retrained successfully.")
                                st.json(response.json())
                            else:
                                st.error(f"❌ Retraining failed (Status Code: {response.status_code})")
                                st.text(response.text)
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out. Please try again or check the server.")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Failed to connect to the server. Check your network or API URL.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Network error during retraining: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ General error during retraining: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload multiple `.wav` files (max 100MB each, 500MB total) to retrain the model.")

# --- Status & Logs Tab ---
with tab3:
    st.subheader("Model Status & Prediction Logs")
    
    # Model Status
    with st.container():
        st.markdown("### API Status")
        try:
            status = requests.get(f"{API_URL}/status").json()
            uptime_seconds = status.get("uptime_seconds", 0)
            uptime_str = time.strftime('%Hh %Mm %Ss', time.gmtime(uptime_seconds))
            model_path = status.get("model_path", "N/A")
            model_version = model_path.split("_v")[-1].replace(".keras", "") if "_v" in model_path else "N/A"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Uptime", value=uptime_str)
            with col2:
                st.metric(label="Model Version", value=model_version)
            st.text(f"Model Path: {model_path}")
        except Exception as e:
            st.error("❌ Unable to fetch API status.")
            st.exception(e)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Logs
    with st.container():
        st.markdown("### Recent Prediction Logs")
        if st.session_state.logs:
            df_logs = pd.DataFrame(st.session_state.logs)
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(df_logs[::-1], use_container_width=True)
            
            # Download logs as CSV
            csv = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Logs as CSV",
                data=csv,
                file_name=f"prediction_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions yet.")
        st.markdown('</div>', unsafe_allow_html=True)