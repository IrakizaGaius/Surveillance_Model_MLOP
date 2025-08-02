from datetime import datetime
from typing import Union, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.prediction import load_model, predict_audio
from src.model import train_model, build_simple_classifier
from src.preprocessing import load_data_from_directory
from pydantic import BaseModel
import os
import shutil
import time
import threading
import logging
import absl.logging
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow INFO/WARNING
absl.logging.set_verbosity('error')
logging.getLogger("absl").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Surveillance Audio Classification API",
    description="An API to detect and classify sounds like gunshots, explosions, and other surveillance-related audio.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants ===
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500MB total

# === Utility Functions ===
def get_latest_model_path() -> str:
    model_dir = "models"
    if not os.path.exists(model_dir):
        logging.warning("Models directory does not exist")
        return "None"
    
    version_dirs = [d for d in os.listdir(model_dir) if d.startswith("model_v") and os.path.isdir(os.path.join(model_dir, d))]
    if not version_dirs:
        logging.warning("No model version folders found in 'models' directory")
        return "None"

    version_dirs.sort(key=lambda d: int(d.split("_v")[-1]), reverse=True)
    latest_dir = version_dirs[0]
    keras_files = [f for f in os.listdir(os.path.join(model_dir, latest_dir)) if f.endswith(".keras")]
    if not keras_files:
        logging.warning(f"No .keras model file found inside {latest_dir}")
        return "None"

    return os.path.join(model_dir, latest_dir, keras_files[0])

def extract_model_version(model_path: str) -> str:
    if not model_path:
        return "v0"
    filename = os.path.basename(model_path)
    if "_v" in filename:
        return filename.split("_v")[-1].replace(".keras", "")
    folder = os.path.basename(os.path.dirname(model_path))
    if folder.startswith("model_v"):
        return folder.split("_v")[-1]
    return "v0"

# === Initialization ===
model_lock = threading.Lock()
start_time = time.time()
MODEL_PATH = get_latest_model_path()
model_version = extract_model_version(MODEL_PATH) if MODEL_PATH else "v0"
model = load_model(MODEL_PATH) if MODEL_PATH else None

# === Response Schemas ===
class RootResponse(BaseModel):
    message: str

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: int
    model_path: Union[str, None]
    model_version: str

class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float
    model_version: Union[str, None]
    timestamp: str

class RetrainResponse(BaseModel):
    message: str
    samples: int
    last_accuracy: str
    model_version: str
    model_path: str

# === Endpoints ===
@app.get("/", response_model=RootResponse, summary="Root status check", tags=["General"])
def read_root():
    return {"message": "Surveillance Sound Classification API"}

@app.get("/status", response_model=StatusResponse, summary="API status check", tags=["Monitoring"])
def get_status():
    uptime = int(time.time() - start_time)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "model_path": MODEL_PATH,
        "model_version": model_version
    }

@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["Monitoring"])
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, summary="Predict sound class", tags=["Model Inference"])
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    if hasattr(file, "size") and file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 100MB limit")

    temp_file = f"data/temp_predict_{file.filename}"
    try:
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Saved temporary file for prediction: {temp_file}")

        with model_lock:
            if model is None:
                raise HTTPException(status_code=503, detail="No model loaded")
            result = predict_audio(temp_file, model)

        if result is None:
            raise HTTPException(status_code=400, detail="Prediction failed")

        return PredictionResponse(
            **result,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.info(f"Removed temporary file: {temp_file}")

@app.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Retrain the model with uploaded data",
    tags=["Model Training"]
)
async def retrain(files: List[UploadFile] = File(...)):
    temp_data_dir = None
    try:
        print("[INFO] Starting retraining...")

        # Validate total file size
        total_size = sum(file.size or 0 for file in files if hasattr(file, "size"))
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(status_code=400, detail=f"Total file size ({total_size // (1024 * 1024)}MB) exceeds 500MB limit")

        model_dir = "models"
        existing_versions = [
            d for d in os.listdir(model_dir)
            if d.startswith("model_v") and os.path.isdir(os.path.join(model_dir, d))
        ]
        current_version_num = max([int(d.split("_v")[-1]) for d in existing_versions], default=0)
        next_version_num = current_version_num + 1

        # Create temporary directory for uploaded data
        temp_data_dir = f"data/temp_train_v{next_version_num}"
        os.makedirs(temp_data_dir, exist_ok=True)
        logging.info(f"Created temporary directory: {temp_data_dir}")

        # Save uploaded files temporarily
        for file in files:
            if file.content_type != "audio/wav":
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a .wav file")
            if hasattr(file, "size") and file.size is not None and file.size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 100MB limit")
            
            filename = file.filename or f"uploaded_{id(file)}.wav"
            temp_file_path = os.path.join(temp_data_dir, filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"Saved uploaded file to: {temp_file_path}")

        # Create new version folder and model path
        new_model_dir = os.path.join(model_dir, f"model_v{next_version_num}")
        os.makedirs(new_model_dir, exist_ok=True)
        new_model_path = os.path.join(new_model_dir, f"sesa_model_v{next_version_num}.keras")
        logging.info(f"New model path: {new_model_path}")

        # Load previous model if exists
        previous_model_path = os.path.join(model_dir, f"model_v{current_version_num}", f"sesa_model_v{current_version_num}.keras")
        base_model = load_model(previous_model_path) if os.path.exists(previous_model_path) else None
        logging.info(f"Loaded previous model from: {previous_model_path if base_model else 'None'}")

        # Load training data from temporary directory
        X, y = load_data_from_directory(temp_data_dir)
        logging.info(f"Loaded {len(X)} samples from {temp_data_dir}")

        # Train model with new data, using previous model as base
        new_model, history = train_model(X, y, model_path=new_model_path, base_model=base_model)
        accuracy = history.history.get("accuracy", [None])[-1]
        logging.info(f"Training completed with accuracy: {accuracy}")

        # Update the global model and paths inside a lock
        with model_lock:
            global model, MODEL_PATH, model_version
            model = new_model
            MODEL_PATH = new_model_path
            model_version = f"v{next_version_num}"
            logging.info(f"Updated global model to version: {model_version}")

        # Clean up temporary data
        if temp_data_dir and os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
            logging.info(f"Removed temporary directory: {temp_data_dir}")

        return RetrainResponse(
            message="Model retrained successfully",
            samples=len(X),
            last_accuracy=str(round(float(accuracy), 4)) if accuracy else "N/A",
            model_version=model_version,
            model_path=new_model_path
        )

    except Exception as e:
        logging.error(f"Retraining failed: {e}")
        if temp_data_dir and os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
            logging.info(f"Removed temporary directory on error: {temp_data_dir}")
        raise HTTPException(status_code=500, detail=str(e))