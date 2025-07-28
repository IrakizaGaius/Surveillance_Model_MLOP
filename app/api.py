from datetime import datetime
from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.prediction import load_model, predict_audio
from src.model import train_model
from src.preprocessing import load_data_from_directory
from pydantic import BaseModel
import os
import shutil
import time
import threading
import logging
import absl.logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow INFO/WARNING
absl.logging.set_verbosity('error')
logging.getLogger("absl").setLevel(logging.ERROR)


app = FastAPI(
    title="Surveillance Audio Classification API",
    description="An API to detect and classify sounds like gunshots, explosions, and other surveillance-related audio.",
    version="1.0.0"
)


# === Utility Functions ===

def get_latest_model_path() -> str:
    model_dir = "models"
    # Find all subdirectories named like model_vX
    version_dirs = [d for d in os.listdir(model_dir) if d.startswith("model_v") and os.path.isdir(os.path.join(model_dir, d))]
    if not version_dirs:
        raise FileNotFoundError("No model version folders found in 'models' directory.")

    # Sort folders by version number descending
    version_dirs.sort(key=lambda d: int(d.split("_v")[-1]), reverse=True)

    latest_dir = version_dirs[0]
    # Find the .keras file inside that folder
    keras_files = [f for f in os.listdir(os.path.join(model_dir, latest_dir)) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras model file found inside {latest_dir}")

    # Assuming only one model file per version folder:
    return os.path.join(model_dir, latest_dir, keras_files[0])


def extract_model_version(model_path: str) -> str:
    # Example path: models/model_v1/sesa_model_v1.keras
    filename = os.path.basename(model_path)
    if "_v" in filename:
        return filename.split("_v")[-1].replace(".keras", "")
    # fallback, maybe check folder name
    folder = os.path.basename(os.path.dirname(model_path))
    if folder.startswith("model_v"):
        return folder.split("_v")[-1]
    return "unknown"


# === Initialization ===

MODEL_PATH = get_latest_model_path()
model_version = extract_model_version(MODEL_PATH)

model_lock = threading.Lock()
model = load_model(MODEL_PATH)
start_time = time.time()

# === Response Schemas ===

class RootResponse(BaseModel):
    message: str

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: int
    model_path: str
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
    last_accuracy: Union[float, str]
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

    temp_file = f"temp_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with model_lock:
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

@app.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Retrain the model",
    tags=["Model Training"]
)
def retrain():
    try:
        print("[INFO] Starting retraining...")

        model_dir = "models"
        # Find existing version folders like model_v1, model_v2, etc.
        existing_versions = [
            d for d in os.listdir(model_dir)
            if d.startswith("model_v") and os.path.isdir(os.path.join(model_dir, d))
        ]
        next_version_num = (
            max([int(d.split("_v")[-1]) for d in existing_versions], default=0) + 1
        )

        # Create new version folder and model path
        new_model_dir = os.path.join(model_dir, f"model_v{next_version_num}")
        os.makedirs(new_model_dir, exist_ok=True)
        new_model_path = os.path.join(new_model_dir, f"sesa_model_v{next_version_num}.keras")

        # Load training data for the new version
        train_dir = f"data/train_v{next_version_num}"
        X, y = load_data_from_directory(train_dir)

        # Train model and save to new_model_path
        new_model, history = train_model(X, y, model_path=new_model_path)
        accuracy = history.history.get("accuracy", [None])[-1]

        # Update the global model and paths inside a lock
        with model_lock:
            global model, MODEL_PATH, model_version
            model = new_model
            MODEL_PATH = new_model_path
            model_version = f"v{next_version_num}"

        return RetrainResponse(
            message="Model retrained successfully",
            samples=len(X),
            last_accuracy=round(float(accuracy), 4) if accuracy else "N/A",
            model_version=model_version,
            model_path=new_model_path
        )

    except Exception as e:
        logging.error(f"Retraining failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
