from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.prediction import load_model, predict_audio
from src.model import train_model
from src.preprocessing import load_data_from_directory
import os
import shutil
import time
import threading
from pydantic import BaseModel


app = FastAPI(title="Surveillance Audio Classification API",
    description="An API to detect and classify sounds like gunshots, explosions, and other surveillance-related audio.",
    version="1.0.0")

MODEL_PATH = 'models/yamnet_sesa_model.keras'
model_lock = threading.Lock()
model = load_model(MODEL_PATH)
start_time = time.time()

class RootResponse(BaseModel):
    message: str


@app.get("/", response_model=RootResponse, summary="Root status check", tags=["General"])
def read_root():
    """
    Health check endpoint for the API.

    Returns:
        A simple message confirming the API is running.
    """
    return {"message": "Surveillance Sound Classification API"}

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: int
    model_path: str

@app.get("/status", response_model=StatusResponse, summary="API status check", tags=["Monitoring"])
def get_status():
    """
    Returns API uptime and model path information.

    Useful for checking if the service and model are available and running properly.
    """
    uptime = time.time() - start_time
    return {
        "status": "ok",
        "uptime_seconds": int(uptime),
        "model_path": MODEL_PATH
    }


class HealthResponse(BaseModel):
    status: str

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"]
)
def health_check():
    """
    Lightweight health check to confirm the API is responsive.

    Useful for uptime monitoring tools like AWS ELB, GCP Load Balancer, or Kubernetes liveness checks.
    """
    return {"status": "ok"}

class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sound class",
    tags=["Model Inference"],
    description="Upload a `.wav` file and receive the predicted sound class with confidence."
)
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

        return result

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


class RetrainResponse(BaseModel):
    message: str
    samples: int
    last_accuracy: Union[float, str]

@app.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Retrain the model",
    tags=["Model Training"],
    description="Triggers model retraining using new data in the `data/train` directory. "
                "The retrained model is then hot-reloaded and saved."
)
def retrain():
    try:
        print("[INFO] Starting retraining...")
        X, y = load_data_from_directory("data/train")
        new_model, history = train_model(X, y)
        accuracy = history.history.get("accuracy", [None])[-1]

        with model_lock:
            global model
            model = new_model
            model.save(MODEL_PATH)

        return RetrainResponse(
            message="Model retrained successfully",
            samples=len(X),
            last_accuracy=round(float(accuracy), 4) if accuracy else "N/A"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
