from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.prediction import load_model, predict_audio
from src.model import train_model
from src.preprocessing import load_data_from_directory
import os
import shutil
import time

app = FastAPI()
MODEL_PATH = 'models/yamnet_sesa_model.h5'
model = load_model(MODEL_PATH)
start_time = time.time()

@app.get("/")
def read_root():
    return {"message": "Surveillance Sound Classification API"}

@app.get("/status")
def get_status():
    uptime = time.time() - start_time
    return {
        "status": "ok",
        "uptime_seconds": int(uptime),
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_audio(temp_file, model)
    os.remove(temp_file)

    if result is None:
        return JSONResponse(content={"error": "Could not process audio"}, status_code=400)
    
    return result

@app.post("/retrain")
def retrain():
    try:
        print("[INFO] Starting retraining...")
        X, y = load_data_from_directory("data/train")
        new_model = train_model(X, y)
        global model
        model = new_model  # hot-replace in-memory model
        return {"message": "Model retrained successfully"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
