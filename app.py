from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# -------------------------------
# CORS middleware
# -------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary for testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load models and scaler (CPU-safe)
# -------------------------------
ridge_model = joblib.load("models/ridge_model.pkl")
scaler = joblib.load("models/ridge_scaler.pkl")

plank_model = joblib.load("models/plank_seconds_model.pkl")
pushups_model = joblib.load("models/pushups_model_cpu.pkl")
situps_model = joblib.load("models/situps_model_cpu.pkl")
squats_model = joblib.load("models/squats_model_cpu.pkl")

models = {
    "plank_seconds": plank_model,
    "pushups": pushups_model,
    "situps": situps_model,
    "squats": squats_model,
}

# -------------------------------
# Feature columns
# -------------------------------
FEATURE_COLS = [
    "week",
    "age", "weight", "experience", "true_strength",
    "progression_rate", "fatigue_sensitivity", "fatigue",
    "plank_seconds_lag1", "plank_seconds_lag2", "plank_seconds_lag3",
    "plank_seconds_rolling_avg_3w", "plank_seconds_trend",
    "pushups_lag1", "pushups_lag2", "pushups_lag3",
    "pushups_rolling_avg_3w", "pushups_trend",
    "situps_lag1", "situps_lag2", "situps_lag3",
    "situps_rolling_avg_3w", "situps_trend",
    "squats_lag1", "squats_lag2", "squats_lag3",
    "squats_rolling_avg_3w", "squats_trend",
]

# -------------------------------
# Input schemas
# -------------------------------
class RidgeInput(BaseModel):
    total_weighted_load_2w: float
    avg_rpe_2w: float
    volume_trend: float
    fatigue_index: float
    monotony: float
    age: int
    weight: float
    experience: int
    progression_rate: float
    fatigue_sensitivity: float
    best_session_volume_pushups: float
    total_volume: float


class PredictionInput(BaseModel):
    week: int
    age: int
    weight: float
    experience: int
    true_strength: float
    progression_rate: float
    fatigue_sensitivity: float
    fatigue: float

    plank_seconds_lag1: float
    plank_seconds_lag2: float
    plank_seconds_lag3: float
    plank_seconds_rolling_avg_3w: float
    plank_seconds_trend: float

    pushups_lag1: float
    pushups_lag2: float
    pushups_lag3: float
    pushups_rolling_avg_3w: float
    pushups_trend: float

    situps_lag1: float
    situps_lag2: float
    situps_lag3: float
    situps_rolling_avg_3w: float
    situps_trend: float

    squats_lag1: float
    squats_lag2: float
    squats_lag3: float
    squats_rolling_avg_3w: float
    squats_trend: float


# -------------------------------
# Helper: convert input → numpy array
# -------------------------------
def build_feature_vector(data: PredictionInput) -> np.ndarray:
    return np.array([[getattr(data, col) for col in FEATURE_COLS]])


# -------------------------------
# Prediction endpoints
# -------------------------------
@app.post("/predict/ridge")
def predict_ridge(data: RidgeInput):
    X = np.array([[
        data.total_weighted_load_2w,
        data.avg_rpe_2w,
        data.volume_trend,
        data.fatigue_index,
        data.monotony,
        data.age,
        data.weight,
        data.experience,
        data.progression_rate,
        data.fatigue_sensitivity,
        data.best_session_volume_pushups,
        data.total_volume
    ]])
    X_scaled = scaler.transform(X)
    prediction = ridge_model.predict(X_scaled)[0]
    return {"predicted_weighted_volume_change": float(prediction)}


@app.post("/predict/plank")
def predict_plank(data: PredictionInput):
    X = build_feature_vector(data)
    pred = models["plank_seconds"].predict(X)[0]
    return {"exercise": "plank_seconds", "prediction": float(pred)}


@app.post("/predict/pushups")
def predict_pushups(data: PredictionInput):
    X = build_feature_vector(data)
    pred = models["pushups"].predict(X)[0]
    return {"exercise": "pushups", "prediction": float(pred)}


@app.post("/predict/situps")
def predict_situps(data: PredictionInput):
    try:
        X = build_feature_vector(data)
        pred = models["situps"].predict(X)[0]
        return {"exercise": "situps", "prediction": float(pred)}
    except Exception:
        print("Situps prediction error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Server prediction error")


@app.post("/predict/squats")
def predict_squats(data: PredictionInput):
    X = build_feature_vector(data)
    pred = models["squats"].predict(X)[0]
    return {"exercise": "squats", "prediction": float(pred)}


@app.post("/predict/all")
def predict_all(data: PredictionInput):
    X = build_feature_vector(data)
    return {
        "plank_seconds": float(models["plank_seconds"].predict(X)[0]),
        "pushups": float(models["pushups"].predict(X)[0]),
        "situps": float(models["situps"].predict(X)[0]),
        "squats": float(models["squats"].predict(X)[0]),
    }
