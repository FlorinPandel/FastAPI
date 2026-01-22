from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all (not recommended in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load once at startup
ridge_model = joblib.load("models/ridge_model.pkl")
scaler = joblib.load("models/ridge_scaler.pkl")


feature_cols = [
    'total_weighted_load_2w',
    'avg_rpe_2w',
    'volume_trend',
    'fatigue_index',
    'monotony',
    'age',
    'weight',
    'experience',
    'progression_rate',
    'fatigue_sensitivity',
    'best_session_volume_pushups',
    'total_volume'
]

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


@app.post("/predict/ridge")
def predict_ridge(data: RidgeInput):
    # Convert to numpy in correct order
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

    # Scale (DO NOT fit again)
    X_scaled = scaler.transform(X)

    prediction = ridge_model.predict(X_scaled)[0]

    return {
        "predicted_weighted_volume_change": float(prediction)
    }

