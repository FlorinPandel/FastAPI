import xgboost as xgb
import joblib

# List of models to convert
models = [
    "pushups_model",
    "situps_model",
    "squats_model"
]

for model_name in models:
    print(f"Converting {model_name}...")

    # Load old GPU-trained pickle
    old_model = joblib.load(f"models/{model_name}.pkl")

    # Export to XGBoost JSON format (CPU-friendly)
    old_model.get_booster().save_model(f"models/{model_name}_cpu.json")

    # Load into a new CPU XGBRegressor
    cpu_model = xgb.XGBRegressor()
    cpu_model.load_model(f"models/{model_name}_cpu.json")
    cpu_model.set_params(tree_method="hist")  # CPU-only

    # Save CPU-compatible pickle
    joblib.dump(cpu_model, f"models/{model_name}_cpu.pkl")
    print(f"{model_name} converted successfully!")
