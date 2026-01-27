import pandas as pd
import numpy as np

# Load all data
user_profiles = pd.read_csv('./data/user_profiles.csv')
max_performance = pd.read_csv('./data/max_performance_by_week.csv')

# Define the 4 exercises
exercises = ['plank_seconds', 'situps', 'pushups', 'squats']

# Create lagged features for each exercise
# We'll create lags for 1, 2, and 3 weeks back
lags = [1, 2, 3]

# Sort by user and week to ensure proper lagging
max_performance = max_performance.sort_values(['user_id', 'week']).reset_index(drop=True)

# Create lagged features
for exercise in exercises:
    for lag in lags:
        col_name = f'{exercise}_lag{lag}'
        max_performance[col_name] = max_performance.groupby('user_id')[exercise].shift(lag)

# Create additional features: rolling averages and trends
for exercise in exercises:
    # Rolling average of last 3 weeks
    max_performance[f'{exercise}_rolling_avg_3w'] = (
        max_performance.groupby('user_id')[exercise]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    
    # Trend: difference between current and 1 week ago
    max_performance[f'{exercise}_trend'] = (
        max_performance[exercise] - max_performance[f'{exercise}_lag1']
    )

# Merge with user profile data
feature_data = max_performance.merge(user_profiles, on='user_id', how='left')

train_data = feature_data[feature_data['week'] >= 3].copy()

print(f"Training data shape (week 3+): {train_data.shape}")
print(f"Weeks in training data: {sorted(train_data['week'].unique())}")

# Define feature columns
feature_cols = [
    'week',  # Time component
    # User profile features
    'age', 'weight', 'experience', 'true_strength', 
    'progression_rate', 'fatigue_sensitivity', 'fatigue',
]

# Add lagged features for all exercises
for exercise in exercises:
    for lag in lags:
        feature_cols.append(f'{exercise}_lag{lag}')
    feature_cols.append(f'{exercise}_rolling_avg_3w')
    feature_cols.append(f'{exercise}_trend')

# Time-based split: Use weeks 3-9 for training, weeks 10-11 for testing
train_set = train_data[train_data['week'] <= 9].copy()
test_set = train_data[train_data['week'] > 9].copy()

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Dictionary to store models and results
models = {}
results = {}

print("Training models for each exercise...\n")
print("="*80)

for exercise in exercises:
    print(f"\n{'='*80}")
    print(f"EXERCISE: {exercise.upper()}")
    print(f"{'='*80}")
    
    # Prepare data
    X_train = train_set[feature_cols]
    y_train = train_set[exercise]
    X_test = test_set[feature_cols]
    y_test = test_set[exercise]
    
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Target range - Train: [{y_train.min():.1f}, {y_train.max():.1f}], Test: [{y_test.min():.1f}, {y_test.max():.1f}]")
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Store models
    models[exercise] = {
        'xgboost': xgb_model,
        'random_forest': rf_model
    }
    
    # Make predictions
    print(X_train.head())
    xgb_pred_train = xgb_model.predict(X_train)
    xgb_pred_test = xgb_model.predict(X_test)
    rf_pred_train = rf_model.predict(X_train)
    rf_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    results[exercise] = {
        'xgboost': {
            'train_mae': mean_absolute_error(y_train, xgb_pred_train),
            'test_mae': mean_absolute_error(y_test, xgb_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, xgb_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred_test)),
            'train_r2': r2_score(y_train, xgb_pred_train),
            'test_r2': r2_score(y_test, xgb_pred_test)
        },
        'random_forest': {
            'train_mae': mean_absolute_error(y_train, rf_pred_train),
            'test_mae': mean_absolute_error(y_test, rf_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
            'train_r2': r2_score(y_train, rf_pred_train),
            'test_r2': r2_score(y_test, rf_pred_test)
        }
    }
    
    print(f"\n{'XGBoost Results':^40}")
    print(f"  Train MAE: {results[exercise]['xgboost']['train_mae']:.3f} | Test MAE: {results[exercise]['xgboost']['test_mae']:.3f}")
    print(f"  Train RMSE: {results[exercise]['xgboost']['train_rmse']:.3f} | Test RMSE: {results[exercise]['xgboost']['test_rmse']:.3f}")
    print(f"  Train R²: {results[exercise]['xgboost']['train_r2']:.3f} | Test R²: {results[exercise]['xgboost']['test_r2']:.3f}")
    
    print(f"\n{'Random Forest Results':^40}")
    print(f"  Train MAE: {results[exercise]['random_forest']['train_mae']:.3f} | Test MAE: {results[exercise]['random_forest']['test_mae']:.3f}")
    print(f"  Train RMSE: {results[exercise]['random_forest']['train_rmse']:.3f} | Test RMSE: {results[exercise]['random_forest']['test_rmse']:.3f}")
    print(f"  Train R²: {results[exercise]['random_forest']['train_r2']:.3f} | Test R²: {results[exercise]['random_forest']['test_r2']:.3f}")

print(f"\n{'='*80}")
print("Model training complete!")

summary_data = []

for exercise in exercises:
    for model_name in ['xgboost', 'random_forest']:
        summary_data.append({
            'Exercise': exercise,
            'Model': model_name.replace('_', ' ').title(),
            'Train MAE': results[exercise][model_name]['train_mae'],
            'Test MAE': results[exercise][model_name]['test_mae'],
            'Train RMSE': results[exercise][model_name]['train_rmse'],
            'Test RMSE': results[exercise][model_name]['test_rmse'],
            'Train R²': results[exercise][model_name]['train_r2'],
            'Test R²': results[exercise][model_name]['test_r2']
        })

summary_df = pd.DataFrame(summary_data)

print("MODEL PERFORMANCE SUMMARY")
print("="*100)
print(summary_df.to_string(index=False))

# Find best model for each exercise
print("\n" + "="*100)
print("BEST MODEL PER EXERCISE (based on Test MAE)")
print("="*100)
best_models = {}
best_model_info = {}


for exercise in exercises:
    xgb_mae = results[exercise]['xgboost']['test_mae']
    rf_mae = results[exercise]['random_forest']['test_mae']
    best_model = 'XGBoost' if xgb_mae < rf_mae else 'Random Forest'
    best_mae = min(xgb_mae, rf_mae)
    best_r2 = results[exercise]['xgboost']['test_r2'] if xgb_mae < rf_mae else results[exercise]['random_forest']['test_r2']
    print(f"{exercise:20s} -> {best_model:15s} (MAE: {best_mae:.3f}, R²: {best_r2:.3f})")
    if xgb_mae < rf_mae:
        best_models[exercise] = models[exercise]['xgboost']
        best_model_info[exercise] = 'xgboost'
    else:
        best_models[exercise] = models[exercise]['random_forest']
        best_model_info[exercise] = 'random_forest'


pushups_model = best_models['pushups']
squats_model = best_models['squats']
situps_model = best_models['situps']
plank_seconds_model = best_models['plank_seconds']


import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(pushups_model, "models/pushups_model.pkl")
joblib.dump(squats_model, "models/squats_model.pkl")
joblib.dump(situps_model, "models/situps_model.pkl")
joblib.dump(plank_seconds_model, "models/plank_seconds_model.pkl")

print("Models have been saved to the 'models' directory.")

feature_importance_data = []

for exercise in exercises:
    xgb_model = models[exercise]['xgboost']
    
    # Get feature importance
    importance = xgb_model.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Add exercise column
    importance_df['exercise'] = exercise
    
    feature_importance_data.append(importance_df)
    
    print(f"\n{'='*80}")
    print(f"TOP 10 FEATURES FOR {exercise.upper()} (XGBoost)")
    print(f"{'='*80}")
    print(importance_df.head(10).to_string(index=False))

# Combine all feature importance
all_importance = pd.concat(feature_importance_data, ignore_index=True)

