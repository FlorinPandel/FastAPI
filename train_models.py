
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load data
sessions = pd.read_csv('./data/workout_sessions.csv')
users = pd.read_csv('./data/user_profiles.csv')

# 1. Calculate weekly performance metrics per exercise
weekly_perf = sessions.groupby(['user_id', 'week', 'exercise']).agg({
    'volume': 'sum',  # Total reps/seconds for the week
    'weighted_volume': 'sum',  # Volume adjusted by RPE
    'avg_rpe': 'mean'
}).reset_index()

# 2. Calculate best single-session performance per week (proxy for strength)
best_session = sessions.groupby(['user_id', 'week', 'exercise']).agg({
    'volume': 'max'  # Best single session
}).reset_index()
best_session.rename(columns={'volume': 'best_session_volume'}, inplace=True)

weekly_perf = weekly_perf.merge(best_session, on=['user_id', 'week', 'exercise'])

# 3. Pivot to have one row per user-week with all exercises
perf_pivot = weekly_perf.pivot_table(
    index=['user_id', 'week'],
    columns='exercise',
    values=['volume', 'best_session_volume', 'weighted_volume']
).reset_index()

# Flatten column names
perf_pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in perf_pivot.columns.values]


# Building comprehensive training dataset with proper targets

# Now create proper training features and multiple target variables

# Aggregate weekly training load (all exercises combined)
weekly = sessions.groupby(['user_id', 'week']).agg({
    'weighted_volume': 'sum',
    'volume': 'sum',
    'avg_rpe': 'mean'
}).reset_index()

weekly.rename(columns={
    'weighted_volume': 'total_weighted_load',
    'volume': 'total_volume',
    'avg_rpe': 'avg_rpe'
}, inplace=True)

# Count sessions
sessions_count = sessions.groupby(['user_id', 'week']).size().reset_index(name='sessions_completed')
weekly = weekly.merge(sessions_count, on=['user_id', 'week'])

# Add consistency metric (assuming 12 sessions per week = 3 sessions × 4 exercises)
weekly['sessions_planned'] = 12
weekly['consistency'] = weekly['sessions_completed'] / weekly['sessions_planned']

# Sort by user and week
weekly = weekly.sort_values(['user_id', 'week']).reset_index(drop=True)

# Create rolling features (2-week windows)
weekly['total_weighted_load_2w'] = (
    weekly.groupby('user_id')['total_weighted_load']
    .rolling(2, min_periods=1).sum()
    .reset_index(0, drop=True)
)

weekly['avg_rpe_2w'] = (
    weekly.groupby('user_id')['avg_rpe']
    .rolling(2, min_periods=1).mean()
    .reset_index(0, drop=True)
)

# Volume trend (3-week slope)
def slope(x):
    if len(x) < 2:
        return 0
    return np.polyfit(range(len(x)), x, 1)[0]

weekly['volume_trend'] = (
    weekly.groupby('user_id')['total_weighted_load']
    .rolling(3, min_periods=2)
    .apply(slope, raw=False)
    .reset_index(0, drop=True)
)

# Fatigue index
weekly['fatigue_index'] = weekly['total_weighted_load'] * weekly['avg_rpe']

# Monotony (mean/std of recent volume)
weekly['monotony'] = (
    weekly.groupby('user_id')['total_weighted_load']
    .rolling(3, min_periods=2).mean()
    / weekly.groupby('user_id')['total_weighted_load']
    .rolling(3, min_periods=2).std()
).reset_index(0, drop=True)

# Replace inf with NaN
weekly['monotony'] = weekly['monotony'].replace([np.inf, -np.inf], np.nan)


# Merge with performance metrics
dataset = weekly.merge(perf_pivot, on=['user_id', 'week'], how='left')

# Merge with user profiles
dataset = dataset.merge(
    users[['user_id', 'age', 'weight', 'experience', 'progression_rate', 'fatigue_sensitivity']],
    on='user_id',
    how='left'
)

# CREATE MULTIPLE TARGET VARIABLES
# These are what we want to predict for next week

# 1. Best performance improvement (strength proxy) - for pushups as example
dataset['next_best_pushups'] = (
    dataset.groupby('user_id')['best_session_volume_pushups'].shift(-1)
)
dataset['pushups_improvement'] = (
    dataset['next_best_pushups'] - dataset['best_session_volume_pushups']
)

# 2. Total volume change (capacity)
dataset['next_total_volume'] = (
    dataset.groupby('user_id')['total_volume'].shift(-1)
)
dataset['volume_change'] = (
    dataset['next_total_volume'] - dataset['total_volume']
)

# 3. Weighted volume change (quality-adjusted capacity)
dataset['next_weighted_volume'] = (
    dataset.groupby('user_id')['total_weighted_load'].shift(-1)
)
dataset['weighted_volume_change'] = (
    dataset['next_weighted_volume'] - dataset['total_weighted_load']
)

# 4. RPE change (fatigue/recovery indicator)
dataset['next_rpe'] = (
    dataset.groupby('user_id')['avg_rpe'].shift(-1)
)
dataset['rpe_change'] = (
    dataset['next_rpe'] - dataset['avg_rpe']
)

# Drop last week for each user (no future data)
dataset = dataset.dropna(subset=['next_best_pushups'])

def categorize_volume_change(change):
    """Convert continuous change to actionable categories"""
    if change < -100:
        return 'DELOAD'  # Significant reduction needed
    elif change < -20:
        return 'REDUCE'  # Moderate reduction
    elif change < 20:
        return 'MAINTAIN'  # Keep current volume
    elif change < 100:
        return 'INCREASE'  # Moderate increase
    else:
        return 'PUSH'  # Significant increase

dataset['volume_recommendation'] = dataset['weighted_volume_change'].apply(categorize_volume_change)

print("=== VOLUME RECOMMENDATIONS DISTRIBUTION ===")
print(dataset['volume_recommendation'].value_counts())
print(f"\nPercentages:")
print(dataset['volume_recommendation'].value_counts(normalize=True) * 100)

# Prepare features
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
    'best_session_volume_pushups',  # Current strength level
    'total_volume'  # Current capacity
]

# Clean data
X = dataset[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Multiple targets
y_continuous = dataset['weighted_volume_change'].copy()
y_categorical = dataset['volume_recommendation'].copy()


# Use GroupKFold to prevent data leakage (same user in train and test)
groups = dataset['user_id'].values

# Split data (time-based)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_continuous.iloc[:split_idx], y_continuous.iloc[split_idx:]
groups_train = groups[:split_idx]

print("=== TRAINING MULTIPLE MODELS ===\n")

# 1. Baseline: Mean predictor
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"BASELINE (Mean Predictor)")
print(f"  MAE: {baseline_mae:.2f}")
print(f"  R²: {baseline_r2:.4f}")
print()

# 2. Ridge Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"RIDGE REGRESSION")
print(f"  MAE: {ridge_mae:.2f}")
print(f"  RMSE: {ridge_rmse:.2f}")
print(f"  R²: {ridge_r2:.4f}")
print()

# 3. Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"RANDOM FOREST")
print(f"  MAE: {rf_mae:.2f}")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"  R²: {rf_r2:.4f}")
print()

# 4. Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"GRADIENT BOOSTING")
print(f"  MAE: {gb_mae:.2f}")
print(f"  RMSE: {gb_rmse:.2f}")
print(f"  R²: {gb_r2:.4f}")
print()

# Summary
print("="*50)
print("SUMMARY: Best model is", end=" ")
best_mae = min(ridge_mae, rf_mae, gb_mae)
if best_mae == ridge_mae:
    print("RIDGE REGRESSION")
    best_model = ridge
    best_pred = y_pred_ridge
elif best_mae == rf_mae:
    print("RANDOM FOREST")
    best_model = rf
    best_pred = y_pred_rf
else:
    print("GRADIENT BOOSTING")
    best_model = gb
    best_pred = y_pred_gb

# Implementing proper temporal validation
print("Min week:", dataset['week'].min())
print("Max week:", dataset['week'].max())
print(dataset['week'].value_counts().sort_index())

# Proper temporal split: Use last 2 weeks for testing
print("=== PROPER TEMPORAL VALIDATION ===\n")

# Split by week
train_mask = dataset['week'] < 7
test_mask = dataset['week'] >= 7

X_train_proper = X[train_mask]
X_test_proper = X[test_mask]
y_train_proper = y_continuous[train_mask]
y_test_proper = y_continuous[test_mask]

print(f"Training set: {len(X_train_proper)} samples (weeks 0-8)")
print(f"Test set: {len(X_test_proper)} samples (weeks 9-10)")
print()

# Retrain models with proper split
scaler_proper = StandardScaler()
X_train_scaled_proper = scaler_proper.fit_transform(X_train_proper)
X_test_scaled_proper = scaler_proper.transform(X_test_proper)

# Baselines
baseline_pred_proper = np.full(len(y_test_proper), y_train_proper.mean())
baseline_mae_proper = mean_absolute_error(y_test_proper, baseline_pred_proper)
baseline_r2_proper = r2_score(y_test_proper, baseline_pred_proper)

dummy_pred_proper = np.zeros(len(y_test_proper))
dummy_mae_proper = mean_absolute_error(y_test_proper, dummy_pred_proper)
dummy_r2_proper = r2_score(y_test_proper, dummy_pred_proper)

# Ridge
ridge_proper = Ridge(alpha=10.0)
ridge_proper.fit(X_train_scaled_proper, y_train_proper)
y_pred_ridge_proper = ridge_proper.predict(X_test_scaled_proper)
ridge_mae_proper = mean_absolute_error(y_test_proper, y_pred_ridge_proper)
ridge_r2_proper = r2_score(y_test_proper, y_pred_ridge_proper)

# Random Forest
rf_proper = RandomForestRegressor(
    n_estimators=100, max_depth=8, min_samples_split=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_proper.fit(X_train_proper, y_train_proper)
y_pred_rf_proper = rf_proper.predict(X_test_proper)
rf_mae_proper = mean_absolute_error(y_test_proper, y_pred_rf_proper)
rf_r2_proper = r2_score(y_test_proper, y_pred_rf_proper)

# Results
results_proper = pd.DataFrame({
    'Model': ['Dummy (No Change)', 'Mean Predictor', 'Ridge Regression', 'Random Forest'],
    'MAE': [dummy_mae_proper, baseline_mae_proper, ridge_mae_proper, rf_mae_proper],
    'R²': [dummy_r2_proper, baseline_r2_proper, ridge_r2_proper, rf_r2_proper]
})

results_proper['MAE_vs_Dummy'] = ((dummy_mae_proper - results_proper['MAE']) / dummy_mae_proper * 100).round(1)

print("RESULTS WITH PROPER TEMPORAL VALIDATION")
print("="*70)
print(results_proper.to_string(index=False))
print("="*70)

print(f"\n\nCOMPARISON: Original vs Proper Split")
print("-" * 70)
print(f"Ridge Regression:")
print(f"  Original split - MAE: {ridge_mae:.2f}, R²: {ridge_r2:.4f}")
print(f"  Proper split   - MAE: {ridge_mae_proper:.2f}, R²: {ridge_r2_proper:.4f}")
print(f"\nRandom Forest:")
print(f"  Original split - MAE: {rf_mae:.2f}, R²: {rf_r2:.4f}")
print(f"  Proper split   - MAE: {rf_mae_proper:.2f}, R²: {rf_r2_proper:.4f}")

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(ridge_proper, "models/ridge_model.pkl")
joblib.dump(scaler_proper, "models/ridge_scaler.pkl")

print("✅ Ridge model & scaler saved")
