"""
Smart Drain - One-Shot Setup Script
Generates the dataset and trains all ML models.

Run this once before launching the app if you prefer not to wait
for auto-training on first run:

    python setup.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("=" * 60)
print("  SMART DRAIN -- Setup & Training")
print("=" * 60)

# Step 1: Generate dataset
print("\n[1/4] Generating synthetic dataset...")
from data.generate_dataset import generate_dataset
df = generate_dataset()
print(f"      [OK] {len(df)} rows generated.")

# Step 2: Train flood model
print("\n[2/4] Training flood-risk model...")
from ml.train_flood_model import train_flood_model
_, _, fm = train_flood_model()
print(f"      [OK] Accuracy: {fm['accuracy']:.4f}  F1: {fm['f1']:.4f}")

# Step 3: Train cause model
print("\n[3/4] Training cause-classification model...")
from ml.train_cause_model import train_cause_model
_, _, cm_metrics = train_cause_model()
print(f"      [OK] Accuracy: {cm_metrics['accuracy']:.4f}  F1: {cm_metrics['f1']:.4f}")

# Step 4: Train baseline model
print("\n[4/4] Training hydraulic baseline model...")
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from utils.config import DATASET_PATH, BASELINE_MODEL_PATH, RANDOM_STATE

dset = pd.read_csv(DATASET_PATH)
X = dset[["rainfall", "water_level"]].fillna(0).values
y = dset["expected_flow"].fillna(0).values
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
reg.fit(X, y)
os.makedirs(os.path.dirname(BASELINE_MODEL_PATH), exist_ok=True)
joblib.dump(reg, BASELINE_MODEL_PATH)
print(f"      [OK] Baseline model saved -> {BASELINE_MODEL_PATH}")

print("\n" + "=" * 60)
print("  Setup complete! Run the app with:")
print("  streamlit run app.py")
print("=" * 60)
