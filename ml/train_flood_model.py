"""
Smart Drain — Train Flood-Risk Model
Binary classification: Flood (1) vs No-Flood (0).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DATASET_PATH,
    FLOOD_MODEL_PATH,
    FLOOD_SCALER_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS_FLOOD,
)
from utils.features import extract_feature_matrix
from utils.metrics import compute_classification_metrics


def train_flood_model(dataset_path: str = DATASET_PATH):
    """Train and save the flood-risk RandomForest classifier."""
    print("[...] Loading dataset...")
    df = pd.read_csv(dataset_path)

    X = extract_feature_matrix(df).fillna(0)
    y = df["flood_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("[...] Training flood-risk model...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_FLOOD,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    metrics = compute_classification_metrics(
        y_test, y_pred, labels=[0, 1], target_names=["No Flood", "Flood"]
    )

    print("\n── Flood Model Performance ──")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"\n{metrics['classification_report']}")

    os.makedirs(os.path.dirname(os.path.abspath(FLOOD_MODEL_PATH)), exist_ok=True)
    joblib.dump(clf, FLOOD_MODEL_PATH)
    joblib.dump(scaler, FLOOD_SCALER_PATH)
    print(f"[OK] Model saved: {FLOOD_MODEL_PATH}")

    return clf, scaler, metrics


if __name__ == "__main__":
    train_flood_model()
