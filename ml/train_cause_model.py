"""
Smart Drain — Train Cause-Classification Model
Multi-class: NORMAL / HEAVY_RAINFALL / BLOCKAGE / CAPACITY_EXCEEDED / FLOW_RESTRICTION
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
    CAUSE_MODEL_PATH,
    CAUSE_SCALER_PATH,
    FEATURE_COLUMNS,
    CAUSE_CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS_CAUSE,
)
from utils.features import extract_feature_matrix
from utils.metrics import compute_classification_metrics


def train_cause_model(dataset_path: str = DATASET_PATH):
    """Train and save the cause-classification RandomForest classifier."""
    print("[...] Loading dataset...")
    df = pd.read_csv(dataset_path)

    X = extract_feature_matrix(df).fillna(0)
    y = df["cause_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("[...] Training cause-classification model...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_CAUSE,
        max_depth=18,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        labels=list(range(len(CAUSE_CLASSES))),
        target_names=CAUSE_CLASSES,
    )

    print("\n── Cause Model Performance ──")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"\n{metrics['classification_report']}")

    os.makedirs(os.path.dirname(os.path.abspath(CAUSE_MODEL_PATH)), exist_ok=True)
    joblib.dump(clf, CAUSE_MODEL_PATH)
    joblib.dump(scaler, CAUSE_SCALER_PATH)
    print(f"[OK] Model saved: {CAUSE_MODEL_PATH}")

    return clf, scaler, metrics


if __name__ == "__main__":
    train_cause_model()
