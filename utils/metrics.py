"""
Smart Drain — Metrics Utilities
Helpers for model evaluation and reporting.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
import pandas as pd


def compute_classification_metrics(
    y_true, y_pred, labels=None, target_names=None, average="weighted"
):
    """
    Compute standard classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    classification_report (str)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_early_warning_lead_time(
    flood_risk_series: np.ndarray,
    water_level_series: np.ndarray,
    flood_threshold_cm: float,
    high_risk_threshold: float = 50.0,
    dt_minutes: float = 1.0,
) -> float | None:
    """
    Compute early-warning lead time in minutes.

    Finds the first step where flood_risk >= high_risk_threshold and
    the first step where water_level >= flood_threshold_cm.  Lead time
    is the difference.

    Returns None if flood never occurs or warning never triggers.
    """
    risk_steps = np.where(np.array(flood_risk_series) >= high_risk_threshold)[0]
    flood_steps = np.where(np.array(water_level_series) >= flood_threshold_cm)[0]

    if len(risk_steps) == 0 or len(flood_steps) == 0:
        return None

    first_warning = risk_steps[0]
    first_flood = flood_steps[0]

    lead_time = (first_flood - first_warning) * dt_minutes
    return max(lead_time, 0.0)
