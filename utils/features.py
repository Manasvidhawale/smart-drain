"""
Smart Drain — Feature Engineering
Extracts and prepares ML-ready feature vectors from simulation DataFrames.
"""

import pandas as pd
import numpy as np
from utils.config import FEATURE_COLUMNS, FLOOD_WATER_LEVEL_THRESHOLD_CM, CAUSE_CLASSES


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all FEATURE_COLUMNS exist in df.  Create any that are missing
    from raw simulation output (rolling averages, etc.).
    """
    out = df.copy()

    # Ensure rolling averages exist
    for col in ["rainfall", "water_level", "flow"]:
        avg_col = f"{col}_avg_5"
        if avg_col not in out.columns:
            out[avg_col] = out[col].rolling(window=5, min_periods=1).mean()

    # Ensure all feature columns exist (fill with 0 if truly missing)
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0

    return out


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns as a DataFrame."""
    df = compute_features(df)
    return df[FEATURE_COLUMNS].copy()


def assign_flood_label(df: pd.DataFrame, threshold_cm: float = FLOOD_WATER_LEVEL_THRESHOLD_CM) -> pd.Series:
    """
    Binary flood label: 1 if water_level >= threshold, else 0.
    Also considers rise-rate for early detection scenarios.
    """
    flood = (df["water_level"] >= threshold_cm).astype(int)
    # Also flag if level is high AND rising fast
    near_flood = (
        (df["water_level"] >= threshold_cm * 0.75)
        & (df["water_level_rise_rate"] > 1.5)
    )
    flood = flood | near_flood.astype(int)
    return flood


def assign_cause_label(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic cause labelling for training data based on underlying
    simulation parameters.  Returns integer-encoded labels.

    Priority logic (highest priority first):
      1. BLOCKAGE — blockage_pct >= 40 AND flow_deviation > 0.5
      2. CAPACITY_EXCEEDED — inflow > capacity AND blockage_pct < 20
      3. FLOW_RESTRICTION — moderate flow_deviation without heavy blockage
      4. HEAVY_RAINFALL — rainfall >= 30 AND blockage_pct < 15
      5. NORMAL — everything else
    """
    labels = pd.Series("NORMAL", index=df.index)

    # Heavy rainfall
    mask_heavy = (df["rainfall"] >= 30) & (df["blockage_pct"] < 15)
    labels[mask_heavy] = "HEAVY_RAINFALL"

    # Flow restriction
    mask_restrict = (
        (df["flow_deviation"] > 0.3)
        & (df["blockage_pct"] < 40)
        & (df["blockage_pct"] >= 5)
    )
    labels[mask_restrict] = "FLOW_RESTRICTION"

    # Capacity exceeded  (inflow > capacity, low blockage)
    inflow = df.get("inflow", df["rainfall"] * 500 * 0.85 / 3600)
    mask_cap = (
        (inflow > df["drainage_capacity"])
        & (df["blockage_pct"] < 20)
    )
    labels[mask_cap] = "CAPACITY_EXCEEDED"

    # Blockage
    mask_block = (df["blockage_pct"] >= 40) & (df["flow_deviation"] > 0.3)
    labels[mask_block] = "BLOCKAGE"

    # Encode to integers
    cause_map = {c: i for i, c in enumerate(CAUSE_CLASSES)}
    return labels.map(cause_map).astype(int)
