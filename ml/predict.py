"""
Smart Drain — Prediction Module
Loads trained models and provides real-time inference.

Performance design
------------------
Use `predict_dataframe(df)` for bulk inference over a simulation DataFrame.
This calls sklearn's predict / predict_proba **once** on the full matrix
instead of row-by-row — typically 50-100× faster for 90-step simulations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from utils.config import (
    FLOOD_MODEL_PATH,
    CAUSE_MODEL_PATH,
    BASELINE_MODEL_PATH,
    FLOOD_SCALER_PATH,
    CAUSE_SCALER_PATH,
    FEATURE_COLUMNS,
    CAUSE_CLASSES,
    FLOOD_RISK_LEVELS,
    SEGMENT_LABELS,
    SEGMENT_CONNECTIONS,
)


def _load(path):
    """Load a joblib model, return None if missing."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _risk_level(flood_prob: float) -> str:
    """Map flood probability (0-100) to risk level string."""
    if flood_prob >= 75:
        return "CRITICAL"
    for level, (lo, hi) in FLOOD_RISK_LEVELS.items():
        if lo <= flood_prob < hi:
            return level
    return "LOW"


class SmartDrainPredictor:
    """Unified predictor for flood risk, cause, and blockage localization."""

    def __init__(self):
        self.flood_model = _load(FLOOD_MODEL_PATH)
        self.flood_scaler = _load(FLOOD_SCALER_PATH)
        self.cause_model = _load(CAUSE_MODEL_PATH)
        self.cause_scaler = _load(CAUSE_SCALER_PATH)
        self.baseline_model = _load(BASELINE_MODEL_PATH)

    @property
    def is_ready(self) -> bool:
        return self.flood_model is not None and self.cause_model is not None

    # ────────── Feature preparation ──────────
    def _prepare_row(self, row: dict) -> pd.DataFrame:
        """Build a named-column DataFrame (1 row) from a sensor reading dict.

        Using a DataFrame (not ndarray) avoids sklearn UserWarnings when the
        scaler was fitted with feature names.
        """
        data = {c: [float(row.get(c, 0.0))] for c in FEATURE_COLUMNS}
        return pd.DataFrame(data)

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the feature matrix from a full simulation DataFrame."""
        X = pd.DataFrame(index=df.index)
        for c in FEATURE_COLUMNS:
            X[c] = df[c].values if c in df.columns else 0.0
        return X.astype(float)

    # ══════════════════════════════════════════════════
    #  BATCH INFERENCE  — use this for simulation DFs
    # ══════════════════════════════════════════════════
    def predict_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run all predictions (flood, cause, blockage) on an entire simulation
        DataFrame **in batch** — a single sklearn call per model.

        Returns
        -------
        list of dicts (one per row) with keys:
            flood_prob, risk_level, cause, cause_probs,
            blockage_seg, blockage_conf, blockage_all, blockage_detected
        """
        X = self._prepare_df(df)
        n = len(X)

        # ── Flood risk (batch) ──
        if self.flood_model is not None:
            X_f = self.flood_scaler.transform(X) if self.flood_scaler else X.values
            flood_probas = self.flood_model.predict_proba(X_f)          # (n, 2)
            flood_probs_pct = (flood_probas[:, 1] * 100).round(1)
        else:
            flood_probs_pct = np.zeros(n)

        # ── Cause diagnosis (batch) ──
        if self.cause_model is not None:
            X_c = self.cause_scaler.transform(X) if self.cause_scaler else X.values
            cause_ids = self.cause_model.predict(X_c)                   # (n,)
            cause_probas = self.cause_model.predict_proba(X_c)          # (n, k)
        else:
            cause_ids = np.zeros(n, dtype=int)
            cause_probas = np.zeros((n, len(CAUSE_CLASSES)))

        # ── Blockage localization — vectorised math ──
        s1 = df["s1_level"].values if "s1_level" in df.columns else np.zeros(n)
        s2 = df["s2_level"].values if "s2_level" in df.columns else np.zeros(n)
        s3 = df["s3_level"].values if "s3_level" in df.columns else np.zeros(n)
        blockage = df["blockage_pct"].values if "blockage_pct" in df.columns else np.zeros(n)
        flow_dev = df["flow_deviation"].values if "flow_deviation" in df.columns else np.zeros(n)

        d12 = s1 - s2
        d23 = s2 - s3

        MIN_FLOW_DEV  = 0.3
        MIN_BLOCKAGE  = 10.0
        MIN_LEVEL_DIFF = 2.0

        has_signal = (
            (blockage >= MIN_BLOCKAGE)
            | (flow_dev >= MIN_FLOW_DEV)
            | (np.abs(d12) >= MIN_LEVEL_DIFF)
            | (np.abs(d23) >= MIN_LEVEL_DIFF)
        )

        seg_s1_s2  = SEGMENT_CONNECTIONS[0]
        seg_s2_s3  = SEGMENT_CONNECTIONS[1]
        seg_s3_out = SEGMENT_CONNECTIONS[2]

        sc_s1_s2  = np.maximum(0.0, d12) * 2.0 + np.maximum(0.0, flow_dev) * 0.5
        sc_s2_s3  = np.maximum(0.0, d23) * 2.0 + np.maximum(0.0, flow_dev) * 0.8
        sc_s3_out = np.maximum(0.0, s3 - 30.0) * 0.5 + np.maximum(0.0, flow_dev) * 0.3

        boost = np.where(blockage > MIN_BLOCKAGE, 1.0 + blockage / 100.0, 1.0)
        sc_s1_s2  *= boost
        sc_s2_s3  *= boost
        sc_s3_out *= boost

        totals = sc_s1_s2 + sc_s2_s3 + sc_s3_out + 1e-9
        p_s1_s2  = (sc_s1_s2  / totals * 100).round(1)
        p_s2_s3  = (sc_s2_s3  / totals * 100).round(1)
        p_s3_out = (sc_s3_out / totals * 100).round(1)

        # Best segment per row
        seg_matrix = np.stack([p_s1_s2, p_s2_s3, p_s3_out], axis=1)  # (n, 3)
        best_idx = seg_matrix.argmax(axis=1)
        seg_names_arr = np.array([seg_s1_s2, seg_s2_s3, seg_s3_out])
        best_segs = seg_names_arr[best_idx]
        best_confs = seg_matrix[np.arange(n), best_idx]

        # ── Assemble results ──
        results = []
        no_seg_default = {k: round(100.0 / 3, 1) for k in SEGMENT_CONNECTIONS}

        for i in range(n):
            fp = float(flood_probs_pct[i])
            cid = int(cause_ids[i])
            cp_arr = cause_probas[i]

            cause_prob_dict = {
                CAUSE_CLASSES[j]: round(float(cp_arr[j]) * 100, 1)
                for j in range(len(CAUSE_CLASSES))
            }

            if has_signal[i]:
                bl_seg   = str(best_segs[i])
                bl_conf  = float(best_confs[i])
                bl_all   = {
                    seg_s1_s2:  float(p_s1_s2[i]),
                    seg_s2_s3:  float(p_s2_s3[i]),
                    seg_s3_out: float(p_s3_out[i]),
                }
                bl_det = True
            else:
                bl_seg  = "None Detected"
                bl_conf = 0.0
                bl_all  = no_seg_default.copy()
                bl_det  = False

            results.append({
                "flood_prob":        round(fp, 1),
                "risk_level":        _risk_level(fp),
                "cause":             CAUSE_CLASSES[cid] if cid < len(CAUSE_CLASSES) else "UNKNOWN",
                "cause_probs":       cause_prob_dict,
                "blockage_seg":      bl_seg,
                "blockage_conf":     bl_conf,
                "blockage_all":      bl_all,
                "blockage_detected": bl_det,
            })

        return results

    # ══════════════════════════════════════════════════
    #  SINGLE-ROW INFERENCE  — kept for API / testing
    # ══════════════════════════════════════════════════
    def predict_flood(self, row: dict) -> Dict[str, Any]:
        """Single-row flood risk prediction."""
        if self.flood_model is None:
            return {"probability": 0.0, "label": 0, "risk_level": "UNKNOWN"}

        X = self._prepare_row(row)
        if self.flood_scaler is not None:
            X = self.flood_scaler.transform(X)

        proba = self.flood_model.predict_proba(X)[0]
        flood_prob = float(proba[1]) * 100 if len(proba) > 1 else 0.0
        label = int(self.flood_model.predict(X)[0])

        return {
            "probability": round(flood_prob, 1),
            "label":       label,
            "risk_level":  _risk_level(flood_prob),
        }

    def predict_cause(self, row: dict) -> Dict[str, Any]:
        """Single-row cause diagnosis."""
        if self.cause_model is None:
            return {"cause": "UNKNOWN", "cause_id": -1, "probabilities": {}}

        X = self._prepare_row(row)
        if self.cause_scaler is not None:
            X = self.cause_scaler.transform(X)

        pred   = int(self.cause_model.predict(X)[0])
        probas = self.cause_model.predict_proba(X)[0]
        prob_dict = {
            CAUSE_CLASSES[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probas)
        }
        cause_name = CAUSE_CLASSES[pred] if pred < len(CAUSE_CLASSES) else "UNKNOWN"

        return {"cause": cause_name, "cause_id": pred, "probabilities": prob_dict}

    def localize_blockage(self, row: dict) -> Dict[str, Any]:
        """Single-row blockage localization."""
        s1  = float(row.get("s1_level", 0))
        s2  = float(row.get("s2_level", 0))
        s3  = float(row.get("s3_level", 0))
        d12 = s1 - s2
        d23 = s2 - s3
        blockage = float(row.get("blockage_pct", 0))
        flow_dev = float(row.get("flow_deviation", 0))

        MIN_FLOW_DEV   = 0.3
        MIN_BLOCKAGE   = 10.0
        MIN_LEVEL_DIFF = 2.0

        has_signal = (
            blockage  >= MIN_BLOCKAGE
            or flow_dev >= MIN_FLOW_DEV
            or abs(d12) >= MIN_LEVEL_DIFF
            or abs(d23) >= MIN_LEVEL_DIFF
        )

        if not has_signal:
            return {
                "segment":    "None Detected",
                "confidence": 0.0,
                "all_segments": {k: round(100.0 / 3, 1) for k in SEGMENT_CONNECTIONS},
                "detected":   False,
            }

        seg_s1_s2  = SEGMENT_CONNECTIONS[0]
        seg_s2_s3  = SEGMENT_CONNECTIONS[1]
        seg_s3_out = SEGMENT_CONNECTIONS[2]

        scores = {
            seg_s1_s2:  max(0.0, d12) * 2.0 + max(0.0, flow_dev) * 0.5,
            seg_s2_s3:  max(0.0, d23) * 2.0 + max(0.0, flow_dev) * 0.8,
            seg_s3_out: max(0.0, s3 - 30.0) * 0.5 + max(0.0, flow_dev) * 0.3,
        }
        if blockage > MIN_BLOCKAGE:
            boost = 1.0 + blockage / 100.0
            scores = {k: v * boost for k, v in scores.items()}

        total = sum(scores.values()) + 1e-9
        probs = {k: round(v / total * 100, 1) for k, v in scores.items()}
        best_seg = max(probs, key=probs.get)

        return {
            "segment":     best_seg,
            "confidence":  probs[best_seg],
            "all_segments": probs,
            "detected":    True,
        }

    # ────────── Feature Importance ──────────
    def get_feature_importance(self, model_name: str = "flood") -> Dict[str, float]:
        """Return feature importances from a trained RF model."""
        model = self.flood_model if model_name == "flood" else self.cause_model
        if model is None or not hasattr(model, "feature_importances_"):
            return {}
        return {
            FEATURE_COLUMNS[i]: round(float(model.feature_importances_[i]), 4)
            for i in range(len(FEATURE_COLUMNS))
        }

    # ────────── Expected Flow (Baseline) ──────────
    def predict_expected_flow(self, rainfall: float, water_level: float) -> float:
        """Use baseline regression model if available, else physics fallback."""
        if self.baseline_model is not None:
            return float(self.baseline_model.predict(np.array([[rainfall, water_level]]))[0])
        from utils.config import CATCHMENT_AREA_M2, CATCHMENT_FACTOR, BASE_DRAIN_CAPACITY_LS
        inflow = rainfall * CATCHMENT_AREA_M2 * CATCHMENT_FACTOR / 3600.0
        return min(inflow, BASE_DRAIN_CAPACITY_LS)
