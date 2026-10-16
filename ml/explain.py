"""
Smart Drain — Explainability Module
Generates human-readable explanations for model predictions.
"""

from typing import Dict, List, Tuple
from utils.config import FEATURE_COLUMNS, CAUSE_CLASSES


# Human-friendly feature names
FEATURE_DISPLAY_NAMES = {
    "rainfall": "Rainfall Intensity",
    "rainfall_change": "Rainfall Change",
    "water_level": "Water Level",
    "water_level_change": "Water Level Change",
    "water_level_rise_rate": "Water-Level Rise Rate",
    "flow": "Flow Rate",
    "flow_change": "Flow Change",
    "expected_flow": "Expected Flow",
    "flow_deviation": "Flow Deviation (Expected − Actual)",
    "drainage_capacity": "Drainage Capacity",
    "blockage_pct": "Blockage Indicator",
    "s1_level": "S1 Water Level (Upstream)",
    "s2_level": "S2 Water Level (Mid)",
    "s3_level": "S3 Water Level (Downstream)",
    "delta_s1_s2": "S1−S2 Level Difference",
    "delta_s2_s3": "S2−S3 Level Difference",
    "rainfall_avg_5": "Recent Rainfall Average",
    "water_level_avg_5": "Recent Water-Level Average",
    "flow_avg_5": "Recent Flow Average",
}


def _contribution_label(importance: float, max_importance: float) -> str:
    """Map relative importance to a textual contribution level."""
    ratio = importance / max_importance if max_importance > 0 else 0
    if ratio >= 0.6:
        return "HIGH"
    elif ratio >= 0.3:
        return "MODERATE"
    else:
        return "LOW"


def explain_prediction(
    feature_importances: Dict[str, float],
    current_values: Dict[str, float],
    top_n: int = 7,
) -> List[Dict[str, str]]:
    """
    Generate an explanation list for the current prediction.

    Returns a list of dicts with keys:
        feature, display_name, importance, contribution, value
    sorted by importance descending.
    """
    if not feature_importances:
        return []

    sorted_feats = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    max_imp = sorted_feats[0][1] if sorted_feats else 1.0

    explanations = []
    for feat, imp in sorted_feats:
        explanations.append({
            "feature": feat,
            "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "importance": f"{imp:.4f}",
            "contribution": _contribution_label(imp, max_imp),
            "value": f"{current_values.get(feat, 0):.2f}",
        })

    return explanations


def generate_recommendation(
    cause: str,
    flood_prob: float,
    blockage_segment: str,
    blockage_confidence: float,
) -> str:
    """Generate a human-readable recommended action string."""
    if flood_prob < 25:
        return "✅ No action required — system operating normally."

    actions = []

    if cause == "BLOCKAGE":
        actions.append(
            f"🔧 **Inspect drainage segment {blockage_segment}** immediately "
            f"(confidence: {blockage_confidence:.0f}%)."
        )
        actions.append("🚿 Deploy maintenance crew for debris removal.")
    elif cause == "HEAVY_RAINFALL":
        actions.append("🌧️ **Heavy rainfall detected** — monitor water levels closely.")
        actions.append("⚠️ Consider activating emergency overflow channels.")
    elif cause == "CAPACITY_EXCEEDED":
        actions.append(
            "⚡ **Drainage capacity exceeded** — rainfall volume surpasses "
            "maximum drain throughput."
        )
        actions.append("🚧 Activate flood barriers and divert traffic.")
        actions.append("📋 Long-term: review drainage infrastructure capacity.")
    elif cause == "FLOW_RESTRICTION":
        actions.append("🔍 **Flow restriction detected** — possible partial obstruction.")
        actions.append(
            f"🔧 Inspect segment {blockage_segment} for debris build-up."
        )
    else:
        actions.append("📊 Continue monitoring — no immediate risk.")

    if flood_prob >= 75:
        actions.insert(0, "🚨 **CRITICAL FLOOD RISK** — initiate emergency protocol.")

    return "\n\n".join(actions)


def cause_description(cause: str) -> str:
    """Return a one-line description for each cause class."""
    descriptions = {
        "NORMAL": "Drainage system operating within normal parameters.",
        "HEAVY_RAINFALL": "High rainfall intensity is the primary driver of elevated water levels. Drainage is functioning but overwhelmed by volume.",
        "BLOCKAGE": "Drainage outflow is significantly reduced compared to expected levels, indicating physical obstruction in the drain channel.",
        "CAPACITY_EXCEEDED": "Rainfall volume exceeds the maximum drainage throughput capacity, even with no blockage present.",
        "FLOW_RESTRICTION": "Moderate flow reduction detected — possible partial obstruction or hydraulic anomaly reducing drainage efficiency.",
    }
    return descriptions.get(cause, "Unknown condition.")
