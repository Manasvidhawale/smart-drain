"""
Smart Drain — Rainfall Model
Generates realistic time-varying rainfall profiles for each scenario.
"""

import numpy as np
from utils.config import RAINFALL_CATEGORIES, NOISE_RAINFALL_MMHR


def generate_rainfall_series(
    pattern: str = "Normal",
    n_steps: int = 120,
    custom_intensity: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a rainfall intensity time-series (mm/h) for *n_steps* minutes.

    Parameters
    ----------
    pattern : str
        One of "Normal", "Moderate", "Heavy", "Extreme", or "Custom".
    n_steps : int
        Number of simulation steps (minutes).
    custom_intensity : float or None
        Used when pattern == "Custom".
    rng : numpy Generator
        For reproducibility.

    Returns
    -------
    np.ndarray of shape (n_steps,) — rainfall in mm/h at each minute.
    """
    if rng is None:
        rng = np.random.default_rng()

    if pattern == "Custom" and custom_intensity is not None:
        base = custom_intensity
    else:
        lo, hi = RAINFALL_CATEGORIES.get(pattern, (5, 15))
        base = rng.uniform(lo, hi)

    # Build a smooth temporal envelope so rain isn't a flat constant
    t = np.linspace(0, 1, n_steps)

    # Ramp-up / peak / ramp-down shape (skewed Gaussian-ish)
    peak_pos = rng.uniform(0.25, 0.55)
    width = rng.uniform(0.15, 0.35)
    envelope = np.exp(-0.5 * ((t - peak_pos) / width) ** 2)

    # Scale so the *peak* equals `base`; tails are ~10-30% of base
    floor = base * rng.uniform(0.10, 0.30)
    rain = floor + (base - floor) * envelope

    # Add small Gaussian noise
    noise = rng.normal(0, NOISE_RAINFALL_MMHR, n_steps)
    rain = np.clip(rain + noise, 0.0, None)

    return rain


def get_rainfall_at_step(
    rain_series: np.ndarray, step: int
) -> float:
    """Safely return rainfall at a given step (clamped to series bounds)."""
    idx = min(max(step, 0), len(rain_series) - 1)
    return float(rain_series[idx])
