"""
Smart Drain — Blockage Model
Controls simulated blockage level and its impact on drainage capacity.
"""

import numpy as np
from utils.config import NOISE_FLOW_LS


def apply_blockage(
    base_capacity_ls: float,
    blockage_pct: float,
) -> float:
    """
    Compute effective drainage capacity given a blockage percentage.

    effective = base_capacity × (1 − blockage_pct / 100)

    Parameters
    ----------
    base_capacity_ls : float
        Maximum drain outflow in L/s with no blockage.
    blockage_pct : float
        0–100 blockage percentage.

    Returns
    -------
    float — effective outflow capacity in L/s.
    """
    factor = max(0.0, 1.0 - blockage_pct / 100.0)
    return base_capacity_ls * factor


def blockage_label(pct: float) -> str:
    """Human-readable blockage severity."""
    if pct <= 0:
        return "None"
    elif pct <= 30:
        return "Minor"
    elif pct <= 60:
        return "Partial"
    elif pct <= 80:
        return "Major"
    else:
        return "Severe"


def generate_blockage_series(
    n_steps: int,
    blockage_pct: float = 0.0,
    onset_step: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a blockage time-series.

    If onset_step is given the blockage ramps from 0 → blockage_pct
    over ~10 steps starting at onset_step (simulating debris build-up).
    Otherwise it is constant throughout.
    """
    if rng is None:
        rng = np.random.default_rng()

    series = np.full(n_steps, blockage_pct, dtype=float)

    if onset_step is not None and 0 <= onset_step < n_steps:
        ramp_len = min(10, n_steps - onset_step)
        for i in range(onset_step):
            series[i] = 0.0
        for i in range(ramp_len):
            t = (i + 1) / ramp_len
            series[onset_step + i] = blockage_pct * t
        # after ramp → stays at blockage_pct (already filled)

    # tiny jitter
    noise = rng.normal(0, 0.5, n_steps)
    series = np.clip(series + noise, 0, 100)
    return series
