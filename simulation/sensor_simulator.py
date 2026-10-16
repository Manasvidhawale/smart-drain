"""
Smart Drain — Sensor Simulator
Wraps the drainage model and produces sensor-like readings with noise.
Provides the complete time-series runner used by both the dataset generator
and the live Streamlit dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from simulation.drainage_model import DrainageModel, DrainState
from simulation.rainfall_model import generate_rainfall_series
from simulation.blockage_model import generate_blockage_series
from utils.config import (
    DEFAULT_DT_SECONDS,
    DEFAULT_SIM_DURATION_STEPS,
    BASE_DRAIN_CAPACITY_LS,
    SEGMENT_LABELS,
)


def run_simulation(
    scenario: str = "Normal",
    n_steps: int = DEFAULT_SIM_DURATION_STEPS,
    rainfall_intensity: float | None = None,
    blockage_pct: float = 0.0,
    blockage_segment: int = 1,
    blockage_onset_step: int | None = None,
    drainage_capacity: float = BASE_DRAIN_CAPACITY_LS,
    initial_water_level: float = 2.0,
    dt_seconds: float = DEFAULT_DT_SECONDS,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Run a complete simulation and return a DataFrame of sensor readings.

    Parameters
    ----------
    scenario : str
        "Normal", "Heavy", "Blockage", "Capacity", or "Custom".
    n_steps : int
        Number of time steps.
    rainfall_intensity : float or None
        Override rainfall intensity (mm/h) for Custom mode.
    blockage_pct : float
        0–100 blockage level.
    blockage_segment : int
        0-based index of the blocked segment.
    blockage_onset_step : int or None
        Step at which blockage begins (ramps up over ~10 steps).
    drainage_capacity : float
        Maximum drain capacity in L/s.
    initial_water_level : float
        Starting water level in cm.
    dt_seconds : float
        Seconds per simulation step.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame with one row per timestep and columns for all sensor
    readings and derived features.
    """
    rng = np.random.default_rng(seed)

    # ── Map scenario to parameters ──
    rain_pattern = {
        "Normal": "Normal",
        "Heavy": "Heavy",
        "Blockage": "Moderate",
        "Capacity": "Extreme",
        "Custom": "Custom",
    }.get(scenario, "Normal")

    if scenario == "Blockage" and blockage_pct < 30:
        blockage_pct = 75.0  # ensure meaningful blockage for this scenario
    if scenario == "Capacity":
        blockage_pct = 0.0
        if drainage_capacity >= 5.0:
            drainage_capacity = 2.5   # reduce capacity so it's exceeded

    # ── Generate rainfall & blockage time-series ──
    rain_series = generate_rainfall_series(
        pattern=rain_pattern,
        n_steps=n_steps,
        custom_intensity=rainfall_intensity,
        rng=rng,
    )
    blk_series = generate_blockage_series(
        n_steps=n_steps,
        blockage_pct=blockage_pct,
        onset_step=blockage_onset_step,
        rng=rng,
    )

    # ── Initialize drainage model ──
    model = DrainageModel(
        base_capacity_ls=drainage_capacity,
        dt_seconds=dt_seconds,
        rng=rng,
    )
    model.reset(initial_water_level)

    # ── Run ──
    records: List[Dict[str, Any]] = []
    prev_state: DrainState | None = None

    for t in range(n_steps):
        rain_t = float(rain_series[t])
        blk_t = float(blk_series[t])

        state = model.step(
            rainfall_mmhr=rain_t,
            blockage_pct=blk_t,
            blockage_segment=blockage_segment,
            capacity_override=drainage_capacity,
        )

        # ── Derived features ──
        rain_change = rain_t - (prev_state.rainfall_mmhr if prev_state else rain_t)
        wl_change = (
            state.water_level_cm - prev_state.water_level_cm
            if prev_state
            else 0.0
        )
        rise_rate = wl_change / (dt_seconds / 60.0)   # cm per minute
        flow_change = (
            state.flow_ls - prev_state.flow_ls if prev_state else 0.0
        )
        flow_deviation = state.expected_flow_ls - state.flow_ls

        rec: Dict[str, Any] = {
            "time_step": t,
            "time_minutes": state.time_minutes,
            "rainfall": max(rain_t, 0.0),
            "rainfall_change": rain_change,
            "water_level": state.water_level_cm,
            "water_level_change": wl_change,
            "water_level_rise_rate": rise_rate,
            "flow": state.flow_ls,
            "flow_change": flow_change,
            "expected_flow": state.expected_flow_ls,
            "flow_deviation": flow_deviation,
            "drainage_capacity": drainage_capacity,
            "blockage_pct": blk_t,
            "inflow": state.inflow_ls,
        }

        # Segment levels
        for i, seg in enumerate(state.segments):
            rec[f"s{i+1}_level"] = seg.water_level_cm
            rec[f"s{i+1}_flow"] = seg.flow_out_ls

        # Spatial differences
        if len(state.segments) >= 3:
            rec["delta_s1_s2"] = (
                state.segments[0].water_level_cm - state.segments[1].water_level_cm
            )
            rec["delta_s2_s3"] = (
                state.segments[1].water_level_cm - state.segments[2].water_level_cm
            )
        else:
            rec["delta_s1_s2"] = 0.0
            rec["delta_s2_s3"] = 0.0

        records.append(rec)
        prev_state = state

    df = pd.DataFrame(records)

    # Rolling averages (window=5)
    for col in ["rainfall", "water_level", "flow"]:
        df[f"{col}_avg_5"] = df[col].rolling(window=5, min_periods=1).mean()

    return df
