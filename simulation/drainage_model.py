"""
Smart Drain — Core Drainage Model
Simplified physics-inspired hydraulic simulation with multi-segment drain.

Water balance per step:
    Water(t+1) = Water(t) + Inflow(t) − Outflow(t)

Inflow  = rainfall × catchment_factor  (converted to L/s)
Outflow = min(effective_capacity, f(water_level))
    where effective_capacity = base_capacity × (1 − blockage%)

Multi-segment model:
    S1 (upstream) → S2 (mid) → S3 (downstream) → Outlet
    Each segment accumulates water independently.
    Flow between segments depends on head difference & blockage location.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from utils.config import (
    NUM_SEGMENTS,
    SEGMENT_LABELS,
    CATCHMENT_FACTOR,
    CATCHMENT_AREA_M2,
    BASE_DRAIN_CAPACITY_LS,
    MAX_WATER_LEVEL_CM,
    NOISE_WATER_LEVEL_CM,
    NOISE_FLOW_LS,
    DEFAULT_DT_SECONDS,
)
from simulation.blockage_model import apply_blockage


# ──────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────
@dataclass
class SegmentState:
    """Hydraulic state of one drain segment."""
    label: str = "S1"
    water_level_cm: float = 0.0          # current water depth (cm)
    flow_out_ls: float = 0.0             # outflow to next segment (L/s)
    volume_litres: float = 0.0           # accumulated water (litres)
    capacity_ls: float = BASE_DRAIN_CAPACITY_LS


@dataclass
class DrainState:
    """Complete system state at one timestep."""
    time_step: int = 0
    time_minutes: float = 0.0
    rainfall_mmhr: float = 0.0
    inflow_ls: float = 0.0
    total_outflow_ls: float = 0.0
    water_level_cm: float = 0.0          # representative (S1 or max)
    flow_ls: float = 0.0                 # representative outflow
    expected_flow_ls: float = 0.0
    blockage_pct: float = 0.0
    drainage_capacity_ls: float = BASE_DRAIN_CAPACITY_LS
    segments: List[SegmentState] = field(default_factory=list)


# ──────────────────────────────────────────────────
# Drainage Simulation Engine
# ──────────────────────────────────────────────────
class DrainageModel:
    """
    Simplified multi-segment drainage simulation.
    """

    def __init__(
        self,
        base_capacity_ls: float = BASE_DRAIN_CAPACITY_LS,
        catchment_area_m2: float = CATCHMENT_AREA_M2,
        catchment_factor: float = CATCHMENT_FACTOR,
        dt_seconds: float = DEFAULT_DT_SECONDS,
        n_segments: int = NUM_SEGMENTS,
        rng: np.random.Generator | None = None,
    ):
        self.base_capacity = base_capacity_ls
        self.catchment_area = catchment_area_m2
        self.catchment_factor = catchment_factor
        self.dt = dt_seconds
        self.n_segments = n_segments
        self.rng = rng or np.random.default_rng()

        # Segment cross-section areas (m²) — slight variation per segment
        self.seg_cross_sections = [0.25, 0.25, 0.25][:n_segments]

        # Volume-to-level conversion: level_cm = volume_L / (cross_section_m² × 10)
        # (10 converts m to cm for 1-litre = 0.001 m³)
        self.vol_to_cm = [
            1.0 / (cs * 10.0) for cs in self.seg_cross_sections
        ]

        self.reset()

    # ────────── Reset ──────────
    def reset(self, initial_level_cm: float = 0.0):
        """Reset all segments to the given water level."""
        self.segments: list[SegmentState] = []
        for i in range(self.n_segments):
            lbl = SEGMENT_LABELS[i] if i < len(SEGMENT_LABELS) else f"S{i+1}"
            vol = initial_level_cm / self.vol_to_cm[i] if self.vol_to_cm[i] else 0.0
            self.segments.append(
                SegmentState(
                    label=lbl,
                    water_level_cm=initial_level_cm,
                    volume_litres=vol,
                    capacity_ls=self.base_capacity,
                )
            )
        self.step_count = 0

    # ────────── Single-step update ──────────
    def step(
        self,
        rainfall_mmhr: float,
        blockage_pct: float,
        blockage_segment: int = 1,       # which segment has the blockage (0-indexed)
        capacity_override: float | None = None,
    ) -> DrainState:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        rainfall_mmhr : float   Current rainfall intensity (mm/h).
        blockage_pct : float    Blockage level 0–100.
        blockage_segment : int  Index of the blocked segment (default=1 → S2).
        capacity_override : float or None  If set, overrides base drain capacity.

        Returns
        -------
        DrainState — snapshot after this step.
        """
        dt = self.dt
        base_cap = capacity_override if capacity_override is not None else self.base_capacity

        # ── Inflow from rainfall ──
        # rainfall (mm/h) → L/s:  mm/h × m² / 3600 × 1L/0.001m³ × 0.001 = area*rain/3600
        inflow_ls = rainfall_mmhr * self.catchment_area * self.catchment_factor / 3600.0

        # ── Distribute inflow to upstream segment (S1) ──
        inflow_volume = inflow_ls * dt   # litres added this step

        # ── Per-segment update ──
        for i, seg in enumerate(self.segments):
            # --- Determine effective capacity for THIS segment ---
            if i == blockage_segment:
                eff_cap = apply_blockage(base_cap, blockage_pct)
            else:
                eff_cap = base_cap
            seg.capacity_ls = eff_cap

            # --- Add inflow ---
            if i == 0:
                seg.volume_litres += inflow_volume
            # else: receives flow from previous segment (handled below)

            # --- Compute outflow (limited by head & capacity) ---
            # Outflow ∝ sqrt(head) capped at effective capacity  (simplified Torricelli)
            head_m = max(seg.water_level_cm / 100.0, 0.0)
            gravity_flow = 0.6 * self.seg_cross_sections[i] * np.sqrt(2 * 9.81 * head_m) * 1000  # L/s
            outflow_ls = min(gravity_flow, eff_cap)
            outflow_volume = outflow_ls * dt

            # Cannot drain more than available
            outflow_volume = min(outflow_volume, max(seg.volume_litres, 0.0))
            outflow_ls = outflow_volume / dt if dt > 0 else 0.0

            seg.volume_litres -= outflow_volume
            seg.volume_litres = max(seg.volume_litres, 0.0)

            # Add sensor noise to flow
            noisy_flow = outflow_ls + self.rng.normal(0, NOISE_FLOW_LS)
            seg.flow_out_ls = max(noisy_flow, 0.0)

            # --- Pass outflow to next segment ---
            if i + 1 < self.n_segments:
                self.segments[i + 1].volume_litres += outflow_volume

            # --- Update water level ---
            seg.water_level_cm = seg.volume_litres * self.vol_to_cm[i]
            seg.water_level_cm += self.rng.normal(0, NOISE_WATER_LEVEL_CM)
            seg.water_level_cm = np.clip(seg.water_level_cm, 0.0, MAX_WATER_LEVEL_CM)

        # ── Representative system values ──
        # Water level = max across segments (conservative)
        sys_water_level = max(s.water_level_cm for s in self.segments)
        # Flow = outflow of last segment (what actually leaves the drain)
        sys_flow = self.segments[-1].flow_out_ls
        total_outflow = sum(s.flow_out_ls for s in self.segments) / self.n_segments

        # Expected flow (physics baseline, ignoring blockage)
        expected_outflow_ls = min(inflow_ls, base_cap)

        self.step_count += 1

        return DrainState(
            time_step=self.step_count,
            time_minutes=self.step_count * dt / 60.0,
            rainfall_mmhr=rainfall_mmhr + self.rng.normal(0, 0.3),
            inflow_ls=inflow_ls,
            total_outflow_ls=total_outflow,
            water_level_cm=sys_water_level,
            flow_ls=sys_flow,
            expected_flow_ls=expected_outflow_ls,
            blockage_pct=blockage_pct,
            drainage_capacity_ls=base_cap,
            segments=[
                SegmentState(
                    label=s.label,
                    water_level_cm=s.water_level_cm,
                    flow_out_ls=s.flow_out_ls,
                    volume_litres=s.volume_litres,
                    capacity_ls=s.capacity_ls,
                )
                for s in self.segments
            ],
        )
