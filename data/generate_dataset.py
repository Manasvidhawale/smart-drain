"""
Smart Drain - Synthetic Dataset Generator
Generates a large multi-scenario dataset for training the flood-risk
and cause-classification ML models.

Usage:
    python -m data.generate_dataset          # from project root
    python data/generate_dataset.py          # direct
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sensor_simulator import run_simulation
from utils.features import compute_features, assign_flood_label, assign_cause_label
from utils.config import DATASET_SIZE, DATASET_PATH


def generate_dataset(
    target_rows: int = DATASET_SIZE,
    output_path: str = DATASET_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset by running many short simulations
    with varied parameters.
    """
    rng = np.random.default_rng(seed)
    n_steps = 60  # each sim = 60 steps (1 hour)
    runs_per_scenario = max(target_rows // (n_steps * 5), 15)

    all_frames = []

    scenarios = [
        # (scenario_name, blockage_pct_range, capacity_range, seg, description)
        ("Normal",   (0, 5),    (4.0, 6.0),  1, "NORMAL"),
        ("Heavy",    (0, 10),   (4.0, 6.0),  1, "HEAVY_RAINFALL"),
        ("Blockage", (50, 95),  (4.0, 6.0),  1, "BLOCKAGE"),
        ("Capacity", (0, 5),    (1.0, 2.5),  1, "CAPACITY_EXCEEDED"),
        ("Custom",   (15, 45),  (3.0, 5.0),  1, "FLOW_RESTRICTION"),
    ]

    run_id = 0
    for scenario_name, blk_range, cap_range, seg, _desc in scenarios:
        for _ in range(runs_per_scenario):
            blk = rng.uniform(*blk_range)
            cap = rng.uniform(*cap_range)
            rain_int = None

            if scenario_name == "Custom":
                rain_int = rng.uniform(10, 40)

            onset = rng.integers(5, 30) if scenario_name == "Blockage" else None

            df = run_simulation(
                scenario=scenario_name,
                n_steps=n_steps,
                rainfall_intensity=rain_int,
                blockage_pct=blk,
                blockage_segment=seg,
                blockage_onset_step=onset,
                drainage_capacity=cap,
                initial_water_level=rng.uniform(0.5, 5.0),
                seed=int(rng.integers(0, 1_000_000)),
            )
            df["run_id"] = run_id
            all_frames.append(df)
            run_id += 1

    dataset = pd.concat(all_frames, ignore_index=True)

    # Compute features & labels
    dataset = compute_features(dataset)
    dataset["flood_label"] = assign_flood_label(dataset)
    dataset["cause_label"] = assign_cause_label(dataset)

    # Trim to target size (shuffle first)
    if len(dataset) > target_rows:
        dataset = dataset.sample(n=target_rows, random_state=seed).reset_index(drop=True)

    # Save
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent_dir, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"[OK] Dataset saved: {output_path}  ({len(dataset)} rows)")
    print(f"   Flood distribution:\n{dataset['flood_label'].value_counts().to_string()}")
    print(f"   Cause distribution:\n{dataset['cause_label'].value_counts().to_string()}")

    return dataset


if __name__ == "__main__":
    generate_dataset()
