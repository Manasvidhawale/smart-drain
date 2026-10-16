"""
Smart Drain — Configuration Constants
All tunable parameters for the simulation, ML models, and UI.
"""

# ──────────────────────────────────────────────
# Drainage Geometry
# ──────────────────────────────────────────────
NUM_SEGMENTS = 3                       # S1 (upstream), S2 (mid), S3 (downstream)
SEGMENT_LABELS = ["S1", "S2", "S3"]
SEGMENT_CONNECTIONS = ["S1 → S2", "S2 → S3", "S3 → Outlet"]

# ──────────────────────────────────────────────
# Physical / Hydraulic Constants (simplified)
# ──────────────────────────────────────────────
CATCHMENT_AREA_M2 = 500.0              # m² of road surface draining into the system
CATCHMENT_FACTOR = 0.85                # fraction of rainfall that becomes inflow
BASE_DRAIN_CAPACITY_LS = 5.0           # litres/second at full capacity
DRAIN_CROSS_SECTION_M2 = 0.25         # cross-section area for depth calc
MAX_WATER_LEVEL_CM = 100.0            # drain depth in cm
GRAVITY = 9.81                         # m/s²

# Scaling factors (simplified physics → simulation units)
INFLOW_SCALE = CATCHMENT_AREA_M2 * CATCHMENT_FACTOR / 3600.0   # mm/h → L/s approx
LEVEL_SCALE = 1.0 / (DRAIN_CROSS_SECTION_M2 * 10.0)           # ΔVolume → Δcm

# ──────────────────────────────────────────────
# Simulation Timing
# ──────────────────────────────────────────────
DEFAULT_DT_SECONDS = 60               # 1 minute per simulation step
DEFAULT_SIM_DURATION_STEPS = 120      # 2 hours total
SIM_SPEED_OPTIONS = [0.5, 1.0, 2.0, 5.0, 10.0]

# ──────────────────────────────────────────────
# Rainfall Categories (mm/h)
# ──────────────────────────────────────────────
RAINFALL_CATEGORIES = {
    "Normal":   (5, 15),
    "Moderate": (15, 30),
    "Heavy":    (30, 60),
    "Extreme":  (60, 100),
}

# ──────────────────────────────────────────────
# Blockage Interpretation
# ──────────────────────────────────────────────
BLOCKAGE_LABELS = {
    (0, 0):      "None",
    (1, 30):     "Minor",
    (31, 60):    "Partial",
    (61, 80):    "Major",
    (81, 100):   "Severe",
}

# ──────────────────────────────────────────────
# Sensor Noise (Gaussian σ)
# ──────────────────────────────────────────────
NOISE_RAINFALL_MMHR = 0.3
NOISE_WATER_LEVEL_CM = 0.2
NOISE_FLOW_LS = 0.08

# ──────────────────────────────────────────────
# Flood Risk Thresholds
# ──────────────────────────────────────────────
FLOOD_RISK_LEVELS = {
    "LOW":      (0, 25),
    "MODERATE": (25, 50),
    "HIGH":     (50, 75),
    "CRITICAL": (75, 100),
}

FLOOD_WATER_LEVEL_THRESHOLD_CM = 70.0  # level at which actual flooding begins

# ──────────────────────────────────────────────
# Cause Classes
# ──────────────────────────────────────────────
CAUSE_CLASSES = [
    "NORMAL",
    "HEAVY_RAINFALL",
    "BLOCKAGE",
    "CAPACITY_EXCEEDED",
    "FLOW_RESTRICTION",
]
CAUSE_CLASS_MAP = {label: idx for idx, label in enumerate(CAUSE_CLASSES)}

# ──────────────────────────────────────────────
# ML Training
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS_FLOOD = 200
N_ESTIMATORS_CAUSE = 200
N_ESTIMATORS_BASELINE = 100

# ──────────────────────────────────────────────
# Dataset Generation
# ──────────────────────────────────────────────
DATASET_SIZE = 20000
DATASET_PATH = "data/simulated_drainage_data.csv"

# ──────────────────────────────────────────────
# Saved Model Paths
# ──────────────────────────────────────────────
FLOOD_MODEL_PATH = "models/flood_model.pkl"
CAUSE_MODEL_PATH = "models/cause_model.pkl"
BASELINE_MODEL_PATH = "models/baseline_model.pkl"
FLOOD_SCALER_PATH = "models/flood_scaler.pkl"
CAUSE_SCALER_PATH = "models/cause_scaler.pkl"

# ──────────────────────────────────────────────
# Feature Names (used consistently across modules)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "rainfall",
    "rainfall_change",
    "water_level",
    "water_level_change",
    "water_level_rise_rate",
    "flow",
    "flow_change",
    "expected_flow",
    "flow_deviation",
    "drainage_capacity",
    "blockage_pct",
    "s1_level",
    "s2_level",
    "s3_level",
    "delta_s1_s2",
    "delta_s2_s3",
    "rainfall_avg_5",
    "water_level_avg_5",
    "flow_avg_5",
]
