"""
Smart Drain — Streamlit Dashboard Application
==============================================
Predict Why a Road Will Flood

An IoT + ML based intelligent drainage monitoring simulation.
Run with:  streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Ensure project root is on path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulation.sensor_simulator import run_simulation
from simulation.drainage_model import DrainageModel
from simulation.rainfall_model import generate_rainfall_series
from simulation.blockage_model import generate_blockage_series, blockage_label
from utils.config import (
    BASE_DRAIN_CAPACITY_LS,
    DEFAULT_SIM_DURATION_STEPS,
    FEATURE_COLUMNS,
    CAUSE_CLASSES,
    FLOOD_RISK_LEVELS,
    FLOOD_WATER_LEVEL_THRESHOLD_CM,
    SEGMENT_LABELS,
    SEGMENT_CONNECTIONS,
    DATASET_PATH,
    FLOOD_MODEL_PATH,
    CAUSE_MODEL_PATH,
)
from utils.features import compute_features
from utils.metrics import compute_early_warning_lead_time
from ml.predict import SmartDrainPredictor
from ml.explain import (
    explain_prediction,
    generate_recommendation,
    cause_description,
    FEATURE_DISPLAY_NAMES,
)

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Drain — Predict Why a Road Will Flood",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    color: #00d2ff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 2px;
}
.main-header p {
    color: #a0d2db;
    font-size: 1.05rem;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
}

/* Status bar */
.status-bar {
    padding: 0.8rem 1.2rem;
    border-radius: 12px;
    text-align: center;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}
.status-normal { background: linear-gradient(90deg, #00b09b, #96c93d); color: #fff; }
.status-warning { background: linear-gradient(90deg, #f7971e, #ffd200); color: #1a1a2e; }
.status-critical { background: linear-gradient(90deg, #eb3349, #f45c43); color: #fff; }

/* Metric card */
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d2ff;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #7f8c8d;
    font-weight: 500;
    margin-top: 0.2rem;
}
.metric-card .unit {
    font-size: 0.75rem;
    color: #566573;
}

/* Flood risk display */
.flood-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.flood-low { background: linear-gradient(145deg, #0a3d1a, #155d27); border: 2px solid #27ae60; }
.flood-moderate { background: linear-gradient(145deg, #4a3f0a, #6b5b0a); border: 2px solid #f39c12; }
.flood-high { background: linear-gradient(145deg, #4a1a0a, #6b2d0a); border: 2px solid #e67e22; }
.flood-critical { background: linear-gradient(145deg, #4a0a0a, #8b0000); border: 2px solid #e74c3c; }
.flood-card .pct { font-size: 3.5rem; font-weight: 800; color: #fff; }
.flood-card .risk-label { font-size: 1.1rem; font-weight: 600; letter-spacing: 2px; }

/* Cause bar */
.cause-bar {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.cause-bar .bar-fill {
    height: 22px;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* Drain visualization */
.drain-viz {
    background: linear-gradient(180deg, #0a1628, #0f2027);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.5rem;
    font-family: 'Courier New', monospace;
    color: #00d2ff;
    line-height: 1.6;
    overflow-x: auto;
}

/* Info panel */
.info-panel {
    background: #0d1b2a;
    border-left: 4px solid #00d2ff;
    padding: 1rem 1.2rem;
    border-radius: 0 10px 10px 0;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #a0b4c8;
}

/* Scenario buttons */
.scenario-btn {
    width: 100%;
    padding: 0.8rem;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════
@st.cache_resource
def load_predictor():
    """Load ML models once (cached for the lifetime of the server process)."""
    return SmartDrainPredictor()


@st.cache_data(show_spinner=False)
def run_simulation_cached(
    scenario, n_steps, rainfall_intensity, blockage_pct,
    blockage_segment, blockage_onset_step, drainage_capacity,
    initial_water_level, seed=42,
):
    """Cache simulation results — identical inputs return instantly."""
    return run_simulation(
        scenario=scenario,
        n_steps=n_steps,
        rainfall_intensity=rainfall_intensity,
        blockage_pct=blockage_pct,
        blockage_segment=blockage_segment,
        blockage_onset_step=blockage_onset_step,
        drainage_capacity=drainage_capacity,
        initial_water_level=initial_water_level,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def compute_model_metrics_cached(dataset_path: str):
    """
    Load the dataset and compute flood + cause test-split data once.
    Cached by dataset_path — re-runs only if the file changes.
    """
    from utils.features import extract_feature_matrix
    from sklearn.model_selection import train_test_split
    from utils.config import TEST_SIZE, RANDOM_STATE

    perf_df = pd.read_csv(dataset_path)
    X_all = extract_feature_matrix(perf_df).fillna(0)

    y_flood = perf_df["flood_label"].values
    _, X_te, _, y_te_flood = train_test_split(
        X_all, y_flood, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_flood
    )

    y_cause = perf_df["cause_label"].values
    _, _, _, y_te_cause = train_test_split(
        X_all, y_cause, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cause
    )

    return {"X_te": X_te, "y_te_flood": y_te_flood, "y_te_cause": y_te_cause}


def ensure_models_exist():
    """Auto-train models if they don't exist."""
    # Resolve paths relative to the project root (the directory containing app.py)
    root = os.path.dirname(os.path.abspath(__file__))
    abs_flood = os.path.join(root, FLOOD_MODEL_PATH)
    abs_cause = os.path.join(root, CAUSE_MODEL_PATH)
    abs_dataset = os.path.join(root, DATASET_PATH)

    if not os.path.exists(abs_flood) or not os.path.exists(abs_cause):
        with st.spinner("🔧 First run — generating dataset & training models (this takes ~30s) …"):
            # Generate dataset
            if not os.path.exists(abs_dataset):
                from data.generate_dataset import generate_dataset
                generate_dataset(output_path=abs_dataset)
            # Train models
            from ml.train_flood_model import train_flood_model
            from ml.train_cause_model import train_cause_model
            train_flood_model(dataset_path=abs_dataset)
            train_cause_model(dataset_path=abs_dataset)
            # Train baseline model
            train_baseline_model(dataset_path=abs_dataset)
        st.cache_resource.clear()


def train_baseline_model(dataset_path: str = DATASET_PATH):
    """Train a simple regression baseline: rainfall + water_level → expected_flow."""
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    from utils.config import BASELINE_MODEL_PATH, RANDOM_STATE

    # Resolve to absolute path if relative
    root = os.path.dirname(os.path.abspath(__file__))
    abs_dataset = dataset_path if os.path.isabs(dataset_path) else os.path.join(root, dataset_path)
    abs_model = BASELINE_MODEL_PATH if os.path.isabs(BASELINE_MODEL_PATH) else os.path.join(root, BASELINE_MODEL_PATH)

    if not os.path.exists(abs_dataset):
        return
    df = pd.read_csv(abs_dataset)
    X = df[["rainfall", "water_level"]].fillna(0).values
    y = df["expected_flow"].fillna(0).values
    reg = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    reg.fit(X, y)
    os.makedirs(os.path.dirname(abs_model), exist_ok=True)
    joblib.dump(reg, abs_model)


def risk_color(level: str) -> str:
    """Map risk level to CSS class suffix."""
    return {"LOW": "low", "MODERATE": "moderate", "HIGH": "high", "CRITICAL": "critical"}.get(level, "low")


def status_class(level: str) -> str:
    if level in ("LOW",):
        return "status-normal"
    elif level in ("MODERATE", "HIGH"):
        return "status-warning"
    else:
        return "status-critical"


def status_text(level: str) -> str:
    if level == "LOW":
        return "🟢 SYSTEM NORMAL"
    elif level == "MODERATE":
        return "🟡 ELEVATED MONITORING"
    elif level == "HIGH":
        return "🟠 WARNING — HIGH FLOOD RISK"
    else:
        return "🔴 CRITICAL — FLOOD IMMINENT"


# ═══════════════════════════════════════════════
#  DRAIN VISUALIZATION  (HTML / Unicode art)
# ═══════════════════════════════════════════════
def render_drain_visualization(
    s1_level: float, s2_level: float, s3_level: float,
    blockage_pct: float, blockage_seg: int,
    rainfall: float, flow: float,
):
    """Generate an animated ASCII-style drain cross-section."""
    # Water fill chars based on level
    def water_bar(level, max_w=20):
        filled = int(level / 100 * max_w)
        return "█" * filled + "░" * (max_w - filled)

    def rain_drops(intensity):
        if intensity < 5:
            return "  .   .   .   .   .  "
        elif intensity < 30:
            return " ╎ ╎ ╎ ╎ ╎ ╎ ╎ ╎ ╎  "
        elif intensity < 60:
            return " ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃  "
        else:
            return " ┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃ "

    blk_marker = lambda seg: " ██ BLOCKAGE" if blockage_pct > 30 and blockage_seg == seg else ""
    flow_arrow = "→→→" if flow > 0.5 else "→" if flow > 0.1 else "·"

    viz = f"""
┌──────────────────── ROAD SURFACE ─────────────────────┐
│                    {rain_drops(rainfall)}                      │
│  Rain: {rainfall:5.1f} mm/h       ↓ ↓ ↓ ↓ ↓ ↓ ↓                │
└───────────────────────────────────────────────────────┘
        ↓                    ↓                    ↓
      [S1]                 [S2]                 [S3]
   {s1_level:5.1f} cm            {s2_level:5.1f} cm            {s3_level:5.1f} cm

╔═══════════════════════════════════════════════════════╗
║  S1  {water_bar(s1_level)}  {flow_arrow}  {water_bar(s2_level)}  S2{blk_marker(1)}
║       {flow_arrow}                              {flow_arrow}
║  S2  {water_bar(s2_level)}  {flow_arrow}  {water_bar(s3_level)}  S3{blk_marker(2)}
╚══════════════════════════════════ {flow_arrow} ══ OUTLET ═══╝
                              Flow: {flow:5.2f} L/s
"""
    return viz


# ═══════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════
CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.05)"


def _base_layout(title=""):
    return dict(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text=title, font=dict(size=14, color="#a0d2db")),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(gridcolor=GRID_COLOR, title="Time (min)"),
        yaxis=dict(gridcolor=GRID_COLOR),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def chart_rainfall(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_minutes"], y=df["rainfall"],
        mode="lines", fill="tozeroy",
        line=dict(color="#3498db", width=2),
        fillcolor="rgba(52,152,219,0.2)",
        name="Rainfall",
    ))
    fig.update_layout(**_base_layout("🌧️ Rainfall (mm/h)"))
    fig.update_yaxes(title="mm/h")
    return fig


def chart_water_level(df):
    fig = go.Figure()
    for seg in ["s1_level", "s2_level", "s3_level"]:
        name = seg.replace("_level", "").upper()
        fig.add_trace(go.Scatter(
            x=df["time_minutes"], y=df[seg],
            mode="lines", name=name,
            line=dict(width=2),
        ))
    fig.add_hline(
        y=FLOOD_WATER_LEVEL_THRESHOLD_CM, line_dash="dash",
        line_color="red", annotation_text="Flood Threshold",
    )
    fig.update_layout(**_base_layout("📏 Water Level (cm)"))
    fig.update_yaxes(title="cm")
    return fig


def chart_flow(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_minutes"], y=df["flow"],
        mode="lines", name="Actual Flow",
        line=dict(color="#2ecc71", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["time_minutes"], y=df["expected_flow"],
        mode="lines", name="Expected Flow",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))
    fig.update_layout(**_base_layout("💧 Flow Rate — Expected vs Actual (L/s)"))
    fig.update_yaxes(title="L/s")
    return fig


def chart_flood_risk(times, risks):
    fig = go.Figure()
    colors = []
    for r in risks:
        if r < 25:
            colors.append("#27ae60")
        elif r < 50:
            colors.append("#f39c12")
        elif r < 75:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")
    fig.add_trace(go.Bar(
        x=times, y=risks,
        marker_color=colors,
        name="Flood Risk %",
    ))
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Critical")
    fig.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="High")
    fig.update_layout(**_base_layout("⚠️ Flood Risk Over Time (%)"))
    fig.update_yaxes(title="%", range=[0, 105])
    return fig


def chart_cause_probabilities(prob_dict):
    """Horizontal bar chart for cause probabilities."""
    causes = list(prob_dict.keys())
    probs = list(prob_dict.values())
    colors = ["#27ae60", "#3498db", "#e74c3c", "#e67e22", "#9b59b6"]
    fig = go.Figure(go.Bar(
        y=causes, x=probs, orientation="h",
        marker_color=colors[:len(causes)],
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=20, t=10, b=10),
        height=220,
        xaxis=dict(range=[0, 105], gridcolor=GRID_COLOR, title="%"),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


def chart_confusion_matrix(cm, labels):
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=labels, y=labels,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=30, b=10),
        height=350,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def chart_feature_importance(importances: dict, top_n: int = 12):
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [FEATURE_DISPLAY_NAMES.get(f, f) for f, _ in sorted_feats]
    vals = [v for _, v in sorted_feats]
    fig = go.Figure(go.Bar(
        y=names[::-1], x=vals[::-1], orientation="h",
        marker=dict(
            color=vals[::-1],
            colorscale="Viridis",
        ),
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=20, t=10, b=10),
        height=380,
        xaxis=dict(gridcolor=GRID_COLOR, title="Importance"),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ═══════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════
def main():
    # Ensure models exist
    ensure_models_exist()
    predictor = load_predictor()

    if not predictor.is_ready:
        st.error("⚠️ ML models not found. Please run the training scripts first or click Reset to auto-train.")
        st.stop()

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🌊 SMART DRAIN</h1>
        <p>Predict Why a Road Will Flood — IoT + ML Drainage Monitoring Simulation</p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        st.markdown("### 📋 Scenario Selection")
        scenario_cols = st.columns(2)
        with scenario_cols[0]:
            btn_normal = st.button("🟢 Normal", use_container_width=True, key="btn_normal")
            btn_blockage = st.button("🔴 Blockage", use_container_width=True, key="btn_blockage")
        with scenario_cols[1]:
            btn_heavy = st.button("🟡 Heavy Rain", use_container_width=True, key="btn_heavy")
            btn_capacity = st.button("🟠 Capacity", use_container_width=True, key="btn_capacity")

        st.markdown("---")

        # Demo mode button
        demo_mode = st.toggle("🎬 Demo Mode", value=False, key="demo_toggle")

        st.markdown("---")
        st.markdown("### ⚙️ Simulation Parameters")

        # Initialize session state for scenario
        if "scenario" not in st.session_state:
            st.session_state.scenario = "Normal"
            st.session_state.rainfall_intensity = 10.0
            st.session_state.blockage_pct = 0.0
            st.session_state.drainage_capacity = 5.0
            st.session_state.initial_water_level = 2.0
            st.session_state.blockage_segment = 1
            st.session_state.sim_speed = 1.0

        # Handle scenario buttons
        if btn_normal:
            st.session_state.scenario = "Normal"
            st.session_state.rainfall_intensity = 10.0
            st.session_state.blockage_pct = 0.0
            st.session_state.drainage_capacity = 5.0
        elif btn_heavy:
            st.session_state.scenario = "Heavy"
            st.session_state.rainfall_intensity = 50.0
            st.session_state.blockage_pct = 0.0
            st.session_state.drainage_capacity = 5.0
        elif btn_blockage:
            st.session_state.scenario = "Blockage"
            st.session_state.rainfall_intensity = 25.0
            st.session_state.blockage_pct = 75.0
            st.session_state.drainage_capacity = 5.0
        elif btn_capacity:
            st.session_state.scenario = "Capacity"
            st.session_state.rainfall_intensity = 85.0
            st.session_state.blockage_pct = 0.0
            st.session_state.drainage_capacity = 2.5

        rainfall_intensity = st.slider(
            "🌧️ Rainfall Intensity (mm/h)", 0.0, 120.0,
            st.session_state.rainfall_intensity, 1.0, key="rain_slider"
        )
        blockage_pct = st.slider(
            "🚧 Blockage Level (%)", 0.0, 100.0,
            st.session_state.blockage_pct, 1.0, key="blk_slider"
        )
        drainage_capacity = st.slider(
            "🔧 Drainage Capacity (L/s)", 0.5, 10.0,
            st.session_state.drainage_capacity, 0.5, key="cap_slider"
        )
        initial_water_level = st.slider(
            "📏 Initial Water Level (cm)", 0.0, 30.0,
            st.session_state.initial_water_level, 0.5, key="init_wl_slider"
        )
        blockage_segment = st.selectbox(
            "📍 Blockage Location", [0, 1, 2],
            format_func=lambda x: SEGMENT_LABELS[x],
            index=st.session_state.blockage_segment,
            key="blk_seg_select"
        )
        sim_steps = st.slider(
            "⏱️ Simulation Duration (steps)", 30, 180, 90, 10, key="steps_slider"
        )

        st.markdown("---")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            run_sim = st.button("▶️ RUN", use_container_width=True, type="primary")
        with col_ctrl2:
            reset_btn = st.button("🔄 RESET", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-panel">
            <strong>SIMULATION MODE</strong><br><br>
            This application digitally simulates the sensor readings
            and hydraulic behavior of the proposed Smart Drain system.<br><br>
            <em>In a real deployment:</em><br>
            • Rainfall sensor → actual tipping-bucket sensor<br>
            • Water level → ultrasonic/pressure sensor<br>
            • Flow → electromagnetic flow meter<br>
            • Controller → ESP32 / IoT node<br>
            • Communication → Wi-Fi / LoRa / NB-IoT
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════
    #  HANDLE RESET
    # ═══════════════════════════════════════
    if reset_btn:
        for key in ["sim_data", "predictions", "demo_step"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.scenario = "Normal"
        st.session_state.rainfall_intensity = 10.0
        st.session_state.blockage_pct = 0.0
        st.session_state.drainage_capacity = 5.0
        st.session_state.initial_water_level = 2.0
        st.rerun()

    # ═══════════════════════════════════════
    #  RUN SIMULATION
    # ═══════════════════════════════════════
    if run_sim or demo_mode:
        scenario = st.session_state.scenario

        if demo_mode:
            # Demo mode: run through stages
            if "demo_step" not in st.session_state:
                st.session_state.demo_step = 0

            demo_configs = [
                {"scenario": "Normal", "rainfall": 10, "blockage": 0, "capacity": 5.0, "desc": "Step 1: Normal drainage — baseline behavior"},
                {"scenario": "Custom", "rainfall": 35, "blockage": 0, "capacity": 5.0, "desc": "Step 2: Increasing rainfall — water levels rising"},
                {"scenario": "Custom", "rainfall": 50, "blockage": 0, "capacity": 5.0, "desc": "Step 3: Heavy rainfall — elevated water levels"},
                {"scenario": "Custom", "rainfall": 30, "blockage": 40, "capacity": 5.0, "desc": "Step 4: Introducing blockage — flow begins to reduce"},
                {"scenario": "Custom", "rainfall": 30, "blockage": 75, "capacity": 5.0, "desc": "Step 5: Severe blockage — flow significantly reduced"},
                {"scenario": "Custom", "rainfall": 40, "blockage": 80, "capacity": 5.0, "desc": "Step 6: Flood risk increasing rapidly"},
                {"scenario": "Blockage", "rainfall": 35, "blockage": 85, "capacity": 5.0, "desc": "Step 7: Cause = BLOCKAGE — system diagnosing root cause"},
                {"scenario": "Blockage", "rainfall": 35, "blockage": 85, "capacity": 5.0, "desc": "Step 8: Blockage localized to S2→S3"},
                {"scenario": "Blockage", "rainfall": 35, "blockage": 85, "capacity": 5.0, "desc": "Step 9: Recommended action generated"},
            ]

            step = st.session_state.demo_step
            if step >= len(demo_configs):
                step = len(demo_configs) - 1

            cfg = demo_configs[step]
            st.info(f"🎬 **Demo Mode** — {cfg['desc']}")

            demo_cols = st.columns(3)
            with demo_cols[0]:
                if st.button("⬅️ Previous Step") and step > 0:
                    st.session_state.demo_step = step - 1
                    st.rerun()
            with demo_cols[1]:
                st.write(f"Step {step + 1} / {len(demo_configs)}")
            with demo_cols[2]:
                if st.button("Next Step ➡️") and step < len(demo_configs) - 1:
                    st.session_state.demo_step = step + 1
                    st.rerun()

            rainfall_intensity = cfg["rainfall"]
            blockage_pct = cfg["blockage"]
            drainage_capacity = cfg["capacity"]
            scenario = cfg["scenario"]

        # Run the simulation (cached — same inputs = instant result)
        df = run_simulation_cached(
            scenario=scenario,
            n_steps=sim_steps,
            rainfall_intensity=rainfall_intensity,
            blockage_pct=blockage_pct,
            blockage_segment=blockage_segment,
            blockage_onset_step=15 if blockage_pct > 20 else None,
            drainage_capacity=drainage_capacity,
            initial_water_level=initial_water_level,
            seed=42,
        )

        st.session_state.sim_data = df

        # ── Batch predictions: single sklearn call for the whole DataFrame ──
        predictions = predictor.predict_dataframe(df)
        st.session_state.predictions = predictions

    # ═══════════════════════════════════════
    #  DISPLAY DASHBOARD
    # ═══════════════════════════════════════
    if "sim_data" not in st.session_state:
        # Show welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h2 style="color: #00d2ff;">Welcome to Smart Drain</h2>
            <p style="color: #7f8c8d; font-size: 1.1rem;">
                Select a scenario from the sidebar and click <strong>RUN</strong> to begin the simulation.<br>
                Or enable <strong>Demo Mode</strong> for a guided walkthrough.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df = st.session_state.sim_data

    # Safety: predictions may be absent after a hot-reload (sim_data is
    # persisted in session state but predictions is not serialisable by
    # the cache layer). Regenerate transparently.
    if "predictions" not in st.session_state:
        st.session_state.predictions = predictor.predict_dataframe(df)

    preds = st.session_state.predictions

    # Use the LAST timestep for current values
    latest = df.iloc[-1].to_dict()
    latest_pred = preds[-1]

    # ══════════════════════════════
    #  TABS
    # ══════════════════════════════
    tab_dash, tab_perf, tab_data, tab_arch = st.tabs([
        "📊 Live Dashboard",
        "📈 Model Performance",
        "📋 Simulated Data",
        "🏗️ System Architecture",
    ])

    # ═══════════════════════════════════════
    #  TAB 1 — LIVE DASHBOARD
    # ═══════════════════════════════════════
    with tab_dash:
        # ── Section 1: System Status ──
        risk = latest_pred["risk_level"]
        st.markdown(f"""
        <div class="status-bar {status_class(risk)}">
            {status_text(risk)}
        </div>
        """, unsafe_allow_html=True)

        # ── Section 2: Live Sensor Values ──
        m_cols = st.columns(6)
        metrics_data = [
            ("🌧️", f"{latest['rainfall']:.1f}", "Rainfall", "mm/h"),
            ("📏", f"{latest['water_level']:.1f}", "Water Level", "cm"),
            ("💧", f"{latest['flow']:.2f}", "Flow Rate", "L/s"),
            ("📈", f"{latest['water_level_rise_rate']:.2f}", "Rise Rate", "cm/min"),
            ("🔧", f"{latest['drainage_capacity']:.1f}", "Drain Capacity", "L/s"),
            ("🚧", f"{latest['blockage_pct']:.0f}%", "Blockage", blockage_label(latest['blockage_pct'])),
        ]
        for i, (icon, val, label, unit) in enumerate(metrics_data):
            with m_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="value">{val}</div>
                    <div class="label">{label}</div>
                    <div class="unit">{unit}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Section 3 & 4: Flood Risk + Cause ──
        col_risk, col_cause = st.columns([1, 1])

        with col_risk:
            rc = risk_color(risk)
            st.markdown(f"""
            <div class="flood-card flood-{rc}">
                <div class="pct">{latest_pred['flood_prob']:.0f}%</div>
                <div class="risk-label" style="color:{'#e74c3c' if risk in ('HIGH','CRITICAL') else '#f39c12' if risk=='MODERATE' else '#27ae60'};">
                    {risk} FLOOD RISK
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Early warning lead time
            flood_probs = [p["flood_prob"] for p in preds]
            water_levels = df["water_level"].tolist()
            ewlt = compute_early_warning_lead_time(
                flood_probs, water_levels, FLOOD_WATER_LEVEL_THRESHOLD_CM
            )
            if ewlt is not None and ewlt > 0:
                st.success(f"⏱️ **Early Warning Lead Time: {ewlt:.0f} minutes** before flood threshold reached")
            elif ewlt == 0:
                st.warning("⏱️ Warning issued simultaneously with flood threshold")

        with col_cause:
            st.markdown(f"### 🔍 Probable Cause: **{latest_pred['cause']}**")
            st.caption(cause_description(latest_pred['cause']))
            st.plotly_chart(
                chart_cause_probabilities(latest_pred["cause_probs"]),
                use_container_width=True,
            )

        st.markdown("---")

        # ── Section 5: Drain Visualization ──
        col_viz, col_loc = st.columns([2, 1])

        with col_viz:
            st.markdown("### 🔩 Drainage System Visualization")
            viz_text = render_drain_visualization(
                latest["s1_level"], latest["s2_level"], latest["s3_level"],
                latest["blockage_pct"], blockage_segment,
                latest["rainfall"], latest["flow"],
            )
            st.markdown(f'<div class="drain-viz"><pre>{viz_text}</pre></div>', unsafe_allow_html=True)

        with col_loc:
            st.markdown("### 📍 Blockage Localization")
            bl_detected = latest_pred.get("blockage_detected", True)
            bl_seg = latest_pred["blockage_seg"]

            if bl_seg != "None Detected" and (
                latest_pred["cause"] in ("BLOCKAGE", "FLOW_RESTRICTION")
                or latest["blockage_pct"] > 20
            ):
                st.markdown(f"""
                **Suspected Segment:**
                ### 🎯 {bl_seg}
                **Confidence: {latest_pred['blockage_conf']:.0f}%**
                """)
                # Show all segment probabilities
                for seg, prob in latest_pred["blockage_all"].items():
                    color = "#e74c3c" if prob > 50 else "#f39c12" if prob > 30 else "#27ae60"
                    st.markdown(f"**{seg}**: {prob:.1f}%")
                    st.progress(min(prob / 100, 1.0))
            else:
                st.info("✅ No significant blockage detected in any segment.")

        st.markdown("---")

        # ── Section 6: Explanation ──
        col_explain, col_action = st.columns([1, 1])

        with col_explain:
            st.markdown("### 🧠 Why Is This Happening?")
            importances = predictor.get_feature_importance("flood")
            explanations = explain_prediction(importances, latest, top_n=7)
            for exp in explanations:
                icon = "🔴" if exp["contribution"] == "HIGH" else "🟡" if exp["contribution"] == "MODERATE" else "🟢"
                st.markdown(
                    f"{icon} **{exp['display_name']}** = {exp['value']}  "
                    f"→ {exp['contribution']} contribution"
                )

        with col_action:
            st.markdown("### 📋 Recommended Action")
            recommendation = generate_recommendation(
                latest_pred["cause"],
                latest_pred["flood_prob"],
                latest_pred["blockage_seg"],
                latest_pred["blockage_conf"],
            )
            st.markdown(recommendation)
            st.caption("_These are simulation-generated recommendations for demonstration purposes._")

        st.markdown("---")

        # ── Section 7: Live Graphs ──
        st.markdown("### 📈 Live Monitoring Graphs")

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.plotly_chart(chart_rainfall(df), use_container_width=True)
        with gcol2:
            st.plotly_chart(chart_water_level(df), use_container_width=True)

        gcol3, gcol4 = st.columns(2)
        with gcol3:
            st.plotly_chart(chart_flow(df), use_container_width=True)
        with gcol4:
            times = df["time_minutes"].tolist()
            risks = [p["flood_prob"] for p in preds]
            st.plotly_chart(chart_flood_risk(times, risks), use_container_width=True)

        # Expected vs Actual Flow (key diagnostic - shown here only, not duplicated)
        st.markdown("#### 🔑 Key Diagnostic: Flow Deviation Chart")
        fig_deviation = go.Figure()
        fig_deviation.add_trace(go.Scatter(
            x=df["time_minutes"],
            y=df["flow_deviation"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#e74c3c", width=2),
            fillcolor="rgba(231,76,60,0.15)",
            name="Flow Deviation (Expected − Actual)",
        ))
        fig_deviation.add_hline(
            y=0.3, line_dash="dash", line_color="orange",
            annotation_text="Restriction threshold",
        )
        fig_deviation.update_layout(**_base_layout("🔑 Flow Deviation (Expected − Actual L/s)"))
        fig_deviation.update_yaxes(title="L/s")
        st.plotly_chart(fig_deviation, use_container_width=True)

    # ═══════════════════════════════════════
    #  TAB 2 — MODEL PERFORMANCE
    # ═══════════════════════════════════════
    with tab_perf:
        st.markdown("## 📈 Model Performance & Evaluation")

        # Resolve dataset path to absolute (works regardless of CWD)
        root = os.path.dirname(os.path.abspath(__file__))
        abs_dataset = os.path.join(root, DATASET_PATH)

        if os.path.exists(abs_dataset):
            from utils.metrics import compute_classification_metrics

            # Cached split — only recomputes if dataset file path changes
            split_data = compute_model_metrics_cached(abs_dataset)
            X_te       = split_data["X_te"]
            y_te_flood = split_data["y_te_flood"]
            y_te_cause = split_data["y_te_cause"]

            pc1, pc2 = st.columns(2)

            with pc1:
                st.markdown("### 🌊 Flood-Risk Model")
                if predictor.flood_model is not None and predictor.flood_scaler is not None:
                    X_te_sc = predictor.flood_scaler.transform(X_te)
                    y_pred  = predictor.flood_model.predict(X_te_sc)
                    fm = compute_classification_metrics(
                        y_te_flood, y_pred, labels=[0, 1], target_names=["No Flood", "Flood"]
                    )
                    st.metric("Accuracy",  f"{fm['accuracy']:.4f}")
                    st.metric("Precision", f"{fm['precision']:.4f}")
                    st.metric("Recall",    f"{fm['recall']:.4f}")
                    st.metric("F1 Score",  f"{fm['f1']:.4f}")
                    st.plotly_chart(
                        chart_confusion_matrix(fm["confusion_matrix"], ["No Flood", "Flood"]),
                        use_container_width=True,
                    )
                    st.text(fm["classification_report"])

            with pc2:
                st.markdown("### 🔍 Cause-Classification Model")
                if predictor.cause_model is not None and predictor.cause_scaler is not None:
                    X_te_sc_c = predictor.cause_scaler.transform(X_te)
                    y_pred_c  = predictor.cause_model.predict(X_te_sc_c)
                    cm_metrics = compute_classification_metrics(
                        y_te_cause, y_pred_c,
                        labels=list(range(len(CAUSE_CLASSES))),
                        target_names=CAUSE_CLASSES,
                    )
                    st.metric("Accuracy",  f"{cm_metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{cm_metrics['precision']:.4f}")
                    st.metric("Recall",    f"{cm_metrics['recall']:.4f}")
                    st.metric("F1 Score",  f"{cm_metrics['f1']:.4f}")
                    st.plotly_chart(
                        chart_confusion_matrix(cm_metrics["confusion_matrix"], CAUSE_CLASSES),
                        use_container_width=True,
                    )
                    st.text(cm_metrics["classification_report"])

            st.markdown("---")
            st.markdown("### 🏆 Feature Importance")
            fi_col1, fi_col2 = st.columns(2)
            with fi_col1:
                st.markdown("**Flood Model**")
                flood_imp = predictor.get_feature_importance("flood")
                if flood_imp:
                    st.plotly_chart(chart_feature_importance(flood_imp), use_container_width=True)
            with fi_col2:
                st.markdown("**Cause Model**")
                cause_imp = predictor.get_feature_importance("cause")
                if cause_imp:
                    st.plotly_chart(chart_feature_importance(cause_imp), use_container_width=True)

            st.markdown("---")
            st.markdown("### ⏱️ Early Warning Analysis")

            # Only run the 4 heavy simulations when the user explicitly asks
            if "ew_results" not in st.session_state:
                if st.button("▶️ Compute Early Warning Lead Times", key="ew_btn"):
                    with st.spinner("Running 4 scenario simulations …"):
                        ew_results = []
                        for scn_name, scn_params in [
                            ("Normal",           {"scenario": "Normal",   "blockage_pct": 0,  "drainage_capacity": 5.0}),
                            ("Heavy Rainfall",   {"scenario": "Heavy",    "blockage_pct": 0,  "drainage_capacity": 5.0}),
                            ("Blockage",         {"scenario": "Blockage", "blockage_pct": 75, "drainage_capacity": 5.0}),
                            ("Capacity Exceeded",{"scenario": "Capacity", "blockage_pct": 0,  "drainage_capacity": 2.5}),
                        ]:
                            test_df = run_simulation_cached(
                                scenario=scn_params["scenario"],
                                n_steps=60,
                                rainfall_intensity=None,
                                blockage_pct=scn_params["blockage_pct"],
                                blockage_segment=1,
                                blockage_onset_step=None,
                                drainage_capacity=scn_params["drainage_capacity"],
                                initial_water_level=2.0,
                                seed=42,
                            )
                            # Batch predict for speed
                            test_batch = predictor.predict_dataframe(test_df)
                            test_probs = [p["flood_prob"] for p in test_batch]
                            lt = compute_early_warning_lead_time(
                                test_probs, test_df["water_level"].tolist(),
                                FLOOD_WATER_LEVEL_THRESHOLD_CM
                            )
                            ew_results.append({
                                "Scenario": scn_name,
                                "Lead Time (min)": f"{lt:.0f}" if lt is not None else "N/A"
                            })
                        st.session_state.ew_results = ew_results
                        st.rerun()
            else:
                st.table(pd.DataFrame(st.session_state.ew_results))
                if st.button("🔄 Re-compute", key="ew_recompute"):
                    del st.session_state["ew_results"]
                    st.rerun()
        else:
            st.warning("Dataset not found. Run the simulation first to generate data.")

    # ═══════════════════════════════════════
    #  TAB 3 — SIMULATED DATA
    # ═══════════════════════════════════════
    with tab_data:
        st.markdown("## 📋 Simulated Sensor Data")

        # Add prediction columns
        display_df = df.copy()
        display_df["flood_risk_%"] = [p["flood_prob"] for p in preds]
        display_df["risk_level"] = [p["risk_level"] for p in preds]
        display_df["probable_cause"] = [p["cause"] for p in preds]
        display_df["blockage_segment"] = [p["blockage_seg"] for p in preds]

        display_cols = [
            "time_minutes", "rainfall", "water_level", "flow",
            "expected_flow", "flow_deviation", "water_level_rise_rate",
            "s1_level", "s2_level", "s3_level",
            "blockage_pct", "flood_risk_%", "risk_level",
            "probable_cause", "blockage_segment",
        ]
        st.dataframe(
            display_df[display_cols].style.format({
                "rainfall": "{:.1f}",
                "water_level": "{:.1f}",
                "flow": "{:.2f}",
                "expected_flow": "{:.2f}",
                "flow_deviation": "{:.2f}",
                "water_level_rise_rate": "{:.2f}",
                "s1_level": "{:.1f}",
                "s2_level": "{:.1f}",
                "s3_level": "{:.1f}",
                "blockage_pct": "{:.0f}",
                "flood_risk_%": "{:.1f}",
            }),
            use_container_width=True,
            height=500,
        )

        # Download
        csv = display_df[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "smart_drain_simulation_data.csv",
            "text/csv",
            use_container_width=True,
        )

    # ═══════════════════════════════════════
    #  TAB 4 — SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════
    with tab_arch:
        st.markdown("## 🏗️ System Architecture")

        st.markdown("""
        <div class="info-panel">
            <strong>Note:</strong> Physical sensors are represented by software-simulated sensor
            streams in this demonstration. The architecture below shows the proposed real-world
            deployment structure alongside its simulation counterpart.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ```
                            ┌──────────────────────────────┐
                            │     SMART DRAIN SYSTEM       │
                            └──────────────┬───────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
            ┌───────▼───────┐    ┌─────────▼─────────┐   ┌───────▼───────┐
            │   Rainfall    │    │   Water Level      │   │   Flow Rate   │
            │   Sensor      │    │   Sensors (×3)     │   │   Sensor      │
            │  (simulated)  │    │   S1, S2, S3       │   │  (simulated)  │
            └───────┬───────┘    │   (simulated)      │   └───────┬───────┘
                    │            └─────────┬──────────┘           │
                    └──────────────────────┼──────────────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │    ESP32 / IoT      │
                                │    Controller       │
                                │    (simulated)      │
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │   Data Processing   │
                                │  Feature Engineering│
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Dynamic Hydraulic  │
                                │     Baseline        │
                                │  (Expected Flow)    │
                                └──────────┬──────────┘
                                           │
                            ┌──────────────┼──────────────┐
                            │              │              │
                    ┌───────▼───────┐ ┌────▼────┐ ┌──────▼──────┐
                    │  Flood Risk   │ │  Cause  │ │  Blockage   │
                    │  ML Model     │ │  ML     │ │  Locali-    │
                    │  (RF Binary)  │ │  Model  │ │  zation     │
                    └───────┬───────┘ │(RF Mult)│ └──────┬──────┘
                            │         └────┬────┘        │
                            └──────────────┼─────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Explainable AI     │
                                │  (Feature Importance│
                                │   + Recommendations)│
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │   Dashboard /       │
                                │   Alert System      │
                                │   (Streamlit)       │
                                └─────────────────────┘
        ```
        """)

        st.markdown("---")
        st.markdown("### 💧 Hydraulic Model")
        st.markdown("""
        The simulation uses a simplified physics-inspired water-balance model:

        ```
        Water(t+1) = Water(t) + Inflow(t) − Outflow(t)

        where:
            Inflow  = Rainfall × Catchment_Area × Catchment_Factor / 3600
            Outflow = min(Gravity_Flow, Effective_Capacity)
            Effective_Capacity = Base_Capacity × (1 − Blockage%)
            Gravity_Flow ∝ √(2 × g × head)
        ```

        Multi-segment drain (S1 → S2 → S3 → Outlet) allows spatial analysis
        of water accumulation patterns for blockage localization.
        """)

        st.markdown("---")
        st.markdown("### 🔬 Cause-Diagnosis Logic")
        st.markdown("""
        | Condition | Rainfall | Water Level | Flow | Blockage | Diagnosis |
        |-----------|----------|-------------|------|----------|-----------|
        | Normal | Low | Stable | Normal | None | ✅ NORMAL |
        | Storm | High | Rising | **High** | None | 🌧️ HEAVY RAINFALL |
        | Clogged | Moderate | **Rapidly Rising** | **Low** | High | 🚧 BLOCKAGE |
        | Overwhelmed | Extreme | Rising | High | None | ⚡ CAPACITY EXCEEDED |
        | Partial | Moderate | Rising | Reduced | Partial | 🔍 FLOW RESTRICTION |
        """)

        st.markdown("---")
        st.markdown("### ⚠️ Academic Limitations")
        st.info("""
        **Simplified Model:** The current implementation uses a simplified physics-inspired
        drainage model for simulation and algorithm validation. Real-world deployment would
        require calibration using actual drainage geometry, rainfall measurements, hydraulic
        characteristics, and field data.

        **Synthetic Data:** Synthetic data are used for the present demonstration because a
        physical prototype and long-term field dataset are not available.
        """)


if __name__ == "__main__":
    main()
