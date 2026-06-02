"""
config.py — Configuration for the Surface Roughness & Hardness predictor.

Two independent regression models predict surface roughness and hardness from
four process parameters. The two datasets come from different experiments, so
each property has its own CSV and its own model.
"""

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ── General ──────────────────────────────────────────────────────────────────
SEED     = 42
CV_FOLDS = 5

# ── Input features ───────────────────────────────────────────────────────────
# Four raw process parameters the user enters.
BASE_FEATURES = [
    "Laser_Power_W",
    "Scan_Speed_mm_s",
    "Hatch_Distance_mm",
    "Layer_Thickness_mm",
]

# Two derived features that consolidate the parameter interactions and measurably
# improve accuracy (see add_derived_features):
#   Energy    = P / (v * h * t)   — energy delivered per unit volume
#   Power_Speed_Ratio = P / v     — energy delivered per unit track length
DERIVED_FEATURES = ["Energy", "Power_Speed_Ratio"]

# Full feature set the models actually train on.
FEATURES = BASE_FEATURES + DERIVED_FEATURES

FEATURE_LABELS = {
    "Laser_Power_W":      "Laser Power (W)",
    "Scan_Speed_mm_s":    "Scan Speed (mm/s)",
    "Hatch_Distance_mm":  "Hatch Distance (mm)",
    "Layer_Thickness_mm": "Layer Thickness (mm)",
    "Energy":             "Energy (P/v·h·t)",
    "Power_Speed_Ratio":  "Power/Speed (P/v)",
}

# Input ranges (min, max, default) for the app sliders.
PARAM_RANGES = {
    "Laser_Power_W":      (100.0, 950.0, 350.0),
    "Scan_Speed_mm_s":    (200.0, 3000.0, 1300.0),
    "Hatch_Distance_mm":  (0.03, 0.40, 0.13),
    "Layer_Thickness_mm": (0.02, 0.06, 0.03),
}

# ── Targets: one model per property ──────────────────────────────────────────
# Each target uses the algorithm that cross-validated best for it:
#   • roughness → tuned Random Forest
#   • hardness  → Extra Trees (clearly higher CV R² on this dataset)
TARGETS = {
    "roughness": {
        "csv":    "surface_roughness_data.csv",
        "column": "Surface_Roughness_um",
        "label":  "Surface Roughness",
        "unit":   "µm",
        "color":  "#1f77b4",
        "model":  "rf",
    },
    "hardness": {
        "csv":    "hardness_data.csv",
        "column": "Hardness_HV",
        "label":  "Hardness",
        "unit":   "HV",
        "color":  "#2ca02c",
        "model":  "extratrees",
    },
}

# ── Model hyperparameters (per algorithm) ────────────────────────────────────
N_ESTIMATORS = 500
RF_PARAMS = dict(
    n_estimators=N_ESTIMATORS, max_depth=None, min_samples_leaf=1,
    max_features=1.0, random_state=SEED, n_jobs=-1,
)
EXTRATREES_PARAMS = dict(
    n_estimators=400, min_samples_leaf=1, random_state=SEED, n_jobs=-1,
)


def add_derived_features(df):
    """Append the two engineered features. Operates on a copy; guards against a
    zero denominator (none occur in the data, but cheap insurance)."""
    df = df.copy()
    denom = (df["Scan_Speed_mm_s"] * df["Hatch_Distance_mm"]
             * df["Layer_Thickness_mm"]).replace(0, float("nan"))
    df["Energy"] = df["Laser_Power_W"] / denom
    df["Power_Speed_Ratio"] = df["Laser_Power_W"] / df["Scan_Speed_mm_s"].replace(0, float("nan"))
    return df


def build_model(kind="rf"):
    """Standardised regressor for the given target. `kind` selects the algorithm
    that validated best for that property."""
    if kind == "extratrees":
        estimator = ExtraTreesRegressor(**EXTRATREES_PARAMS)
    else:
        estimator = RandomForestRegressor(**RF_PARAMS)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
