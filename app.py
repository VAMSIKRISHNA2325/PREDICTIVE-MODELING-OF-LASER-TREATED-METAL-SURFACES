"""
app.py — Surface Roughness & Hardness Predictor
Run with:  python -m streamlit run app.py

Predicts surface roughness and hardness from four process parameters using two
independent Random Forest models.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import (DATA_DIR, MODEL_DIR, SEED, N_ESTIMATORS, CV_FOLDS,
                    FEATURES, BASE_FEATURES, FEATURE_LABELS, PARAM_RANGES,
                    TARGETS, add_derived_features, build_model)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_and_train():
    """Load each dataset, train or reload its model, compute CV metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {}
    for key, spec in TARGETS.items():
        df = add_derived_features(pd.read_csv(DATA_DIR / spec["csv"]))
        X, y = df[FEATURES], df[spec["column"]].values
        kind = spec["model"]

        path = MODEL_DIR / f"model_{key}.pkl"
        if path.exists():
            model = joblib.load(path)
        else:
            model = build_model(kind); model.fit(X, y); joblib.dump(model, path)

        cv = LeaveOneOut() if len(df) < 40 else KFold(CV_FOLDS, shuffle=True, random_state=SEED)
        cvname = "LOO" if len(df) < 40 else f"{CV_FOLDS}-fold"
        yp = cross_val_predict(build_model(kind), X, y, cv=cv)
        bundle[key] = {
            "model": model, "df": df, "rows": len(df), "cv": cvname,
            "y_true": y, "y_pred": yp,
            "r2": r2_score(y, yp), "mae": mean_absolute_error(y, yp),
            "rmse": np.sqrt(mean_squared_error(y, yp)),
        }
    return bundle


def predict_one(model, input_df):
    """Mean ±1σ across the ensemble's trees."""
    ens = model.named_steps["model"]
    Xt = model.named_steps["scaler"].transform(input_df[FEATURES])
    preds = np.array([t.predict(Xt) for t in ens.estimators_])
    return float(preds.mean()), float(preds.std())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Roughness & Hardness Predictor", layout="wide", page_icon="📈")
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: var(--background-color, #f9f9f9);
    border-radius: 8px; padding: 10px 16px; border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

with st.spinner("Loading models … (first run trains them)"):
    B = load_and_train()

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Process Parameters")
    p_min, p_max, p_def = PARAM_RANGES["Laser_Power_W"]
    s_min, s_max, s_def = PARAM_RANGES["Scan_Speed_mm_s"]
    h_min, h_max, h_def = PARAM_RANGES["Hatch_Distance_mm"]
    t_min, t_max, t_def = PARAM_RANGES["Layer_Thickness_mm"]

    power = st.number_input("Laser Power (W)",      p_min, p_max, p_def, step=10.0)
    speed = st.number_input("Scan Speed (mm/s)",    s_min, s_max, s_def, step=50.0)
    hatch = st.number_input("Hatch Distance (mm)",  h_min, h_max, h_def, step=0.01, format="%.3f")
    layer = st.number_input("Layer Thickness (mm)", t_min, t_max, t_def, step=0.005, format="%.3f")

    st.markdown("---")
    st.caption(f"🌲 {N_ESTIMATORS} trees · seed {SEED}")

input_df = add_derived_features(pd.DataFrame([{
    "Laser_Power_W": power, "Scan_Speed_mm_s": speed,
    "Hatch_Distance_mm": hatch, "Layer_Thickness_mm": layer,
}]))

# ── Header + predictions ─────────────────────────────────────────────────────
st.title("📈 Surface Roughness & Hardness Predictor")
st.markdown("Predicting **surface roughness** and **hardness** from process "
            "parameters · predictions shown with ±1σ")
st.markdown("---")

preds = {}
cols = st.columns(len(TARGETS))
for col, (key, spec) in zip(cols, TARGETS.items()):
    mean, std = predict_one(B[key]["model"], input_df)
    preds[key] = (mean, std)
    with col:
        st.metric(spec["label"], f"{mean:.2f} {spec['unit']}",
                  delta=f"± {std:.2f} {spec['unit']}", delta_color="off")
        st.caption(f"CV R² = {B[key]['r2']:.3f}  ·  {B[key]['rows']} samples")

# Out-of-range warning
warns = []
for f in BASE_FEATURES:
    lo, hi, _ = PARAM_RANGES[f]
    v = input_df[f].iloc[0]
    if v < lo or v > hi:
        warns.append(f"{FEATURE_LABELS[f]} = {v:g} outside [{lo:g}, {hi:g}]")
if warns:
    st.warning("⚠️ Input outside the training range — " + "; ".join(warns)
               + ". Predictions outside the data range are unreliable.")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Process Map", "📊 Model Performance", "📈 Feature Importance", "🔎 Data Explorer",
])
TARGET_LABELS = [s["label"] for s in TARGETS.values()]
KEY_BY_LABEL = {s["label"]: k for k, s in TARGETS.items()}

# Process map
with tab1:
    st.subheader("Predicted Property — Power × Scan Speed")
    st.caption("Sweep Power × Speed with hatch and layer fixed. ★ = current setting.")
    c0, c1 = st.columns([1, 3])
    with c0:
        map_label = st.selectbox("Property to map", TARGET_LABELS, index=0)
        fixed_h = st.slider("Hatch Distance (mm)", float(h_min), float(h_max),
                            float(hatch), step=0.01, key="map_h")
        fixed_t = st.slider("Layer Thickness (mm)", float(t_min), float(t_max),
                            float(layer), step=0.005, key="map_t")
    with c1:
        mkey = KEY_BY_LABEL[map_label]; spec = TARGETS[mkey]
        xx, yy = np.meshgrid(np.linspace(p_min, p_max, 60), np.linspace(s_min, s_max, 60))
        grid = add_derived_features(pd.DataFrame({
            "Laser_Power_W": xx.ravel(), "Scan_Speed_mm_s": yy.ravel(),
            "Hatch_Distance_mm": fixed_h, "Layer_Thickness_mm": fixed_t,
        }))
        Z = B[mkey]["model"].predict(grid[FEATURES]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(9, 5))
        cf = ax.contourf(xx, yy, Z, levels=25, cmap="viridis")
        fig.colorbar(cf, ax=ax).set_label(f"{spec['label']} ({spec['unit']})")
        ax.scatter(power, speed, color="red", s=400, marker="*",
                   edgecolors="white", linewidth=1.2, label="Current", zorder=5)
        ax.set_xlabel("Laser Power (W)"); ax.set_ylabel("Scan Speed (mm/s)")
        ax.set_title(f"{spec['label']} Map  (hatch={fixed_h:.3f} mm, layer={fixed_t:.3f} mm)")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# Performance
with tab2:
    st.subheader("Model Validation — Actual vs Predicted (cross-validated)")
    mcols = st.columns(len(TARGETS))
    for col, (key, spec) in zip(mcols, TARGETS.items()):
        b = B[key]
        col.metric(spec["label"], f"R² = {b['r2']:.3f}",
                   delta=f"MAE {b['mae']:.2f} {spec['unit']} · {b['rows']} samples · {b['cv']}",
                   delta_color="off")
    st.markdown("---")
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 5), squeeze=False)
    for ax, (key, spec) in zip(axes[0], TARGETS.items()):
        b = B[key]; y_t, y_p = b["y_true"], b["y_pred"]
        lo = min(y_t.min(), y_p.min()) * 0.96; hi = max(y_t.max(), y_p.max()) * 1.04
        ax.scatter(y_t, y_p, alpha=0.55, s=26, color=spec["color"], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"Actual ({spec['unit']})"); ax.set_ylabel(f"Predicted ({spec['unit']})")
        ax.set_title(f"{spec['label']}\nR²={b['r2']:.3f}  MAE={b['mae']:.2f}", fontsize=10)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.table(pd.DataFrame([
        {"Property": s["label"], "Samples": B[k]["rows"], "CV": B[k]["cv"],
         "R²": f"{B[k]['r2']:.3f}", "MAE": f"{B[k]['mae']:.2f} {s['unit']}",
         "RMSE": f"{B[k]['rmse']:.2f} {s['unit']}"}
        for k, s in TARGETS.items()]).set_index("Property"))

# Feature importance
with tab3:
    st.subheader("Parameter Importance per Property")
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 4), squeeze=False)
    for ax, (key, spec) in zip(axes[0], TARGETS.items()):
        imp = B[key]["model"].named_steps["model"].feature_importances_
        order = np.argsort(imp)
        ax.barh([FEATURE_LABELS[FEATURES[i]] for i in order], imp[order],
                color=spec["color"], alpha=0.8)
        ax.set_title(spec["label"], fontsize=11); ax.set_xlabel("Importance")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# Data explorer
with tab4:
    st.subheader("Dataset Explorer")
    which = st.selectbox("Dataset", TARGET_LABELS, index=0)
    key = KEY_BY_LABEL[which]; spec = TARGETS[key]; df = B[key]["df"]; col = spec["column"]
    d1, d2, d3 = st.columns(3)
    d1.metric("Samples", len(df))
    d2.metric(f"{spec['label']} range", f"{df[col].min():.1f}–{df[col].max():.1f} {spec['unit']}")
    d3.metric("Power range", f"{df['Laser_Power_W'].min():.0f}–{df['Laser_Power_W'].max():.0f} W")
    with st.expander("Full dataset"):
        st.dataframe(df, use_container_width=True)
