"""
train.py — Train the surface roughness and hardness models
==========================================================
Trains one Random Forest per property from the four process parameters, reports
cross-validated accuracy, and saves the models + diagnostic plots.

Usage:  python train.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATA_DIR, MODEL_DIR, FEATURES, FEATURE_LABELS, TARGETS,
                    SEED, CV_FOLDS, add_derived_features, build_model)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def cv_for(n):
    """Leave-One-Out for small datasets, otherwise K-fold."""
    if n < 40:
        return LeaveOneOut(), "Leave-One-Out"
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED), f"{CV_FOLDS}-fold"


def train_one(key, spec):
    df = add_derived_features(pd.read_csv(DATA_DIR / spec["csv"]))
    X, y = df[FEATURES], df[spec["column"]].values
    label, unit, kind = spec["label"], spec["unit"], spec["model"]

    print("\n" + "=" * 55)
    print(f"  {label} ({unit})  —  {len(df)} samples  [{kind}]")
    print("=" * 55)

    # Cross-validated accuracy on pooled held-out predictions.
    cv, cvname = cv_for(len(df))
    y_pred = cross_val_predict(build_model(kind), X, y, cv=cv)
    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  {cvname} CV R²   = {r2:+.4f}")
    print(f"  {cvname} CV MAE  = {mae:.3f} {unit}")
    print(f"  {cvname} CV RMSE = {rmse:.3f} {unit}")

    # Fit on all data and save.
    model = build_model(kind)
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"model_{key}.pkl")
    print(f"  Saved → model_{key}.pkl")

    _plot_actual_vs_pred(y, y_pred, spec, r2, mae)
    _plot_importance(model, spec)
    return {"rows": len(df), "cv": cvname, "r2": r2, "mae": mae, "rmse": rmse}


def _plot_actual_vs_pred(y, y_pred, spec, r2, mae):
    lo = min(y.min(), y_pred.min()) * 0.96
    hi = max(y.max(), y_pred.max()) * 1.04
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, alpha=0.6, color=spec["color"], edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"Actual ({spec['unit']})"); ax.set_ylabel(f"Predicted ({spec['unit']})")
    ax.set_title(f"{spec['label']} — cross-validated\nR²={r2:.3f}  MAE={mae:.2f}")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(MODEL_DIR / f"actual_vs_predicted_{key_safe(spec)}.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


def _plot_importance(model, spec):
    imp = model.named_steps["model"].feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_LABELS[FEATURES[i]] for i in order], imp[order],
            color=spec["color"], alpha=0.8)
    ax.set_title(f"Feature Importance — {spec['label']}")
    ax.set_xlabel("Relative importance")
    for i, v in enumerate(imp[order]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"feature_importance_{key_safe(spec)}.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


def key_safe(spec):
    return spec["column"]


def main():
    results = {k: train_one(k, s) for k, s in TARGETS.items()}
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for k, r in results.items():
        print(f"  {TARGETS[k]['label']:18s}: R²={r['r2']:+.3f}  "
              f"MAE={r['mae']:.2f} {TARGETS[k]['unit']}  ({r['rows']} samples)")
    print("\n  Run the app:  python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
