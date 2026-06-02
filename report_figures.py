"""
report_figures.py — Generate all figures used in the project report
====================================================================
Produces, into results/:
  • workflow.png            — end-to-end ML pipeline
  • random_forest.png       — how an ensemble of trees is averaged
  • before_after.png        — accuracy comparison (basic vs improved)
  • relational_<col>.png    — each property vs the four process parameters
  • workflow_<col>.png      — detailed per-model workflow (one per property)

Run:  python report_figures.py   (also called by generate_report.py)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import _tree

from config import (DATA_DIR, BASE_DIR, MODEL_DIR, BASE_FEATURES, FEATURES, FEATURE_LABELS,
                    TARGETS, SEED, add_derived_features, build_model)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

OUT = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)


# ── helpers for schematic boxes ──────────────────────────────────────────────
def _box(ax, x, y, w, h, text, fc, ec="#333", fs=9, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color=color))


# ── 1. WORKFLOW — clean technical infographic ────────────────────────────────
def workflow():
    """Clean, minimal technical workflow — dark header, accent bars, real numbers."""
    BG      = "#f8f9fa"
    DARK    = "#1e293b"
    MUTED   = "#64748b"
    ACCENT_COLORS = ["#3b82f6","#8b5cf6","#ec4899","#6366f1","#22c55e","#f59e0b"]

    stages = [
        ("01",  "Collect Data",
         ["89 roughness samples (µm)", "30 hardness samples (HV)",
          "Source: published literature", "4 raw process parameters"],
         "#3b82f6"),
        ("02",  "Feature Engineering",
         ["4 raw inputs → 6 model inputs",
          "Energy = P / (v · h · t)",
          "Power/Speed = P / v",
          "Captures parameter interactions"],
         "#8b5cf6"),
        ("03",  "Pre-process",
         ["StandardScaler applied",
          "Mean = 0,  Std = 1",
          "Prevents large-scale features",
          "from dominating splits"],
         "#ec4899"),
        ("04",  "Train Models",
         ["Roughness → Random Forest",
          "  500 trees, max_features=1.0",
          "Hardness  → Extra Trees",
          "  400 trees, random splits"],
         "#6366f1"),
        ("05",  "Cross-Validate",
         ["Roughness: 5-fold CV",
          "Hardness:  Leave-One-Out",
          "R² pooled over all folds",
          "Reports MAE and RMSE"],
         "#22c55e"),
        ("06",  "Deploy",
         ["Streamlit interactive app",
          "Slider inputs → live prediction",
          "Mean ± 1σ uncertainty band",
          "Out-of-range warning flag"],
         "#f59e0b"),
    ]

    fig = plt.figure(figsize=(18, 5.8), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, 5.8); ax.axis("off")
    ax.set_facecolor(BG)

    # header banner
    ax.add_patch(patches.Rectangle((0, 4.85), 18, 0.95, fc=DARK, zorder=1))
    ax.text(9, 5.33, "Machine-Learning Pipeline  —  Surface Roughness & Hardness Prediction",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="white", zorder=2)

    card_w, card_h, gap = 2.68, 3.95, 0.24
    x0 = 0.20

    for i, (num, title, details, col) in enumerate(stages):
        x = x0 + i * (card_w + gap)
        y = 0.55

        # card shadow (slightly offset grey rect)
        ax.add_patch(patches.FancyBboxPatch((x + 0.07, y - 0.07), card_w, card_h,
            boxstyle="round,pad=0.0,rounding_size=0.15",
            fc="#d1d5db", ec="none", zorder=2))
        # card body
        ax.add_patch(patches.FancyBboxPatch((x, y), card_w, card_h,
            boxstyle="round,pad=0.0,rounding_size=0.15",
            fc="white", ec="#e2e8f0", lw=1.2, zorder=3))
        # colored top accent bar
        ax.add_patch(patches.FancyBboxPatch((x, y + card_h - 0.72), card_w, 0.72,
            boxstyle="round,pad=0.0,rounding_size=0.15",
            fc=col, ec="none", zorder=4))
        # cover the bottom-round part of accent bar so it looks like flat bottom
        ax.add_patch(patches.Rectangle((x, y + card_h - 0.40), card_w, 0.40,
            fc=col, ec="none", zorder=4))

        # step number bubble on accent bar
        ax.add_patch(plt.Circle((x + 0.42, y + card_h - 0.36), 0.22,
                     fc="white", ec="none", zorder=5))
        ax.text(x + 0.42, y + card_h - 0.36, num, ha="center", va="center",
                fontsize=9, fontweight="bold", color=col, zorder=6)

        # title on accent bar
        ax.text(x + card_w / 2 + 0.12, y + card_h - 0.36, title,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color="white", zorder=6)

        # divider line
        ax.plot([x + 0.25, x + card_w - 0.25], [y + card_h - 0.90, y + card_h - 0.90],
                color="#e2e8f0", lw=1.0, zorder=4)

        # detail lines
        for j, line in enumerate(details):
            ax.text(x + 0.28, y + card_h - 1.25 - j * 0.62, f"• {line}",
                    ha="left", va="center", fontsize=8.4, color="#334155", zorder=5)

        # connector arrow between cards
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + card_w + gap, y + card_h / 2 + 0.26),
                        xytext=(x + card_w, y + card_h / 2 + 0.26),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                        color="#94a3b8", mutation_scale=18), zorder=6)

    # bottom caption
    ax.text(9, 0.24,
            "Both models share steps 1–3 and 5–6. Step 4 trains separate algorithms "
            "per property (Random Forest for roughness, Extra Trees for hardness).",
            ha="center", va="center", fontsize=8.8, color=MUTED, style="italic")

    plt.savefig(OUT / "workflow.png", dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved: workflow.png")


# ── 2. RANDOM FOREST — real tree structure from trained model ─────────────────
SHORT = {"Laser_Power_W": "Power (W)", "Scan_Speed_mm_s": "Speed (mm/s)",
         "Hatch_Distance_mm": "Hatch (mm)", "Layer_Thickness_mm": "Layer (mm)",
         "Energy": "Energy", "Power_Speed_Ratio": "Power/Speed"}


def _extract_nodes(estimator, feature_names, max_depth=3):
    """Walk a fitted sklearn tree and return a list of node dicts for drawing."""
    t = estimator.tree_
    feat = [SHORT.get(feature_names[i], feature_names[i])
            if i != _tree.TREE_UNDEFINED else "leaf"
            for i in t.feature]

    nodes = []
    def walk(node_id, depth, x, x_lo, x_hi):
        if depth > max_depth:
            return
        is_leaf = t.feature[node_id] == _tree.TREE_UNDEFINED
        val     = float(t.value[node_id][0][0])
        nodes.append(dict(id=node_id, depth=depth, x=x, is_leaf=is_leaf,
                          feat=feat[node_id], thresh=float(t.threshold[node_id]),
                          val=val, n_samples=int(t.n_node_samples[node_id])))
        if not is_leaf:
            mid_l = (x_lo + x) / 2
            mid_r = (x + x_hi) / 2
            walk(t.children_left[node_id],  depth + 1, mid_l, x_lo, x)
            walk(t.children_right[node_id], depth + 1, mid_r, x, x_hi)
    walk(0, 0, 0.5, 0.0, 1.0)
    return nodes


def _draw_real_tree(ax, nodes, x_off, y_top, width, v_gap,
                    node_fc, leaf_fc, edge_color, lw_edge=1.4):
    """Draw a real tree (extracted from estimator) in the given axes region."""
    max_d = max(n["depth"] for n in nodes) + 1

    def data2ax(nx, depth):
        return x_off + nx * width, y_top - depth * v_gap

    id2node = {n["id"]: n for n in nodes}

    # edges first
    for n in nodes:
        if n["is_leaf"]:
            continue
        t_obj = None  # we only have rendered nodes, not the raw tree
        cx, cy = data2ax(n["x"], n["depth"])
        # find children by id (left/right child ids are unknown here; match by depth+x)
        children = [c for c in nodes if c["depth"] == n["depth"] + 1
                    and abs(c["x"] - n["x"]) < 0.45 / (2 ** n["depth"] + 0.001)]
        for ch in children:
            chx, chy = data2ax(ch["x"], ch["depth"])
            ax.plot([cx, chx], [cy, chy], color=edge_color, lw=lw_edge,
                    zorder=2, solid_capstyle="round")
            # yes/no label on edge
            is_left = ch["x"] < n["x"]
            mid_x   = (cx + chx) / 2 + (-0.018 * width if is_left else 0.018 * width)
            mid_y   = (cy + chy) / 2
            ax.text(mid_x, mid_y, "Y" if is_left else "N", ha="center",
                    va="center", fontsize=6.0, color=edge_color, fontweight="bold")

    # nodes
    r_int  = 0.028 * width
    r_leaf = 0.024 * width
    for n in nodes:
        cx, cy = data2ax(n["x"], n["depth"])
        if n["is_leaf"]:
            ax.add_patch(plt.Circle((cx, cy), r_leaf, fc=leaf_fc, ec=edge_color,
                         lw=1.0, zorder=4))
            ax.text(cx, cy, f"{n['val']:.1f}", ha="center", va="center",
                    fontsize=5.8, color="white", fontweight="bold", zorder=5)
        else:
            ax.add_patch(plt.Rectangle((cx - r_int, cy - r_int * 0.7),
                         2 * r_int, 1.4 * r_int, fc=node_fc, ec=edge_color,
                         lw=1.2, zorder=4))
            thresh_str = f"{n['thresh']:.3f}" if n['thresh'] < 1 else f"{n['thresh']:.1f}"
            ax.text(cx, cy + r_int * 0.28, n["feat"], ha="center", va="center",
                    fontsize=5.5, color="white", fontweight="bold", zorder=5)
            ax.text(cx, cy - r_int * 0.25, f"≤ {thresh_str}", ha="center", va="center",
                    fontsize=5.0, color="#dde", zorder=5)


def random_forest():
    """Two real trees (one from each trained model) side-by-side with explanation."""
    import joblib

    BG    = "#f8f9fa"
    DARK  = "#1e293b"
    MUTED = "#64748b"

    # load trained models
    try:
        rf_model = joblib.load(MODEL_DIR / "model_roughness.pkl")
        et_model = joblib.load(MODEL_DIR / "model_hardness.pkl")
        rf_est   = rf_model.named_steps["model"].estimators_[7]
        et_est   = et_model.named_steps["model"].estimators_[7]
        rf_nodes = _extract_nodes(rf_est, FEATURES, max_depth=3)
        et_nodes = _extract_nodes(et_est, FEATURES, max_depth=3)
    except Exception:
        rf_nodes = et_nodes = []

    fig = plt.figure(figsize=(16, 9.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── header ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 0.895, 1, 0.105])
    hax.set_xlim(0, 1); hax.set_ylim(0, 1); hax.axis("off")
    hax.set_facecolor(DARK)
    hax.text(0.5, 0.55, "How Tree-Ensemble Models Predict: Decision Trees → Average → Result",
             ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    hax.text(0.5, 0.18, "Left tree: actual Random Forest tree (roughness model)   ·   "
             "Right tree: actual Extra Trees tree (hardness model)   ·   "
             "Both shown to depth 3",
             ha="center", va="center", fontsize=9, color="#94a3b8")

    # ── concept strip (how it works) ──────────────────────────────────────────
    cax = fig.add_axes([0.02, 0.79, 0.96, 0.095])
    cax.set_xlim(0, 10); cax.set_ylim(0, 1); cax.axis("off")
    cax.set_facecolor(BG)
    concepts = [
        ("#3b82f6", "BOOTSTRAP SAMPLING",
         "Each tree trains on a\nrandom subset of rows"),
        ("#8b5cf6", "RANDOM FEATURE SUBSET",
         "At every split, only a\nrandom subset of features\nis considered"),
        ("#22c55e", "INDEPENDENT SPLITS",
         "Each tree grows\nindependently; splits\ndiffer across trees"),
        ("#f59e0b", "AGGREGATION",
         "Final answer = mean\nof all tree predictions\n(N = 500 trees)"),
        ("#ec4899", "UNCERTAINTY ±1σ",
         "Spread of predictions\nacross trees gives a\ncalibrated error band"),
    ]
    step = 2.0
    for i, (col, title, detail) in enumerate(concepts):
        px = 0.1 + i * step
        cax.add_patch(patches.FancyBboxPatch((px, 0.06), 1.78, 0.88,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            fc="white", ec=col, lw=1.5))
        cax.add_patch(patches.Rectangle((px, 0.70), 1.78, 0.24, fc=col, ec="none"))
        cax.add_patch(patches.FancyBboxPatch((px, 0.70), 1.78, 0.24,
            boxstyle="round,pad=0.0,rounding_size=0.12", fc=col, ec="none"))
        cax.text(px + 0.89, 0.82, title, ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color="white")
        for j, ln in enumerate(detail.split("\n")):
            cax.text(px + 0.89, 0.53 - j * 0.17, ln, ha="center", va="center",
                     fontsize=7.0, color="#334155")
        if i < len(concepts) - 1:
            cax.annotate("", xy=(px + 1.78 + 0.11, 0.5), xytext=(px + 1.78, 0.5),
                         arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#94a3b8",
                                         mutation_scale=12))

    # ── tree panels ───────────────────────────────────────────────────────────
    panels = [
        (0.01, 0.08, 0.48, "#3b82f6", "#1d4ed8", "#bfdbfe", "#93c5fd",
         "Surface Roughness Model — Random Forest", "Tree #8 of 500",
         rf_nodes, "SR (µm)"),
        (0.51, 0.08, 0.48, "#22c55e", "#15803d", "#bbf7d0", "#86efac",
         "Hardness Model — Extra Trees",           "Tree #8 of 400",
         et_nodes, "HV"),
    ]

    for (lx, ly, pw, ec, ec2, leaf_fc, node_fc,
         panel_title, tree_label, nodes, unit) in panels:
        pax = fig.add_axes([lx, ly, pw, 0.70])
        pax.set_facecolor("white")
        pax.set_xlim(0, 1); pax.set_ylim(-0.1, 1.05)
        for spine in pax.spines.values():
            spine.set_edgecolor("#e2e8f0"); spine.set_linewidth(1.0)
        pax.set_xticks([]); pax.set_yticks([])

        # panel title strip
        pax.add_patch(patches.Rectangle((0, 0.945), 1.0, 0.055, fc=ec, ec="none",
                      transform=pax.transAxes, clip_on=False, zorder=10))
        pax.text(0.50, 0.968, panel_title, ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white",
                 transform=pax.transAxes, zorder=11)

        # tree label badge
        pax.text(0.97, 0.908, tree_label, ha="right", va="center",
                 fontsize=7.5, color=ec, style="italic",
                 transform=pax.transAxes)

        if nodes:
            _draw_real_tree(pax, nodes,
                            x_off=0.05, y_top=0.88, width=0.90, v_gap=0.22,
                            node_fc=ec2, leaf_fc=ec, edge_color=ec)
        else:
            pax.text(0.5, 0.5, "Run train.py to generate", ha="center",
                     fontsize=10, color=MUTED, transform=pax.transAxes)

        # legend
        pax.add_patch(patches.Rectangle((-0.005, -0.095), 0.25, 0.085,
            fc=ec2, ec=ec, lw=0.8, transform=pax.transAxes))
        pax.text(0.125, -0.055, "Decision node", ha="center", va="center",
                 fontsize=7, color="white", transform=pax.transAxes)
        pax.add_patch(plt.Circle((0.33, -0.053), 0.025, fc=ec, ec="none",
                      transform=pax.transAxes))
        pax.text(0.41, -0.053, f"Leaf (predicted {unit})", ha="left", va="center",
                 fontsize=7, color="#334155", transform=pax.transAxes)
        pax.text(0.73, -0.053, "Y = yes (≤ threshold)   N = no (> threshold)",
                 ha="left", va="center", fontsize=6.8, color="#64748b",
                 transform=pax.transAxes)

    # ── aggregation bar ───────────────────────────────────────────────────────
    aax = fig.add_axes([0.08, 0.01, 0.84, 0.065])
    aax.set_xlim(0, 1); aax.set_ylim(0, 1); aax.axis("off")
    aax.set_facecolor(DARK)
    aax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.0,rounding_size=0.02", fc=DARK, ec="none"))
    aax.text(0.5, 0.72,
             "Final prediction  =  Average of ALL trees           "
             "ŷ  =  (1/N) · Σᵢ fᵢ(x)           "
             "Uncertainty  =  std dev across trees  (≈ ±1σ)",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="white", family="monospace")
    aax.text(0.5, 0.24,
             "Each tree votes independently.  Averaging cancels individual errors.  "
             "More trees → more stable, lower-variance prediction.",
             ha="center", va="center", fontsize=8.5, color="#94a3b8", style="italic")

    plt.savefig(OUT / "random_forest.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved: random_forest.png")


# ── 3. before/after comparison (computed live) ───────────────────────────────
def _basic_model():
    """The original 'before' model: plain RandomForest on the 4 raw features."""
    return Pipeline([("scaler", StandardScaler()),
                     ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                     random_state=SEED, n_jobs=-1))])


def compute_before_after():
    """Cross-validated R²/MAE for the basic vs improved configuration."""
    rows = {}
    for key, spec in TARGETS.items():
        raw = pd.read_csv(DATA_DIR / spec["csv"])
        eng = add_derived_features(raw)
        y = raw[spec["column"]].values
        cv = LeaveOneOut() if len(raw) < 40 else KFold(5, shuffle=True, random_state=SEED)

        # before: basic RF, 4 raw features
        yb = cross_val_predict(_basic_model(), raw[BASE_FEATURES], y, cv=cv, n_jobs=-1)
        # after: tuned per-target model, 6 features
        ya = cross_val_predict(build_model(spec["model"]), eng[FEATURES], y, cv=cv, n_jobs=-1)
        rows[key] = {
            "label": spec["label"], "unit": spec["unit"],
            "before": (r2_score(y, yb), mean_absolute_error(y, yb)),
            "after":  (r2_score(y, ya), mean_absolute_error(y, ya)),
        }
    return rows


def before_after(rows):
    labels = [rows[k]["label"] for k in rows]
    r2_b = [rows[k]["before"][0] for k in rows]
    r2_a = [rows[k]["after"][0] for k in rows]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels)); w = 0.36

    axl.bar(x - w/2, r2_b, w, label="Before (basic RF, 4 features)", color="#bbbbbb")
    axl.bar(x + w/2, r2_a, w, label="After (tuned + derived features)", color="#2ca02c")
    axl.set_xticks(x); axl.set_xticklabels(labels); axl.set_ylabel("Cross-validated R²")
    axl.set_ylim(0, 1); axl.set_title("Accuracy (R²) — higher is better")
    axl.legend(fontsize=8); axl.grid(axis="y", alpha=0.3, linestyle="--")
    for xi, (b, a) in enumerate(zip(r2_b, r2_a)):
        axl.text(xi - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        axl.text(xi + w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, fontweight="bold")

    mae_b = [rows[k]["before"][1] for k in rows]
    mae_a = [rows[k]["after"][1] for k in rows]
    # normalise MAE to each property's own scale for a shared axis
    axr2 = axr
    axr2.bar(x - w/2, mae_b, w, label="Before", color="#bbbbbb")
    axr2.bar(x + w/2, mae_a, w, label="After", color="#1f77b4")
    axr2.set_xticks(x)
    axr2.set_xticklabels([f"{rows[k]['label']}\n({rows[k]['unit']})" for k in rows])
    axr2.set_ylabel("Mean Absolute Error"); axr2.set_title("Error (MAE) — lower is better")
    axr2.legend(fontsize=8); axr2.grid(axis="y", alpha=0.3, linestyle="--")
    for xi, (b, a) in enumerate(zip(mae_b, mae_a)):
        axr2.text(xi - w/2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
        axr2.text(xi + w/2, a + 0.05, f"{a:.2f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Model Improvement: Before vs After", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT / "before_after.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  Saved: before_after.png")


# ── 4. relational plots: property vs each parameter ──────────────────────────
def relational():
    for key, spec in TARGETS.items():
        df = pd.read_csv(DATA_DIR / spec["csv"])
        col, unit, label, color = spec["column"], spec["unit"], spec["label"], spec["color"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, feat in zip(axes.ravel(), BASE_FEATURES):
            ax.scatter(df[feat], df[col], alpha=0.6, color=color, edgecolors="white", linewidth=0.3)
            ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
            ax.set_ylabel(f"{label} ({unit})", fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")
        fig.suptitle(f"{label} vs Process Parameters  ({len(df)} samples)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(OUT / f"relational_{col}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: relational_{col}.png")


# ── 5. per-model detailed workflow ───────────────────────────────────────────
def _model_metrics(key, spec):
    """Cross-validated R², MAE, RMSE and feature importances for one model."""
    df = add_derived_features(pd.read_csv(DATA_DIR / spec["csv"]))
    X, y = df[FEATURES], df[spec["column"]].values
    n = len(df)
    cv = LeaveOneOut() if n < 40 else KFold(5, shuffle=True, random_state=SEED)
    cvname = "Leave-One-Out CV" if n < 40 else "5-fold CV"
    yp = cross_val_predict(build_model(spec["model"]), X, y, cv=cv, n_jobs=-1)
    model = build_model(spec["model"]).fit(X, y)
    imp = dict(zip(FEATURES, model.named_steps["model"].feature_importances_))
    return {
        "n": n, "cvname": cvname,
        "r2": r2_score(y, yp), "mae": mean_absolute_error(y, yp),
        "rmse": float(np.sqrt(np.mean((y - yp) ** 2))),
        "range": (float(y.min()), float(y.max())), "importance": imp,
    }


def per_model_workflow(key, spec):
    """A detailed, stage-by-stage workflow diagram for a single property's model.

    Shows: data → feature engineering → standardisation → the specific ensemble
    algorithm → cross-validation → final metrics, with that model's real numbers
    and the top driving features.
    """
    m = _model_metrics(key, spec)
    color = spec["color"]
    algo_full = ("Random Forest Regressor" if spec["model"] == "rf"
                 else "Extra Trees Regressor")
    n_trees = 500 if spec["model"] == "rf" else 400
    algo_note = ("each tree trained on a bootstrap sample;\nbest split chosen per node"
                 if spec["model"] == "rf"
                 else "split thresholds chosen at random →\nlower variance on small data")
    # top-3 features by importance (short names so they fit the side box)
    short = {"Laser_Power_W": "Power", "Scan_Speed_mm_s": "Speed",
             "Hatch_Distance_mm": "Hatch", "Layer_Thickness_mm": "Layer",
             "Energy": "Energy", "Power_Speed_Ratio": "Power/Speed"}
    top3 = sorted(m["importance"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_txt = "\n".join(f"{short[f]}: {v:.2f}" for f, v in top3)

    fig, ax = plt.subplots(figsize=(9.4, 10.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 12.4); ax.axis("off")
    cx, w = 1.6, 6.0

    stages = [
        (f"INPUT DATA\n{m['n']} samples · {spec['label']} ({spec['unit']})\n"
         f"range {m['range'][0]:.1f}–{m['range'][1]:.1f} {spec['unit']}", "#dbeafe"),
        ("RAW FEATURES (4)\nLaser Power · Scan Speed ·\nHatch Distance · Layer Thickness", "#e0e7ff"),
        ("FEATURE ENGINEERING (+2)\nEnergy = P/(v·h·t)\nPower/Speed = P/v", "#ede9fe"),
        ("STANDARDISE\nzero mean, unit variance\n(6 features total)", "#fce7f3"),
        (f"MODEL: {algo_full}\n{n_trees} trees\n{algo_note}", "#ffffff"),
        (f"VALIDATION\n{m['cvname']} — pooled predictions", "#dcfce7"),
        (f"RESULT\nR² = {m['r2']:.3f}   MAE = {m['mae']:.2f} {spec['unit']}\n"
         f"RMSE = {m['rmse']:.2f} {spec['unit']}", "#fef9c3"),
    ]
    h = 1.15; gap = 0.42; y = 12.4 - h - 0.2
    centres = []
    for i, (txt, fc) in enumerate(stages):
        ec = color if i in (4, 6) else "#333"
        lw = 2.0 if i in (4, 6) else 1.3
        ax.add_patch(FancyBboxPatch((cx, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=lw, edgecolor=ec, facecolor=fc))
        ax.text(cx + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9, color="#111")
        centres.append((cx + w / 2, y))
        if i > 0:
            px, py = centres[i - 1]
            _arrow(ax, px, py, cx + w / 2, y + h, color="#666")
        y -= h + gap

    # side annotation: top driving features
    sb_x, sb_w = cx + w + 0.30, 2.7
    ax.add_patch(FancyBboxPatch((sb_x, centres[4][1] - 0.15), sb_w, h + 0.3,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.0, edgecolor=color, facecolor="#fafafa"))
    ax.text(sb_x + sb_w / 2, centres[4][1] + h / 2,
            "Top features\n(importance)\n\n" + top_txt,
            ha="center", va="center", fontsize=8, color="#333")

    ax.set_title(f"{spec['label']} — Model Workflow ({algo_full})",
                 fontsize=12, color=color, pad=10)
    out = OUT / f"workflow_{spec['column']}.png"
    plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out.name}")


def make_all():
    workflow(); random_forest(); relational()
    for key, spec in TARGETS.items():
        per_model_workflow(key, spec)
    rows = compute_before_after(); before_after(rows)
    return rows


if __name__ == "__main__":
    print("\nGenerating report figures …")
    make_all()
    print("Done. See results/.\n")
