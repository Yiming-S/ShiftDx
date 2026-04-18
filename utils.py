"""
Utility constants and helper functions for the ShiftDx dashboard.
"""

import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Dataset auto-discovery
# ---------------------------------------------------------------------------

# Known metadata (extend as new builders are added). Datasets not listed here
# still work; they just show blank metadata in the Overview card.
DATASET_META = {
    "bnci004":     {"channels": 3,  "subjects": 9,  "sessions": "5",    "role": "Low-channel supplement"},
    "stieger2021": {"channels": 62, "subjects": 38, "sessions": "6-11", "role": "Primary analysis"},
    "ma2020":      {"channels": 62, "subjects": 25, "sessions": "15",   "role": "Validation"},
    "zhou2016":    {"channels": 14, "subjects": 4,  "sessions": "3",    "role": "Small MOABB demo"},
    "bnci2015001": {"channels": 13, "subjects": 12, "sessions": "2-3",  "role": "Small MOABB binary MI"},
}


def discover_datasets(data_dir: str) -> list[str]:
    """Return dataset names found in `data/` by scanning drift_trajectories_*.csv."""
    pattern = os.path.join(data_dir, "drift_trajectories_*.csv")
    names = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        # drift_trajectories_<name>.csv
        stem = base[len("drift_trajectories_"):-len(".csv")]
        names.append(stem)
    return sorted(names)


# ---------------------------------------------------------------------------
# Features / strategies
# ---------------------------------------------------------------------------

FEATURES = ["CSP", "logvar", "TS"]

STRATEGIES = ["train_once", "train_once_da", "retrain"]
STRATEGY_LABEL = {
    "train_once":    "No DA",
    "train_once_da": "DA",
    "retrain":       "Retrain",
}

# ---------------------------------------------------------------------------
# Classifiers (naming follows CrossPython)
# ---------------------------------------------------------------------------

CLASSIFIERS = ["lda", "svm_linear", "svm_radial"]

CLASSIFIER_LABEL = {
    "lda":        "LDA",
    "svm_linear": "SVM (linear)",
    "svm_radial": "SVM (radial)",
    "mdm":        "MDM",
    "lr":         "LogReg",
    "el":         "ElasticNet",
}


def available_classifiers(df) -> list[str]:
    """Classifiers actually present in a DataFrame."""
    if df is None or df.empty or "classifier" not in df.columns:
        return []
    present = set(df["classifier"].dropna().astype(str).unique())
    return [c for c in CLASSIFIERS if c in present] + \
           sorted(c for c in present if c not in CLASSIFIERS)


# ---------------------------------------------------------------------------
# DA methods — all 10 from DA4BCI
# ---------------------------------------------------------------------------

DA_METHODS = ["sa", "pt", "coral", "tca", "gfk", "rd", "art", "ot", "mida", "m3d"]

DA_LABEL = {
    "none":  "No DA",
    "sa":    "SA (Subspace Alignment)",
    "pt":    "PT (Parallel Transport)",
    "coral": "CORAL (Correlation Alignment)",
    "tca":   "TCA (Transfer Component Analysis)",
    "gfk":   "GFK (Geodesic Flow Kernel)",
    "rd":    "RD (Riemannian Distance)",
    "art":   "ART (Aligned Riemannian Transport)",
    "ot":    "OT (Sinkhorn barycentric)",
    "mida":  "MIDA (Max-Independence DA)",
    "m3d":   "M3D (Manifold Multi-step DA)",
}


# ---------------------------------------------------------------------------
# Distance metrics — all 5 from DA4BCI
# ---------------------------------------------------------------------------

DISTANCE_METRICS = ["dist_mmd", "dist_energy", "dist_wasserstein",
                    "dist_mahalanobis", "dist_euclidean"]

DISTANCE_LABEL = {
    "dist_mmd":         "MMD",
    "dist_energy":      "Energy",
    "dist_wasserstein": "Wasserstein",
    "dist_mahalanobis": "Mahalanobis",
    "dist_euclidean":   "Euclidean",
}

# Map distance column name to its corresponding drift_z column (built by
# scripts/build_moabb.py; z-scored within dataset × feature × classifier).
DRIFT_Z_COL = {
    "dist_mmd":         "drift_z_mmd",
    "dist_energy":      "drift_z_energy",
    "dist_wasserstein": "drift_z_wasserstein",
    "dist_mahalanobis": "drift_z_mahalanobis",
    "dist_euclidean":   "drift_z_euclidean",
}


def pick_metric_with_drift_z(df) -> list[str]:
    """Return distance metrics present AND with a matching drift_z_* column."""
    if df is None or df.empty:
        return []
    return [
        m for m in DISTANCE_METRICS
        if m in df.columns and DRIFT_Z_COL[m] in df.columns
    ]


def apply_drift_metric(df, metric: str):
    """Return a copy of `df` whose `drift_z` column equals the chosen metric.

    If the expected drift_z_<metric> column is missing, falls back to the
    existing `drift_z` column (backwards-compatible with legacy CSVs that
    only carry an MMD-based drift_z).
    """
    if df is None or df.empty:
        return df
    zcol = DRIFT_Z_COL.get(metric)
    out = df.copy()
    if zcol and zcol in out.columns:
        out["drift_z"] = out[zcol]
    return out

DISTANCE_DESCRIPTION = {
    "dist_mmd":         "Maximum Mean Discrepancy (RBF kernel)",
    "dist_energy":      "Energy distance (pairwise distance based)",
    "dist_wasserstein": "Wasserstein (optimal transport cost)",
    "dist_mahalanobis": "Mahalanobis (whitening-aware with shrinkage)",
    "dist_euclidean":   "Euclidean (mean of pairwise distance matrix)",
}


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

DATASET_COLORS = {
    "bnci004":     "#F59E0B",
    "stieger2021": "#3B82F6",
    "ma2020":      "#10B981",
    "zhou2016":    "#8B5CF6",
    "bnci2015001": "#EC4899",
}

FEATURE_COLORS = {
    "CSP":    "#4F46E5",
    "logvar": "#EF4444",
    "TS":     "#8B5CF6",
}

STRATEGY_COLORS = {
    "No DA":   "#94A3B8",
    "DA":      "#F59E0B",
    "Retrain": "#10B981",
}

DA_COLORS = {
    "none":  "#94A3B8",
    "sa":    "#F59E0B",
    "pt":    "#8B5CF6",
    "coral": "#EC4899",
    "tca":   "#06B6D4",
    "gfk":   "#22C55E",
    "rd":    "#F97316",
    "art":   "#0EA5E9",
    "ot":    "#A855F7",
    "mida":  "#EAB308",
    "m3d":   "#EF4444",
}

CLASSIFIER_COLORS = {
    "lda":        "#4F46E5",
    "svm_linear": "#F59E0B",
    "svm_radial": "#EF4444",
    "mdm":        "#10B981",
    "lr":         "#EC4899",
    "el":         "#8B5CF6",
}

COOL_LIGHT_SEQUENTIAL = [
    [0.00, "#F7FAFD"],
    [0.20, "#E6EEF8"],
    [0.40, "#CCDDF4"],
    [0.60, "#A8C0E8"],
    [0.80, "#6F97DB"],
    [1.00, "#2F6FED"],
]

COOL_LIGHT_DIVERGING = [
    [0.00, "#C97A66"],
    [0.20, "#E8BFB4"],
    [0.48, "#F8F9FB"],
    [0.52, "#F8F9FB"],
    [0.80, "#ABC3EA"],
    [1.00, "#2F6FED"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_acc(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def style_figure(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13),
        title_font=dict(size=16, color="#1E293B"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=50, b=50),
        hoverlabel=dict(
            bgcolor="white", font_size=12,
            font_family="Inter, system-ui, sans-serif", bordercolor="#E2E8F0",
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def filter_by_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "All" or "dataset" not in df.columns:
        return df
    return df[df["dataset"] == dataset].copy()


def available_da_methods(df: pd.DataFrame) -> list[str]:
    """DA methods actually present in a sequential_eval or merged DataFrame."""
    if df.empty or "da" not in df.columns:
        return []
    present = set(df["da"].dropna().astype(str).unique())
    # Preserve canonical order; exclude 'none'
    return [m for m in DA_METHODS if m in present]


def available_distance_metrics(df: pd.DataFrame) -> list[str]:
    """Distance columns actually present in a drift DataFrame."""
    if df.empty:
        return []
    return [m for m in DISTANCE_METRICS if m in df.columns]
