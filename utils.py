"""
Utility constants, shared UI components, and the global-context sidebar
for the ShiftDx dashboard.
"""

import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# Dataset auto-discovery
# ============================================================================

DATASET_META = {
    "bnci004":     {"channels": 3,  "subjects": 9,  "sessions": "5",    "role": "Low-channel supplement"},
    "stieger2021": {"channels": 62, "subjects": 62, "sessions": "6-11", "role": "Primary analysis"},
    "ma2020":      {"channels": 65, "subjects": 25, "sessions": "15",   "role": "Validation"},
    "zhou2016":    {"channels": 14, "subjects": 4,  "sessions": "3",    "role": "Small MOABB demo"},
    "bnci2015001": {"channels": 13, "subjects": 12, "sessions": "2-3",  "role": "Small MOABB binary MI"},
}


def discover_datasets(data_dir: str) -> list[str]:
    """Return dataset names found in `data/` by scanning drift_trajectories_*.csv."""
    pattern = os.path.join(data_dir, "drift_trajectories_*.csv")
    names = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        stem = base[len("drift_trajectories_"):-len(".csv")]
        names.append(stem)
    return sorted(names)


# ============================================================================
# Features / strategies  (D0.2: locked display labels)
# ============================================================================

FEATURES = ["CSP", "logvar", "TS"]

STRATEGIES = ["train_once", "train_once_da", "retrain"]

STRATEGY_LABEL = {
    "train_once":    "No DA",
    "train_once_da": "DA",
    "retrain":       "Retrain",
}


def strategy_display(strategy: str, da: str | None = None) -> str:
    """User-facing strategy label. Appends DA method suffix when relevant."""
    base = STRATEGY_LABEL.get(strategy, strategy)
    if strategy == "train_once_da" and da and da != "none":
        return f"DA · {da.upper()}"
    return base


# ============================================================================
# Classifiers
# ============================================================================

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
    if df is None or df.empty or "classifier" not in df.columns:
        return []
    present = set(df["classifier"].dropna().astype(str).unique())
    return [c for c in CLASSIFIERS if c in present] + \
           sorted(c for c in present if c not in CLASSIFIERS)


# ============================================================================
# DA methods
# ============================================================================

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

DA_SHORT_LABEL = {
    "sa": "SA", "pt": "PT", "coral": "CORAL", "tca": "TCA", "gfk": "GFK",
    "rd": "RD", "art": "ART", "ot": "OT", "mida": "MIDA", "m3d": "M3D",
}


def available_da_methods(df: pd.DataFrame) -> list[str]:
    if df.empty or "da" not in df.columns:
        return []
    present = set(df["da"].dropna().astype(str).unique())
    return [m for m in DA_METHODS if m in present]


# ============================================================================
# Distance metrics
# ============================================================================

DISTANCE_METRICS = ["dist_mmd", "dist_energy", "dist_wasserstein",
                    "dist_mahalanobis", "dist_euclidean"]

DISTANCE_LABEL = {
    "dist_mmd":         "MMD",
    "dist_energy":      "Energy",
    "dist_wasserstein": "Wasserstein",
    "dist_mahalanobis": "Mahalanobis",
    "dist_euclidean":   "Euclidean",
}

DRIFT_Z_COL = {
    "dist_mmd":         "drift_z_mmd",
    "dist_energy":      "drift_z_energy",
    "dist_wasserstein": "drift_z_wasserstein",
    "dist_mahalanobis": "drift_z_mahalanobis",
    "dist_euclidean":   "drift_z_euclidean",
}

DISTANCE_DESCRIPTION = {
    "dist_mmd":         "Maximum Mean Discrepancy (RBF kernel).",
    "dist_energy":      "Energy distance (pairwise distance based).",
    "dist_wasserstein": "Wasserstein (optimal transport cost).",
    "dist_mahalanobis": "Mahalanobis (whitening-aware with shrinkage).",
    "dist_euclidean":   "Euclidean (mean of pairwise distance matrix).",
}


def pick_metric_with_drift_z(df) -> list[str]:
    if df is None or df.empty:
        return []
    return [m for m in DISTANCE_METRICS
            if m in df.columns and DRIFT_Z_COL[m] in df.columns]


def apply_drift_metric(df, metric: str):
    if df is None or df.empty:
        return df
    zcol = DRIFT_Z_COL.get(metric)
    out = df.copy()
    if zcol and zcol in out.columns:
        out["drift_z"] = out[zcol]
    return out


def available_distance_metrics(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return [m for m in DISTANCE_METRICS if m in df.columns]


# ============================================================================
# Colour palettes
# ============================================================================

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
    [0.00, "#F7FAFD"], [0.20, "#E6EEF8"], [0.40, "#CCDDF4"],
    [0.60, "#A8C0E8"], [0.80, "#6F97DB"], [1.00, "#2F6FED"],
]

COOL_LIGHT_DIVERGING = [
    [0.00, "#C97A66"], [0.20, "#E8BFB4"], [0.48, "#F8F9FB"],
    [0.52, "#F8F9FB"], [0.80, "#ABC3EA"], [1.00, "#2F6FED"],
]


# ── Colorblind-safe (Okabe-Ito) variants ─────────────────────────────────────
# Activated by the sidebar toggle (st.session_state['ctx_colorblind']). The
# accessor functions below return the active palette so views never branch.

FEATURE_COLORS_CB = {"CSP": "#0072B2", "logvar": "#D55E00", "TS": "#009E73"}

STRATEGY_COLORS_CB = {"No DA": "#999999", "DA": "#E69F00", "Retrain": "#009E73"}

DATASET_COLORS_CB = {
    "bnci004": "#E69F00", "stieger2021": "#0072B2", "ma2020": "#009E73",
    "zhou2016": "#CC79A7", "bnci2015001": "#D55E00",
}

# Okabe-Ito has 8 distinct hues; with 10 DA methods a couple necessarily repeat.
DA_COLORS_CB = {
    "none": "#999999", "sa": "#E69F00", "pt": "#0072B2", "coral": "#CC79A7",
    "tca": "#56B4E9", "gfk": "#009E73", "rd": "#D55E00", "art": "#882255",
    "ot": "#F0E442", "mida": "#000000", "m3d": "#117733",
}

CLASSIFIER_COLORS_CB = {
    "lda": "#0072B2", "svm_linear": "#E69F00", "svm_radial": "#D55E00",
    "mdm": "#009E73", "lr": "#CC79A7", "el": "#56B4E9",
}


def colorblind_mode() -> bool:
    """True when the user enabled the colorblind-safe palette in the sidebar."""
    return bool(st.session_state.get("ctx_colorblind", False))


def feature_colors() -> dict:
    return FEATURE_COLORS_CB if colorblind_mode() else FEATURE_COLORS


def strategy_colors() -> dict:
    return STRATEGY_COLORS_CB if colorblind_mode() else STRATEGY_COLORS


def dataset_colors() -> dict:
    return DATASET_COLORS_CB if colorblind_mode() else DATASET_COLORS


def da_colors() -> dict:
    return DA_COLORS_CB if colorblind_mode() else DA_COLORS


def classifier_colors() -> dict:
    return CLASSIFIER_COLORS_CB if colorblind_mode() else CLASSIFIER_COLORS


# ============================================================================
# Formatting helpers
# ============================================================================

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


# ============================================================================
# Global sidebar context (Phase 1 / D1.1)
# ============================================================================

DA_LAB_PAGES = {"live-da", "multi-metric", "da-sweep", "drift-detection"}


# Mapping of session_state ctx keys -> short URL query-param names.
_QP_KEYS = {"dataset": "ds", "classifier": "clf", "metric": "metric", "da": "da"}


def _init_ctx_from_query():
    """Seed the global context from URL query params on first load, so a shared
    link reproduces the selection. Only fills keys not already in session_state;
    invalid values are corrected later by each selectbox's option guard."""
    qp = st.query_params
    for skey, qkey in _QP_KEYS.items():
        sk = f"ctx_{skey}"
        if sk not in st.session_state and qkey in qp:
            st.session_state[sk] = qp.get(qkey)
    if "ctx_colorblind" not in st.session_state and "cb" in qp:
        st.session_state["ctx_colorblind"] = qp.get("cb") in ("1", "true", "True")


def _sync_ctx_to_query():
    """Write the active context back to the URL so it is bookmarkable."""
    updates = {qkey: str(st.session_state.get(f"ctx_{skey}"))
               for skey, qkey in _QP_KEYS.items()
               if st.session_state.get(f"ctx_{skey}") is not None}
    updates["cb"] = "1" if st.session_state.get("ctx_colorblind") else "0"
    try:
        st.query_params.update(updates)
    except Exception:
        pass


def _render_display_controls():
    cb = st.toggle(
        "Colorblind-safe palette",
        value=bool(st.session_state.get("ctx_colorblind", False)),
        key="ctx_colorblind_widget",
        help="Recolor feature / DA / strategy series with an Okabe-Ito palette.",
    )
    st.session_state["ctx_colorblind"] = cb


def _render_selection_summary(store):
    ctx = get_ctx()
    ds = ctx["dataset"]
    try:
        n_subj = len(store.subjects(ds))
    except Exception:
        n_subj = 0
    merged = store.merged_df
    if merged is not None and not merged.empty and ds != "All" and "dataset" in merged.columns:
        n_obs = int((merged["dataset"] == ds).sum())
    else:
        n_obs = int(len(merged)) if merged is not None else 0
    clf = CLASSIFIER_LABEL.get(ctx["classifier"], ctx["classifier"])
    met = DISTANCE_LABEL.get(ctx["metric"], ctx["metric"])
    da = DA_SHORT_LABEL.get(ctx["da"], ctx["da"].upper())
    st.markdown(
        f'''<div style="background:#F1F5F9;border:1px solid #E2E8F0;border-radius:10px;
        padding:10px 12px;margin:6px 0;font-size:0.74rem;color:#334155;line-height:1.6;">
        <b>{ds}</b> · {n_subj} subj · {n_obs:,} rows<br>
        clf <b>{clf}</b> · metric <b>{met}</b> · DA <b>{da}</b>
        </div>''',
        unsafe_allow_html=True,
    )


def render_global_sidebar(store, current_page_path: str | None = None):
    """Render the 4-selector global context in the sidebar.

    On DA-Lab pages (D0.5) we skip the dropdowns and show a notice. The
    selection is deep-linked via st.query_params and summarized in a card.
    """
    is_da_lab = current_page_path in DA_LAB_PAGES
    _init_ctx_from_query()

    with st.sidebar:
        st.markdown(
            '<div style="margin:12px 0 6px 0;font-size:0.72rem;'
            'font-weight:700;letter-spacing:0.8px;color:#64748B;">'
            'DATASET CONTEXT</div>',
            unsafe_allow_html=True,
        )

        if is_da_lab:
            st.caption(
                "Global filters don't apply on DA Lab pages — "
                "each page has its own controls."
            )
            _render_display_controls()
            if st.button("↻ Refresh data", use_container_width=True, key="ctx_refresh_da"):
                st.cache_resource.clear()
                st.rerun()
            _sync_ctx_to_query()
            return

        # Dataset
        ds_options = (["All"] + list(store.datasets)
                      if len(store.datasets) > 1
                      else list(store.datasets) or ["(none)"])
        default_ds = st.session_state.get("ctx_dataset") or ds_options[0]
        if default_ds not in ds_options:
            default_ds = ds_options[0]
        ds = st.selectbox(
            "Dataset", ds_options,
            index=ds_options.index(default_ds), key="ctx_dataset_widget",
        )
        st.session_state["ctx_dataset"] = ds

        # Classifier
        clf_pool = _union_classifiers(store)
        clf_options = ["All pooled"] + clf_pool
        default_clf = st.session_state.get("ctx_classifier", "All pooled")
        if default_clf not in clf_options:
            default_clf = "All pooled"
        clf = st.selectbox(
            "Classifier", clf_options,
            index=clf_options.index(default_clf), key="ctx_classifier_widget",
            format_func=lambda x: x if x == "All pooled" else CLASSIFIER_LABEL.get(x, x),
        )
        st.session_state["ctx_classifier"] = clf

        # Drift metric
        metric_pool = _union_metrics(store)
        default_m = st.session_state.get("ctx_metric", "dist_mmd")
        if default_m not in metric_pool:
            default_m = metric_pool[0] if metric_pool else "dist_mmd"
        m = st.selectbox(
            "Drift metric", metric_pool,
            index=metric_pool.index(default_m) if metric_pool else 0,
            key="ctx_metric_widget",
            format_func=lambda x: DISTANCE_LABEL.get(x, x),
        )
        st.session_state["ctx_metric"] = m

        # DA method
        da_pool = _union_da_methods(store)
        default_da = st.session_state.get("ctx_da", "sa")
        if default_da not in da_pool:
            default_da = da_pool[0] if da_pool else "sa"
        da = st.selectbox(
            "DA method", da_pool,
            index=da_pool.index(default_da) if da_pool else 0,
            key="ctx_da_widget",
            format_func=lambda x: DA_LABEL.get(x, x),
        )
        st.session_state["ctx_da"] = da

        _render_selection_summary(store)
        _render_display_controls()

        b1, b2 = st.columns(2)
        with b1:
            if st.button("↻ Refresh", use_container_width=True, key="ctx_refresh"):
                st.cache_resource.clear()
                st.rerun()
        with b2:
            if st.button("⟲ Reset", use_container_width=True, key="ctx_reset",
                         help="Clear the selection and the shareable URL params."):
                for k in ("ctx_dataset", "ctx_classifier", "ctx_metric",
                          "ctx_da", "ctx_colorblind"):
                    st.session_state.pop(k, None)
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                st.rerun()

        _sync_ctx_to_query()


def _union_classifiers(store) -> list[str]:
    pool: set[str] = set()
    for df in (store.merged_df, store.eval_df):
        if df is not None and not df.empty and "classifier" in df.columns:
            pool.update(str(c) for c in df["classifier"].dropna().unique())
    return [c for c in CLASSIFIERS if c in pool] + \
           sorted(c for c in pool if c not in CLASSIFIERS)


def _union_metrics(store) -> list[str]:
    out: list[str] = []
    for m in DISTANCE_METRICS:
        for df in (store.drift_df, store.merged_df):
            if df is not None and not df.empty and m in df.columns:
                out.append(m)
                break
    return out or ["dist_mmd"]


def _union_da_methods(store) -> list[str]:
    pool: set[str] = set()
    for df in (store.merged_df, store.eval_df):
        if df is not None and not df.empty and "da" in df.columns:
            pool.update(str(d) for d in df["da"].dropna().unique())
    pool.discard("none")
    return [m for m in DA_METHODS if m in pool] or ["sa"]


def get_ctx() -> dict:
    """Read global context with safe defaults."""
    return {
        "dataset":    st.session_state.get("ctx_dataset", "All"),
        "classifier": st.session_state.get("ctx_classifier", "All pooled"),
        "metric":     st.session_state.get("ctx_metric", "dist_mmd"),
        "da":         st.session_state.get("ctx_da", "sa"),
    }


def apply_ctx_dataset(df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    ctx = ctx or get_ctx()
    return filter_by_dataset(df, ctx["dataset"])


def apply_ctx_classifier(df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    ctx = ctx or get_ctx()
    if df is None or df.empty or "classifier" not in df.columns:
        return df
    if ctx["classifier"] == "All pooled":
        return df
    return df[df["classifier"] == ctx["classifier"]].copy()


def apply_ctx_metric(df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    ctx = ctx or get_ctx()
    return apply_drift_metric(df, ctx["metric"])


# ============================================================================
# Reusable UI components (Phase 2)
# ============================================================================

def empty_state(title: str, reason: str,
                cmd: str | None = None, cmd_label: str | None = None,
                dataset: str | None = None):
    """Render a friendly empty-state panel.

    If `dataset` is given (and no explicit `cmd`), suggest the build command for
    that specific dataset so the hint names the thing the user actually filtered
    to instead of a generic zhou2016 example.
    """
    if cmd is None and dataset and dataset != "All":
        cmd = f"python scripts/build_moabb.py --dataset {dataset}"
        cmd_label = cmd_label or f"Build `{dataset}`"
    with st.container(border=True):
        st.markdown(f"#### ⚠ {title}")
        st.markdown(reason)
        if cmd:
            st.markdown(f"**{cmd_label or 'Suggested fix'}:**")
            st.code(cmd, language="bash")


def about_page(what_you_see: list[str], how_to_read: list[str],
               paper_ref: str | None = None,
               key_terms: list[tuple[str, str]] | None = None):
    """Collapsible 'About this page' expander shown near the page header."""
    with st.expander("ℹ About this page", expanded=False):
        st.markdown("**What you see:**")
        for item in what_you_see:
            st.markdown(f"- {item}")
        st.markdown("**How to read:**")
        for item in how_to_read:
            st.markdown(f"- {item}")
        if paper_ref:
            st.markdown(f"**Paper reference:** {paper_ref}")
        if key_terms:
            st.markdown("**Key terms on this page:**")
            for term, desc in key_terms:
                st.markdown(f"- `{term}` — {desc}")


def download_bar(key: str, df: pd.DataFrame | None = None, filename: str = "data"):
    """Compact CSV-download row. Plotly's toolbar already provides PNG export."""
    if df is None or df.empty:
        return
    c1, _ = st.columns([1, 6])
    with c1:
        st.download_button(
            "📥 CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_{key}",
        )


# Glossary entries (D1.2: double placement — sidebar + page-local About expanders)
GLOSSARY_ENTRIES = [
    ("Session k", "The k-th recording session for a subject. Session 0 is calibration."),
    ("Feature", "Feature family: CSP, logvar, or TS (tangent space)."),
    ("No DA", "Train once on session 0, predict later sessions unchanged."),
    ("DA", "Train once on session 0, then adapt source/target features before predicting session k."),
    ("Retrain", "5-fold CV on session k itself — oracle upper bound."),
    ("drift_z", "Standardized drift: z-score of the chosen distance (within dataset × feature × classifier)."),
    ("acc_centered", "Session-k accuracy minus that subject's session-0 baseline."),
    ("η₂", "Claim 2: level shift — DA's constant accuracy gain."),
    ("η₄", "Claim 2: slope change — DA's reduction of drift sensitivity."),
    ("ρ₀, ρ₁", "Claim 3: intercept and slope of R_g(z) = retrain − DA vs drift."),
    ("R_g(z)", "Retraining gap at drift level z: acc_retrain − acc_DA."),
]


def render_glossary():
    with st.sidebar:
        st.markdown(
            '<div style="margin:16px 0 6px 0;font-size:0.72rem;'
            'font-weight:700;letter-spacing:0.8px;color:#64748B;">'
            'GLOSSARY</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Key terms", expanded=False):
            for term, desc in GLOSSARY_ENTRIES:
                st.markdown(f"**{term}** — {desc}")


# ============================================================================
# Shared data filters & guards
# ============================================================================

def is_retrain(df: pd.DataFrame) -> pd.DataFrame:
    """The canonical retrain slice: strategy == 'retrain' on the diagonal
    (ref_session == target_session). Centralized so a protocol change touches
    one place instead of every claim page."""
    if df is None or df.empty:
        return df
    needed = {"strategy", "ref_session", "target_session"}
    if not needed.issubset(df.columns):
        return df.iloc[0:0]
    return df[(df["strategy"] == "retrain") &
              (df["ref_session"] == df["target_session"])]


def require_columns(df: pd.DataFrame, cols, what: str = "this view") -> bool:
    """Return True if df has the needed columns; otherwise render an empty_state
    panel (instead of crashing with a raw KeyError) and return False."""
    if df is None or df.empty:
        empty_state("No data for this selection",
                    f"No rows are available for {what}.")
        return False
    missing = [c for c in cols if c not in df.columns]
    if missing:
        empty_state(
            "Missing expected columns",
            f"{what} needs column(s) `{', '.join(missing)}`, which the loaded "
            "CSV does not have. Rebuild the dataset with `scripts/build_moabb.py`.",
        )
        return False
    return True


# ============================================================================
# DA4BCI control parameters & synthetic shift scenarios (shared by DA-Lab pages)
# ============================================================================

def da_control_params(method: str, d: int) -> dict:
    """Default control dict for `da4bci.domain_adaptation`, using the dispatcher's
    ACTUAL key names. The dispatcher reads e.g. control['lambda'] (coral),
    control['dim_subspace'] (gfk), control['eps'] (ot), control['max_dim'] (m3d),
    control['k'] (sa/tca/mida). Passing the wrong key silently falls back to the
    library default, so these names matter."""
    k = min(10, max(2, d - 1)) if d > 1 else 1
    table = {
        "sa":    {"k": k},
        "tca":   {"k": k},
        "mida":  {"k": k},
        "coral": {"lambda": 1e-5},
        "gfk":   {"dim_subspace": k},
        "ot":    {"eps": 0.1},
        "m3d":   {"max_dim": max(5, k)},
        "pt":    {},
        "rd":    {},
        "art":   {},
    }
    return dict(table.get(method, {}))


# Methods that cannot run on the unlabeled synthetic sandbox (m3d needs source
# labels). DA-Lab pages exclude these from their method pickers.
DA_LAB_UNSUPPORTED = {"m3d"}


def _rotate(X: np.ndarray, angle: float) -> np.ndarray:
    d = X.shape[1]
    R = np.eye(d)
    c, s = np.cos(angle), np.sin(angle)
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return X @ R


# name -> generator(n, d, rng) -> (source, target). Single source of truth for
# both the Live DA Sandbox and the DA Method Sweep.
SHIFT_SCENARIOS = {
    "Different Means": lambda n, d, rng: (rng.normal(0, 1, (n, d)),
                                          rng.normal(0.8, 1, (n, d))),
    "Different SD":    lambda n, d, rng: (rng.normal(0, 1, (n, d)),
                                          rng.normal(0, 2, (n, d))),
    "Rotation":        lambda n, d, rng: (rng.normal(0, 1, (n, d)),
                                          _rotate(rng.normal(0, 1, (n, d)), np.pi / 6)),
    "Heavy-tail (t)":  lambda n, d, rng: (rng.standard_t(5, (n, d)),
                                          rng.standard_t(3, (n, d)) + 0.3),
    "Covariance shift": lambda n, d, rng: (
        rng.multivariate_normal(np.zeros(d), np.eye(d), n),
        rng.multivariate_normal(np.zeros(d), np.diag(np.linspace(0.5, 2.0, d)), n),
    ),
}

SCENARIO_PAIR_LABEL = {
    "Different Means":  ("N(0,1)", "N(0.8,1)"),
    "Different SD":     ("N(0,1)", "N(0,2)"),
    "Rotation":         ("X", "rotate(X, π/6)"),
    "Heavy-tail (t)":   ("t(5)", "t(3)+0.3"),
    "Covariance shift": ("I_d", "diag(0.5..2.0)"),
}

SCENARIO_TIPS = {
    "Different Means":  "Mean shifts ⇒ CORAL / PT excel (fast + effective).",
    "Different SD":     "Scale shifts ⇒ SA, ART, PT typically reach high Mahalanobis reduction.",
    "Rotation":         "Pure rotation ⇒ subspace methods (SA, GFK) realign the axes.",
    "Heavy-tail (t)":   "Outliers break many methods; TCA is usually most robust.",
    "Covariance shift": "SPD methods (PT, ART, Riemannian) are designed for this regime.",
}


def generate_shift(scenario: str, n: int, d: int, seed: int):
    """Reproducible (source, target) for a named shift scenario."""
    rng = np.random.default_rng(int(seed))
    return SHIFT_SCENARIOS[scenario](int(n), int(d), rng)


def mmd_distance(x, y) -> float:
    """MMD with the median-heuristic bandwidth (lazy da4bci import)."""
    from da4bci import compute_mmd, sigma_med
    try:
        sigma = sigma_med(x, y)
    except Exception:
        sigma = 1.0
    return float(compute_mmd(x, y, sigma))


def euclid_distance(x, y) -> float:
    """Mean pairwise Euclidean distance, with a pure-numpy fallback."""
    try:
        from da4bci import compute_distance_matrix
        D = compute_distance_matrix(x, y)
        return float(np.mean(D))
    except Exception:
        x_ = np.asarray(x); y_ = np.asarray(y)
        return float(np.mean(np.linalg.norm(
            x_[:, None, :] - y_[None, :, :], axis=-1)))


# ============================================================================
# Statistical helpers (additive — never replace the published model fits)
# ============================================================================

def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def cluster_bootstrap_ci(df: pd.DataFrame, stat_fn, cluster_col: str = "uid",
                         n_boot: int = 1000, seed: int = 0, ci: float = 95.0,
                         min_clusters: int = 3):
    """Subject-clustered bootstrap CI for an arbitrary statistic.

    Resamples whole clusters (subjects) with replacement — the correct unit of
    analysis for repeated within-subject measurements — and recomputes
    `stat_fn(resampled_df)`. Returns (point, lo, hi); lo/hi are NaN when there
    are too few clusters or too many failed refits. The point estimate is the
    statistic on the original data (never altered by the bootstrap).
    """
    point = _safe_stat(stat_fn, df)
    if df is None or df.empty or cluster_col not in df.columns:
        return (point, np.nan, np.nan)
    work = df.reset_index(drop=True)
    clusters = work[cluster_col].dropna().unique()
    if len(clusters) < min_clusters:
        return (point, np.nan, np.nan)
    idx_by_cluster = [work.index[work[cluster_col] == c].to_numpy()
                      for c in clusters]
    rng = np.random.default_rng(seed)
    n = len(clusters)
    stats = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        v = _safe_stat(stat_fn, work.iloc[idx])
        if not _is_nan(v):
            stats.append(v)
    if len(stats) < max(20, n_boot // 20):
        return (point, np.nan, np.nan)
    a = (100.0 - ci) / 2.0
    lo, hi = np.percentile(stats, [a, 100.0 - a])
    return (point, float(lo), float(hi))


def cluster_bootstrap_p(df: pd.DataFrame, stat_fn, cluster_col: str = "uid",
                        n_boot: int = 2000, seed: int = 0, min_clusters: int = 3):
    """Two-sided bootstrap p-value for H0: statistic == 0, via the subject
    cluster bootstrap. Returns (point, p) with p = 2·min(P(b≤0), P(b≥0))."""
    point = _safe_stat(stat_fn, df)
    if df is None or df.empty or cluster_col not in df.columns:
        return (point, np.nan)
    work = df.reset_index(drop=True)
    clusters = work[cluster_col].dropna().unique()
    if len(clusters) < min_clusters:
        return (point, np.nan)
    idx_by_cluster = [work.index[work[cluster_col] == c].to_numpy()
                      for c in clusters]
    rng = np.random.default_rng(seed)
    n = len(clusters)
    boots = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        v = _safe_stat(stat_fn, work.iloc[idx])
        if not _is_nan(v):
            boots.append(v)
    if len(boots) < max(20, n_boot // 20):
        return (point, np.nan)
    boots = np.asarray(boots)
    frac_le = float(np.mean(boots <= 0))
    frac_ge = float(np.mean(boots >= 0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return (point, p)


def _safe_stat(stat_fn, df):
    try:
        v = stat_fn(df)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def ols_slope(df: pd.DataFrame, x: str, y: str) -> float:
    """Plain OLS slope of y on x (NaN if <2 valid rows). Handy as a stat_fn."""
    g = df[[x, y]].dropna()
    if len(g) < 2:
        return np.nan
    return float(np.polyfit(g[x].to_numpy(), g[y].to_numpy(), 1)[0])


def mixedlm_pseudo_r2(result) -> tuple[float, float]:
    """Nakagawa & Schielzeth marginal / conditional pseudo-R² for a fitted
    statsmodels MixedLM (random-intercept). marginal = fixed-effect variance
    share; conditional adds the random-effect variance. Descriptive only."""
    try:
        fe_pred = np.asarray(result.model.exog) @ np.asarray(result.fe_params)
        var_f = float(np.var(fe_pred, ddof=1))
        var_a = float(np.trace(np.atleast_2d(np.asarray(result.cov_re))))
        var_e = float(result.scale)
        denom = var_f + var_a + var_e
        if denom <= 0:
            return (np.nan, np.nan)
        return (var_f / denom, (var_f + var_a) / denom)
    except Exception:
        return (np.nan, np.nan)


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-FDR adjusted p-values. Uses statsmodels when available, else a
    self-contained implementation. NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return out
    try:
        from statsmodels.stats.multitest import multipletests
        out[mask] = multipletests(p[mask], method="fdr_bh")[1]
        return out
    except Exception:
        pm = p[mask]
        order = np.argsort(pm)
        m = len(pm)
        ranked = pm[order] * m / (np.arange(m) + 1)
        # enforce monotonicity from the largest p downward
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        adj = np.clip(adj, 0, 1)
        res = np.empty(m)
        res[order] = adj
        out[mask] = res
        return out


# ── Significance / formatting badges (shared across claim pages) ──────────────

SIG_LEGEND = "●●● p<0.001 · ●● p<0.01 · ● p<0.05 · — ns"


def pvalue_badge(p) -> str:
    if _is_nan(p):
        return "—"
    return ("●●●" if p < 0.001 else
            "●●" if p < 0.01 else
            "●" if p < 0.05 else "—")


def ci_excludes_zero(lo, hi) -> bool:
    if _is_nan(lo) or _is_nan(hi):
        return False
    return (lo > 0) or (hi < 0)


def format_pvalue(p) -> str:
    if _is_nan(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3g}"


def format_coef(v, decimals: int = 4) -> str:
    if _is_nan(v):
        return "—"
    return f"{v:+.{decimals}f}"


def format_pct(v, decimals: int = 1) -> str:
    if _is_nan(v):
        return "—"
    return f"{v*100:+.{decimals}f}%"


# ============================================================================
# Context-filter chaining (used by cached claim helpers)
# ============================================================================

def apply_ctx_all(df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    """Apply the three global context filters (dataset → classifier → metric)
    in one call. Mirrors the per-filter helpers above."""
    ctx = ctx or get_ctx()
    out = apply_ctx_dataset(df, ctx)
    out = apply_ctx_classifier(out, ctx)
    out = apply_ctx_metric(out, ctx)
    return out
