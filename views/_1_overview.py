"""Page 1: Dataset Overview."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    DATASET_META, DATASET_COLORS, FEATURES,
    filter_by_dataset, style_figure,
)


def render(store, dataset):
    st.header("ShiftDx — Dataset Overview")
    st.markdown(
        "Interactive diagnostic dashboard for cross-session drift in MI-EEG BCIs. "
        "Based on Shen & Degras: **Drift Diagnostics, Adaptation, and Recalibration "
        "in Multi-Session Motor-Imagery EEG**."
    )

    drift = filter_by_dataset(store.drift_df, dataset)
    eval_df = filter_by_dataset(store.eval_df, dataset)
    merged = filter_by_dataset(store.merged_df, dataset)

    if drift.empty:
        st.warning("No drift data loaded. Check `data/` directory.")
        return

    # ── KPI cards ────────────────────────────────────────────────────────────
    n_datasets = drift["dataset"].nunique()
    n_subjects = drift.groupby("dataset")["subject"].nunique().sum()
    n_obs_noda = int(((eval_df["strategy"] == "train_once") & (eval_df["da"] == "none")).sum()) if not eval_df.empty else 0
    n_obs_total = len(eval_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", n_datasets)
    c2.metric("Subjects", int(n_subjects))
    c3.metric("No-DA rows", f"{n_obs_noda:,}")
    c4.metric("Total eval rows", f"{n_obs_total:,}")

    # ── Paper anchor ─────────────────────────────────────────────────────────
    st.success(
        "**Four reduced-form claims** anchor this dashboard: "
        "(1) drift predicts loss · (2) DA decomposes into level shift + slope change · "
        "(3) retraining gap is positive in high-drift regimes · "
        "(4) feature robustness requires a ceiling anchor."
    )

    st.markdown("")

    # ── Dataset summary + drift summary ──────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("Datasets")
            rows = []
            for ds in sorted(drift["dataset"].unique()):
                if dataset != "All" and ds != dataset:
                    continue
                meta = DATASET_META.get(ds, {"channels": "?", "subjects": "?",
                                              "sessions": "?", "role": ""})
                sub = drift[drift["dataset"] == ds]
                rows.append({
                    "Dataset": ds,
                    "Channels": meta["channels"],
                    "Subjects": sub["subject"].nunique() if not sub.empty else meta["subjects"],
                    "Sessions": meta["sessions"],
                    "Role": meta["role"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_right:
        with st.container(border=True):
            st.subheader("Mean MMD drift by dataset x feature")
            if "dist_mmd" in drift.columns:
                agg = (
                    drift.groupby(["dataset", "feature"])["dist_mmd"]
                    .mean()
                    .reset_index()
                    .pivot(index="dataset", columns="feature", values="dist_mmd")
                    .reindex(columns=[f for f in FEATURES if f in drift["feature"].unique()])
                )
                st.dataframe(
                    agg.style.format("{:.3f}").background_gradient(cmap="Blues"),
                    use_container_width=True,
                )
            else:
                st.info("No `dist_mmd` column in drift_df.")

    # ── MMD distribution by dataset (feature=CSP) ────────────────────────────
    with st.container(border=True):
        st.subheader("MMD distribution (CSP features)")
        csp = drift[drift["feature"] == "CSP"]
        if csp.empty:
            st.info("No CSP rows in drift_df.")
        else:
            fig = px.box(
                csp, x="dataset", y="dist_mmd", color="dataset",
                color_discrete_map=DATASET_COLORS, points="outliers",
                category_orders={"dataset": sorted(csp["dataset"].unique())},
            )
            fig.update_layout(showlegend=False, yaxis_title="MMD(session 0, session k)")
            style_figure(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)

    # ── Strategy coverage ────────────────────────────────────────────────────
    if not eval_df.empty:
        with st.expander("Strategy x DA coverage"):
            cov = eval_df.groupby(["strategy", "da"]).size().reset_index(name="rows")
            st.dataframe(cov, use_container_width=True, hide_index=True)

    # ── Data freshness ───────────────────────────────────────────────────────
    with st.expander("Data freshness"):
        import os
        from datetime import datetime
        try:
            files = [
                os.path.join(store.data_dir, f)
                for f in os.listdir(store.data_dir) if f.endswith(".csv")
            ]
            if files:
                newest = max(os.path.getmtime(p) for p in files)
                st.markdown(f"Newest CSV: **{datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}**")
                st.caption(f"Files: {len(files)}")
        except Exception as exc:
            st.caption(f"Could not stat files: {exc}")
