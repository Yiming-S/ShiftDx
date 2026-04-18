"""Page 7: Subject Explorer — per-subject drift + strategy-wise accuracy evolution."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    FEATURE_COLORS, STRATEGY_COLORS, STRATEGY_LABEL,
    filter_by_dataset, style_figure,
)


def render(store, dataset):
    st.header("Subject Explorer")
    st.caption("Single-subject drift trajectory and strategy-wise accuracy evolution across sessions.")

    drift = filter_by_dataset(store.drift_df, dataset)
    eval_df = filter_by_dataset(store.eval_df, dataset)
    if drift.empty or eval_df.empty:
        st.warning("Missing drift or eval data.")
        return

    # ── Subject picker ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    if dataset == "All":
        datasets = sorted(drift["dataset"].unique())
        ds_pick = c1.selectbox("Dataset", datasets)
    else:
        ds_pick = dataset
        c1.markdown(f"**Dataset:** `{ds_pick}`")

    subjects = sorted(drift.loc[drift["dataset"] == ds_pick, "subject"].unique())
    subject = c2.selectbox("Subject", subjects)

    features_avail = sorted(drift.loc[(drift["dataset"] == ds_pick), "feature"].unique())
    feature = c3.selectbox("Feature", features_avail,
                           index=features_avail.index("CSP") if "CSP" in features_avail else 0)

    from utils import available_classifiers, CLASSIFIER_LABEL
    clf_opts = available_classifiers(eval_df)
    if len(clf_opts) > 1:
        clf_pick = st.radio("Classifier", clf_opts, horizontal=True, index=0,
                             format_func=lambda x: CLASSIFIER_LABEL.get(x, x))
        eval_df = eval_df[eval_df["classifier"] == clf_pick].copy()

    subj_drift = drift[(drift["dataset"] == ds_pick) & (drift["subject"] == subject)
                       & (drift["feature"] == feature)].sort_values("session_k")
    subj_eval = eval_df[(eval_df["dataset"] == ds_pick) & (eval_df["subject"] == subject)
                        & (eval_df["feature"] == feature)]

    if subj_drift.empty or subj_eval.empty:
        st.warning("No data for selected subject / feature combination.")
        return

    # ── KPIs ────────────────────────────────────────────────────────────────
    noda = subj_eval[(subj_eval["strategy"] == "train_once") & (subj_eval["da"] == "none")]
    sa = subj_eval[(subj_eval["strategy"] == "train_once_da") & (subj_eval["da"] == "sa")]
    retr = subj_eval[(subj_eval["strategy"] == "retrain")
                     & (subj_eval["ref_session"] == subj_eval["target_session"])]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean MMD", f"{subj_drift['dist_mmd'].mean():.3f}")
    k2.metric("No-DA mean acc", f"{noda['accuracy'].mean():.3f}" if not noda.empty else "---")
    k3.metric("SA mean acc", f"{sa['accuracy'].mean():.3f}" if not sa.empty else "---")
    k4.metric("Retrain mean acc", f"{retr['accuracy'].mean():.3f}" if not retr.empty else "---")

    # ── Drift + accuracy dual plot ──────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Subject {subject} · {feature}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=("MMD drift (session 0 → session k)",
                                             "Strategy-wise accuracy"))

        fig.add_trace(go.Scatter(
            x=subj_drift["session_k"], y=subj_drift["dist_mmd"],
            mode="lines+markers", line=dict(color=FEATURE_COLORS.get(feature, "#4F46E5"), width=3),
            marker=dict(size=8), name="MMD", showlegend=False,
        ), row=1, col=1)

        from utils import available_da_methods, DA_LABEL
        da_opts = available_da_methods(subj_eval) or ["sa"]
        pick = st.selectbox("DA method shown", da_opts,
                             format_func=lambda m: DA_LABEL.get(m, m),
                             key="subject_da_method")
        for strat, label, dflt_da in [
            ("train_once", "No DA", "none"),
            ("train_once_da", f"DA ({pick.upper()})", pick),
            ("retrain", "Retrain", "none"),
        ]:
            if strat == "retrain":
                sub = subj_eval[(subj_eval["strategy"] == "retrain")
                                & (subj_eval["ref_session"] == subj_eval["target_session"])]
                sub = sub.sort_values("target_session")
            else:
                sub = subj_eval[(subj_eval["strategy"] == strat) & (subj_eval["da"] == dflt_da)]
                sub = sub.sort_values("target_session")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["target_session"], y=sub["accuracy"],
                mode="lines+markers", name=label,
                line=dict(color=STRATEGY_COLORS[STRATEGY_LABEL[strat]], width=3),
                marker=dict(size=8),
            ), row=2, col=1)

        fig.update_xaxes(title_text="session k", row=2, col=1)
        fig.update_yaxes(title_text="MMD", row=1, col=1)
        fig.update_yaxes(title_text="accuracy", row=2, col=1)
        fig.update_layout(legend=dict(orientation="h", y=-0.1))
        style_figure(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-session strategy table ──────────────────────────────────────────
    with st.expander("Per-session accuracy table"):
        noda_s = (subj_eval[(subj_eval["strategy"] == "train_once") & (subj_eval["da"] == "none")]
                  .set_index("target_session")["accuracy"].rename("No DA"))
        pivot = noda_s.to_frame()
        from utils import available_da_methods
        for m in available_da_methods(subj_eval):
            s = (subj_eval[(subj_eval["strategy"] == "train_once_da") & (subj_eval["da"] == m)]
                 .set_index("target_session")["accuracy"].rename(m.upper()))
            pivot = pivot.join(s, how="outer")
        retr_s = (subj_eval[(subj_eval["strategy"] == "retrain")
                            & (subj_eval["ref_session"] == subj_eval["target_session"])]
                  .set_index("target_session")["accuracy"].rename("Retrain"))
        pivot = pivot.join(retr_s, how="outer")
        pivot.index.name = "session_k"
        st.dataframe(
            pivot.style.format("{:.3f}", na_rep="---")
                 .background_gradient(cmap="Blues", axis=None, vmin=0.3, vmax=1.0),
            use_container_width=True,
        )
