"""Page 11: Online Drift Detection Demo — Page-Hinkley on per-subject accuracy series."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import STRATEGY_COLORS, filter_by_dataset, style_figure

try:
    from da4bci import ph_init, ph_update
    HAS_DA4BCI = True
except Exception as _exc:
    HAS_DA4BCI = False
    _DA4BCI_ERR = str(_exc)


def _run_page_hinkley(series: np.ndarray, delta: float, lam: float, alpha: float):
    """Sequentially feed `series` into Page-Hinkley and record state + triggers."""
    state = ph_init(delta=delta, lambda_=lam, alpha=alpha)
    means, cums, mins, triggers = [], [], [], []
    for x in series:
        out = ph_update(state, x)
        state = out["state"]
        means.append(state["mean"])
        cums.append(state["cum"])
        mins.append(state["min_cum"])
        triggers.append(out["change"])
    return np.array(means), np.array(cums), np.array(mins), np.array(triggers)


def render(store, dataset):
    st.header("Drift Detection Demo")
    st.caption(
        "Feed one subject's per-session accuracy into the Page-Hinkley detector "
        "from DA4BCI and mark the earliest session at which a retraining trigger fires. "
        "Links the paper's static retraining-gap analysis to an online deployment rule."
    )

    if not HAS_DA4BCI:
        st.error(f"DA4BCI not importable: {_DA4BCI_ERR}")
        return

    eval_df = filter_by_dataset(store.eval_df, dataset)
    if eval_df.empty:
        st.warning("No evaluation data.")
        return

    # ── Subject picker ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    if dataset == "All":
        ds_pick = c1.selectbox("Dataset", sorted(eval_df["dataset"].unique()))
    else:
        ds_pick = dataset
        c1.markdown(f"**Dataset:** `{ds_pick}`")
    subj_df = eval_df[eval_df["dataset"] == ds_pick]
    subject = c2.selectbox("Subject", sorted(subj_df["subject"].unique()))
    feature = c3.selectbox("Feature", sorted(subj_df["feature"].unique()),
                            index=0)
    from utils import available_da_methods, DA_LABEL
    da_opts = available_da_methods(subj_df)
    stream_options = ["No DA"] + [f"DA · {DA_LABEL.get(m, m)}" for m in da_opts] + ["Retrain"]
    stream_choice = c4.selectbox("Stream", stream_options, index=0)

    if stream_choice == "No DA":
        series_df = subj_df[(subj_df["subject"] == subject) & (subj_df["feature"] == feature)
                             & (subj_df["strategy"] == "train_once") & (subj_df["da"] == "none")]
    elif stream_choice == "Retrain":
        series_df = subj_df[(subj_df["subject"] == subject) & (subj_df["feature"] == feature)
                             & (subj_df["strategy"] == "retrain")
                             & (subj_df["ref_session"] == subj_df["target_session"])]
    else:
        idx = stream_options.index(stream_choice) - 1  # offset by "No DA"
        da_pick = da_opts[idx]
        series_df = subj_df[(subj_df["subject"] == subject) & (subj_df["feature"] == feature)
                             & (subj_df["strategy"] == "train_once_da") & (subj_df["da"] == da_pick)]

    series_df = series_df.sort_values("target_session")
    if series_df.empty:
        st.warning("No series for selected combo.")
        return

    # Page-Hinkley fires on *increases* in the input signal; to detect accuracy
    # *drops*, feed in the negated accuracy or equivalently the "loss" = 1 - acc.
    losses = 1.0 - series_df["accuracy"].to_numpy()
    sessions = series_df["target_session"].to_numpy()

    # ── Detector params ─────────────────────────────────────────────────────
    st.markdown("**Page-Hinkley parameters**")
    p1, p2, p3 = st.columns(3)
    delta = p1.slider("δ (min magnitude)", 0.0, 0.1, 0.005, 0.001, format="%.3f")
    lam = p2.slider("λ (threshold)", 0.0, 2.0, 0.3, 0.05)
    alpha = p3.slider("α (EMA decay)", 0.5, 0.999, 0.9, 0.01)

    means, cums, mins, triggers = _run_page_hinkley(losses, delta, lam, alpha)
    first_trigger = int(np.argmax(triggers)) if triggers.any() else None

    # ── KPI ─────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Sessions observed", len(sessions))
    k2.metric("Trigger events", int(triggers.sum()))
    k3.metric("First trigger @ session",
              int(sessions[first_trigger]) if first_trigger is not None else "—")

    # ── Plot ────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"S{subject} · {feature} · {stream_choice}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Accuracy", "Page-Hinkley CUSUM"))
        fig.add_trace(go.Scatter(
            x=sessions, y=1 - losses, mode="lines+markers", name="accuracy",
            line=dict(color="#4F46E5", width=3),
            marker=dict(size=8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sessions, y=cums - mins, mode="lines+markers",
            line=dict(color="#EF4444", width=2), name="cum − min",
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=lam, line_dash="dash", line_color="#64748B", row=2, col=1,
                      annotation_text=f"λ={lam:.2f}")
        trigger_idx = np.where(triggers)[0]
        if len(trigger_idx) > 0:
            for i in trigger_idx:
                fig.add_vline(x=sessions[i], line_color="#EF4444", line_dash="dot", opacity=0.4)
        fig.update_xaxes(title_text="session_k", row=2, col=1)
        fig.update_yaxes(title_text="accuracy", row=1, col=1)
        fig.update_yaxes(title_text="PH statistic", row=2, col=1)
        style_figure(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Once the PH statistic exceeds λ, a retraining trigger fires. "
        "Compare the firing session against the paper's high-drift regime "
        "(upper Q75 of `drift_z`) to see if online detection catches drift "
        "before accuracy noticeably degrades."
    )
