"""Page 2: Drift Trajectory."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    DATASET_COLORS, DISTANCE_LABEL,
    available_distance_metrics, filter_by_dataset, style_figure,
)


def _non_monotonic_pct(drift: pd.DataFrame, metric_col: str) -> float:
    """Fraction of (subject, feature) groups whose metric is NOT monotonically increasing."""
    if drift.empty:
        return 0.0
    bad = 0
    total = 0
    for _, grp in drift.sort_values("session_k").groupby(["dataset", "subject", "feature"]):
        total += 1
        diffs = grp[metric_col].diff().dropna()
        if (diffs < 0).any():
            bad += 1
    return bad / total if total else 0.0


def render(store, dataset):
    st.header("Drift Trajectory")
    st.caption(
        "Distance from session 0 to each subsequent session, per subject. "
        "Non-monotonic trajectories indicate session-to-session variability "
        "rather than progressive degradation (paper §5.1)."
    )

    drift = filter_by_dataset(store.drift_df, dataset)
    if drift.empty:
        st.warning("No drift data.")
        return

    # Metric + feature selectors
    metric_opts = available_distance_metrics(drift)
    if not metric_opts:
        st.error("No distance-metric columns found in drift data.")
        return

    c1, c2 = st.columns(2)
    metric = c1.radio(
        "Distance metric", metric_opts, horizontal=True, index=0,
        format_func=lambda m: DISTANCE_LABEL.get(m, m),
    )
    features_present = sorted(drift["feature"].unique())
    feature = c2.radio("Feature", features_present, horizontal=True, index=0)

    metric_name = DISTANCE_LABEL.get(metric, metric)
    subdf = drift[drift["feature"] == feature]

    # ── KPI ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Mean {metric_name}", f"{subdf[metric].mean():.3f}")
    k2.metric(f"SD {metric_name}",   f"{subdf[metric].std():.3f}")
    k3.metric("Non-monotonic %", f"{_non_monotonic_pct(subdf, metric)*100:.0f}%")

    # ── Per-subject curves + feature-mean envelope ──────────────────────────
    with st.container(border=True):
        st.subheader(f"Per-subject {metric_name} trajectory ({feature})")
        datasets_here = sorted(subdf["dataset"].unique())
        tabs = st.tabs(datasets_here) if len(datasets_here) > 1 else [st.container()]
        for tab, ds in zip(tabs, datasets_here):
            with tab:
                ds_df = subdf[subdf["dataset"] == ds]
                fig = go.Figure()
                for subj, g in ds_df.groupby("subject"):
                    g = g.sort_values("session_k")
                    fig.add_trace(go.Scatter(
                        x=g["session_k"], y=g[metric],
                        mode="lines", line=dict(width=1, color="rgba(100,116,139,0.35)"),
                        showlegend=False,
                        hovertemplate=(f"S{subj}<br>k=%{{x}}<br>"
                                       f"{metric_name}=%{{y:.3f}}<extra></extra>"),
                    ))
                mean_curve = ds_df.groupby("session_k")[metric].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=mean_curve["session_k"], y=mean_curve[metric],
                    mode="lines+markers",
                    line=dict(width=3, color=DATASET_COLORS.get(ds, "#4F46E5")),
                    name=f"{ds} mean", marker=dict(size=8),
                ))
                fig.update_layout(
                    xaxis_title="session_k",
                    yaxis_title=f"{metric_name}(session 0, session k)",
                    title=ds, legend=dict(orientation="h", y=-0.15),
                )
                style_figure(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)

    # ── Subject × session heatmap ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Subject × session heatmap ({metric_name})")
        st.caption("Rows = subject, columns = session_k. Color intensity = drift magnitude.")
        import plotly.express as px
        for ds in datasets_here:
            ds_df = subdf[subdf["dataset"] == ds]
            pivot = ds_df.pivot_table(
                index="subject", columns="session_k", values=metric, aggfunc="mean",
            ).sort_index()
            if pivot.empty:
                continue
            if len(datasets_here) > 1:
                st.markdown(f"**{ds}**")
            fig_h = px.imshow(
                pivot.values,
                x=[f"k={c}" for c in pivot.columns],
                y=[f"S{r}" for r in pivot.index],
                color_continuous_scale="YlOrRd",
                aspect="auto", text_auto=".3f",
                labels=dict(color=metric_name),
            )
            fig_h.update_layout(
                xaxis_title="target session", yaxis_title="subject",
                coloraxis_colorbar=dict(title=metric_name),
            )
            style_figure(fig_h, height=max(260, 32 * len(pivot) + 120))
            st.plotly_chart(fig_h, use_container_width=True)

    # ── Drift summary table ─────────────────────────────────────────────────
    with st.expander("Drift summary by dataset"):
        summary = (
            subdf.groupby("dataset")[metric]
            .agg(["count", "mean", "std", "min", "max"])
            .round(3)
        )
        st.dataframe(summary, use_container_width=True)
