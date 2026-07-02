"""Page 2: Drift Trajectory."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    DISTANCE_LABEL,
    about_page, apply_ctx_dataset, available_distance_metrics,
    dataset_colors, download_bar, empty_state, get_ctx, style_figure,
)


def _non_monotonic_subjects(drift: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Return table of (dataset, subject, feature, drops) for non-monotonic trajectories."""
    if drift.empty:
        return pd.DataFrame()
    rows = []
    for (ds, subj, feat), grp in drift.sort_values("session_k").groupby(
        ["dataset", "subject", "feature"]
    ):
        diffs = grp[metric_col].diff().dropna()
        drops = int((diffs < 0).sum())
        if drops > 0:
            rows.append({"dataset": ds, "subject": subj, "feature": feat, "drops": drops})
    return pd.DataFrame(rows)


def render(store):
    ctx = get_ctx()
    dscolors = dataset_colors()

    st.header("Drift Trajectory")
    st.caption(
        "Distance from session 0 to each subsequent session, per subject. "
        "Non-monotonic trajectories indicate session-to-session variability "
        "rather than progressive degradation (paper §5.1)."
    )

    about_page(
        what_you_see=[
            "Per-subject distance trajectories with the dataset mean overlaid.",
            "Subject × session heatmap showing where drift concentrates.",
        ],
        how_to_read=[
            "Flat / low trajectory ⇒ subject is stable over sessions.",
            "Trajectories that jump up *and down* indicate session variability rather than drift.",
        ],
        paper_ref="§5.1",
        key_terms=[
            ("session k", "the k-th session of a subject; k=0 is calibration."),
            ("Non-monotonic", "a trajectory with at least one session-over-session drop."),
        ],
    )

    drift = apply_ctx_dataset(store.drift_df)
    if drift.empty:
        empty_state(
            "No drift data for this selection",
            f"No `drift_trajectories_*.csv` loaded for dataset `{ctx['dataset']}`. "
            "Check the Overview page for loading status.",
            dataset=ctx["dataset"],
        )
        return

    # ── Page-local controls ─────────────────────────────────────────────────
    metric_opts = available_distance_metrics(drift)
    if not metric_opts:
        empty_state("No distance-metric columns",
                    "The loaded drift CSV has no `dist_*` columns.")
        return

    # Default to the global metric if it exists in this dataset; otherwise first available.
    default_metric = ctx["metric"] if ctx["metric"] in metric_opts else metric_opts[0]

    c1, c2, c3 = st.columns([2, 2, 1])
    metric = c1.radio(
        "Distance metric (this page)", metric_opts, horizontal=True,
        index=metric_opts.index(default_metric),
        format_func=lambda m: DISTANCE_LABEL.get(m, m),
        help="Drift distance to plot. The global metric from the sidebar is used by default.",
    )
    features_present = sorted(drift["feature"].unique())
    feature = c2.radio("Feature", features_present, horizontal=True, index=0)
    top_k_noisy = c3.number_input(
        "Top-K noisy subjects",
        min_value=0, max_value=100, value=0, step=5,
        help="0 = show all subjects. >0 = show only the K subjects with the most "
             "non-monotonic drops.",
    )

    metric_name = DISTANCE_LABEL.get(metric, metric)
    subdf = drift[drift["feature"] == feature]

    # ── KPI row ─────────────────────────────────────────────────────────────
    nm_tbl = _non_monotonic_subjects(subdf, metric)
    nm_pct = len(nm_tbl) / subdf.groupby(
        ["dataset", "subject", "feature"]
    ).ngroups if not subdf.empty else 0.0
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Mean {metric_name}", f"{subdf[metric].mean():.3f}")
    k2.metric(f"SD {metric_name}", f"{subdf[metric].std():.3f}")
    k3.metric("Non-monotonic %", f"{nm_pct*100:.0f}%",
              help="Share of (subject, feature) trajectories with at least one "
                   "session-over-session drop.")

    # Optional top-K filter (vectorized via a MultiIndex membership test)
    if top_k_noisy > 0 and not nm_tbl.empty:
        worst = (nm_tbl.sort_values("drops", ascending=False)
                       .head(int(top_k_noisy))[["dataset", "subject"]]
                       .drop_duplicates())
        keep_idx = pd.MultiIndex.from_frame(worst)
        mask = pd.MultiIndex.from_frame(subdf[["dataset", "subject"]]).isin(keep_idx)
        subdf = subdf[mask]

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
                        mode="lines",
                        line=dict(width=1, color="rgba(100,116,139,0.35)"),
                        showlegend=False,
                        hovertemplate=(f"S{subj}<br>k=%{{x}}<br>"
                                       f"{metric_name}=%{{y:.3f}}<extra></extra>"),
                    ))
                mean_curve = ds_df.groupby("session_k")[metric].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=mean_curve["session_k"], y=mean_curve[metric],
                    mode="lines+markers",
                    line=dict(width=3, color=dscolors.get(ds, "#4F46E5")),
                    name=f"{ds} mean", marker=dict(size=8),
                ))
                fig.update_layout(
                    xaxis_title="session_k",
                    yaxis_title=f"{metric_name}(session 0, session k)",
                    title=ds, legend=dict(orientation="h", y=-0.15),
                )
                style_figure(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)
        download_bar("drift_traj_curves", subdf, f"drift_trajectories_{feature}_{metric}")

    # ── Subject × session heatmap ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Subject × session heatmap ({metric_name})")
        st.caption("Rows = subject, columns = session_k. Colour intensity = drift magnitude.")
        if ctx["dataset"] == "All" and len(datasets_here) > 1:
            # Horizontal facet when multiple datasets are loaded in "All" mode
            cols = st.columns(min(len(datasets_here), 3))
            for idx, ds in enumerate(datasets_here):
                with cols[idx % len(cols)]:
                    ds_df = subdf[subdf["dataset"] == ds]
                    pivot = ds_df.pivot_table(
                        index="subject", columns="session_k",
                        values=metric, aggfunc="mean",
                    ).sort_index()
                    if pivot.empty:
                        continue
                    fig_h = px.imshow(
                        pivot.values,
                        x=[f"k={c}" for c in pivot.columns],
                        y=[f"S{r}" for r in pivot.index],
                        color_continuous_scale="YlOrRd",
                        aspect="auto", text_auto=".3f",
                        labels=dict(color=metric_name),
                        title=ds,
                    )
                    fig_h.update_layout(
                        xaxis_title="target session", yaxis_title="subject",
                        coloraxis_colorbar=dict(title=metric_name),
                    )
                    style_figure(fig_h, height=max(220, 24 * len(pivot) + 100))
                    st.plotly_chart(fig_h, use_container_width=True)
        else:
            for ds in datasets_here:
                ds_df = subdf[subdf["dataset"] == ds]
                pivot = ds_df.pivot_table(
                    index="subject", columns="session_k",
                    values=metric, aggfunc="mean",
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

    # ── Non-monotonic subjects ──────────────────────────────────────────────
    if not nm_tbl.empty:
        with st.expander(f"Subjects with non-monotonic {metric_name} trajectories "
                         f"({len(nm_tbl)} rows)"):
            st.dataframe(
                nm_tbl.sort_values("drops", ascending=False),
                use_container_width=True, hide_index=True,
            )
            download_bar("drift_nm_subjects", nm_tbl, f"non_monotonic_{metric}")

    # ── Drift summary table ─────────────────────────────────────────────────
    with st.expander("Drift summary by dataset"):
        summary = (
            subdf.groupby("dataset")[metric]
            .agg(["count", "mean", "std", "min", "max"])
            .round(3)
        )
        st.dataframe(summary, use_container_width=True)
        download_bar("drift_summary", summary.reset_index(), f"drift_summary_{metric}")

    # ── Drift forecast (experimental) ────────────────────────────────────────
    with st.expander("🔮 Drift forecast (experimental — linear extrapolation)"):
        st.caption(
            "Linear continuation of the observed drift trajectory. **Illustrative "
            "only**: it assumes drift keeps growing at the same rate, which short "
            "multi-session recordings cannot validate. No accuracy projection is "
            "made (a borrowed pooled slope is not a sound per-subject forecast)."
        )
        horizon = st.slider("Forecast horizon (extra sessions)", 1, 10, 3, 1,
                            key="drift_fc_h")
        datasets_fc = sorted(subdf["dataset"].unique())
        fig_fc = go.Figure()
        rows = []
        for ds in datasets_fc:
            ds_df = subdf[subdf["dataset"] == ds]
            if ds_df.empty:
                continue
            q75 = ds_df[metric].quantile(0.75)
            mean_curve = ds_df.groupby("session_k")[metric].mean()
            if mean_curve.index.nunique() >= 3:
                xx = mean_curve.index.to_numpy(dtype=float)
                yy = mean_curve.to_numpy(dtype=float)
                slope, intercept = np.polyfit(xx, yy, 1)
                xf = np.arange(xx.max(), xx.max() + horizon + 1)
                yf = slope * xf + intercept
                col = dscolors.get(ds, "#4F46E5")
                fig_fc.add_trace(go.Scatter(
                    x=xx, y=yy, mode="lines+markers", name=f"{ds} observed",
                    line=dict(color=col, width=3),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=xf, y=yf, mode="lines", name=f"{ds} forecast",
                    line=dict(color=col, width=2, dash="dash"),
                ))
            for subj, g in ds_df.groupby("subject"):
                g = g.sort_values("session_k")
                if g["session_k"].nunique() < 3:
                    continue
                x = g["session_k"].to_numpy(dtype=float)
                y = g[metric].to_numpy(dtype=float)
                s, b = np.polyfit(x, y, 1)
                cross = np.nan
                if s > 1e-9:
                    s_cross = (q75 - b) / s
                    if s_cross > x.max():
                        cross = s_cross - x.max()
                rows.append({
                    "dataset": ds, "subject": subj,
                    "slope/session": s, "current": y[-1],
                    f"forecast +{horizon}": s * (x.max() + horizon) + b,
                    "sessions to Q75": cross,
                })
        if fig_fc.data:
            fig_fc.add_hline(y=subdf[metric].quantile(0.75), line_dash="dot",
                             line_color="#EF4444",
                             annotation_text="Q75 (high-drift)")
            fig_fc.update_layout(xaxis_title="session_k",
                                 yaxis_title=metric_name,
                                 legend=dict(orientation="h", y=-0.2))
            style_figure(fig_fc, height=360)
            st.plotly_chart(fig_fc, use_container_width=True)
        if rows:
            fdf = pd.DataFrame(rows)
            st.dataframe(
                fdf.style.format({
                    "slope/session": "{:+.4f}", "current": "{:.3f}",
                    f"forecast +{horizon}": "{:.3f}", "sessions to Q75": "{:.1f}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
            download_bar("drift_forecast", fdf, f"drift_forecast_{feature}_{metric}")
        else:
            st.info("Need ≥3 sessions per subject to fit a trend.")
