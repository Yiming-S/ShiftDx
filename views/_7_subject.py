"""Page 7: Subject Explorer — per-subject drift + strategy-wise accuracy evolution."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    STRATEGY_LABEL, DA_LABEL,
    about_page, apply_ctx_classifier, apply_ctx_dataset,
    available_da_methods, download_bar, empty_state, feature_colors,
    get_ctx, is_retrain, strategy_colors, style_figure,
)


def render(store):
    ctx = get_ctx()
    fcolors = feature_colors()
    scolors = strategy_colors()

    st.header("Subject Explorer")
    st.caption(
        "Single-subject drift trajectory and strategy-wise accuracy evolution "
        "across sessions."
    )

    about_page(
        what_you_see=[
            "Top chart: MMD drift trajectory (session 0 → session k).",
            "Bottom chart: accuracy under No DA, chosen DA, and Retrain.",
            "Detailed per-session accuracy table with all DA methods.",
        ],
        how_to_read=[
            "Accuracy dropping while drift rises = classic drift scenario.",
            "Retrain line well above DA = DA is insufficient for this subject.",
        ],
        paper_ref="§5 subject-level view",
    )

    drift = apply_ctx_dataset(store.drift_df)
    eval_df = apply_ctx_dataset(store.eval_df)
    eval_df = apply_ctx_classifier(eval_df, ctx)
    if drift.empty or eval_df.empty:
        empty_state(
            "Missing drift or eval data",
            f"No data for dataset=`{ctx['dataset']}`, classifier=`{ctx['classifier']}`.",
            dataset=ctx["dataset"],
        )
        return

    # ── Page-local pickers ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    if ctx["dataset"] == "All":
        datasets = sorted(drift["dataset"].unique())
        ds_pick = c1.selectbox("Dataset", datasets)
    else:
        ds_pick = ctx["dataset"]
        c1.markdown(f"**Dataset:** `{ds_pick}`")

    subjects = sorted(drift.loc[drift["dataset"] == ds_pick, "subject"].unique())
    if not subjects:
        empty_state("No subjects", f"No subjects for dataset `{ds_pick}`.")
        return
    subject = c2.selectbox("Subject", subjects)

    features_avail = sorted(drift.loc[drift["dataset"] == ds_pick, "feature"].unique())
    feature = c3.selectbox(
        "Feature", features_avail,
        index=features_avail.index("CSP") if "CSP" in features_avail else 0,
    )

    subj_drift = drift[(drift["dataset"] == ds_pick) &
                       (drift["subject"] == subject) &
                       (drift["feature"] == feature)].sort_values("session_k")
    subj_eval = eval_df[(eval_df["dataset"] == ds_pick) &
                        (eval_df["subject"] == subject) &
                        (eval_df["feature"] == feature)]

    if subj_drift.empty or subj_eval.empty:
        empty_state("No data for this subject / feature",
                    "Pick a different (subject, feature) combination.")
        return

    # ── KPIs with deltas ────────────────────────────────────────────────────
    noda = subj_eval[(subj_eval["strategy"] == "train_once") &
                     (subj_eval["da"] == "none")]
    da_opts = available_da_methods(subj_eval) or ["sa"]
    default_da = ctx["da"] if ctx["da"] in da_opts else da_opts[0]
    pick = st.selectbox(
        "DA method to plot", da_opts, index=da_opts.index(default_da),
        format_func=lambda m: DA_LABEL.get(m, m),
        key="subject_da_method",
    )
    da_rows = subj_eval[(subj_eval["strategy"] == "train_once_da") &
                        (subj_eval["da"] == pick)]
    retr = is_retrain(subj_eval)

    noda_mean = noda["accuracy"].mean() if not noda.empty else np.nan
    da_mean = da_rows["accuracy"].mean() if not da_rows.empty else np.nan
    retr_mean = retr["accuracy"].mean() if not retr.empty else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean MMD", f"{subj_drift['dist_mmd'].mean():.3f}")
    k2.metric("No DA mean accuracy",
              f"{noda_mean:.3f}" if not np.isnan(noda_mean) else "---")
    k3.metric(f"{pick.upper()} mean accuracy",
              f"{da_mean:.3f}" if not np.isnan(da_mean) else "---",
              delta=(f"{(da_mean - noda_mean)*100:+.1f} pp"
                     if not np.isnan(da_mean) and not np.isnan(noda_mean) else None),
              help="Delta vs No DA.")
    k4.metric("Retrain mean accuracy",
              f"{retr_mean:.3f}" if not np.isnan(retr_mean) else "---",
              delta=(f"{(retr_mean - noda_mean)*100:+.1f} pp"
                     if not np.isnan(retr_mean) and not np.isnan(noda_mean) else None),
              help="Delta vs No DA.")

    # ── Drift + accuracy dual plot ──────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Subject {subject} · {feature}")
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("MMD drift (session 0 → session k)",
                            "Strategy-wise accuracy"),
        )

        fig.add_trace(go.Scatter(
            x=subj_drift["session_k"], y=subj_drift["dist_mmd"],
            mode="lines+markers",
            line=dict(color=fcolors.get(feature, "#4F46E5"), width=3),
            marker=dict(size=8), name="MMD", showlegend=False,
        ), row=1, col=1)

        for strat, label, dflt_da in [
            ("train_once", "No DA", "none"),
            ("train_once_da", f"DA ({pick.upper()})", pick),
            ("retrain", "Retrain", "none"),
        ]:
            if strat == "retrain":
                sub = is_retrain(subj_eval)
            else:
                sub = subj_eval[(subj_eval["strategy"] == strat) &
                                (subj_eval["da"] == dflt_da)]
            if sub.empty:
                continue
            # Average over any pooled dimension (e.g. multiple classifiers) so a
            # duplicate target_session index can't appear.
            agg = (sub.groupby("target_session")["accuracy"].mean()
                      .reset_index().sort_values("target_session"))
            fig.add_trace(go.Scatter(
                x=agg["target_session"], y=agg["accuracy"],
                mode="lines+markers", name=label,
                line=dict(color=scolors[STRATEGY_LABEL[strat]], width=3),
                marker=dict(size=8),
            ), row=2, col=1)

        fig.update_xaxes(title_text="session k", row=2, col=1)
        fig.update_yaxes(title_text="MMD", row=1, col=1)
        fig.update_yaxes(title_text="accuracy", row=2, col=1)
        fig.update_layout(legend=dict(orientation="h", y=-0.1))
        style_figure(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-session accuracy table (promoted from expander) ─────────────────
    with st.container(border=True):
        st.subheader("Per-session accuracy by strategy")
        # Aggregate by session (mean over any pooled classifiers) BEFORE the
        # joins — otherwise a duplicate target_session index makes the outer
        # joins blow up combinatorially (3^k rows).
        noda_s = (subj_eval[(subj_eval["strategy"] == "train_once") &
                            (subj_eval["da"] == "none")]
                  .groupby("target_session")["accuracy"].mean().rename("No DA"))
        pivot = noda_s.to_frame()
        for m in available_da_methods(subj_eval):
            s = (subj_eval[(subj_eval["strategy"] == "train_once_da") &
                           (subj_eval["da"] == m)]
                 .groupby("target_session")["accuracy"].mean().rename(m.upper()))
            pivot = pivot.join(s, how="outer")
        retr_s = (is_retrain(subj_eval)
                  .groupby("target_session")["accuracy"].mean().rename("Retrain"))
        pivot = pivot.join(retr_s, how="outer")
        pivot.index.name = "session_k"
        st.dataframe(
            pivot.style.format("{:.3f}", na_rep="---")
                 .background_gradient(cmap="Blues", axis=None, vmin=0.3, vmax=1.0),
            use_container_width=True,
        )
        download_bar(
            "subject_per_session", pivot.reset_index(),
            f"subject_{subject}_{feature}_per_session",
        )

    # ── Per-subject drift-vs-loss small multiples (all subjects) ─────────────
    with st.container(border=True):
        st.subheader("All subjects — drift vs accuracy (No DA)")
        st.caption(
            "Each panel is one subject: standardized drift vs baseline-centered "
            "accuracy under No-DA, with an OLS fit. Surfaces between-subject "
            "heterogeneity (a paper Limitation). Exploratory — per-panel fits "
            "carry no multiplicity claim."
        )
        merged = apply_ctx_dataset(store.merged_df)
        merged = apply_ctx_classifier(merged, ctx)
        sm = merged[(merged["dataset"] == ds_pick) &
                    (merged["feature"] == feature) &
                    (merged["strategy"] == "train_once") &
                    (merged["da"] == "none")].copy()
        sm = sm.dropna(subset=["drift_z", "acc_centered"])
        if sm.empty or sm["subject"].nunique() < 2:
            st.info("Not enough No-DA rows with drift_z / acc_centered for a grid.")
        else:
            sm["subject"] = sm["subject"].astype(str)
            # Cap the number of panels: large datasets (e.g. 62-subject
            # Stieger2021) would otherwise build a huge, memory-heavy grid.
            all_subs = sorted(sm["subject"].unique(), key=lambda s: int(s))
            MAX_PANELS = 12
            if len(all_subs) > MAX_PANELS:
                picks = st.multiselect(
                    "Subjects to show", all_subs, default=all_subs[:MAX_PANELS],
                    key="subj_sm_pick",
                    help=f"Capped at {MAX_PANELS} panels for performance.",
                ) or all_subs[:MAX_PANELS]
            else:
                picks = all_subs
            sm = sm[sm["subject"].isin(picks)]
            # Manual subplot grid with np.polyfit lines — avoids plotly's
            # statsmodels-backed trendline (heavy / thread-fragile under load).
            wrap = max(1, min(4, len(picks)))
            n_rows = int(np.ceil(len(picks) / wrap))
            col_feat = fcolors.get(feature, "#4F46E5")
            fig_sm = make_subplots(
                rows=n_rows, cols=wrap,
                subplot_titles=[f"S{p}" for p in picks],
                vertical_spacing=0.12, horizontal_spacing=0.06,
            )
            for i, subj_id in enumerate(picks):
                rr, cc = i // wrap + 1, i % wrap + 1
                g = sm[sm["subject"] == subj_id]
                fig_sm.add_trace(go.Scatter(
                    x=g["drift_z"], y=g["acc_centered"], mode="markers",
                    marker=dict(color=col_feat, size=5, opacity=0.6),
                    showlegend=False,
                ), row=rr, col=cc)
                if len(g) >= 2:
                    coef = np.polyfit(g["drift_z"], g["acc_centered"], 1)
                    xs = np.linspace(g["drift_z"].min(), g["drift_z"].max(), 20)
                    fig_sm.add_trace(go.Scatter(
                        x=xs, y=np.polyval(coef, xs), mode="lines",
                        line=dict(color="#EF4444", width=2), showlegend=False,
                    ), row=rr, col=cc)
            style_figure(fig_sm, height=max(280, 200 * n_rows))
            st.plotly_chart(fig_sm, use_container_width=True)
