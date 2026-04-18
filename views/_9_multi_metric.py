"""Page 9: Multi-Metric Drift Panel.

Shows the full set of distance metrics produced by the build pipeline
(MMD, Energy, Wasserstein, Mahalanobis, Euclidean), lets the user pick
one to re-define drift_z, and re-fits Claim 1 to test robustness of the
drift → loss relationship against metric choice.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.formula.api as smf

from utils import (
    DATASET_COLORS, DISTANCE_LABEL, DISTANCE_DESCRIPTION,
    available_distance_metrics, filter_by_dataset, style_figure,
)


def _zscore_by_block(df: pd.DataFrame, col: str, block: list[str]) -> pd.Series:
    return df.groupby(block)[col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    )


def render(store, dataset):
    st.header("Multi-Metric Drift Panel")
    st.caption(
        "Compare all distance metrics side-by-side. "
        "Paper Limitation 4: MMD and Mahalanobis/AIRM are complementary, not interchangeable — "
        "this page extends that to the full DA4BCI suite."
    )

    drift = filter_by_dataset(store.drift_df, dataset)
    merged = filter_by_dataset(store.merged_df, dataset)
    if drift.empty:
        st.warning("No drift data.")
        return

    metrics = available_distance_metrics(drift)
    if not metrics:
        st.error("No distance-metric columns found in drift data.")
        return

    st.markdown(f"**Metrics available:** {', '.join(DISTANCE_LABEL[m] for m in metrics)}")

    feats = sorted(drift["feature"].unique())
    feature = st.radio("Feature", feats, horizontal=True,
                        index=feats.index("CSP") if "CSP" in feats else 0)
    sub = drift[drift["feature"] == feature].copy()

    # ── Cross-metric correlation matrix ─────────────────────────────────────
    with st.container(border=True):
        st.subheader("Cross-metric correlation")
        metric_values = sub[metrics].rename(columns=DISTANCE_LABEL)
        pearson = metric_values.corr().round(3)
        spearman = metric_values.corr(method="spearman").round(3)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pearson r**")
            st.dataframe(
                pearson.style.format("{:+.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True,
            )
        with c2:
            st.markdown("**Spearman ρ**")
            st.dataframe(
                spearman.style.format("{:+.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True,
            )
        st.caption("Near-zero off-diagonal values → metrics disagree on what counts as 'drift'.")

    # ── Scatter of chosen pair ──────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Pairwise scatter")
        c1, c2 = st.columns(2)
        x_metric = c1.selectbox("X-axis metric", metrics,
                                 format_func=lambda m: DISTANCE_LABEL[m],
                                 index=0)
        y_metric = c2.selectbox("Y-axis metric", metrics,
                                 format_func=lambda m: DISTANCE_LABEL[m],
                                 index=min(3, len(metrics) - 1))
        fig = px.scatter(
            sub, x=x_metric, y=y_metric, color="dataset",
            color_discrete_map=DATASET_COLORS, opacity=0.5,
            hover_data=["subject", "session_k"],
        )
        fig.update_layout(
            xaxis_title=DISTANCE_LABEL[x_metric],
            yaxis_title=DISTANCE_LABEL[y_metric],
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Re-fit Claim 1 per metric ───────────────────────────────────────────
    if merged.empty:
        st.info("No merged eval data available, skipping Claim 1 re-fit.")
        return

    with st.container(border=True):
        st.subheader("Claim 1 drift β under each metric")
        st.caption(
            "For each metric, z-score within (dataset × feature) blocks, then "
            "refit the mixed-effects model on No-DA data."
        )

        noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")].copy()
        rows = []

        for col in metrics:
            if col not in noda.columns:
                # Merged CSV may not yet carry all metrics; skip
                continue
            work = noda.dropna(subset=[col, "acc_centered"]).copy()
            if work.empty or work["subject"].nunique() < 2:
                continue
            work["drift_z_new"] = _zscore_by_block(work, col, ["dataset", "feature"])
            work = work.dropna(subset=["drift_z_new"])
            work["feature"] = work["feature"].astype("category")
            work["dataset"] = work["dataset"].astype("category")
            work["subject_tag"] = work["dataset"].astype(str) + "_" + work["subject"].astype(str)
            try:
                model = smf.mixedlm(
                    "acc_centered ~ drift_z_new * C(feature) + C(dataset)",
                    data=work, groups=work["subject_tag"],
                )
                res = model.fit(method="lbfgs", disp=False)
                beta = res.params.get("drift_z_new", np.nan)
                p = res.pvalues.get("drift_z_new", np.nan)
                # Logvar interaction
                inter_key = [k for k in res.params.index
                             if k.startswith("drift_z_new:") and "logvar" in k]
                inter = res.params[inter_key[0]] if inter_key else np.nan
                inter_p = res.pvalues[inter_key[0]] if inter_key else np.nan
            except Exception:
                beta = p = inter = inter_p = np.nan
            rows.append({
                "metric": DISTANCE_LABEL[col],
                "β(drift_z)": beta, "p": p,
                "β(×logvar)": inter, "p_int": inter_p,
                "n": len(work),
            })

        if rows:
            beta_df = pd.DataFrame(rows)
            st.dataframe(
                beta_df.style.format({
                    "β(drift_z)": "{:+.4f}", "p": "{:.3g}",
                    "β(×logvar)": "{:+.4f}", "p_int": "{:.3g}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Consistent sign and magnitude across metrics → paper's Claim 1 is metric-robust. "
                "Large swings → metric choice drives the reported effect."
            )
        else:
            st.info("Merged CSV does not contain all metric columns; re-run the build script.")

    # ── Per-dataset mean drift under each metric ────────────────────────────
    with st.expander("Mean drift magnitude per (dataset × feature × metric)"):
        agg = drift.groupby(["dataset", "feature"])[metrics].mean().round(4)
        agg = agg.rename(columns=DISTANCE_LABEL)
        st.dataframe(agg.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

    # ── Metric descriptions ────────────────────────────────────────────────
    with st.expander("Metric descriptions"):
        for m in metrics:
            st.markdown(f"**{DISTANCE_LABEL[m]}** — {DISTANCE_DESCRIPTION[m]}")
