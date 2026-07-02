"""Page 9: Multi-Metric Drift Panel.

Shows the full set of distance metrics produced by the build pipeline
(MMD, Energy, Wasserstein, Mahalanobis, Euclidean), lets the user pick
one to re-define drift_z, and re-fits Claim 1 to test robustness of the
drift → loss relationship against metric choice.
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.formula.api as smf

from utils import (
    DISTANCE_LABEL, DISTANCE_DESCRIPTION,
    about_page, available_distance_metrics, dataset_colors, download_bar,
    empty_state, filter_by_dataset, style_figure,
)

logger = logging.getLogger(__name__)


def _zscore_by_block(df: pd.DataFrame, col: str, block: list[str]) -> pd.Series:
    return df.groupby(block)[col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    )


def _fisher_ci(r: float, n: int, alpha: float = 0.05):
    """Fisher z 95% CI for a Pearson correlation."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    crit = norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


@st.cache_data(show_spinner=False)
def _per_metric_refit(noda: pd.DataFrame, metrics: tuple) -> pd.DataFrame:
    """Refit the Claim-1 mixed model with drift_z derived from each metric in
    turn (z-scored within dataset × feature). Cached on the input frame."""
    rows = []
    for col in metrics:
        if col not in noda.columns:
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
            inter_key = [k for k in res.params.index
                         if k.startswith("drift_z_new:") and "logvar" in k]
            inter = res.params[inter_key[0]] if inter_key else np.nan
            inter_p = res.pvalues[inter_key[0]] if inter_key else np.nan
        except Exception as exc:
            logger.debug("per-metric refit failed for %s: %s", col, exc)
            beta = p = inter = inter_p = np.nan
        rows.append({
            "metric_col": col, "metric": DISTANCE_LABEL[col],
            "β(drift)": beta, "p": p,
            "β(×logvar)": inter, "p_int": inter_p, "n": len(work),
        })
    return pd.DataFrame(rows)


def render(store):
    st.header("Multi-Metric Drift Panel")
    st.caption(
        "Compare all distance metrics side-by-side — paper Limitation 4 (MMD vs "
        "Mahalanobis complementarity), generalised to the full DA4BCI suite. "
        "This page has its own controls (DA Lab section)."
    )

    about_page(
        what_you_see=[
            "Pearson + Spearman cross-metric correlation.",
            "Pairwise scatter for any two metrics.",
            "Claim-1 refit under each metric as a robustness check.",
        ],
        how_to_read=[
            "Metrics disagreeing (off-diagonal ≈ 0) ⇒ 'drift' is not one-dimensional.",
            "Stable β(drift) sign across all metrics ⇒ Claim 1 is metric-robust.",
        ],
        paper_ref="§6 Limitation 4",
    )

    # DA Lab page: use its own dataset picker (independent of sidebar)
    datasets = sorted(store.drift_df["dataset"].unique()) if not store.drift_df.empty else []
    if not datasets:
        empty_state("No drift data loaded", "Load a dataset first (see Overview).")
        return

    ds_options = ["All"] + datasets if len(datasets) > 1 else datasets
    dataset = st.selectbox("Dataset (page)", ds_options, index=0)

    drift = filter_by_dataset(store.drift_df, dataset)
    merged = filter_by_dataset(store.merged_df, dataset)
    if drift.empty:
        empty_state("No drift data for this dataset", "")
        return

    metrics = available_distance_metrics(drift)
    if not metrics:
        empty_state("No distance-metric columns",
                    "Loaded CSV has no `dist_*` columns.")
        return

    st.markdown(f"**Metrics available:** {', '.join(DISTANCE_LABEL[m] for m in metrics)}")

    feats = sorted(drift["feature"].unique())
    feature = st.radio(
        "Feature", feats, horizontal=True,
        index=feats.index("CSP") if "CSP" in feats else 0,
    )
    sub = drift[drift["feature"] == feature].copy()

    # ── Cross-metric correlation matrix (tabs) ─────────────────────────────
    with st.container(border=True):
        st.subheader("Cross-metric correlation")
        metric_values = sub[metrics].rename(columns=DISTANCE_LABEL)
        n_corr = int(len(metric_values.dropna()))
        pearson = metric_values.corr().round(3)
        spearman = metric_values.corr(method="spearman").round(3)

        tab_p, tab_s = st.tabs(["Pearson r", "Spearman ρ"])
        with tab_p:
            st.dataframe(
                pearson.style.format("{:+.3f}")
                     .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True,
            )
        with tab_s:
            st.dataframe(
                spearman.style.format("{:+.3f}")
                      .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True,
            )
        st.caption(
            f"n = {n_corr} (subject × session) points. Near-zero off-diagonal "
            "values ⇒ metrics disagree on what counts as 'drift'."
        )
        if n_corr < 100:
            st.warning(
                f"Only {n_corr} points — correlation CIs are wide and Spearman "
                "is noisy. Interpret 'metrics disagree' cautiously."
            )
        with st.expander("Pairwise Pearson 95% CIs (Fisher z)"):
            labels = list(metric_values.columns)
            ci_rows = []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    r = pearson.iloc[i, j]
                    lo, hi = _fisher_ci(r, n_corr)
                    ci_rows.append({
                        "metric A": labels[i], "metric B": labels[j],
                        "r": r, "95% CI": (f"[{lo:+.2f}, {hi:+.2f}]"
                                           if np.isfinite(lo) else "—"),
                    })
            st.dataframe(pd.DataFrame(ci_rows).style.format({"r": "{:+.3f}"}),
                         use_container_width=True, hide_index=True)

    # ── Scatter of chosen pair ──────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Pairwise scatter")
        c1, c2 = st.columns(2)
        x_metric = c1.selectbox(
            "X-axis metric", metrics,
            format_func=lambda m: DISTANCE_LABEL[m], index=0,
        )
        y_metric = c2.selectbox(
            "Y-axis metric", metrics,
            format_func=lambda m: DISTANCE_LABEL[m],
            index=min(3, len(metrics) - 1),
        )
        fig = px.scatter(
            sub, x=x_metric, y=y_metric, color="dataset",
            color_discrete_map=dataset_colors(), opacity=0.5,
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
        st.info("No merged eval data available; Claim-1 refit skipped.")
        return

    with st.container(border=True):
        st.subheader("Claim 1 drift slope under each metric")
        st.caption(
            "For each metric, z-score within (dataset × feature), then refit "
            "the mixed-effects model on No-DA rows. Δ vs MMD is shown as a "
            "sanity check."
        )

        noda = merged[(merged["strategy"] == "train_once") &
                      (merged["da"] == "none")].copy()

        # Stability guard: z-scores within tiny (dataset × feature) blocks are
        # unreliable. Warn when any block is small.
        block_sizes = noda.groupby(["dataset", "feature"]).size()
        if len(block_sizes) and block_sizes.min() < 10:
            st.warning(
                f"Smallest (dataset × feature) block has {int(block_sizes.min())} "
                "rows; within-block z-scores (and the refit below) are unstable there."
            )

        with st.spinner("Refitting Claim 1 under each metric…"):
            beta_df = _per_metric_refit(noda, tuple(metrics))

        if not beta_df.empty:
            # Δ vs MMD baseline
            mmd_row = beta_df[beta_df["metric_col"] == "dist_mmd"]
            if not mmd_row.empty:
                mmd_beta = float(mmd_row["β(drift)"].iloc[0])
                beta_df["Δ vs MMD"] = beta_df["β(drift)"] - mmd_beta
            else:
                beta_df["Δ vs MMD"] = np.nan
            show_df = beta_df.drop(columns=["metric_col"])
            st.dataframe(
                show_df.style.format({
                    "β(drift)": "{:+.4f}", "p": "{:.3g}",
                    "β(×logvar)": "{:+.4f}", "p_int": "{:.3g}",
                    "Δ vs MMD": "{:+.4f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Consistent sign and magnitude across metrics ⇒ Claim 1 is "
                "metric-robust. Large `Δ vs MMD` swings ⇒ metric choice drives "
                "the reported effect."
            )
            download_bar("mm_claim1_refit", show_df, "mm_claim1_refit")
        else:
            st.info("Merged CSV does not contain all metric columns; re-run the build script.")

    # ── Per-dataset mean drift ─────────────────────────────────────────────
    with st.expander("Mean drift magnitude per (dataset × feature × metric)"):
        agg = drift.groupby(["dataset", "feature"])[metrics].mean().round(4)
        agg = agg.rename(columns=DISTANCE_LABEL)
        st.dataframe(agg.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

    # ── Metric descriptions ────────────────────────────────────────────────
    with st.expander("Metric descriptions"):
        for m in metrics:
            st.markdown(f"**{DISTANCE_LABEL[m]}** — {DISTANCE_DESCRIPTION[m]}")
