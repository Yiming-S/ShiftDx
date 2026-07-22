"""Page 3: Claim 1 — Drift Predicts Performance Loss."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import (
    CLASSIFIER_LABEL, DISTANCE_LABEL, DRIFT_Z_COL,
    SIG_LEGEND, about_page, apply_ctx_classifier, apply_ctx_dataset,
    apply_ctx_metric, benjamini_hochberg, build_claim_formula, download_bar,
    empty_state, feature_colors, fit_lmm_or_fallback, get_ctx,
    mixedlm_pseudo_r2, pvalue_badge, style_figure,
)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _fit_claim1(df: pd.DataFrame) -> dict:
    """Fit the baseline-centered mixed-effects model from the paper (Table 1).

    Cached directly on the DataFrame (Streamlit hashes it natively). Robust to
    degenerate slices: single-level factors are dropped from the formula, a
    no-variance drift slice is reported clearly, and a rank-deficient (singular)
    mixed fit falls back to OLS with subject-cluster-robust SEs.
    """
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough subjects in this slice to fit the model "
                         "(need at least 2). Widen the dataset / classifier filter."}
    df = df.copy()
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    d = df.dropna(subset=["drift_z", "acc_centered"])
    if len(d) < 3 or d["subject_tag"].nunique() < 2:
        return {"error": "Not enough usable rows after dropping missing "
                         "drift_z / acc_centered."}
    if float(d["drift_z"].std(ddof=1)) < 1e-8:
        return {"error": "Drift has essentially no variation in this slice, so a "
                         "drift slope cannot be estimated. Pick a slice with more "
                         "sessions/subjects or a different drift metric."}

    formula = build_claim_formula(d, "acc_centered", interact_da=False)
    try:
        result, method, is_mixed = fit_lmm_or_fallback(d, formula, "subject_tag")
    except Exception as exc:
        return {"error": f"Model fit failed: {exc}"}

    rows = []
    for k in result.params.index:
        if str(k).endswith("Var") or "Group" in str(k):
            continue  # hide the random-effect variance row
        rows.append({
            "term": k,
            "coef": result.params[k],
            "SE": result.bse.get(k, np.nan),
            "t": result.tvalues.get(k, np.nan),
            "p": result.pvalues.get(k, np.nan),
        })
    r2m, r2c = (mixedlm_pseudo_r2(result) if is_mixed
                else (float(getattr(result, "rsquared", np.nan)), np.nan))
    return {
        "table": pd.DataFrame(rows),
        "n": int(len(d)),
        "n_subj": int(d["subject_tag"].nunique()),
        "r2_marginal": r2m,
        "r2_conditional": r2c,
        "method": method,
        "is_mixed": is_mixed,
    }


@st.cache_data(show_spinner=False)
def _fit_random_slope(df: pd.DataFrame) -> dict:
    """SUPPLEMENTARY random-slope MixedLM (re_formula='~drift_z').

    NOT the published model — offered only as a robustness panel. Convergence
    with few subjects per dataset is fragile, hence the explicit status flag.
    """
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough groups."}
    df = df.copy()
    df["feature"] = df["feature"].astype("category")
    df["dataset"] = df["dataset"].astype("category")
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    try:
        model = smf.mixedlm(
            "acc_centered ~ drift_z * C(feature) + C(dataset)",
            data=df, groups=df["subject_tag"], re_formula="~drift_z",
        )
        result = model.fit(method="lbfgs", disp=False)
    except Exception as exc:
        return {"error": f"Random-slope fit failed: {exc}"}
    rows = [{"term": k, "coef": result.params[k], "SE": result.bse.get(k, np.nan),
             "p": result.pvalues.get(k, np.nan)} for k in result.params.index]
    return {"table": pd.DataFrame(rows),
            "converged": bool(getattr(result, "converged", True))}


@st.cache_data(show_spinner=False)
def _build_forest(raw_noda: pd.DataFrame) -> tuple:
    """Per-(feature × classifier × metric) OLS drift slopes with BH-FDR.

    The FDR correction is computed over the FULL set of fitted cells (the true
    multiplicity), before any display filtering. Returns (forest_df, skipped).
    """
    forest_rows = []
    skipped = 0
    for (feat, clf), g in raw_noda.groupby(["feature", "classifier"]):
        for m_col, z_col in DRIFT_Z_COL.items():
            if z_col not in g.columns:
                continue
            gg = g.dropna(subset=[z_col, "acc_centered"])
            if len(gg) < 5:
                skipped += 1
                continue
            try:
                mod = smf.ols(f"acc_centered ~ {z_col}", data=gg).fit()
                beta = mod.params[z_col]
                se = mod.bse[z_col]
                p = mod.pvalues[z_col]
                r2 = mod.rsquared
                ci = mod.conf_int(alpha=0.05).loc[z_col].values
            except Exception as exc:
                logger.debug("forest OLS failed for %s/%s/%s: %s",
                             feat, clf, m_col, exc)
                skipped += 1
                continue
            forest_rows.append({
                "feature": feat, "classifier": clf, "metric": m_col,
                "metric_label": DISTANCE_LABEL[m_col], "n": len(gg),
                "beta": beta, "se": se, "lo": ci[0], "hi": ci[1],
                "p": p, "R2": r2,
            })
    fdf = pd.DataFrame(forest_rows)
    if not fdf.empty:
        fdf["p_adj"] = benjamini_hochberg(fdf["p"].to_numpy())
        fdf["sig_fdr"] = fdf["p_adj"] < 0.05
    return fdf, skipped


def render(store):
    ctx = get_ctx()
    fcolors = feature_colors()

    st.header("Claim 1 · Drift predicts loss")
    st.caption(
        "Does accuracy drop as drift grows? — "
        r"$acc\_centered \sim drift\_z \times feature + dataset$, "
        "subject random intercept. Replicates paper Table 1."
    )

    about_page(
        what_you_see=[
            "Scatter of standardized drift vs baseline-centered accuracy, one OLS line per feature.",
            "Mixed-effects coefficient table (paper-style) with pseudo-R².",
            "Forest plot of OLS drift slope across every (feature × classifier × metric) cell, "
            "with Benjamini-Hochberg FDR correction.",
        ],
        how_to_read=[
            "Negative drift slope ⇒ accuracy drops as drift grows — Claim 1 is supported.",
            "Colored markers on the forest plot are significant after FDR correction across all cells.",
        ],
        paper_ref="§5.2, Table 1",
        key_terms=[
            ("drift_z", "standardized drift (chosen in the sidebar)."),
            ("acc_centered", "accuracy minus subject baseline (session 0, 5-fold CV)."),
            ("FDR", "Benjamini-Hochberg false-discovery-rate adjusted p across all forest cells."),
        ],
    )

    merged = apply_ctx_dataset(store.merged_df)
    merged = apply_ctx_classifier(merged, ctx)
    merged = apply_ctx_metric(merged, ctx)
    if merged.empty:
        empty_state(
            "No merged data for this selection",
            f"`merged_drift_accuracy_*.csv` is empty under "
            f"dataset=`{ctx['dataset']}`, classifier=`{ctx['classifier']}`.",
            dataset=ctx["dataset"],
        )
        return

    # Filter to No-DA only (paper Claim 1 uses No-DA)
    noda = merged[(merged["strategy"] == "train_once") &
                  (merged["da"] == "none")].copy()
    if noda.empty:
        empty_state("No No-DA rows",
                    "The merged file has no rows with strategy=No DA.")
        return

    features = st.multiselect(
        "Features to include",
        sorted(noda["feature"].unique()),
        default=[f for f in ["CSP", "logvar", "TS"] if f in noda["feature"].unique()],
    )
    if not features:
        st.info("Select at least one feature.")
        return
    noda = noda[noda["feature"].isin(features)]

    # ── Scatter with linear fits per feature ────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Drift vs No-DA accuracy "
                     f"({DISTANCE_LABEL.get(ctx['metric'], ctx['metric'])})")
        fig = go.Figure()
        for feat, g in noda.groupby("feature"):
            col = fcolors.get(feat, "#4F46E5")
            opacity = min(0.6, max(0.15, 300 / max(len(g), 1)))
            fig.add_trace(go.Scatter(
                x=g["drift_z"], y=g["acc_centered"],
                mode="markers", name=feat,
                marker=dict(color=col, size=5, opacity=opacity),
            ))
            if len(g) >= 2:
                coef = np.polyfit(g["drift_z"], g["acc_centered"], 1)
                xs = np.linspace(g["drift_z"].min(), g["drift_z"].max(), 50)
                ys = np.polyval(coef, xs)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=col, width=3, dash="dash"),
                    name=f"{feat} fit (slope={coef[0]:+.3f})",
                ))
        fig.update_layout(
            xaxis_title="drift (z-score)",
            yaxis_title="accuracy vs baseline",
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        download_bar("claim1_scatter", noda, "claim1_noda_points")

    # ── Fit MixedLM ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Mixed-effects coefficients")
        with st.spinner("Fitting mixed-effects model…"):
            fit = _fit_claim1(noda)
        if "error" in fit:
            st.info(fit["error"])
        else:
            if not fit.get("is_mixed", True):
                st.warning(f"⚠ Estimator fell back to **{fit['method']}** — the "
                           "mixed model was rank-deficient on this slice "
                           "(e.g. near-constant drift). Point estimates are still "
                           "valid; interpret with the wider selection in mind.")
            c1, c2, c3 = st.columns(3)
            c1.metric("n observations", f"{fit['n']:,}")
            c2.metric("n subjects", fit["n_subj"])
            r2m, r2c = fit.get("r2_marginal"), fit.get("r2_conditional")
            c3.metric("pseudo-R² (marg / cond)",
                      f"{r2m:.2f} / {r2c:.2f}" if r2m == r2m else "—",
                      help="Nakagawa marginal (fixed effects) / conditional "
                           "(incl. subject random intercept).")
            tbl = fit["table"].copy()
            tbl["sig"] = tbl["p"].apply(pvalue_badge)
            st.dataframe(
                tbl.style.format({"coef": "{:+.4f}", "SE": "{:.4f}",
                                  "t": "{:+.2f}", "p": "{:.4g}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"Legend: {SIG_LEGEND}.  Paper Table 1 (all 3 datasets, both "
                "features): drift β = −0.029 (p<0.001); drift × logvar = +0.034 "
                "(p<0.001).  MixedLM p-values are asymptotic (Z-based) and can be "
                "anticonservative with few subjects per dataset."
            )
            download_bar("claim1_mixedlm", tbl, "claim1_mixedlm_coefs")

        # OPT-1: supplementary random-slope model (opt-in, not the published fit)
        with st.expander("Robustness: random-slope model (supplementary — not the published fit)"):
            st.caption(
                "Adds a per-subject random slope for `drift_z` "
                "(`re_formula='~drift_z'`). This is **not** the model behind "
                "paper Table 1; it is shown only to gauge sensitivity. "
                "Convergence is fragile with few subjects."
            )
            if st.checkbox("Fit random-slope model", value=False, key="c1_rslope"):
                with st.spinner("Fitting random-slope model…"):
                    rs = _fit_random_slope(noda)
                if "error" in rs:
                    st.warning(rs["error"])
                else:
                    if not rs["converged"]:
                        st.warning("⚠ Model did not fully converge — interpret with caution.")
                    rtbl = rs["table"].copy()
                    rtbl["sig"] = rtbl["p"].apply(pvalue_badge)
                    st.dataframe(
                        rtbl.style.format({"coef": "{:+.4f}", "SE": "{:.4f}",
                                           "p": "{:.4g}"}),
                        use_container_width=True, hide_index=True,
                    )

    # ── Forest plot: β across all (feature × classifier × metric) cells ────
    raw = apply_ctx_dataset(store.merged_df)
    raw_noda = raw[(raw["strategy"] == "train_once") & (raw["da"] == "none")]
    with st.spinner("Fitting forest-plot cells…"):
        fdf_full, skipped = _build_forest(raw_noda)

    if not fdf_full.empty:
        with st.container(border=True):
            st.subheader("Forest plot — slope across (feature × classifier × metric)")
            n_total = len(fdf_full)
            n_sig_fdr = int(fdf_full["sig_fdr"].sum())
            st.caption(
                f"Each row is an independent OLS fit. Bars show 95% CI; markers "
                f"are colored when significant after **Benjamini-Hochberg FDR** "
                f"across all {n_total} fitted cells "
                f"({n_sig_fdr}/{n_total} significant)."
                + (f" {skipped} cell(s) skipped (insufficient data)." if skipped else "")
            )
            fdf = fdf_full.copy()

            show_all = st.checkbox(
                "Show all cells (all classifiers × all metrics)",
                value=False,
                help="Uncheck to restrict to the current sidebar classifier and metric. "
                     "FDR is always computed over ALL cells, not just the shown subset.",
            )
            if not show_all:
                filt = fdf.copy()
                if ctx["classifier"] != "All pooled":
                    filt = filt[filt["classifier"] == ctx["classifier"]]
                filt = filt[filt["metric"] == ctx["metric"]]
                if not filt.empty:
                    fdf = filt

            sort_mode = st.radio(
                "Sort by",
                ["feature → classifier → metric", "slope (ascending)",
                 "classifier → feature → metric"],
                horizontal=True,
            )
            if sort_mode == "slope (ascending)":
                fdf = fdf.sort_values("beta")
            elif sort_mode.startswith("classifier"):
                fdf = fdf.sort_values(["classifier", "feature", "metric"])
            else:
                fdf = fdf.sort_values(["feature", "classifier", "metric"])
            fdf = fdf.reset_index(drop=True)
            fdf["label"] = (
                fdf["feature"] + " · "
                + fdf["classifier"].map(lambda x: CLASSIFIER_LABEL.get(x, x)) + " · "
                + fdf["metric_label"]
            )
            # Color rule: significant after FDR ⇒ feature color; else grey.
            colors = [
                fcolors.get(r["feature"], "#4F46E5") if r["sig_fdr"] else "#94A3B8"
                for _, r in fdf.iterrows()
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fdf["beta"], y=fdf["label"],
                mode="markers",
                marker=dict(size=10, color=colors, line=dict(width=1, color="#1E293B")),
                error_x=dict(
                    type="data",
                    array=fdf["hi"] - fdf["beta"],
                    arrayminus=fdf["beta"] - fdf["lo"],
                    color="rgba(100,116,139,0.6)",
                ),
                customdata=np.stack([fdf["n"], fdf["p"], fdf["p_adj"], fdf["R2"]], axis=-1),
                hovertemplate=(
                    "%{y}<br>slope=%{x:+.4f}<br>n=%{customdata[0]}<br>"
                    "p=%{customdata[1]:.3g}<br>p(FDR)=%{customdata[2]:.3g}<br>"
                    "R²=%{customdata[3]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ))
            fig.add_vline(x=0, line_color="#64748B", line_dash="dash")
            fig.update_layout(
                xaxis_title="slope (drift → accuracy)",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                height=max(280, 22 * len(fdf) + 100),
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
            n_sig_shown = int(fdf["sig_fdr"].sum())
            st.caption(f"{n_sig_shown}/{len(fdf)} shown cells significant at FDR-BH 5%.")
            download_bar("claim1_forest", fdf, "claim1_forest_cells")

    # ── Per-(dataset, feature) slopes ───────────────────────────────────────
    with st.expander("Per (dataset × feature) drift slope"):
        rows = []
        skipped_pd = 0
        for (ds, feat), g in noda.groupby(["dataset", "feature"]):
            if len(g) < 5:
                skipped_pd += 1
                continue
            slope, _ = np.polyfit(g["drift_z"], g["acc_centered"], 1)
            try:
                mod = smf.ols("acc_centered ~ drift_z", data=g).fit()
                p = mod.pvalues.get("drift_z", np.nan)
                r2 = mod.rsquared
            except Exception as exc:
                logger.debug("per-(ds,feat) OLS failed for %s/%s: %s", ds, feat, exc)
                p = r2 = np.nan
            rows.append({"dataset": ds, "feature": feat,
                         "n": len(g), "slope": slope, "p": p, "R2": r2})
        if rows:
            df = pd.DataFrame(rows)
            df["sig"] = df["p"].apply(pvalue_badge)
            st.dataframe(
                df.style.format({"slope": "{:+.4f}", "p": "{:.4g}", "R2": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
            if skipped_pd:
                st.caption(f"{skipped_pd} (dataset × feature) group(s) skipped (n<5).")
            download_bar("claim1_per_ds_feat", df, "claim1_per_dataset_feature")
