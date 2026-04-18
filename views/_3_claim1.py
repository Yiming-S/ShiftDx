"""Page 3: Claim 1 — Drift Predicts Performance Loss."""

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import FEATURE_COLORS, filter_by_dataset, format_acc, style_figure


@st.cache_data
def _fit_claim1(df_json: str) -> dict:
    """Fit the baseline-centered mixed-effects model from the paper (Table 1)."""
    df = pd.read_json(StringIO(df_json), orient="split")
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough groups to fit mixed-effects model."}
    # Treat subject as grouping variable (random intercept)
    # Use categorical feature/dataset for fixed effects
    df = df.copy()
    df["feature"] = df["feature"].astype("category")
    df["dataset"] = df["dataset"].astype("category")
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    try:
        model = smf.mixedlm(
            "acc_centered ~ drift_z * C(feature) + C(dataset)",
            data=df,
            groups=df["subject_tag"],
        )
        result = model.fit(method="lbfgs", disp=False)
    except Exception as exc:
        return {"error": f"MixedLM fit failed: {exc}"}

    coef = result.params.to_dict()
    pval = result.pvalues.to_dict()
    se = result.bse.to_dict()
    tval = result.tvalues.to_dict()
    rows = []
    for k in coef:
        rows.append({
            "term": k,
            "coef": coef[k],
            "SE": se.get(k, np.nan),
            "t": tval.get(k, np.nan),
            "p": pval.get(k, np.nan),
        })
    return {"table": pd.DataFrame(rows), "n": int(len(df)), "n_subj": int(df["subject_tag"].nunique())}


def render(store, dataset):
    st.header("Claim 1 — Drift Predicts Performance Loss")
    st.caption(
        "Baseline-centered mixed-effects model: "
        r"$acc\_centered \sim drift\_z \times feature + dataset$, "
        "subject random intercept. Replicates paper Table 1."
    )

    merged = filter_by_dataset(store.merged_df, dataset)
    if merged.empty:
        st.warning("No merged data.")
        return

    # Classifier filter
    from utils import (
        available_classifiers, CLASSIFIER_LABEL,
        pick_metric_with_drift_z, apply_drift_metric, DISTANCE_LABEL,
    )
    clf_opts = available_classifiers(merged)
    if len(clf_opts) > 1:
        c1, c2 = st.columns([1, 3])
        clf_mode = c1.radio("Classifier", ["All pooled"] + clf_opts,
                             horizontal=True, index=0,
                             format_func=lambda x: x if x == "All pooled" else CLASSIFIER_LABEL.get(x, x))
        if clf_mode != "All pooled":
            merged = merged[merged["classifier"] == clf_mode].copy()

    # Distance-metric selector (redefines drift_z to the chosen metric)
    metric_opts = pick_metric_with_drift_z(merged)
    if len(metric_opts) > 1:
        metric = st.radio(
            "Drift metric (drift_z)", metric_opts,
            horizontal=True, index=0,
            format_func=lambda m: DISTANCE_LABEL.get(m, m),
        )
        merged = apply_drift_metric(merged, metric)

    # Filter to No-DA only (paper Claim 1 uses No-DA)
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")].copy()
    if noda.empty:
        st.warning("No rows match (strategy=train_once, da=none).")
        return

    features = st.multiselect(
        "Features to include",
        sorted(noda["feature"].unique()),
        default=[f for f in ["CSP", "logvar"] if f in noda["feature"].unique()],
    )
    if not features:
        st.info("Select at least one feature.")
        return
    noda = noda[noda["feature"].isin(features)]

    # ── Scatter with linear fits per feature ────────────────────────────────
    with st.container(border=True):
        st.subheader("Drift vs No-DA accuracy (baseline-centered)")
        fig = go.Figure()
        for feat, g in noda.groupby("feature"):
            col = FEATURE_COLORS.get(feat, "#4F46E5")
            fig.add_trace(go.Scatter(
                x=g["drift_z"], y=g["acc_centered"],
                mode="markers", name=feat,
                marker=dict(color=col, size=5, opacity=0.4),
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
            xaxis_title="drift_z (standardized MMD)",
            yaxis_title="acc_centered (acc − subject baseline)",
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Fit MixedLM ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Mixed-effects coefficients")
        fit = _fit_claim1(noda.to_json(orient="split"))
        if "error" in fit:
            st.error(fit["error"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("n observations", f"{fit['n']:,}")
            c2.metric("n subjects", fit["n_subj"])
            tbl = fit["table"].copy()
            tbl["sig"] = tbl["p"].apply(
                lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            )
            st.dataframe(
                tbl.style.format({"coef": "{:+.4f}", "SE": "{:.4f}",
                                  "t": "{:+.2f}", "p": "{:.4g}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Paper Table 1 (all 3 datasets, both features): "
                "drift_z β = −0.029 (p<0.001); drift_z × feature[logvar] = +0.034 (p<0.001)."
            )

    # ── Forest plot: β across all (feature × classifier × metric) cells ────
    raw = filter_by_dataset(store.merged_df, dataset)
    raw_noda = raw[(raw["strategy"] == "train_once") & (raw["da"] == "none")]
    from utils import DRIFT_Z_COL, DISTANCE_LABEL, CLASSIFIER_LABEL
    forest_rows = []
    for (feat, clf), g in raw_noda.groupby(["feature", "classifier"]):
        for m_col, z_col in DRIFT_Z_COL.items():
            if z_col not in g.columns:
                continue
            gg = g.dropna(subset=[z_col, "acc_centered"])
            if len(gg) < 5:
                continue
            try:
                mod = smf.ols(f"acc_centered ~ {z_col}", data=gg).fit()
                beta = mod.params[z_col]
                se = mod.bse[z_col]
                p = mod.pvalues[z_col]
                ci = mod.conf_int(alpha=0.05).loc[z_col].values
            except Exception:
                continue
            forest_rows.append({
                "feature": feat, "classifier": clf, "metric": m_col,
                "metric_label": DISTANCE_LABEL[m_col], "n": len(gg),
                "beta": beta, "se": se, "lo": ci[0], "hi": ci[1], "p": p,
            })

    if forest_rows:
        with st.container(border=True):
            st.subheader("Forest plot — β across (feature × classifier × metric)")
            st.caption(
                "Each row is an independent OLS fit on the matching cell. "
                "Bars show 95% CI; colored markers ≠ gray cross zero."
            )
            fdf = pd.DataFrame(forest_rows)
            sort_mode = st.radio(
                "Sort by", ["feature → classifier → metric", "β (ascending)",
                             "classifier → feature → metric"],
                horizontal=True,
            )
            if sort_mode == "β (ascending)":
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
            # Colour non-zero-crossing by feature; gray if CI crosses 0
            colors = []
            for _, r in fdf.iterrows():
                if r["lo"] <= 0 <= r["hi"]:
                    colors.append("#94A3B8")
                else:
                    colors.append(FEATURE_COLORS.get(r["feature"], "#4F46E5"))
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
                customdata=np.stack([fdf["n"], fdf["p"]], axis=-1),
                hovertemplate=(
                    "%{y}<br>β=%{x:+.4f}<br>"
                    "n=%{customdata[0]}<br>p=%{customdata[1]:.3g}<extra></extra>"
                ),
                showlegend=False,
            ))
            fig.add_vline(x=0, line_color="#64748B", line_dash="dash")
            fig.update_layout(
                xaxis_title="β (drift_z slope)",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                height=max(280, 22 * len(fdf) + 100),
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
            n_sig = int(((fdf["lo"] > 0) | (fdf["hi"] < 0)).sum())
            st.caption(f"{n_sig}/{len(fdf)} cells have 95% CI excluding zero.")

    # ── Per-(dataset, feature) slopes ───────────────────────────────────────
    with st.expander("Per (dataset × feature) drift slope"):
        rows = []
        for (ds, feat), g in noda.groupby(["dataset", "feature"]):
            if len(g) < 5:
                continue
            slope, intercept = np.polyfit(g["drift_z"], g["acc_centered"], 1)
            # t-test on slope
            try:
                mod = smf.ols("acc_centered ~ drift_z", data=g).fit()
                p = mod.pvalues.get("drift_z", np.nan)
            except Exception:
                p = np.nan
            rows.append({
                "dataset": ds, "feature": feat,
                "n": len(g), "slope": slope, "p": p,
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({"slope": "{:+.4f}", "p": "{:.4g}"}),
                use_container_width=True, hide_index=True,
            )
