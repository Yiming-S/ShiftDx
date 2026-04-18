"""Page 4: Claim 2 — DA Benefit Decomposition."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import FEATURE_COLORS, filter_by_dataset, style_figure


@st.cache_data
def _fit_claim2(df_json: str) -> dict:
    df = pd.read_json(df_json, orient="split")
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough groups."}
    df = df.copy()
    df["feature"] = df["feature"].astype("category")
    df["dataset"] = df["dataset"].astype("category")
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    try:
        model = smf.mixedlm(
            "acc_centered ~ drift_z * has_da * C(feature) + C(dataset)",
            data=df, groups=df["subject_tag"],
        )
        result = model.fit(method="lbfgs", disp=False)
    except Exception as exc:
        return {"error": f"Fit failed: {exc}"}
    tbl = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "SE": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
    })
    return {"table": tbl, "n": int(len(df)), "n_subj": int(df["subject_tag"].nunique())}


def render(store, dataset):
    st.header("Claim 2 — DA Benefit Decomposition")
    st.caption(
        r"Pool No-DA and DA rows; fit $acc\_centered \sim drift\_z \times has\_da \times feature + dataset$. "
        r"$\eta_2$ = level shift; $\eta_4$ = slope change (paper Table 2)."
    )

    merged = filter_by_dataset(store.merged_df, dataset)
    if merged.empty:
        st.warning("No merged data.")
        return

    from utils import (
        available_da_methods, available_classifiers, DA_LABEL, CLASSIFIER_LABEL,
        pick_metric_with_drift_z, apply_drift_metric, DISTANCE_LABEL,
    )
    clf_opts = available_classifiers(merged)
    if len(clf_opts) > 1:
        clf_mode = st.radio("Classifier", ["All pooled"] + clf_opts,
                             horizontal=True, index=0,
                             format_func=lambda x: x if x == "All pooled" else CLASSIFIER_LABEL.get(x, x))
        if clf_mode != "All pooled":
            merged = merged[merged["classifier"] == clf_mode].copy()

    metric_opts = pick_metric_with_drift_z(merged)
    if len(metric_opts) > 1:
        metric = st.radio(
            "Drift metric (drift_z)", metric_opts,
            horizontal=True, index=0,
            format_func=lambda m: DISTANCE_LABEL.get(m, m),
        )
        merged = apply_drift_metric(merged, metric)

    da_opts = available_da_methods(merged)
    if not da_opts:
        st.warning("No DA methods present in the eval data.")
        return
    da_method = st.selectbox(
        "DA method (vs No-DA)", da_opts,
        format_func=lambda m: DA_LABEL.get(m, m),
    )

    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")].copy()
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)].copy()

    if noda.empty or da.empty:
        st.warning(f"No rows for DA={da_method}.")
        return

    noda["has_da"] = 0
    da["has_da"] = 1
    pooled = pd.concat([noda, da], ignore_index=True)

    # ── Scatter: No-DA vs DA regression lines by feature ────────────────────
    with st.container(border=True):
        st.subheader(f"No-DA vs {da_method.upper()} regression")
        fig = go.Figure()
        for feat in sorted(pooled["feature"].unique()):
            col = FEATURE_COLORS.get(feat, "#4F46E5")
            g_nd = noda[noda["feature"] == feat]
            g_da = da[da["feature"] == feat]
            if len(g_nd) >= 2:
                c = np.polyfit(g_nd["drift_z"], g_nd["acc_centered"], 1)
                xs = np.linspace(g_nd["drift_z"].min(), g_nd["drift_z"].max(), 40)
                fig.add_trace(go.Scatter(
                    x=xs, y=np.polyval(c, xs), mode="lines",
                    line=dict(color=col, width=3, dash="dot"),
                    name=f"{feat} No-DA (slope={c[0]:+.3f})",
                ))
            if len(g_da) >= 2:
                c = np.polyfit(g_da["drift_z"], g_da["acc_centered"], 1)
                xs = np.linspace(g_da["drift_z"].min(), g_da["drift_z"].max(), 40)
                fig.add_trace(go.Scatter(
                    x=xs, y=np.polyval(c, xs), mode="lines",
                    line=dict(color=col, width=3),
                    name=f"{feat} {da_method.upper()} (slope={c[0]:+.3f})",
                ))
        fig.update_layout(
            xaxis_title="drift_z", yaxis_title="acc_centered",
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Fit ─────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Interaction model coefficients")
        fit = _fit_claim2(pooled.to_json(orient="split"))
        if "error" in fit:
            st.error(fit["error"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("n observations", f"{fit['n']:,}")
            c2.metric("n subjects", fit["n_subj"])
            tbl = fit["table"]
            key_rows = tbl[tbl["term"].str.contains("has_da", case=False, na=False)]
            st.markdown("**Key terms (DA effect):**")
            st.dataframe(
                key_rows.style.format({"coef": "{:+.4f}", "SE": "{:.4f}",
                                       "t": "{:+.2f}", "p": "{:.4g}"}),
                use_container_width=True, hide_index=True,
            )
            with st.expander("Full coefficient table"):
                st.dataframe(
                    tbl.style.format({"coef": "{:+.4f}", "SE": "{:.4f}",
                                      "t": "{:+.2f}", "p": "{:.4g}"}),
                    use_container_width=True, hide_index=True,
                )
            st.caption(
                "Paper Table 2 (SA): level shift η₂ = +0.035 (p<0.001), "
                "slope change η₄ = +0.007 (p=0.114). "
                "PT yields significant slope change (paper §5.3)."
            )

    # ── DA fan chart: all 10 DA methods × drift quartiles ──────────────────
    with st.container(border=True):
        st.subheader("DA method fan chart — accuracy by drift quartile")
        st.caption(
            "One curve per strategy. Compare how every DA method tracks drift, "
            "bounded below by No-DA (gray) and above by Retrain (green)."
        )
        feat_pick = st.radio(
            "Feature", sorted(merged["feature"].unique()),
            horizontal=True, key="c2_fan_feat",
        )
        work = merged[merged["feature"] == feat_pick].copy()
        # Bin drift_z into quartiles (within the selected slice)
        work["drift_bin"] = pd.qcut(
            work["drift_z"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop",
        )

        from utils import DA_COLORS
        fig_fan = go.Figure()
        # No-DA
        nd_slice = work[(work["strategy"] == "train_once") & (work["da"] == "none")]
        agg = nd_slice.groupby("drift_bin", observed=True)["accuracy"].agg(["mean", "sem"]).reset_index()
        fig_fan.add_trace(go.Scatter(
            x=agg["drift_bin"], y=agg["mean"],
            error_y=dict(type="data", array=agg["sem"]),
            mode="lines+markers",
            name="No DA", line=dict(color="#475569", width=4),
            marker=dict(size=10, symbol="circle"),
        ))
        # Retrain
        rt_slice = work[(work["strategy"] == "retrain")
                         & (work["ref_session"] == work["target_session"])]
        if not rt_slice.empty:
            agg = rt_slice.groupby("drift_bin", observed=True)["accuracy"].agg(["mean", "sem"]).reset_index()
            fig_fan.add_trace(go.Scatter(
                x=agg["drift_bin"], y=agg["mean"],
                error_y=dict(type="data", array=agg["sem"]),
                mode="lines+markers",
                name="Retrain", line=dict(color="#10B981", width=4),
                marker=dict(size=10, symbol="square"),
            ))
        # Each DA method
        for dm in da_opts:
            slc = work[(work["strategy"] == "train_once_da") & (work["da"] == dm)]
            if slc.empty:
                continue
            agg = slc.groupby("drift_bin", observed=True)["accuracy"].agg(["mean", "sem"]).reset_index()
            fig_fan.add_trace(go.Scatter(
                x=agg["drift_bin"], y=agg["mean"],
                error_y=dict(type="data", array=agg["sem"]),
                mode="lines+markers",
                name=DA_LABEL.get(dm, dm),
                line=dict(color=DA_COLORS.get(dm, "#4F46E5"), width=2),
                marker=dict(size=6),
            ))
        fig_fan.update_layout(
            xaxis_title="drift_z quartile (low → high)",
            yaxis_title="mean accuracy",
            legend=dict(orientation="v", x=1.02, y=1.0),
        )
        style_figure(fig_fan, height=480)
        st.plotly_chart(fig_fan, use_container_width=True)
        st.caption(
            "Hugging the green line = DA reaches retrain ceiling. "
            "Dropping below the gray line = DA is actively harmful."
        )

    # ── Marginal gain by drift bin ──────────────────────────────────────────
    with st.expander("Mean DA gain by drift bin"):
        key = ["dataset", "subject", "feature", "target_session", "drift_z"]
        merged_pair = pd.merge(
            noda[key + ["accuracy"]].rename(columns={"accuracy": "acc_noda"}),
            da[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"}),
            on=key, how="inner",
        )
        if not merged_pair.empty:
            merged_pair["gain"] = merged_pair["acc_da"] - merged_pair["acc_noda"]
            merged_pair["drift_bin"] = pd.qcut(
                merged_pair["drift_z"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop",
            )
            agg = merged_pair.groupby(["drift_bin", "feature"], observed=True)["gain"].mean().reset_index()
            pivot = agg.pivot(index="drift_bin", columns="feature", values="gain")
            st.dataframe(
                pivot.style.format("{:+.4f}").background_gradient(cmap="RdBu_r", vmin=-0.05, vmax=0.05),
                use_container_width=True,
            )
