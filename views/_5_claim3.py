"""Page 5: Claim 3 — Positive Retraining Gap."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import DATASET_COLORS, filter_by_dataset, style_figure


def _compute_rg_table(merged: pd.DataFrame, da_method: str) -> pd.DataFrame:
    """Build R_g(z) = Retrain - DA table per (dataset, feature, subject, target_session)."""
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)]
    # For retrain: ref_session == target_session (within-session CV)
    retrain = merged[(merged["strategy"] == "retrain") & (merged["ref_session"] == merged["target_session"])]

    key = ["dataset", "subject", "feature", "target_session", "drift_z"]
    tbl = (
        noda[key + ["accuracy"]].rename(columns={"accuracy": "acc_noda"})
        .merge(da[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"}), on=key, how="inner")
        .merge(retrain[["dataset", "subject", "feature", "target_session", "accuracy"]]
               .rename(columns={"accuracy": "acc_retrain"}),
               on=["dataset", "subject", "feature", "target_session"], how="inner")
    )
    tbl["Rg"] = tbl["acc_retrain"] - tbl["acc_da"]
    tbl["gap_noda_retrain"] = tbl["acc_retrain"] - tbl["acc_noda"]
    return tbl


def render(store, dataset):
    st.header("Claim 3 — Positive Retraining Gap")
    st.caption(
        r"For each (dataset, feature): fit $R_g(z) = \rho_0 + \rho_1 z$. "
        "Compute fraction of the No-DA→Retrain gap that DA closes. Replicates paper Tables 3 & 4."
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
    da_method = st.selectbox("DA method", da_opts, format_func=lambda m: DA_LABEL.get(m, m))
    tbl = _compute_rg_table(merged, da_method)
    if tbl.empty:
        st.warning(f"No matched rows for DA={da_method}.")
        return

    # ── Per (dataset, feature) linear fit ───────────────────────────────────
    with st.container(border=True):
        st.subheader(r"$R_g(z)$ linear fit per (dataset × feature)")
        rows = []
        for (ds, feat), g in tbl.groupby(["dataset", "feature"]):
            if len(g) < 5:
                continue
            try:
                mod = smf.ols("Rg ~ drift_z", data=g).fit()
                rho0, rho1 = mod.params["Intercept"], mod.params["drift_z"]
                p1 = mod.pvalues["drift_z"]
            except Exception:
                rho0 = rho1 = p1 = np.nan
            q75 = g["Rg"].quantile(0.75)
            hd = g[g["drift_z"] >= g["drift_z"].quantile(0.75)]
            hd_mean = hd["Rg"].mean()
            hd_t = hd["Rg"].mean() / (hd["Rg"].std(ddof=1) / np.sqrt(len(hd))) if len(hd) > 1 else np.nan
            rows.append({
                "dataset": ds, "feature": feat, "n": len(g),
                "ρ₀": rho0, "ρ₁": rho1, "p(ρ₁)": p1,
                "Q75 Rg": q75, "high-drift mean Rg": hd_mean, "t": hd_t,
            })
        rg_df = pd.DataFrame(rows)
        st.dataframe(
            rg_df.style.format({
                "ρ₀": "{:+.3f}", "ρ₁": "{:+.3f}", "p(ρ₁)": "{:.3g}",
                "Q75 Rg": "{:+.3f}", "high-drift mean Rg": "{:+.3f}", "t": "{:+.2f}",
            }).background_gradient(
                subset=["high-drift mean Rg"], cmap="Reds", vmin=0, vmax=0.5,
            ),
            use_container_width=True, hide_index=True,
        )

    # ── "DA closes X% of gap" bar ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Fraction of No-DA → Retrain gap closed by {da_method.upper()}")
        rows = []
        for (ds, feat), g in tbl.groupby(["dataset", "feature"]):
            noda_mean = g["acc_noda"].mean()
            da_mean = g["acc_da"].mean()
            retr_mean = g["acc_retrain"].mean()
            gap = retr_mean - noda_mean
            closed = (da_mean - noda_mean) / gap if abs(gap) > 1e-6 else np.nan
            rows.append({
                "dataset": ds, "feature": feat,
                "No DA": noda_mean, da_method.upper(): da_mean, "Retrain": retr_mean,
                "closed_frac": closed,
            })
        sf = pd.DataFrame(rows)
        fig = go.Figure()
        for ds in sorted(sf["dataset"].unique()):
            sub = sf[sf["dataset"] == ds]
            fig.add_trace(go.Bar(
                x=sub["feature"] + "<br>(" + sub["dataset"] + ")",
                y=sub["closed_frac"] * 100, name=ds,
                marker_color=DATASET_COLORS.get(ds, "#4F46E5"),
                text=[f"{v*100:.0f}%" for v in sub["closed_frac"]],
                textposition="outside",
            ))
        fig.update_layout(
            yaxis_title=f"% of gap closed by {da_method.upper()}",
            xaxis_title="feature × dataset",
            yaxis=dict(range=[-20, 110]),
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sf.style.format({
                "No DA": "{:.3f}", da_method.upper(): "{:.3f}", "Retrain": "{:.3f}",
                "closed_frac": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Paper §5.4 (SA): BNCI2014_004 closes 78–87%; Stieger2021 closes 20–26%; "
            "Ma2020 closes 5–6% — confirming DA is insufficient in high-drift regimes."
        )

    # ── Dumbbell chart: No-DA → DA → Retrain per (dataset × feature × clf) ─
    with st.container(border=True):
        st.subheader("Dumbbell — strategy progression")
        st.caption(
            "One dumbbell per (dataset × feature × classifier). "
            "Line length = full No-DA→Retrain gap; DA marker shows how far "
            f"{da_method.upper()} closes it."
        )
        dumb_rows = []
        has_clf = "classifier" in tbl.columns
        group_cols = ["dataset", "feature", "classifier"] if has_clf else ["dataset", "feature"]
        for keys, g in tbl.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(group_cols, keys))
            info["No DA"] = g["acc_noda"].mean()
            info[da_method.upper()] = g["acc_da"].mean()
            info["Retrain"] = g["acc_retrain"].mean()
            dumb_rows.append(info)
        db = pd.DataFrame(dumb_rows)
        if has_clf:
            db["label"] = (db["feature"] + " · "
                            + db["classifier"] + " · " + db["dataset"])
        else:
            db["label"] = db["feature"] + " · " + db["dataset"]
        # Sort by gap size for visual impact
        db["_gap"] = db["Retrain"] - db["No DA"]
        db = db.sort_values("_gap", ascending=False).reset_index(drop=True)

        fig_db = go.Figure()
        for i, r in db.iterrows():
            # Base line from No DA to Retrain
            fig_db.add_trace(go.Scatter(
                x=[r["No DA"], r["Retrain"]], y=[r["label"]] * 2,
                mode="lines", line=dict(color="#CBD5E1", width=6),
                showlegend=False, hoverinfo="skip",
            ))
            # DA segment from No DA to DA (colored)
            da_val = r[da_method.upper()]
            fig_db.add_trace(go.Scatter(
                x=[r["No DA"], da_val], y=[r["label"]] * 2,
                mode="lines", line=dict(color="#F59E0B", width=6),
                showlegend=False, hoverinfo="skip",
            ))
        # No DA markers (gray)
        fig_db.add_trace(go.Scatter(
            x=db["No DA"], y=db["label"], mode="markers",
            marker=dict(color="#475569", size=14,
                         line=dict(width=2, color="#1E293B")),
            name="No DA",
        ))
        # DA markers (amber)
        fig_db.add_trace(go.Scatter(
            x=db[da_method.upper()], y=db["label"], mode="markers",
            marker=dict(color="#F59E0B", size=14,
                         line=dict(width=2, color="#1E293B")),
            name=da_method.upper(),
        ))
        # Retrain markers (green)
        fig_db.add_trace(go.Scatter(
            x=db["Retrain"], y=db["label"], mode="markers",
            marker=dict(color="#10B981", size=14,
                         line=dict(width=2, color="#1E293B"), symbol="square"),
            name="Retrain",
        ))
        fig_db.update_layout(
            xaxis_title="mean accuracy",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.12),
            height=max(300, 40 * len(db) + 120),
        )
        style_figure(fig_db)
        st.plotly_chart(fig_db, use_container_width=True)

    # ── R_g vs drift scatter ────────────────────────────────────────────────
    with st.expander("R_g vs drift_z scatter"):
        fig = go.Figure()
        for (ds, feat), g in tbl.groupby(["dataset", "feature"]):
            fig.add_trace(go.Scatter(
                x=g["drift_z"], y=g["Rg"], mode="markers", name=f"{ds}/{feat}",
                marker=dict(size=4, opacity=0.4),
            ))
        fig.update_layout(
            xaxis_title="drift_z", yaxis_title=f"R_g = acc_retrain − acc_{da_method}",
            legend=dict(orientation="h", y=-0.15),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
        style_figure(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
