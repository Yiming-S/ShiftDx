"""Page 5: Claim 3 — Positive Retraining Gap."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import (
    about_page, apply_ctx_classifier, apply_ctx_dataset, apply_ctx_metric,
    available_da_methods, cluster_bootstrap_ci, cluster_bootstrap_p,
    dataset_colors, download_bar, empty_state, get_ctx, is_retrain,
    ols_slope, pvalue_badge, style_figure,
)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _compute_rg_table(merged: pd.DataFrame, da_method: str) -> pd.DataFrame:
    """Per (dataset, subject, feature, session_k): R_g = retrain − DA."""
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)]
    retrain = is_retrain(merged)

    key = ["dataset", "subject", "feature", "target_session", "drift_z"]
    tbl = (
        noda[key + ["accuracy"]].rename(columns={"accuracy": "acc_noda"})
        .merge(da[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"}),
               on=key, how="inner")
        .merge(retrain[["dataset", "subject", "feature", "target_session", "accuracy"]]
               .rename(columns={"accuracy": "acc_retrain"}),
               on=["dataset", "subject", "feature", "target_session"], how="inner")
    )
    tbl["Rg"] = tbl["acc_retrain"] - tbl["acc_da"]
    tbl["gap_noda_retrain"] = tbl["acc_retrain"] - tbl["acc_noda"]
    tbl["uid"] = tbl["dataset"].astype(str) + "_" + tbl["subject"].astype(str)
    return tbl


@st.cache_data(show_spinner=False)
def _rg_fits(tbl: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Per (dataset × feature): OLS R_g(z) = ρ₀ + ρ₁z with subject-clustered
    bootstrap CIs, plus a cluster-bootstrap test of the high-drift mean R_g."""
    def _intercept(d):
        g = d[["drift_z", "Rg"]].dropna()
        if len(g) < 2:
            return np.nan
        return float(np.polyfit(g["drift_z"].to_numpy(), g["Rg"].to_numpy(), 1)[1])

    rows = []
    for (ds, feat), g in tbl.groupby(["dataset", "feature"]):
        if len(g) < 5:
            continue
        try:
            mod = smf.ols("Rg ~ drift_z", data=g).fit()
            rho0, rho1 = mod.params["Intercept"], mod.params["drift_z"]
            p1 = mod.pvalues["drift_z"]
            r2 = mod.rsquared
        except Exception as exc:
            logger.debug("Rg OLS failed for %s/%s: %s", ds, feat, exc)
            rho0 = rho1 = p1 = r2 = np.nan
        _, rho0_lo, rho0_hi = cluster_bootstrap_ci(
            g, _intercept, cluster_col="uid", n_boot=n_boot, seed=11)
        _, rho1_lo, rho1_hi = cluster_bootstrap_ci(
            g, lambda d: ols_slope(d, "drift_z", "Rg"),
            cluster_col="uid", n_boot=n_boot, seed=12)

        hd = g[g["drift_z"] >= g["drift_z"].quantile(0.75)]
        hd_mean, hd_lo, hd_hi = cluster_bootstrap_ci(
            hd, lambda d: d["Rg"].mean(), cluster_col="uid",
            n_boot=n_boot, seed=13)
        _, hd_p = cluster_bootstrap_p(
            hd, lambda d: d["Rg"].mean(), cluster_col="uid",
            n_boot=2 * n_boot, seed=14)

        rows.append({
            "dataset": ds, "feature": feat, "n": len(g),
            "intercept": rho0, "ρ₀ CI": _fmt_ci(rho0_lo, rho0_hi),
            "slope": rho1, "ρ₁ CI": _fmt_ci(rho1_lo, rho1_hi),
            "p(slope)": p1, "R²": r2,
            "high-drift mean R_g": hd_mean, "R_g CI": _fmt_ci(hd_lo, hd_hi),
            "p(R_g>0, boot)": hd_p,
        })
    return pd.DataFrame(rows)


def _fmt_ci(lo, hi) -> str:
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
        return "—"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def render(store):
    ctx = get_ctx()
    da_method = ctx["da"]
    dscolors = dataset_colors()

    st.header("Claim 3 · Retraining gap")
    st.caption(
        r"Is there residual loss after DA? — fit $R_g(z) = \rho_0 + \rho_1 z$ "
        "per (dataset × feature). Compute the fraction of the No-DA → Retrain "
        "gap that DA closes. Replicates paper Tables 3 & 4."
    )

    about_page(
        what_you_see=[
            "Per (dataset × feature) intercept/slope of R_g vs drift, with "
            "subject-clustered bootstrap CIs.",
            "Bar chart: fraction of the No-DA → Retrain gap closed by DA.",
            "Dumbbell: strategy progression No DA → DA → Retrain.",
        ],
        how_to_read=[
            "Positive intercept (ρ₀) or positive slope (ρ₁) ⇒ retraining still beats DA.",
            "A high-drift R_g CI excluding 0 ⇒ residual loss is real, not noise.",
        ],
        paper_ref="§5.4, Tables 3–4",
        key_terms=[
            ("R_g(z)", "retrain minus DA accuracy at drift z."),
            ("ρ₀ / ρ₁", "intercept / slope of R_g(z)."),
            ("bootstrap CI", "resamples subjects (the cluster), not rows."),
        ],
    )

    merged = apply_ctx_dataset(store.merged_df)
    merged = apply_ctx_classifier(merged, ctx)
    merged = apply_ctx_metric(merged, ctx)
    if merged.empty:
        empty_state(
            "No merged data for this selection",
            f"dataset=`{ctx['dataset']}`, classifier=`{ctx['classifier']}` gives no rows.",
            dataset=ctx["dataset"],
        )
        return

    da_opts = available_da_methods(merged)
    if not da_opts:
        empty_state("No DA methods present", "The merged file has no `train_once_da` rows.")
        return
    if da_method not in da_opts:
        da_method = da_opts[0]
        st.caption(f"Sidebar DA `{ctx['da']}` not present here — using `{da_method}`.")

    tbl = _compute_rg_table(merged, da_method)
    if tbl.empty:
        empty_state(f"No matched rows for DA = {da_method.upper()}",
                    "Need (No-DA, DA, Retrain) all present for the same (subject, session_k).")
        return

    # ── Per (dataset, feature) linear fit ───────────────────────────────────
    with st.container(border=True):
        st.subheader(r"$R_g(z)$ linear fit per (dataset × feature)")
        with st.spinner("Bootstrapping ρ₀ / ρ₁ CIs…"):
            rg_df = _rg_fits(tbl)
        rg_show = rg_df.copy()
        rg_show["sig"] = rg_show["p(R_g>0, boot)"].apply(pvalue_badge)
        st.dataframe(
            rg_show.style.format({
                "intercept": "{:+.3f}", "slope": "{:+.3f}", "p(slope)": "{:.3g}",
                "R²": "{:.3f}", "high-drift mean R_g": "{:+.3f}",
                "p(R_g>0, boot)": "{:.3g}",
            }).background_gradient(
                subset=["high-drift mean R_g"], cmap="Reds", vmin=0, vmax=0.5,
            ),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "ρ₀ = intercept, ρ₁ = slope (Greek kept in formulas). CIs and the "
            "`p(R_g>0)` are subject-clustered bootstraps; `sig` flags that "
            "high-drift residual loss differs from zero."
        )
        download_bar("claim3_rg_fits", rg_df, f"claim3_rg_fits_{da_method}")

    # ── "DA closes X%" bar ──────────────────────────────────────────────────
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
                marker_color=dscolors.get(ds, "#4F46E5"),
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
            "Paper §5.4 (SA reference): BNCI2014_004 closes 78–87%; "
            "Stieger2021 closes 20–26%; Ma2020 closes 5–6% — confirming DA is "
            "insufficient in high-drift regimes."
        )
        download_bar("claim3_closed_frac", sf, f"claim3_gap_closed_{da_method}")

    # ── Dumbbell chart ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Dumbbell — strategy progression")
        st.caption(
            "One dumbbell per (dataset × feature × classifier). "
            "Grey line = full No-DA → Retrain gap; amber segment = how far "
            f"{da_method.upper()} closes it."
        )
        dumb_rows = []
        has_clf = "classifier" in tbl.columns
        group_cols = (["dataset", "feature", "classifier"]
                      if has_clf else ["dataset", "feature"])
        for keys, g in tbl.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(group_cols, keys))
            info["No DA"] = g["acc_noda"].mean()
            info[da_method.upper()] = g["acc_da"].mean()
            info["Retrain"] = g["acc_retrain"].mean()
            info["gap_closed_pct"] = (
                (info[da_method.upper()] - info["No DA"])
                / max(info["Retrain"] - info["No DA"], 1e-6) * 100
            )
            dumb_rows.append(info)
        db = pd.DataFrame(dumb_rows)
        if has_clf:
            db["label"] = (db["feature"] + " · "
                           + db["classifier"] + " · " + db["dataset"])
        else:
            db["label"] = db["feature"] + " · " + db["dataset"]
        db["_gap"] = db["Retrain"] - db["No DA"]
        db = db.sort_values("_gap", ascending=False).reset_index(drop=True)

        fig_db = go.Figure()
        for _, r in db.iterrows():
            fig_db.add_trace(go.Scatter(
                x=[r["No DA"], r["Retrain"]], y=[r["label"]] * 2,
                mode="lines", line=dict(color="#CBD5E1", width=6),
                showlegend=False, hoverinfo="skip",
            ))
            fig_db.add_trace(go.Scatter(
                x=[r["No DA"], r[da_method.upper()]], y=[r["label"]] * 2,
                mode="lines", line=dict(color="#F59E0B", width=6),
                showlegend=False, hoverinfo="skip",
            ))
        fig_db.add_trace(go.Scatter(
            x=db["No DA"], y=db["label"], mode="markers",
            marker=dict(color="#475569", size=14,
                         line=dict(width=2, color="#1E293B")),
            name="No DA",
            hovertemplate="%{y}<br>No DA = %{x:.3f}<extra></extra>",
        ))
        fig_db.add_trace(go.Scatter(
            x=db[da_method.upper()], y=db["label"], mode="markers",
            marker=dict(color="#F59E0B", size=14,
                         line=dict(width=2, color="#1E293B")),
            name=da_method.upper(),
            customdata=db[["gap_closed_pct"]],
            hovertemplate=(
                f"%{{y}}<br>{da_method.upper()} = %{{x:.3f}}"
                "<br>gap closed = %{customdata[0]:.0f}%<extra></extra>"
            ),
        ))
        fig_db.add_trace(go.Scatter(
            x=db["Retrain"], y=db["label"], mode="markers",
            marker=dict(color="#10B981", size=14,
                         line=dict(width=2, color="#1E293B"), symbol="square"),
            name="Retrain",
            hovertemplate="%{y}<br>Retrain = %{x:.3f}<extra></extra>",
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
        download_bar("claim3_dumbbell", db, f"claim3_dumbbell_{da_method}")

    # ── R_g vs drift scatter ────────────────────────────────────────────────
    with st.expander("R_g vs drift (z-score) scatter"):
        fig = go.Figure()
        for (ds, feat), g in tbl.groupby(["dataset", "feature"]):
            fig.add_trace(go.Scatter(
                x=g["drift_z"], y=g["Rg"], mode="markers", name=f"{ds}/{feat}",
                marker=dict(size=4, opacity=0.4),
            ))
        fig.update_layout(
            xaxis_title="drift (z-score)",
            yaxis_title=f"R_g = accuracy_retrain − accuracy_{da_method}",
            legend=dict(orientation="h", y=-0.15),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
        style_figure(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
