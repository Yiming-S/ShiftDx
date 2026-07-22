"""Page 4: Claim 2 — DA Benefit Decomposition (interactive fan chart, D3.1=A)."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.formula.api as smf

from utils import (
    DA_LABEL, DA_SHORT_LABEL, DISTANCE_LABEL,
    SIG_LEGEND, about_page, apply_ctx_classifier, apply_ctx_dataset,
    apply_ctx_metric, available_da_methods, build_claim_formula,
    cluster_bootstrap_ci, da_colors, download_bar, empty_state, feature_colors,
    fit_lmm_or_fallback, get_ctx, is_retrain, mixedlm_pseudo_r2, pvalue_badge,
    style_figure,
)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _fit_claim2(df: pd.DataFrame) -> dict:
    """Interaction MixedLM (paper Table 2). Robust to degenerate slices: drops
    single-level factors, reports a no-variance drift slice clearly, and falls
    back to OLS with subject-cluster-robust SEs if the mixed fit is singular."""
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough subjects in this slice (need at least 2)."}
    df = df.copy()
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    d = df.dropna(subset=["drift_z", "acc_centered"])
    if len(d) < 3 or d["subject_tag"].nunique() < 2 or d["has_da"].nunique() < 2:
        return {"error": "Not enough usable rows (need both No-DA and DA rows "
                         "with drift_z / acc_centered)."}
    if float(d["drift_z"].std(ddof=1)) < 1e-8:
        return {"error": "Drift has essentially no variation in this slice, so the "
                         "drift × DA interaction cannot be estimated. Widen the "
                         "selection or pick a different drift metric."}

    formula = build_claim_formula(d, "acc_centered", interact_da=True)
    try:
        result, method, is_mixed = fit_lmm_or_fallback(d, formula, "subject_tag")
    except Exception as exc:
        return {"error": f"Model fit failed: {exc}"}

    tbl = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "SE": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
    })
    tbl = tbl[~tbl["term"].astype(str).str.contains("Var|Group", na=False)]
    r2m, r2c = (mixedlm_pseudo_r2(result) if is_mixed
                else (float(getattr(result, "rsquared", np.nan)), np.nan))
    return {"table": tbl, "n": int(len(d)),
            "n_subj": int(d["subject_tag"].nunique()),
            "r2_marginal": r2m, "r2_conditional": r2c,
            "method": method, "is_mixed": is_mixed}


@st.cache_data(show_spinner=False)
def _gee_claim2(df: pd.DataFrame) -> dict:
    """SUPPLEMENTARY cluster-robust GEE (exchangeable working correlation).
    Not the published model — robustness check on the DA-effect terms."""
    if df.empty or df["subject"].nunique() < 2:
        return {"error": "Not enough groups."}
    df = df.copy()
    df["feature"] = df["feature"].astype("category")
    df["dataset"] = df["dataset"].astype("category")
    df["subject_tag"] = df["dataset"].astype(str) + "_" + df["subject"].astype(str)
    try:
        import statsmodels.api as sm
        model = smf.gee(
            "acc_centered ~ drift_z * has_da * C(feature) + C(dataset)",
            groups="subject_tag", data=df,
            cov_struct=sm.cov_struct.Exchangeable(), family=sm.families.Gaussian(),
        )
        res = model.fit()
    except Exception as exc:
        return {"error": f"GEE failed: {exc}"}
    tbl = pd.DataFrame({
        "term": res.params.index, "coef": res.params.values,
        "robust SE": res.bse.values, "p": res.pvalues.values,
    })
    return {"table": tbl[tbl["term"].str.contains("has_da", case=False, na=False)]}


@st.cache_data(show_spinner=False)
def _perm_eta4(df: pd.DataFrame, n_perm: int = 500, seed: int = 0) -> dict:
    """Within-subject permutation p for η₄ (the drift_z:has_da interaction),
    using the OLS interaction coefficient as the statistic. Permuting the DA
    label WITHIN subject preserves clustering under the null of no DA effect."""
    work = df.dropna(subset=["acc_centered", "drift_z"]).copy()
    if work.empty or work["subject"].nunique() < 2:
        return {"error": "Not enough data."}
    work["subject_tag"] = work["dataset"].astype(str) + "_" + work["subject"].astype(str)

    def _eta4(d):
        try:
            m = smf.ols("acc_centered ~ drift_z * has_da", data=d).fit()
            return float(m.params.get("drift_z:has_da", np.nan))
        except Exception:
            return np.nan

    obs = _eta4(work)
    if not np.isfinite(obs):
        return {"error": "Could not fit observed model."}
    rng = np.random.default_rng(seed)
    null = []
    groups = [g for _, g in work.groupby("subject_tag")]
    for _ in range(int(n_perm)):
        permuted = []
        for g in groups:
            gg = g.copy()
            gg["has_da"] = rng.permutation(gg["has_da"].to_numpy())
            permuted.append(gg)
        stat = _eta4(pd.concat(permuted, ignore_index=True))
        if np.isfinite(stat):
            null.append(stat)
    if len(null) < 20:
        return {"error": "Too many failed permutations."}
    null = np.asarray(null)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return {"obs": obs, "p": p, "n_perm": len(null)}


def _quartile_codes(s: pd.Series, q: int = 4):
    """Integer quartile codes (0..n_bins-1; NaN outside), guarded against the
    qcut(duplicates='drop') trap where ties collapse bins below q."""
    try:
        codes = pd.qcut(s, q=q, labels=False, duplicates="drop")
    except Exception:
        codes = pd.Series(np.nan, index=s.index)
    valid = codes.dropna()
    n_bins = int(valid.max()) + 1 if len(valid) else 0
    return codes, n_bins


@st.cache_data(show_spinner=False)
def _fan_data(work: pd.DataFrame, da_method: str, methods: tuple,
              n_boot: int = 600) -> tuple:
    """Per (series × drift-quartile) mean accuracy with subject-clustered
    bootstrap 95% CIs. `work` is already filtered to one feature."""
    w = work.copy()
    codes, n_bins = _quartile_codes(w["drift_z"])
    w["drift_bin"] = codes
    bins = sorted(int(b) for b in pd.Series(codes).dropna().unique())
    cluster_col = "uid" if "uid" in w.columns else "subject"

    series = [("No DA", w[(w["strategy"] == "train_once") & (w["da"] == "none")])]
    rt = is_retrain(w)
    if not rt.empty:
        series.append(("Retrain", rt))
    for m in methods:
        series.append((DA_LABEL.get(m, m),
                       w[(w["strategy"] == "train_once_da") & (w["da"] == m)]))

    rows = []
    for sname, sl in series:
        for b in bins:
            cell = sl[sl["drift_bin"] == b]
            if cell.empty:
                continue
            pt, lo, hi = cluster_bootstrap_ci(
                cell, lambda d: d["accuracy"].mean(),
                cluster_col=cluster_col, n_boot=n_boot, seed=1000 * b + len(sname),
            )
            rows.append({"series": sname, "bin": b, "bin_label": f"Q{b + 1}",
                         "mean": pt, "lo": lo, "hi": hi, "n": len(cell)})
    counts = {f"Q{b + 1}": int((codes == b).sum()) for b in bins}
    return pd.DataFrame(rows), n_bins, counts


def render(store):
    ctx = get_ctx()
    da_method = ctx["da"]
    fcolors = feature_colors()
    dcolors = da_colors()

    st.header("Claim 2 · DA decomposition")
    st.caption(
        "Does DA close the loss as a **level shift** (constant gain) or a **slope change** "
        "(reduced drift sensitivity)? "
        r"Fit $acc\_centered \sim drift\_z \times has\_da \times feature + dataset$. "
        r"$\eta_2$ = level shift; $\eta_4$ = slope change (paper Table 2)."
    )

    about_page(
        what_you_see=[
            "OLS lines of No DA vs chosen DA per feature.",
            "Mixed-effects interaction table: level shift + slope change.",
            "Fan chart with subject-clustered bootstrap CIs across drift quartiles.",
        ],
        how_to_read=[
            "Parallel lines higher up = level shift (η₂).",
            "Shallower DA slope than No-DA slope = slope change (η₄ > 0).",
            "Fan-chart error bars are between-subject bootstrap CIs (not naive SEM).",
        ],
        paper_ref="§5.3, Table 2",
        key_terms=[
            ("η₂", "level shift: constant DA gain."),
            ("η₄", "slope change: reduction in drift sensitivity."),
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
        empty_state("No DA methods present",
                    "The merged file has no `train_once_da` rows.")
        return

    if da_method not in da_opts:
        da_method = da_opts[0]
        st.caption(f"Sidebar DA `{ctx['da']}` is not in this slice — "
                   f"falling back to `{da_method}`.")

    noda = merged[(merged["strategy"] == "train_once") &
                  (merged["da"] == "none")].copy()
    da = merged[(merged["strategy"] == "train_once_da") &
                (merged["da"] == da_method)].copy()

    if noda.empty or da.empty:
        empty_state(f"No rows for DA = {da_method.upper()}",
                    "Try a different DA method in the sidebar.")
        return

    noda["has_da"] = 0
    da["has_da"] = 1
    pooled = pd.concat([noda, da], ignore_index=True)

    # ── Scatter: No-DA vs DA regression lines by feature ────────────────────
    with st.container(border=True):
        st.subheader(f"No DA vs DA · {da_method.upper()} regression lines")
        fig = go.Figure()
        for feat in sorted(pooled["feature"].unique()):
            col = fcolors.get(feat, "#4F46E5")
            g_nd = noda[noda["feature"] == feat]
            g_da = da[da["feature"] == feat]
            if len(g_nd) >= 2:
                c = np.polyfit(g_nd["drift_z"], g_nd["acc_centered"], 1)
                xs = np.linspace(g_nd["drift_z"].min(), g_nd["drift_z"].max(), 40)
                fig.add_trace(go.Scatter(
                    x=xs, y=np.polyval(c, xs), mode="lines",
                    line=dict(color=col, width=3, dash="dot"),
                    name=f"{feat} No DA (slope={c[0]:+.3f})",
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
            xaxis_title="drift (z-score)",
            yaxis_title="accuracy vs baseline",
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── MixedLM fit ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Interaction model coefficients")
        with st.spinner("Fitting interaction model…"):
            fit = _fit_claim2(pooled)
        if "error" in fit:
            st.info(fit["error"])
        else:
            if not fit.get("is_mixed", True):
                st.warning(f"⚠ Estimator fell back to **{fit['method']}** — the "
                           "mixed model was rank-deficient on this slice. Point "
                           "estimates are still valid.")
            c1, c2, c3 = st.columns(3)
            c1.metric("n observations", f"{fit['n']:,}")
            c2.metric("n subjects", fit["n_subj"])
            r2m, r2c = fit.get("r2_marginal"), fit.get("r2_conditional")
            c3.metric("pseudo-R² (marg / cond)",
                      f"{r2m:.2f} / {r2c:.2f}" if r2m == r2m else "—")
            tbl = fit["table"]
            key_rows = tbl[tbl["term"].str.contains("has_da", case=False, na=False)].copy()
            key_rows["sig"] = key_rows["p"].apply(pvalue_badge)
            st.markdown("**Key DA-effect terms (η₂ / η₄ + interactions):**")
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
                download_bar("claim2_full_coefs", tbl, "claim2_mixedlm_full")
            st.caption(
                f"Legend: {SIG_LEGEND}.  Paper Table 2 (SA reference): "
                "η₂ = +0.035 (p<0.001), η₄ = +0.007 (p=0.114). PT yields a "
                "significant slope change.  MixedLM p-values are asymptotic."
            )

        # OPT-2: supplementary cluster-robust inference (not the published fit)
        with st.expander("Robustness: cluster-robust inference (supplementary — not the published fit)"):
            st.caption(
                "Two cluster-aware alternatives to the MixedLM, for the DA-effect "
                "terms only. **Not** the model behind paper Table 2."
            )
            mode = st.radio("Method", ["GEE (exchangeable, robust SE)",
                                       "Within-subject permutation (η₄)"],
                            horizontal=True, key="c2_robust_mode")
            if mode.startswith("GEE"):
                if st.checkbox("Fit GEE", value=False, key="c2_gee"):
                    with st.spinner("Fitting GEE…"):
                        g = _gee_claim2(pooled)
                    if "error" in g:
                        st.warning(g["error"])
                    else:
                        gt = g["table"].copy()
                        gt["sig"] = gt["p"].apply(pvalue_badge)
                        st.dataframe(
                            gt.style.format({"coef": "{:+.4f}", "robust SE": "{:.4f}",
                                             "p": "{:.4g}"}),
                            use_container_width=True, hide_index=True,
                        )
            else:
                if st.checkbox("Run permutation test (slower)", value=False, key="c2_perm"):
                    with st.spinner("Permuting DA labels within subject…"):
                        pr = _perm_eta4(pooled, n_perm=500, seed=0)
                    if "error" in pr:
                        st.warning(pr["error"])
                    else:
                        cc1, cc2 = st.columns(2)
                        cc1.metric("observed η₄ (OLS)", f"{pr['obs']:+.4f}")
                        cc2.metric(f"permutation p ({pr['n_perm']} perms)",
                                   f"{pr['p']:.3g}")
                        st.caption("Null: DA label shuffled within each subject; "
                                   "two-sided p on |η₄|.")

    # ── DA fan chart (interactive, D3.1=A) ──────────────────────────────────
    with st.container(border=True):
        st.subheader("DA method fan chart — accuracy by drift quartile")
        st.caption(
            "Default view: **No DA**, **Retrain**, and your **sidebar DA method**. "
            "Tick others in the right-hand list to add them. Error bars are "
            "subject-clustered bootstrap 95% CIs."
        )

        feat_pick = st.radio(
            "Feature", sorted(merged["feature"].unique()),
            horizontal=True, key="c2_fan_feat",
        )

        layout_main, layout_picker = st.columns([4, 1])

        with layout_picker:
            st.markdown("**Add DA methods**")
            other_methods = [m for m in da_opts if m != da_method]
            selected_extra = []
            for m in other_methods:
                if st.checkbox(DA_SHORT_LABEL.get(m, m.upper()),
                               value=False, key=f"c2_fan_{m}"):
                    selected_extra.append(m)

        work = merged[merged["feature"] == feat_pick].copy()
        methods_to_plot = tuple([da_method] + selected_extra)

        with st.spinner("Bootstrapping fan-chart CIs…"):
            fan_df, n_bins, counts = _fan_data(work, da_method, methods_to_plot)

        if n_bins < 4:
            st.warning(
                f"Only {n_bins} distinct drift quartile(s) on this slice "
                f"(ties collapsed the bins). Bin sizes: {counts}."
            )

        with layout_main:
            fig_fan = go.Figure()

            def _add(series_name, color, width, symbol, size):
                s = fan_df[fan_df["series"] == series_name].sort_values("bin")
                if s.empty:
                    return
                arr_up = (s["hi"] - s["mean"]).fillna(0.0)
                arr_dn = (s["mean"] - s["lo"]).fillna(0.0)
                fig_fan.add_trace(go.Scatter(
                    x=s["bin_label"], y=s["mean"],
                    error_y=dict(type="data", array=arr_up, arrayminus=arr_dn,
                                 thickness=1.2),
                    mode="lines+markers", name=series_name,
                    line=dict(color=color, width=width),
                    marker=dict(size=size, symbol=symbol),
                ))

            _add("No DA", "#475569", 4, "circle", 10)
            _add("Retrain", "#10B981", 4, "square", 10)
            for dm in methods_to_plot:
                lbl = DA_LABEL.get(dm, dm)
                width = 3 if dm == da_method else 1.6
                _add(lbl, dcolors.get(dm, "#4F46E5"), width, "circle",
                     7 if dm == da_method else 5)

            fig_fan.update_layout(
                xaxis_title="drift quartile (low → high)",
                yaxis_title="mean accuracy",
                legend=dict(orientation="v", x=1.02, y=1.0),
            )
            style_figure(fig_fan, height=480)
            st.plotly_chart(fig_fan, use_container_width=True)
            st.caption(
                "Hugging the green line = DA reaches the retrain ceiling. "
                "Dropping below the grey line = DA is actively harmful."
            )
            download_bar("claim2_fan", fan_df, f"claim2_fan_{da_method}_{feat_pick}")

    # ── Marginal gain by drift bin ──────────────────────────────────────────
    with st.expander(f"Mean DA gain by drift bin · {da_method.upper()} − No DA"):
        key = ["dataset", "subject", "feature", "target_session", "drift_z"]
        merged_pair = pd.merge(
            noda[key + ["accuracy"]].rename(columns={"accuracy": "acc_noda"}),
            da[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"}),
            on=key, how="inner",
        )
        if not merged_pair.empty:
            merged_pair["gain"] = merged_pair["acc_da"] - merged_pair["acc_noda"]
            codes, gain_bins = _quartile_codes(merged_pair["drift_z"])
            merged_pair["drift_bin"] = codes.map(
                lambda b: f"Q{int(b) + 1}" if pd.notna(b) else np.nan)
            agg = (merged_pair.dropna(subset=["drift_bin"])
                              .groupby(["drift_bin", "feature"], observed=True)["gain"]
                              .mean().reset_index())
            pivot = agg.pivot(index="drift_bin", columns="feature", values="gain")
            st.dataframe(
                pivot.style.format("{:+.4f}")
                     .background_gradient(cmap="RdBu_r", vmin=-0.05, vmax=0.05),
                use_container_width=True,
            )
            if gain_bins < 4:
                st.caption(f"Only {gain_bins} distinct drift bin(s) on this slice.")
            download_bar("claim2_gain_table", pivot.reset_index(),
                         f"claim2_gain_{da_method}")
