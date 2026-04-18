"""Page 6: Claim 4 — Feature Robustness (joint 4-condition criterion)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import filter_by_dataset, style_figure


def _compute_conditions(merged: pd.DataFrame, da_method: str, epsilon: float,
                        low_q: float = 0.25, high_q: float = 0.75) -> pd.DataFrame:
    """Evaluate the 4 joint robustness conditions per (dataset, feature)."""
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)]
    retrain = merged[(merged["strategy"] == "retrain") & (merged["ref_session"] == merged["target_session"])]

    rows = []
    for (ds, feat), g_nd in noda.groupby(["dataset", "feature"]):
        g_da = da[(da["dataset"] == ds) & (da["feature"] == feat)]
        g_rt = retrain[(retrain["dataset"] == ds) & (retrain["feature"] == feat)]

        # (a) low-drift No-DA
        low_cut = g_nd["drift_z"].quantile(low_q)
        a_val = g_nd.loc[g_nd["drift_z"] <= low_cut, "accuracy"].mean()

        # (b) drift slope (No-DA)
        if len(g_nd) >= 5:
            b_val = np.polyfit(g_nd["drift_z"], g_nd["accuracy"], 1)[0]
        else:
            b_val = np.nan

        # (c) high-drift R_g = retrain - DA
        key = ["dataset", "subject", "feature", "target_session", "drift_z"]
        pair = (g_da[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"})
                .merge(g_rt[["dataset", "subject", "feature", "target_session", "accuracy"]]
                       .rename(columns={"accuracy": "acc_retrain"}),
                       on=["dataset", "subject", "feature", "target_session"], how="inner"))
        if not pair.empty:
            high_cut = pair["drift_z"].quantile(high_q)
            hd = pair[pair["drift_z"] >= high_cut]
            c_val = (hd["acc_retrain"] - hd["acc_da"]).mean()
        else:
            c_val = np.nan

        # (d) retrain ceiling
        d_val = g_rt["accuracy"].mean() if not g_rt.empty else np.nan

        rows.append({
            "dataset": ds, "feature": feat,
            "(a) low-drift No-DA": a_val,
            "(b) drift slope": b_val,
            "(c) high-drift Rg": c_val,
            "(d) retrain ceiling": d_val,
        })
    return pd.DataFrame(rows)


def _pass_fail(df: pd.DataFrame, target_feat: str, baseline_feat: str, eps: float) -> pd.DataFrame:
    """For each dataset, evaluate whether `target_feat` dominates `baseline_feat`."""
    out_rows = []
    for ds, g in df.groupby("dataset"):
        tg = g[g["feature"] == target_feat]
        bl = g[g["feature"] == baseline_feat]
        if tg.empty or bl.empty:
            continue
        tg = tg.iloc[0]
        bl = bl.iloc[0]
        conds = {
            "(a) low-drift No-DA": tg["(a) low-drift No-DA"] >= bl["(a) low-drift No-DA"] - eps,
            "(b) drift slope":     tg["(b) drift slope"]     >  bl["(b) drift slope"],
            "(c) high-drift Rg":   tg["(c) high-drift Rg"]   <  bl["(c) high-drift Rg"],
            "(d) retrain ceiling": tg["(d) retrain ceiling"] >= bl["(d) retrain ceiling"] - eps,
        }
        for cname, passed in conds.items():
            out_rows.append({
                "dataset": ds, "condition": cname,
                baseline_feat: bl[cname], target_feat: tg[cname],
                "pass": "✅" if passed else "❌",
            })
    return pd.DataFrame(out_rows)


def render(store, dataset):
    st.header("Claim 4 — Feature Robustness")
    st.caption(
        "Joint 4-condition test: low-drift competence, drift slope, high-drift Rg, "
        "retrained ceiling. Condition (d) is the ceiling anchor that prevents confounding "
        "low information content with genuine drift immunity. Replicates paper Table 5."
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
    c1, c2, c3, c4 = st.columns(4)
    da_method = c1.selectbox("DA method", da_opts,
                              format_func=lambda m: DA_LABEL.get(m, m), index=0)
    eps = c2.slider("ε (tolerance)", 0.01, 0.15, 0.05, 0.01)
    features_avail = sorted(merged["feature"].unique())
    # Default: logvar challenges CSP (paper's main comparison)
    default_target = "logvar" if "logvar" in features_avail else features_avail[0]
    default_base = "CSP" if "CSP" in features_avail else features_avail[-1]
    target_feat = c3.selectbox("Target feature (claimant)", features_avail,
                               index=features_avail.index(default_target))
    baseline_feat = c4.selectbox(
        "Baseline feature (challenged)",
        [f for f in features_avail if f != target_feat],
        index=max(0, [f for f in features_avail if f != target_feat].index(default_base))
        if default_base in features_avail and default_base != target_feat else 0,
    )

    cond_df = _compute_conditions(merged, da_method, eps)
    cond_df = cond_df[cond_df["feature"].isin([target_feat, baseline_feat])]
    if cond_df.empty:
        st.warning("No rows for selected features.")
        return

    # ── Raw values ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Raw condition values")
        display = cond_df.set_index(["dataset", "feature"])
        st.dataframe(
            display.style.format({
                "(a) low-drift No-DA": "{:.3f}",
                "(b) drift slope": "{:+.3f}",
                "(c) high-drift Rg": "{:+.3f}",
                "(d) retrain ceiling": "{:.3f}",
            }),
            use_container_width=True,
        )

    # ── Pass/fail table ─────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Is **{target_feat}** robustly dominant over **{baseline_feat}**?")
        pf = _pass_fail(cond_df, target_feat, baseline_feat, eps)
        if pf.empty:
            st.info("Not enough data.")
        else:
            st.dataframe(
                pf.style.format({baseline_feat: "{:.3f}", target_feat: "{:.3f}"})
                  .apply(lambda s: ["background-color:#FEE2E2" if v == "❌" else
                                     "background-color:#DCFCE7" if v == "✅" else ""
                                     for v in s],
                          subset=["pass"], axis=0),
                use_container_width=True, hide_index=True,
            )
            all_pass = (pf["pass"] == "✅").groupby(pf["dataset"]).all()
            for ds, passed in all_pass.items():
                if passed:
                    st.success(f"**{ds}**: {target_feat} **IS** robustly dominant over {baseline_feat}.")
                else:
                    fails = pf[(pf["dataset"] == ds) & (pf["pass"] == "❌")]["condition"].tolist()
                    st.error(f"**{ds}**: {target_feat} fails {len(fails)}/4 conditions → {', '.join(fails)}")

    # ── Radar chart ─────────────────────────────────────────────────────────
    with st.expander("Per-dataset radar (normalized conditions)"):
        for ds in sorted(cond_df["dataset"].unique()):
            sub = cond_df[cond_df["dataset"] == ds]
            st.markdown(f"**{ds}**")
            fig = go.Figure()
            cat = ["(a)", "(b)", "(c)", "(d)"]
            for feat in [target_feat, baseline_feat]:
                r = sub[sub["feature"] == feat]
                if r.empty:
                    continue
                r = r.iloc[0]
                # Normalize so larger is "better" on a common 0-1 scale per condition
                # Here we just show raw values to stay interpretable
                vals = [
                    r["(a) low-drift No-DA"],
                    max(0, r["(b) drift slope"] + 0.05),  # shift so negative slopes are smaller
                    max(0, 0.5 - abs(r["(c) high-drift Rg"])),
                    r["(d) retrain ceiling"],
                ]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cat + [cat[0]],
                    fill="toself", name=feat,
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            style_figure(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
