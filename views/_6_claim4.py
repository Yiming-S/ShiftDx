"""Page 6: Claim 4 — Feature Robustness (joint 4-condition criterion)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    about_page, apply_ctx_classifier, apply_ctx_dataset, apply_ctx_metric,
    available_da_methods, DA_LABEL,
    download_bar, empty_state, get_ctx, style_figure,
)


def _cond_values(nd: pd.DataFrame, da_: pd.DataFrame, rt: pd.DataFrame,
                 low_q: float, high_q: float):
    """The 4 raw condition values (a,b,c,d) from already-sliced frames."""
    # (a) low-drift No-DA competence
    if nd.empty:
        a_val = np.nan
    else:
        low_cut = nd["drift_z"].quantile(low_q)
        a_val = nd.loc[nd["drift_z"] <= low_cut, "accuracy"].mean()

    # (b) drift slope under No-DA
    b_val = (np.polyfit(nd["drift_z"], nd["accuracy"], 1)[0]
             if len(nd) >= 5 else np.nan)

    # (c) high-drift R_g = retrain − DA
    key = ["dataset", "subject", "feature", "target_session", "drift_z"]
    pair = (da_[key + ["accuracy"]].rename(columns={"accuracy": "acc_da"})
            .merge(rt[["dataset", "subject", "feature", "target_session", "accuracy"]]
                   .rename(columns={"accuracy": "acc_retrain"}),
                   on=["dataset", "subject", "feature", "target_session"], how="inner"))
    if not pair.empty:
        high_cut = pair["drift_z"].quantile(high_q)
        hd = pair[pair["drift_z"] >= high_cut]
        c_val = (hd["acc_retrain"] - hd["acc_da"]).mean()
    else:
        c_val = np.nan

    # (d) retrain ceiling
    d_val = rt["accuracy"].mean() if not rt.empty else np.nan
    return a_val, b_val, c_val, d_val


@st.cache_data(show_spinner=False)
def _compute_conditions(merged: pd.DataFrame, da_method: str,
                        low_q: float = 0.25, high_q: float = 0.75) -> pd.DataFrame:
    """Evaluate the 4 joint robustness conditions per (dataset, feature)."""
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)]
    retrain = merged[(merged["strategy"] == "retrain") &
                     (merged["ref_session"] == merged["target_session"])]

    rows = []
    for (ds, feat), g_nd in noda.groupby(["dataset", "feature"]):
        g_da = da[(da["dataset"] == ds) & (da["feature"] == feat)]
        g_rt = retrain[(retrain["dataset"] == ds) & (retrain["feature"] == feat)]
        a, b, c, d = _cond_values(g_nd, g_da, g_rt, low_q, high_q)
        rows.append({
            "dataset": ds, "feature": feat,
            "(a) low-drift No-DA": a,
            "(b) drift slope": b,
            "(c) high-drift R_g": c,
            "(d) retrain ceiling": d,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _condition_cis(merged: pd.DataFrame, da_method: str, low_q: float,
                   high_q: float, n_boot: int = 500) -> pd.DataFrame:
    """Subject-clustered bootstrap 95% CIs for each condition per (dataset,
    feature). Resamples subjects and recomputes (a,b,c,d) on their rows."""
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    da = merged[(merged["strategy"] == "train_once_da") & (merged["da"] == da_method)]
    retrain = merged[(merged["strategy"] == "retrain") &
                     (merged["ref_session"] == merged["target_session"])]

    rows = []
    for (ds, feat), g_nd in noda.groupby(["dataset", "feature"]):
        g_da = da[(da["dataset"] == ds) & (da["feature"] == feat)]
        g_rt = retrain[(retrain["dataset"] == ds) & (retrain["feature"] == feat)]
        subs = np.unique(np.concatenate([
            g_nd["subject"].unique(), g_da["subject"].unique(), g_rt["subject"].unique(),
        ])) if len(g_nd) else np.array([])
        if len(subs) < 3:
            rows.append({"dataset": ds, "feature": feat})
            continue
        nd_by = {s: g_nd[g_nd["subject"] == s] for s in subs}
        da_by = {s: g_da[g_da["subject"] == s] for s in subs}
        rt_by = {s: g_rt[g_rt["subject"] == s] for s in subs}
        rng = np.random.default_rng(7)
        acc = {0: [], 1: [], 2: [], 3: []}
        for _ in range(int(n_boot)):
            pick = rng.choice(subs, size=len(subs), replace=True)
            nd_s = pd.concat([nd_by[s] for s in pick], ignore_index=True)
            da_s = pd.concat([da_by[s] for s in pick], ignore_index=True)
            rt_s = pd.concat([rt_by[s] for s in pick], ignore_index=True)
            vals = _cond_values(nd_s, da_s, rt_s, low_q, high_q)
            for i, v in enumerate(vals):
                if v == v:  # not NaN
                    acc[i].append(v)
        out = {"dataset": ds, "feature": feat}
        names = ["a", "b", "c", "d"]
        for i, nm in enumerate(names):
            if len(acc[i]) >= 20:
                lo, hi = np.percentile(acc[i], [2.5, 97.5])
                out[f"{nm}_lo"], out[f"{nm}_hi"] = float(lo), float(hi)
            else:
                out[f"{nm}_lo"] = out[f"{nm}_hi"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _pass_fail(df: pd.DataFrame, target_feat: str, baseline_feat: str,
               eps: float) -> pd.DataFrame:
    """For each dataset, evaluate whether `target_feat` dominates `baseline_feat`."""
    out_rows = []
    for ds, g in df.groupby("dataset"):
        tg = g[g["feature"] == target_feat]
        bl = g[g["feature"] == baseline_feat]
        if tg.empty or bl.empty:
            continue
        tg = tg.iloc[0]
        bl = bl.iloc[0]
        a_gap = bl["(a) low-drift No-DA"] - tg["(a) low-drift No-DA"]
        d_gap = bl["(d) retrain ceiling"] - tg["(d) retrain ceiling"]
        rows = [
            (
                "(a) low-drift No-DA", True,
                tg["(a) low-drift No-DA"] >= bl["(a) low-drift No-DA"] - eps,
                tg["(a) low-drift No-DA"], bl["(a) low-drift No-DA"],
                max(0.0, a_gap),
            ),
            (
                "(b) drift slope", False,
                tg["(b) drift slope"] > bl["(b) drift slope"],
                tg["(b) drift slope"], bl["(b) drift slope"], np.nan,
            ),
            (
                "(c) high-drift R_g", False,
                tg["(c) high-drift R_g"] < bl["(c) high-drift R_g"],
                tg["(c) high-drift R_g"], bl["(c) high-drift R_g"], np.nan,
            ),
            (
                "(d) retrain ceiling", True,
                tg["(d) retrain ceiling"] >= bl["(d) retrain ceiling"] - eps,
                tg["(d) retrain ceiling"], bl["(d) retrain ceiling"],
                max(0.0, d_gap),
            ),
        ]
        for cname, uses_eps, passed, tv, bv, eps_need in rows:
            out_rows.append({
                "dataset": ds, "condition": cname,
                f"{target_feat} (tg)": tv, f"{baseline_feat} (bl)": bv,
                "Δ (tg−bl)": tv - bv,
                "ε needed to pass": eps_need if uses_eps else np.nan,
                "uses ε?": "✓" if uses_eps else "—",
                "pass": "✅" if passed else "❌",
            })
    return pd.DataFrame(out_rows)


def _ci_string(ci_df: pd.DataFrame, ds: str, feat: str, letter: str) -> str:
    if ci_df is None or ci_df.empty:
        return "—"
    r = ci_df[(ci_df["dataset"] == ds) & (ci_df["feature"] == feat)]
    if r.empty:
        return "—"
    lo, hi = r.iloc[0].get(f"{letter}_lo"), r.iloc[0].get(f"{letter}_hi")
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
        return "—"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def render(store):
    ctx = get_ctx()

    st.header("Claim 4 · Feature robustness")
    st.caption(
        "Is your target feature genuinely more drift-robust than the baseline? "
        "Joint 4-condition test: low-drift competence, drift slope, high-drift R_g, "
        "retrain ceiling. Condition (d) prevents confounding low information content "
        "with genuine drift immunity. Replicates paper Table 5."
    )

    about_page(
        what_you_see=[
            "Raw value of the 4 conditions per (dataset × feature), with bootstrap CIs.",
            "Conclusion: does the target feature dominate the baseline on all 4?",
            "Radar of condition values for each dataset.",
        ],
        how_to_read=[
            "All 4 ✅ ⇒ target feature is **robustly dominant**.",
            "Wide / overlapping CIs ⇒ the verdict is fragile, not decisive.",
        ],
        paper_ref="§5.5, Table 5",
        key_terms=[
            ("ε (tolerance)", "allowed slack in conditions (a) and (d)."),
            ("target / baseline", "the feature you're claiming vs the one you're comparing to."),
            ("CI", "subject-clustered bootstrap 95% interval for each condition value."),
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
        empty_state("No DA methods present", "No `train_once_da` rows in the merged file.")
        return

    c1, c2, c3, c4 = st.columns(4)
    default_da = ctx["da"] if ctx["da"] in da_opts else da_opts[0]
    da_method = c1.selectbox(
        "DA method (page)", da_opts, index=da_opts.index(default_da),
        format_func=lambda m: DA_LABEL.get(m, m),
    )
    eps = c2.slider(
        "ε (tolerance)", 0.00, 0.30, 0.05, 0.01,
        help="Tolerance applied to conditions (a) low-drift No-DA and "
             "(d) retrain ceiling. Conditions (b) drift slope and "
             "(c) high-drift R_g are hard inequalities — ε does not touch them.",
    )
    features_avail = sorted(merged["feature"].unique())
    default_target = "logvar" if "logvar" in features_avail else features_avail[0]
    default_base = "CSP" if "CSP" in features_avail else features_avail[-1]
    target_feat = c3.selectbox("Target feature (claimant)", features_avail,
                               index=features_avail.index(default_target))
    base_choices = [f for f in features_avail if f != target_feat]
    base_idx = (base_choices.index(default_base)
                if default_base in base_choices else 0)
    baseline_feat = c4.selectbox(
        "Baseline feature (challenged)", base_choices, index=base_idx,
    )

    cond_df = _compute_conditions(merged, da_method)
    cond_df = cond_df[cond_df["feature"].isin([target_feat, baseline_feat])]
    if cond_df.empty:
        empty_state("No rows for selected features", "Try a different feature pair.")
        return
    with st.spinner("Bootstrapping condition CIs…"):
        ci_df = _condition_cis(merged, da_method, 0.25, 0.75)

    # ── Headline verdict ────────────────────────────────────────────────────
    pf_preview = _pass_fail(cond_df, target_feat, baseline_feat, eps)
    with st.container(border=True):
        st.subheader(f"Does **{target_feat}** robustly dominate **{baseline_feat}**?")
        if pf_preview.empty:
            st.info("Not enough data to decide.")
        else:
            all_pass = (pf_preview["pass"] == "✅").groupby(pf_preview["dataset"]).all()
            cols = st.columns(len(all_pass))
            for i, (ds, passed) in enumerate(all_pass.items()):
                with cols[i]:
                    if passed:
                        st.success(f"**{ds}** — {target_feat} dominates ✅")
                    else:
                        fails = pf_preview[(pf_preview["dataset"] == ds) &
                                           (pf_preview["pass"] == "❌")]["condition"].tolist()
                        st.error(f"**{ds}** — fails {len(fails)}/4: "
                                 + ", ".join(c.split(")")[-1].strip() for c in fails))

    # ── Raw values (with bootstrap CIs) ─────────────────────────────────────
    with st.container(border=True):
        st.subheader("Raw condition values (with bootstrap 95% CI)")
        letters = {"(a) low-drift No-DA": "a", "(b) drift slope": "b",
                   "(c) high-drift R_g": "c", "(d) retrain ceiling": "d"}
        disp_rows = []
        for _, r in cond_df.iterrows():
            row = {"dataset": r["dataset"], "feature": r["feature"]}
            for col, letter in letters.items():
                row[col] = r[col]
                row[f"{col} CI"] = _ci_string(ci_df, r["dataset"], r["feature"], letter)
            disp_rows.append(row)
        disp = pd.DataFrame(disp_rows)
        st.dataframe(
            disp.style.format({
                "(a) low-drift No-DA": "{:.3f}",
                "(b) drift slope": "{:+.3f}",
                "(c) high-drift R_g": "{:+.3f}",
                "(d) retrain ceiling": "{:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption("CIs are subject-clustered bootstraps; wide intervals warn that "
                   "the point-estimate verdict above may not be decisive.")
        download_bar("claim4_conds", disp, "claim4_conditions")

    # ── Pass/fail detail table ──────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Per-condition detail")
        st.caption(
            "**Reading guide.** `Δ (tg−bl)` is the raw gap. "
            "For (a) / (d), the cell passes iff `Δ ≥ −ε` — i.e. you need "
            "`ε ≥ |Δ|` whenever `Δ < 0`. That threshold is shown as **ε needed to pass**. "
            "(b) / (c) are hard inequalities — ε has no effect."
        )
        if pf_preview.empty:
            st.info("Not enough data.")
        else:
            near_flip = pf_preview["ε needed to pass"].apply(
                lambda v: (not pd.isna(v)) and abs(v - eps) < 0.05
            )
            styled = (
                pf_preview.style.format({
                    f"{target_feat} (tg)": "{:+.3f}",
                    f"{baseline_feat} (bl)": "{:+.3f}",
                    "Δ (tg−bl)": "{:+.3f}",
                    "ε needed to pass": "{:.3f}",
                }, na_rep="—")
                .apply(lambda s: [
                    "background-color:#FEE2E2" if v == "❌" else
                    "background-color:#DCFCE7" if v == "✅" else ""
                    for v in s
                ], subset=["pass"], axis=0)
                .apply(lambda _: [
                    "background-color:#FEF3C7" if nf else ""
                    for nf in near_flip
                ], subset=["ε needed to pass"], axis=0)
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                "Yellow highlight on `ε needed to pass` = within ±0.05 of the "
                f"current ε={eps:.2f}; a small slider nudge flips the cell."
            )
            download_bar("claim4_pass_fail", pf_preview, "claim4_pass_fail")

    # ── Radar — promoted from expander to main body ─────────────────────────
    with st.container(border=True):
        st.subheader("Per-dataset condition radar")
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
                vals = [
                    r["(a) low-drift No-DA"],
                    max(0, r["(b) drift slope"] + 0.05),
                    max(0, 0.5 - abs(r["(c) high-drift R_g"])),
                    r["(d) retrain ceiling"],
                ]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cat + [cat[0]],
                    fill="toself", name=feat,
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                              showlegend=True)
            style_figure(fig, height=340)
            st.plotly_chart(fig, use_container_width=True)
