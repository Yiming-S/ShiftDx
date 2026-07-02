"""Page 12: DA Method Leaderboard — real-data ranking of all DA methods.

The DA Method Sweep (page 10) ranks methods on *synthetic* shifts. This page
ranks them on the *actual* benchmark data: mean accuracy gain over No-DA,
win-rate, and the fraction of the No-DA → Retrain gap that each method closes,
under the current global context (dataset × classifier × drift metric).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    DA_LABEL, DA_SHORT_LABEL,
    about_page, apply_ctx_classifier, apply_ctx_dataset, apply_ctx_metric,
    available_da_methods, cluster_bootstrap_ci, da_colors, download_bar,
    empty_state, get_ctx, is_retrain, pvalue_badge, style_figure,
)

_KEY = ["dataset", "subject", "feature", "target_session"]


@st.cache_data(show_spinner=False)
def _leaderboard(merged: pd.DataFrame, methods: tuple, n_boot: int = 800) -> pd.DataFrame:
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    retr = is_retrain(merged)
    nd = noda[_KEY + ["accuracy", "drift_z"]].rename(columns={"accuracy": "acc_noda"})
    rt = retr[_KEY + ["accuracy"]].rename(columns={"accuracy": "acc_retrain"})

    rows = []
    for i, m in enumerate(methods):
        da = (merged[(merged["strategy"] == "train_once_da") & (merged["da"] == m)]
              [_KEY + ["accuracy"]].rename(columns={"accuracy": "acc_da"}))
        pair = nd.merge(da, on=_KEY, how="inner")
        if pair.empty:
            continue
        pair["gain"] = pair["acc_da"] - pair["acc_noda"]
        pair["uid"] = pair["dataset"].astype(str) + "_" + pair["subject"].astype(str)
        mean_gain, lo, hi = cluster_bootstrap_ci(
            pair, lambda d: d["gain"].mean(), cluster_col="uid",
            n_boot=n_boot, seed=100 + i)
        win = float((pair["gain"] > 0).mean())

        pg = pair.merge(rt, on=_KEY, how="inner")
        gap = np.nan
        if not pg.empty:
            denom = pg["acc_retrain"] - pg["acc_noda"]
            valid = denom.abs() > 1e-6
            if valid.any():
                gap = float(((pg.loc[valid, "acc_da"] - pg.loc[valid, "acc_noda"])
                             / denom[valid]).mean())
        # CI excluding 0 ⇒ a (cluster-bootstrap) significant gain
        sig = "●" if (np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)) else "—"
        rows.append({
            "method": m, "DA": DA_SHORT_LABEL.get(m, m.upper()),
            "n_pairs": len(pair), "mean_gain": mean_gain,
            "gain_lo": lo, "gain_hi": hi, "sig": sig,
            "win_rate": win, "mean_gap_closed": gap,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean_gain", ascending=False).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def _gain_by_quartile(merged: pd.DataFrame, methods: tuple) -> pd.DataFrame:
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    nd = noda[_KEY + ["accuracy", "drift_z"]].rename(columns={"accuracy": "acc_noda"})
    try:
        nd = nd.assign(q=pd.qcut(nd["drift_z"], 4, labels=False, duplicates="drop"))
    except Exception:
        nd = nd.assign(q=np.nan)
    rows = []
    for m in methods:
        da = (merged[(merged["strategy"] == "train_once_da") & (merged["da"] == m)]
              [_KEY + ["accuracy"]].rename(columns={"accuracy": "acc_da"}))
        pair = nd.merge(da, on=_KEY, how="inner").dropna(subset=["q"])
        if pair.empty:
            continue
        pair["gain"] = pair["acc_da"] - pair["acc_noda"]
        for q, g in pair.groupby("q"):
            rows.append({"DA": DA_SHORT_LABEL.get(m, m.upper()),
                         "quartile": f"Q{int(q) + 1}", "mean_gain": g["gain"].mean()})
    return pd.DataFrame(rows)


def render(store):
    ctx = get_ctx()
    dcolors = da_colors()

    st.header("DA Method Leaderboard")
    st.caption(
        "Ranks every DA method on the **real** benchmark data (not synthetic "
        "shifts) under the current sidebar context. Mean gain is DA − No-DA "
        "accuracy; CIs are subject-clustered bootstraps."
    )

    about_page(
        what_you_see=[
            "Sortable table: mean gain (with bootstrap CI), win-rate, % gap closed.",
            "Bar chart of mean gain per method.",
            "Gain vs win-rate scatter, and mean gain by drift quartile.",
        ],
        how_to_read=[
            "Mean gain CI excluding 0 (`●`) ⇒ the method reliably beats No-DA.",
            "High win-rate but small mean gain ⇒ consistent but modest.",
            "Watch the quartile tab: some methods only help in high drift.",
        ],
        paper_ref="§5.3–5.4 (DA benefit & retraining gap)",
        key_terms=[
            ("mean gain", "average DA − No-DA accuracy over paired observations."),
            ("win-rate", "share of pairs where DA > No-DA."),
            ("% gap closed", "(DA − No-DA) / (Retrain − No-DA)."),
        ],
    )

    merged = apply_ctx_dataset(store.merged_df)
    merged = apply_ctx_classifier(merged, ctx)
    merged = apply_ctx_metric(merged, ctx)
    if merged.empty:
        empty_state("No merged data for this selection",
                    f"dataset=`{ctx['dataset']}`, classifier=`{ctx['classifier']}` "
                    "gives no rows.", dataset=ctx["dataset"])
        return

    methods = available_da_methods(merged)
    if not methods:
        empty_state("No DA methods present", "The merged file has no `train_once_da` rows.")
        return

    with st.spinner("Ranking DA methods…"):
        lb = _leaderboard(merged, tuple(methods))
    if lb.empty:
        empty_state("No paired rows", "Could not pair No-DA with any DA method.")
        return

    # ── KPI: best method ─────────────────────────────────────────────────────
    best = lb.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Top method (mean gain)", best["DA"],
              delta=f"{best['mean_gain']*100:+.1f} pp")
    k2.metric("Its win-rate", f"{best['win_rate']*100:.0f}%")
    k3.metric("Its % gap closed",
              f"{best['mean_gap_closed']*100:.0f}%"
              if np.isfinite(best["mean_gap_closed"]) else "—")

    # ── Table ────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Leaderboard")
        show = lb.copy()
        show["mean gain"] = show["mean_gain"]
        show["95% CI"] = [
            f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "—"
            for lo, hi in zip(show["gain_lo"], show["gain_hi"])
        ]
        disp = show[["DA", "n_pairs", "mean gain", "95% CI", "sig",
                     "win_rate", "mean_gap_closed"]]
        st.dataframe(
            disp.style.format({
                "mean gain": "{:+.4f}", "win_rate": "{:.0%}",
                "mean_gap_closed": "{:.0%}",
            }, na_rep="—").background_gradient(subset=["mean gain"],
                                              cmap="RdBu_r", vmin=-0.05, vmax=0.05),
            use_container_width=True, hide_index=True,
        )
        st.caption("`sig ●` = bootstrap 95% CI for mean gain excludes 0.")
        download_bar("leaderboard", lb, f"da_leaderboard_{ctx['dataset']}")

    # ── Bar of mean gain ─────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Mean gain per method")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lb["DA"], y=lb["mean_gain"] * 100,
            marker_color=[dcolors.get(m, "#4F46E5") for m in lb["method"]],
            error_y=dict(
                type="data",
                array=[(hi - g) * 100 if np.isfinite(hi) else 0
                       for g, hi in zip(lb["mean_gain"], lb["gain_hi"])],
                arrayminus=[(g - lo) * 100 if np.isfinite(lo) else 0
                            for g, lo in zip(lb["mean_gain"], lb["gain_lo"])],
            ),
        ))
        fig.add_hline(y=0, line_color="#64748B", line_dash="dash")
        fig.update_layout(xaxis_title="DA method", yaxis_title="mean gain (pp)")
        style_figure(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Detail tabs ──────────────────────────────────────────────────────────
    t_scatter, t_quart = st.tabs(["Gain vs win-rate", "Gain by drift quartile"])
    with t_scatter:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lb["win_rate"] * 100, y=lb["mean_gain"] * 100,
            mode="markers+text", text=lb["DA"], textposition="top center",
            marker=dict(size=14, color=[dcolors.get(m, "#4F46E5") for m in lb["method"]],
                        line=dict(width=1, color="#1E293B")),
            hovertemplate="%{text}<br>win=%{x:.0f}%<br>gain=%{y:+.2f} pp<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="#64748B", line_dash="dash")
        fig.add_vline(x=50, line_color="#64748B", line_dash="dot")
        fig.update_layout(xaxis_title="win-rate (%)", yaxis_title="mean gain (pp)")
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Upper-right = reliably and substantially better than No-DA.")

    with t_quart:
        gq = _gain_by_quartile(merged, tuple(methods))
        if gq.empty:
            st.info("Not enough drift spread to form quartiles.")
        else:
            pivot = gq.pivot(index="DA", columns="quartile", values="mean_gain")
            st.dataframe(
                pivot.style.format("{:+.4f}", na_rep="—")
                     .background_gradient(cmap="RdBu_r", vmin=-0.05, vmax=0.05),
                use_container_width=True,
            )
            st.caption("Mean gain (DA − No-DA) within each drift quartile "
                       "(Q1 = low drift → Q4 = high drift).")
            download_bar("leaderboard_quartile", gq.reset_index(drop=True),
                         "da_leaderboard_by_quartile")
