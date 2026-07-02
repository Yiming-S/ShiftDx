"""Page 13: Statistical Report — one-click cross-claim summary export.

Re-runs the existing claim fits for the current sidebar context and assembles a
single Markdown / HTML report with a download button. No new inference: it calls
the same cached functions the Claim pages use.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CLASSIFIER_LABEL, DA_LABEL, DISTANCE_LABEL,
    about_page, apply_ctx_classifier, apply_ctx_dataset, apply_ctx_metric,
    available_da_methods, empty_state, get_ctx,
)
from views._3_claim1 import _fit_claim1, _build_forest
from views._4_claim2 import _fit_claim2
from views._5_claim3 import _compute_rg_table, _rg_fits
from views._6_claim4 import _compute_conditions, _pass_fail
from views._12_da_leaderboard import _leaderboard


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(no rows)_"
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def _build_sections(store, ctx) -> list:
    """Return a list of (heading, dataframe-or-text) sections for the report."""
    sections = []
    merged = apply_ctx_metric(apply_ctx_classifier(apply_ctx_dataset(
        store.merged_df, ctx), ctx), ctx)
    da_opts = available_da_methods(merged)
    da_method = ctx["da"] if ctx["da"] in da_opts else (da_opts[0] if da_opts else None)

    # Claim 1
    noda = merged[(merged["strategy"] == "train_once") & (merged["da"] == "none")]
    try:
        fit1 = _fit_claim1(noda)
        if "error" not in fit1:
            t = fit1["table"].copy()
            t = t[t["term"].str.contains("drift", case=False, na=False)]
            sections.append((
                f"Claim 1 — Drift predicts loss  "
                f"(n={fit1['n']}, subjects={fit1['n_subj']}, "
                f"R²m={fit1['r2_marginal']:.2f}/R²c={fit1['r2_conditional']:.2f})",
                t,
            ))
        forest, _ = _build_forest(noda)
        if not forest.empty:
            n_sig = int(forest["sig_fdr"].sum())
            sections.append((
                f"Claim 1 — Forest plot: {n_sig}/{len(forest)} cells significant "
                "(Benjamini-Hochberg FDR 5%)",
                forest[["feature", "classifier", "metric_label", "beta",
                        "p", "p_adj", "R2"]].rename(columns={"metric_label": "metric"}),
            ))
    except Exception as exc:
        sections.append(("Claim 1 — error", str(exc)))

    # Claim 2
    if da_method:
        try:
            nd = noda.copy(); nd["has_da"] = 0
            da = merged[(merged["strategy"] == "train_once_da") &
                        (merged["da"] == da_method)].copy(); da["has_da"] = 1
            fit2 = _fit_claim2(pd.concat([nd, da], ignore_index=True))
            if "error" not in fit2:
                t = fit2["table"]
                t = t[t["term"].str.contains("has_da", case=False, na=False)]
                sections.append((
                    f"Claim 2 — DA decomposition (DA = {da_method.upper()}, "
                    f"R²m={fit2['r2_marginal']:.2f})", t,
                ))
        except Exception as exc:
            sections.append(("Claim 2 — error", str(exc)))

        # Claim 3
        try:
            tbl = _compute_rg_table(merged, da_method)
            if not tbl.empty:
                sections.append((
                    f"Claim 3 — Retraining gap R_g(z) (DA = {da_method.upper()})",
                    _rg_fits(tbl)[["dataset", "feature", "intercept", "slope",
                                   "p(slope)", "high-drift mean R_g",
                                   "p(R_g>0, boot)"]],
                ))
        except Exception as exc:
            sections.append(("Claim 3 — error", str(exc)))

        # Claim 4 (verdict for logvar vs CSP if present)
        try:
            cond = _compute_conditions(merged, da_method)
            feats = cond["feature"].unique().tolist()
            tgt = "logvar" if "logvar" in feats else (feats[0] if feats else None)
            bl = "CSP" if "CSP" in feats else (feats[-1] if len(feats) > 1 else None)
            if tgt and bl and tgt != bl:
                pf = _pass_fail(cond[cond["feature"].isin([tgt, bl])], tgt, bl, 0.05)
                sections.append((
                    f"Claim 4 — Does {tgt} dominate {bl}? (ε=0.05)",
                    pf[["dataset", "condition", "Δ (tg−bl)", "pass"]],
                ))
        except Exception as exc:
            sections.append(("Claim 4 — error", str(exc)))

    # Leaderboard
    if da_opts:
        try:
            lb = _leaderboard(merged, tuple(da_opts))
            if not lb.empty:
                sections.append((
                    "DA Method Leaderboard (mean gain over No-DA)",
                    lb[["DA", "n_pairs", "mean_gain", "win_rate", "mean_gap_closed"]],
                ))
        except Exception as exc:
            sections.append(("Leaderboard — error", str(exc)))

    return sections


def _to_markdown(ctx, sections, ts) -> str:
    out = ["# ShiftDx Statistical Report", "",
           f"_Generated {ts}_", "",
           "## Context", "",
           f"- **Dataset:** {ctx['dataset']}",
           f"- **Classifier:** {CLASSIFIER_LABEL.get(ctx['classifier'], ctx['classifier'])}",
           f"- **Drift metric:** {DISTANCE_LABEL.get(ctx['metric'], ctx['metric'])}",
           f"- **DA method:** {DA_LABEL.get(ctx['da'], ctx['da'])}",
           ""]
    for heading, content in sections:
        out.append(f"## {heading}")
        out.append("")
        out.append(_md_table(content) if isinstance(content, pd.DataFrame)
                   else str(content))
        out.append("")
    out.append("---")
    out.append("_ShiftDx · Shen & Degras (2026). Generated from the current "
               "dashboard selection; headline models replicate the published tables._")
    return "\n".join(out)


def _to_html(ctx, sections, ts) -> str:
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Inter,system-ui,sans-serif;color:#1E293B;max-width:900px;margin:2rem auto;padding:0 1rem;}",
        "h1{color:#4F46E5;} h2{border-bottom:1px solid #E2E8F0;padding-bottom:4px;margin-top:1.6rem;}",
        "table{border-collapse:collapse;width:100%;font-size:0.85rem;} ",
        "th,td{border:1px solid #E2E8F0;padding:4px 8px;text-align:right;} th{background:#F8FAFC;}",
        "</style></head><body>",
        "<h1>ShiftDx Statistical Report</h1>",
        f"<p><em>Generated {ts}</em></p>",
        "<h2>Context</h2><ul>",
        f"<li><b>Dataset:</b> {ctx['dataset']}</li>",
        f"<li><b>Classifier:</b> {CLASSIFIER_LABEL.get(ctx['classifier'], ctx['classifier'])}</li>",
        f"<li><b>Drift metric:</b> {DISTANCE_LABEL.get(ctx['metric'], ctx['metric'])}</li>",
        f"<li><b>DA method:</b> {DA_LABEL.get(ctx['da'], ctx['da'])}</li></ul>",
    ]
    for heading, content in sections:
        parts.append(f"<h2>{heading}</h2>")
        if isinstance(content, pd.DataFrame):
            parts.append(content.to_html(index=False, float_format=lambda v: f"{v:.4g}"))
        else:
            parts.append(f"<pre>{content}</pre>")
    parts.append("<hr><p><em>ShiftDx · Shen &amp; Degras (2026).</em></p></body></html>")
    return "\n".join(parts)


def render(store):
    ctx = get_ctx()

    st.header("Statistical Report")
    st.caption(
        "A single cross-claim summary for the current sidebar context, ready to "
        "download as Markdown or HTML. It re-runs the same fits as the Claim "
        "pages — no new inference."
    )

    about_page(
        what_you_see=[
            "Claims 1–4 key results + the DA leaderboard for the active selection.",
            "Download buttons for a Markdown and an HTML report.",
        ],
        how_to_read=[
            "Change the sidebar context, then regenerate to retarget the report.",
            "Headline models replicate the published tables; supplementary panels "
            "are not included here.",
        ],
        paper_ref="§5 (all claims)",
    )

    merged = apply_ctx_dataset(store.merged_df, ctx)
    if merged is None or merged.empty:
        empty_state("No data for this selection",
                    f"dataset=`{ctx['dataset']}` has no merged rows.",
                    dataset=ctx["dataset"])
        return

    if st.button("📝 Generate report", type="primary"):
        st.session_state["_report_built"] = True

    if not st.session_state.get("_report_built"):
        st.info("Click **Generate report** to build the summary for the current context.")
        return

    with st.spinner("Assembling report…"):
        sections = _build_sections(store, ctx)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        md = _to_markdown(ctx, sections, ts)
        html = _to_html(ctx, sections, ts)

    c1, c2 = st.columns(2)
    c1.download_button("📥 Markdown (.md)", md.encode("utf-8"),
                       file_name=f"shiftdx_report_{ctx['dataset']}.md",
                       mime="text/markdown", use_container_width=True)
    c2.download_button("📥 HTML (.html)", html.encode("utf-8"),
                       file_name=f"shiftdx_report_{ctx['dataset']}.html",
                       mime="text/html", use_container_width=True)

    st.markdown("---")
    for heading, content in sections:
        st.markdown(f"#### {heading}")
        if isinstance(content, pd.DataFrame):
            st.dataframe(content, use_container_width=True, hide_index=True)
        else:
            st.caption(str(content))
