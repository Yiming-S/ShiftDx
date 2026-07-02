"""Page 14: Data Quality & Sanity — automated checks on the loaded CSVs.

Most valuable when a freshly built or non-standard dataset is loaded; on the two
clean bundled datasets the checks should be green. Reuses the schema validation
from data_loader.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DA_METHODS, FEATURES, STRATEGIES,
    about_page, download_bar, empty_state,
)


def _status_icon(level: str) -> str:
    return {"ok": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}.get(level, "•")


def render(store):
    st.header("Data Quality & Sanity")
    st.caption(
        "Automated integrity checks on the loaded CSVs: schema, value ranges, "
        "internal consistency, and coverage. Run this after building a new dataset."
    )

    about_page(
        what_you_see=[
            "A pass/warn/fail summary of structural checks.",
            "Per-check detail with the offending rows.",
            "A coverage matrix of strategies × features per dataset.",
        ],
        how_to_read=[
            "✅ pass · ⚠️ warning (investigate) · ❌ fail (likely a build bug).",
            "Coverage gaps explain why a Claim page shows 'no rows'.",
        ],
        paper_ref="build pipeline / data integrity",
    )

    drift = store.drift_df
    eval_df = store.eval_df
    merged = store.merged_df
    if (drift is None or drift.empty) and (merged is None or merged.empty):
        empty_state("No data loaded", "Nothing was discovered in `data/`.",
                    cmd="python scripts/gen_synthetic_demo.py",
                    cmd_label="Generate a synthetic demo dataset")
        return

    checks = []          # (level, name, message)
    details = {}         # name -> DataFrame

    # 1. Schema validation (from the loader)
    if store.schema_issues:
        for fname, cols in store.schema_issues.items():
            checks.append(("fail", f"Schema · {fname}",
                           f"missing columns: {', '.join(cols)}"))
    else:
        checks.append(("ok", "Schema", "all loaded CSVs have required columns"))

    # 2. Accuracy range [0, 1]
    for nm, df in [("sequential_eval", eval_df), ("merged", merged)]:
        if df is not None and not df.empty and "accuracy" in df.columns:
            bad = df[(df["accuracy"] < 0) | (df["accuracy"] > 1)]
            if len(bad):
                checks.append(("fail", f"Accuracy range · {nm}",
                               f"{len(bad)} rows with accuracy outside [0, 1]"))
                details[f"Accuracy range · {nm}"] = bad.head(200)
            else:
                checks.append(("ok", f"Accuracy range · {nm}", "all in [0, 1]"))

    # 3. acc_centered == accuracy − baseline_acc
    if merged is not None and not merged.empty and {
            "acc_centered", "accuracy", "baseline_acc"}.issubset(merged.columns):
        m = merged.dropna(subset=["acc_centered", "accuracy", "baseline_acc"])
        resid = (m["acc_centered"] - (m["accuracy"] - m["baseline_acc"])).abs()
        bad = m[resid > 1e-6]
        if len(bad):
            checks.append(("warn", "acc_centered consistency",
                           f"{len(bad)} rows where acc_centered ≠ accuracy − baseline_acc"))
            details["acc_centered consistency"] = bad.head(200)
        else:
            checks.append(("ok", "acc_centered consistency", "matches accuracy − baseline"))

    # 4. drift_z columns present
    if merged is not None and not merged.empty:
        zcols = [c for c in merged.columns if c.startswith("drift_z")]
        if "drift_z" in merged.columns:
            checks.append(("ok", "drift_z columns",
                           f"present ({len(zcols)} z-score column(s))"))
        else:
            checks.append(("fail", "drift_z columns", "no drift_z column in merged"))

    # 5. Session consistency: max(session_k)+1 ≤ n_sessions
    if drift is not None and not drift.empty and {
            "session_k", "n_sessions"}.issubset(drift.columns):
        agg = drift.groupby(["dataset", "subject"]).agg(
            max_k=("session_k", "max"), n_sessions=("n_sessions", "max")).reset_index()
        bad = agg[agg["max_k"] + 1 > agg["n_sessions"]]
        if len(bad):
            checks.append(("warn", "Session consistency",
                           f"{len(bad)} (dataset, subject) with session_k ≥ n_sessions"))
            details["Session consistency"] = bad
        else:
            checks.append(("ok", "Session consistency", "session_k within n_sessions"))

    # 6. Duplicate rows on the full merged key
    if merged is not None and not merged.empty:
        key = [c for c in ["dataset", "subject", "feature", "classifier",
                           "da", "strategy", "target_session"] if c in merged.columns]
        if key:
            dup = merged.duplicated(subset=key).sum()
            if dup:
                checks.append(("warn", "Duplicate rows",
                               f"{int(dup)} duplicate rows on the eval key"))
            else:
                checks.append(("ok", "Duplicate rows", "no duplicates on the eval key"))

    # 7. Expected vocab present
    if merged is not None and not merged.empty:
        feats = set(merged.get("feature", pd.Series(dtype=str)).unique())
        unknown_feat = feats - set(FEATURES)
        strats = set(merged.get("strategy", pd.Series(dtype=str)).unique())
        unknown_strat = strats - set(STRATEGIES)
        das = set(merged.get("da", pd.Series(dtype=str)).unique()) - {"none"}
        unknown_da = das - set(DA_METHODS)
        extras = []
        if unknown_feat:
            extras.append(f"features {sorted(unknown_feat)}")
        if unknown_strat:
            extras.append(f"strategies {sorted(unknown_strat)}")
        if unknown_da:
            extras.append(f"DA {sorted(unknown_da)}")
        if extras:
            checks.append(("info", "Vocabulary", "unrecognized values: " + "; ".join(extras)))
        else:
            checks.append(("ok", "Vocabulary", "features / strategies / DA all known"))

    # ── Summary ──────────────────────────────────────────────────────────────
    n_fail = sum(1 for c in checks if c[0] == "fail")
    n_warn = sum(1 for c in checks if c[0] == "warn")
    k1, k2, k3 = st.columns(3)
    k1.metric("Checks run", len(checks))
    k2.metric("Failures", n_fail)
    k3.metric("Warnings", n_warn)
    if n_fail == 0 and n_warn == 0:
        st.success("All checks passed.")
    elif n_fail:
        st.error(f"{n_fail} failing check(s) — likely a build bug.")
    else:
        st.warning(f"{n_warn} warning(s) — review below.")

    with st.container(border=True):
        st.subheader("Check results")
        cdf = pd.DataFrame(
            [{"": _status_icon(lvl), "check": nm, "detail": msg}
             for lvl, nm, msg in checks])
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        download_bar("qa_checks", cdf, "data_quality_checks")

    for name, df in details.items():
        with st.expander(f"Detail — {name} ({len(df)} rows shown)"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Coverage matrix ──────────────────────────────────────────────────────
    if eval_df is not None and not eval_df.empty:
        with st.container(border=True):
            st.subheader("Coverage — rows per (dataset × feature × strategy)")
            cov = (eval_df.groupby(["dataset", "feature", "strategy"])
                          .size().reset_index(name="rows"))
            pivot = cov.pivot_table(index=["dataset", "feature"],
                                    columns="strategy", values="rows",
                                    fill_value=0)
            st.dataframe(pivot, use_container_width=True)
            download_bar("qa_coverage", cov, "data_quality_coverage")
