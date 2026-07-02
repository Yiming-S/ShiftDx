"""Page 1: Dataset Overview."""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    DATASET_META, FEATURES,
    about_page, apply_ctx_dataset, dataset_colors, download_bar,
    empty_state, get_ctx, style_figure,
)


def render(store):
    ctx = get_ctx()

    st.header("ShiftDx — Dataset Overview")
    st.markdown(
        "Interactive diagnostic dashboard for cross-session drift in MI-EEG BCIs. "
        "Based on Shen & Degras: **Drift Diagnostics, Adaptation, and "
        "Recalibration in Multi-Session Motor-Imagery EEG**."
    )

    about_page(
        what_you_see=[
            "Which datasets are currently loaded and their paper roles.",
            "Per-(dataset × feature) mean MMD drift.",
            "Distribution of MMD drift under the CSP feature family.",
        ],
        how_to_read=[
            "Status ✅ means `drift_trajectories_<name>.csv` is in `data/` and was loaded.",
            "Higher MMD ⇒ stronger distribution shift session 0 → session k.",
        ],
        paper_ref="§4 (datasets), §5 (claims overview)",
    )

    drift = apply_ctx_dataset(store.drift_df)
    eval_df = apply_ctx_dataset(store.eval_df)

    if drift.empty:
        empty_state(
            "No drift data loaded",
            "Nothing was discovered in `data/`. The dashboard needs at least one "
            "`drift_trajectories_<dataset>.csv` file to show anything.",
            cmd="python scripts/build_moabb.py --dataset zhou2016 --no-slow",
            cmd_label="Build the smallest demo dataset",
        )
        return

    # ── KPI cards ────────────────────────────────────────────────────────────
    n_datasets = drift["dataset"].nunique()
    n_subjects = drift.groupby("dataset")["subject"].nunique().sum()
    n_obs_noda = int(((eval_df["strategy"] == "train_once") &
                      (eval_df["da"] == "none")).sum()) if not eval_df.empty else 0
    n_obs_total = len(eval_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets loaded", n_datasets,
              help="Datasets discovered in `data/`.")
    c2.metric("Subjects", int(n_subjects),
              help="Unique subjects across all loaded datasets.")
    c3.metric("No-DA rows", f"{n_obs_noda:,}",
              help="Sequential-eval rows using the No-DA baseline strategy.")
    c4.metric("Total eval rows", f"{n_obs_total:,}",
              help="Every (subject, session, feature, strategy, DA) combination.")

    # ── Paper anchor ─────────────────────────────────────────────────────────
    st.success(
        "**Four reduced-form claims** anchor this dashboard: "
        "(1) drift predicts loss · (2) DA decomposes into level shift + slope change · "
        "(3) retraining gap is positive in high-drift regimes · "
        "(4) feature robustness requires a ceiling anchor."
    )

    # ── Where to start (navigation cards) ────────────────────────────────────
    with st.container(border=True):
        st.subheader("Where to start")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(
            "**📈 Drift Trajectory**  \n"
            "See per-subject drift over sessions.  \n"
            "→ *Overview → Drift Trajectory*"
        )
        cc2.markdown(
            "**📉 Claim 1 — Drift predicts loss**  \n"
            "Run the paper's regression.  \n"
            "→ *Claim Explorer → Claim 1*"
        )
        cc3.markdown(
            "**🧑 Subject Explorer**  \n"
            "Zoom into one subject's evolution.  \n"
            "→ *Deep Dive → Subject Explorer*"
        )
        cc4.markdown(
            "**🔬 Live DA Sandbox**  \n"
            "Try any of 10 DA methods on a synthetic shift.  \n"
            "→ *DA Lab → Live DA Sandbox*"
        )

    st.markdown("")

    # ── Dataset status table + drift summary ────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("Datasets")
            rows = []
            loaded = set(drift["dataset"].unique())
            # List everything we know about, with status
            all_known = sorted(set(list(DATASET_META.keys()) + list(loaded)))
            for ds in all_known:
                if ctx["dataset"] != "All" and ds != ctx["dataset"]:
                    continue
                meta = DATASET_META.get(ds, {"channels": "?", "subjects": "?",
                                              "sessions": "?", "role": ""})
                sub = drift[drift["dataset"] == ds]
                is_loaded = ds in loaded
                rows.append({
                    "Status": "✅" if is_loaded else "❌",
                    "Dataset": ds,
                    "Channels": meta["channels"],
                    "Subjects": sub["subject"].nunique() if is_loaded else meta["subjects"],
                    "Sessions": meta["sessions"],
                    "Role": meta["role"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("✅ = CSV present in `data/`. ❌ = needs `build_moabb.py`.")

    with col_right:
        with st.container(border=True):
            st.subheader("Mean MMD drift (dataset × feature)")
            if "dist_mmd" in drift.columns:
                agg = (
                    drift.groupby(["dataset", "feature"])["dist_mmd"]
                    .mean()
                    .reset_index()
                    .pivot(index="dataset", columns="feature", values="dist_mmd")
                    .reindex(columns=[f for f in FEATURES if f in drift["feature"].unique()])
                )
                st.dataframe(
                    agg.style.format("{:.3f}").background_gradient(cmap="Blues"),
                    use_container_width=True,
                )
                download_bar("overview_mean_mmd", agg.reset_index(), "overview_mean_mmd")
            else:
                st.info("No `dist_mmd` column in drift_df.")

    # ── MMD distribution by dataset (feature=CSP) ───────────────────────────
    with st.container(border=True):
        st.subheader("MMD distribution (CSP features)")
        csp = drift[drift["feature"] == "CSP"]
        if csp.empty:
            st.info("No CSP rows in drift_df.")
        else:
            fig = px.box(
                csp, x="dataset", y="dist_mmd", color="dataset",
                color_discrete_map=dataset_colors(), points="outliers",
                category_orders={"dataset": sorted(csp["dataset"].unique())},
            )
            fig.update_layout(showlegend=False,
                              yaxis_title="MMD(session 0, session k)")
            style_figure(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)

    # ── Strategy coverage ────────────────────────────────────────────────────
    if not eval_df.empty:
        with st.expander("Strategy × DA coverage"):
            cov = eval_df.groupby(["strategy", "da"]).size().reset_index(name="rows")
            st.dataframe(cov, use_container_width=True, hide_index=True)
            download_bar("overview_coverage", cov, "strategy_coverage")

    # ── Data freshness & provenance ──────────────────────────────────────────
    with st.expander("Data freshness & provenance"):
        try:
            files = [
                os.path.join(store.data_dir, f)
                for f in os.listdir(store.data_dir) if f.endswith(".csv")
            ]
            if files:
                newest = max(os.path.getmtime(p) for p in files)
                st.markdown(
                    f"Newest CSV: **{datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}**"
                )
                st.caption(f"Files: {len(files)}")
        except Exception as exc:
            st.caption(f"Could not stat files: {exc}")

        manifests = store.manifests
        if manifests:
            st.markdown("**Build manifests:**")
            man_rows = []
            for ds, man in manifests.items():
                man_rows.append({
                    "dataset": ds,
                    "built": man.get("build_date", "?"),
                    "subjects": man.get("n_subjects", man.get("subjects", "?")),
                    "da4bci": man.get("da4bci_version", "?"),
                })
            st.dataframe(pd.DataFrame(man_rows), use_container_width=True,
                         hide_index=True)

        if store.schema_issues:
            st.warning("Some CSVs are missing expected columns:")
            for fname, cols in store.schema_issues.items():
                st.caption(f"`{fname}` — missing: {', '.join(cols)}")
        else:
            st.caption("✅ All loaded CSVs pass schema validation.")
