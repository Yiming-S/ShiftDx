"""Page 11: Online Drift Detection — Page-Hinkley detector + FPR/TPR calibration."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    about_page, available_da_methods, download_bar,
    empty_state, filter_by_dataset, is_retrain, require_columns, style_figure,
)

HAS_DA4BCI = None
_DA4BCI_ERR = ""


def _ensure_da4bci() -> bool:
    global HAS_DA4BCI, _DA4BCI_ERR
    if HAS_DA4BCI is None:
        try:
            import da4bci  # noqa: F401
            HAS_DA4BCI = True
        except Exception as exc:
            HAS_DA4BCI = False
            _DA4BCI_ERR = str(exc)
    return HAS_DA4BCI


PH_PRESETS = {
    "Conservative (few triggers)": dict(delta=0.01, lam=0.6, alpha=0.95),
    "Default":                     dict(delta=0.005, lam=0.3, alpha=0.9),
    "Aggressive (early alarms)":   dict(delta=0.002, lam=0.15, alpha=0.8),
}


def _run_page_hinkley(series: np.ndarray, delta: float, lam: float, alpha: float):
    from da4bci import ph_init, ph_update
    state = ph_init(delta=delta, lambda_=lam, alpha=alpha)
    means, cums, mins, triggers = [], [], [], []
    for x in series:
        out = ph_update(state, x)
        state = out["state"]
        means.append(state["mean"])
        cums.append(state["cum"])
        mins.append(state["min_cum"])
        triggers.append(out["change"])
    return np.array(means), np.array(cums), np.array(mins), np.array(triggers)


def _fires(series: np.ndarray, delta: float, lam: float, alpha: float) -> bool:
    """Whether the Page-Hinkley detector triggers at least once on a series."""
    _, _, _, trig = _run_page_hinkley(series, delta, lam, alpha)
    return bool(trig.any())


@st.cache_data(show_spinner=False)
def _calibrate(n_series: int, length: int, sigma: float, drift_mag: float,
               drift_kind: str, alpha: float, seed: int,
               lam_grid: tuple, delta_grid: tuple) -> tuple:
    """Monte-Carlo empirical FPR (stationary series) and TPR (drifting series)
    over a (λ, δ) grid. Loss = 1 − accuracy; drift raises loss."""
    rng = np.random.default_rng(seed)
    base = 0.30  # baseline loss level
    stat = base + rng.normal(0, sigma, (n_series, length))
    if drift_kind == "linear ramp":
        ramp = np.linspace(0.0, drift_mag, length)
    else:  # step at midpoint
        ramp = np.where(np.arange(length) >= length // 2, drift_mag, 0.0)
    drift = base + ramp[None, :] + rng.normal(0, sigma, (n_series, length))

    fpr = np.zeros((len(lam_grid), len(delta_grid)))
    tpr = np.zeros_like(fpr)
    for i, lam in enumerate(lam_grid):
        for j, delta in enumerate(delta_grid):
            fpr[i, j] = np.mean([_fires(s, delta, lam, alpha) for s in stat])
            tpr[i, j] = np.mean([_fires(s, delta, lam, alpha) for s in drift])
    return fpr, tpr


def _detector_tab(store):
    # Page-local dataset picker (independent of sidebar)
    datasets = sorted(store.eval_df["dataset"].unique()) if not store.eval_df.empty else []
    if not datasets:
        empty_state("No evaluation data", "Load a dataset first (see Overview).")
        return

    ds_options = ["All"] + datasets if len(datasets) > 1 else datasets
    ds_pick = st.selectbox("Dataset (page)", ds_options, index=0)
    eval_df = filter_by_dataset(store.eval_df, ds_pick)
    if not require_columns(eval_df, ["subject", "feature", "strategy", "da",
                                     "accuracy", "target_session", "ref_session"],
                           "the drift detector"):
        return

    # ── Subject / feature / stream ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    if ds_pick == "All":
        ds_final = c1.selectbox("Sub-dataset", sorted(eval_df["dataset"].unique()))
    else:
        ds_final = ds_pick
        c1.markdown(f"**Dataset:** `{ds_final}`")
    subj_df = eval_df[eval_df["dataset"] == ds_final]
    subject = c2.selectbox("Subject", sorted(subj_df["subject"].unique()))
    feature = c3.selectbox("Feature", sorted(subj_df["feature"].unique()), index=0)

    da_opts = available_da_methods(subj_df)
    stream_options = (["No DA"]
                      + [f"DA · {da.upper()}" for da in da_opts]
                      + ["Retrain"])
    stream_choice = st.selectbox("Stream", stream_options, index=0)

    base = subj_df[(subj_df["subject"] == subject) & (subj_df["feature"] == feature)]
    if stream_choice == "No DA":
        series_df = base[(base["strategy"] == "train_once") & (base["da"] == "none")]
    elif stream_choice == "Retrain":
        series_df = is_retrain(base)
    else:
        idx = stream_options.index(stream_choice) - 1  # offset by "No DA"
        da_pick = da_opts[idx]
        series_df = base[(base["strategy"] == "train_once_da") & (base["da"] == da_pick)]

    series_df = series_df.sort_values("target_session")
    if series_df.empty:
        empty_state("No series for this combo",
                    "Try a different subject / feature / stream.")
        return

    losses = 1.0 - series_df["accuracy"].to_numpy()
    sessions = series_df["target_session"].to_numpy()

    # ── PH parameters with presets ──────────────────────────────────────────
    st.markdown("**Page-Hinkley parameters**")
    preset_col, _ = st.columns([2, 3])
    with preset_col:
        preset_name = st.selectbox("Preset", list(PH_PRESETS.keys()), index=1,
                                   key="ph_preset")
    preset = PH_PRESETS[preset_name]

    p1, p2, p3 = st.columns(3)
    delta = p1.slider("δ (min magnitude)", 0.0, 0.1, preset["delta"], 0.001, format="%.3f")
    lam   = p2.slider("λ (threshold)",     0.0, 2.0, preset["lam"], 0.05)
    alpha = p3.slider("α (EMA decay)",     0.5, 0.999, preset["alpha"], 0.01)

    means, cums, mins, triggers = _run_page_hinkley(losses, delta, lam, alpha)
    first_trigger = int(np.where(triggers)[0][0]) if triggers.any() else None

    # ── KPI ─────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Sessions observed", len(sessions))
    k2.metric("Trigger events", int(triggers.sum()))
    k3.metric("First trigger @ session",
              int(sessions[first_trigger]) if first_trigger is not None else "—")

    # ── Plot ────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"S{subject} · {feature} · {stream_choice}")
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("Accuracy", "Page-Hinkley CUSUM"),
        )
        fig.add_trace(go.Scatter(
            x=sessions, y=1 - losses, mode="lines+markers", name="accuracy",
            line=dict(color="#4F46E5", width=3), marker=dict(size=8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sessions, y=cums - mins, mode="lines+markers",
            line=dict(color="#EF4444", width=2), name="cum − min",
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=lam, line_dash="dash", line_color="#64748B",
                      row=2, col=1, annotation_text=f"λ={lam:.2f}")
        for i in np.where(triggers)[0]:
            fig.add_vline(x=sessions[i], line_color="#EF4444",
                          line_dash="dot", opacity=0.4)
        if first_trigger is not None:
            fig.add_annotation(
                x=sessions[first_trigger], y=1 - losses[first_trigger],
                text="First trigger", showarrow=True, arrowhead=2,
                arrowcolor="#EF4444", font=dict(color="#EF4444", size=12),
                row=1, col=1,
            )
        fig.update_xaxes(title_text="session_k", row=2, col=1)
        fig.update_yaxes(title_text="accuracy", row=1, col=1)
        fig.update_yaxes(title_text="PH statistic", row=2, col=1)
        style_figure(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Download series"):
        out_df = pd.DataFrame({
            "session_k": sessions, "accuracy": 1 - losses,
            "ph_stat": cums - mins, "trigger": triggers,
        })
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        download_bar("detection_series", out_df,
                     f"detection_{ds_final}_S{subject}_{feature}")

    st.info(
        "Once the PH statistic exceeds λ, a retraining trigger fires. "
        "Compare the firing session against the paper's high-drift regime "
        "(upper Q75 of `drift_z`) to see if online detection catches drift "
        "before accuracy noticeably degrades. Cross-reference with **Claim 3** "
        "for this subject's retraining-gap evolution."
    )


def _calibration_tab(store):
    st.caption(
        "Simulation-based calibration: how often does the detector cry wolf on a "
        "*stationary* stream (false-alarm rate), and how reliably does it catch a "
        "*drifting* one (detection rate)? Pick (λ, δ) for a target FPR. This is a "
        "guide from synthetic series — not a guarantee for real EEG."
    )
    c1, c2, c3 = st.columns(3)
    n_series = c1.number_input("Series per condition", 20, 500, 100, 20,
                               key="calib_n")
    length = c2.number_input("Sessions per series", 5, 60, 12, 1, key="calib_len")
    alpha = c3.slider("α (EMA decay)", 0.5, 0.999, 0.9, 0.01, key="calib_alpha")
    c4, c5, c6 = st.columns(3)
    sigma = c4.slider("Noise SD (loss units)", 0.01, 0.20, 0.05, 0.01,
                      key="calib_sigma")
    drift_mag = c5.slider("Drift magnitude (loss rise)", 0.05, 0.50, 0.20, 0.05,
                          key="calib_drift")
    drift_kind = c6.selectbox("Drift shape", ["linear ramp", "step (midpoint)"],
                              key="calib_kind")

    lam_grid = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
    delta_grid = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)

    with st.spinner("Simulating false-alarm / detection rates…"):
        fpr, tpr = _calibrate(int(n_series), int(length), float(sigma),
                              float(drift_mag), drift_kind, float(alpha), 0,
                              lam_grid, delta_grid)

    x = [f"{v:.3f}" for v in delta_grid]
    y = [f"{v:.2f}" for v in lam_grid]
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**False-alarm rate (stationary)** — lower is better")
        fig = px.imshow(fpr, x=x, y=y, color_continuous_scale="Reds",
                        aspect="auto", text_auto=".2f",
                        labels=dict(x="δ", y="λ", color="FPR"), zmin=0, zmax=1)
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.markdown("**Detection rate (drifting)** — higher is better")
        fig = px.imshow(tpr, x=x, y=y, color_continuous_scale="Greens",
                        aspect="auto", text_auto=".2f",
                        labels=dict(x="δ", y="λ", color="TPR"), zmin=0, zmax=1)
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    target_fpr = st.slider("Target false-alarm rate", 0.0, 0.5, 0.05, 0.01,
                           key="calib_target_fpr")
    # Among cells meeting the FPR target, pick the one maximizing TPR.
    rows = []
    for i, lam in enumerate(lam_grid):
        for j, delta in enumerate(delta_grid):
            rows.append({"λ": lam, "δ": delta, "FPR": fpr[i, j], "TPR": tpr[i, j]})
    grid_df = pd.DataFrame(rows)
    feasible = grid_df[grid_df["FPR"] <= target_fpr]
    if feasible.empty:
        st.warning(f"No (λ, δ) on the grid achieves FPR ≤ {target_fpr:.2f}; "
                   "raise the target or λ.")
    else:
        best = feasible.sort_values("TPR", ascending=False).iloc[0]
        st.success(
            f"For FPR ≤ {target_fpr:.2f}: use **λ={best['λ']:.2f}, δ={best['δ']:.3f}** "
            f"→ detection rate **{best['TPR']:.0%}** (false-alarm {best['FPR']:.0%})."
        )
    download_bar("ph_calibration", grid_df, "ph_calibration_grid")


def render(store):
    st.header("Drift Detection")
    st.caption(
        "Feed one subject's per-session accuracy into the Page-Hinkley detector, "
        "and calibrate its false-alarm / detection rates by simulation. "
        "Links static retraining-gap analysis to an online deployment rule. "
        "This page has its own controls (DA Lab section)."
    )

    about_page(
        what_you_see=[
            "Detector tab: accuracy series + the Page-Hinkley statistic with λ threshold.",
            "Calibration tab: empirical FPR / TPR heatmaps over the (λ, δ) grid.",
            "Red dotted vertical lines mark sessions where a trigger fires.",
        ],
        how_to_read=[
            "A trigger in the high-drift regime ⇒ the online rule catches drift early.",
            "Use the calibration tab to choose (λ, δ) for a tolerable false-alarm rate.",
        ],
        paper_ref="§7 online-detection link",
        key_terms=[
            ("δ", "minimum effect size for the detector to count a change."),
            ("λ", "threshold: trigger fires when (cum − min) > λ."),
            ("α", "EMA decay for the running mean."),
        ],
    )

    if not _ensure_da4bci():
        empty_state("DA4BCI not importable", _DA4BCI_ERR,
                    cmd="pip install -e /path/to/DA4BCI-Python")
        return

    tab_detect, tab_calib = st.tabs(["Detector", "Calibration (FPR / TPR)"])
    with tab_detect:
        _detector_tab(store)
    with tab_calib:
        _calibration_tab(store)
