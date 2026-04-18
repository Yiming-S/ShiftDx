"""Page 10: DA Method Sweep — all 10 DA methods from DA4BCI on a controlled shift."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import style_figure

try:
    from da4bci import (
        domain_adaptation, compute_mmd, compute_energy, compute_wasserstein,
        compute_mahalanobis, compute_distance_matrix, sigma_med,
    )
    HAS_DA4BCI = True
except Exception as _exc:
    HAS_DA4BCI = False
    _DA4BCI_ERR = str(_exc)


def _mmd(x, y):
    try:
        sigma = sigma_med(x, y)
    except Exception:
        sigma = 1.0
    return compute_mmd(x, y, sigma)


def _euclid(x, y):
    try:
        D = compute_distance_matrix(x, y)
        return float(np.mean(D))
    except Exception:
        return float(np.nan)


def _wass_safe(x, y):
    try:
        return compute_wasserstein(x, y)
    except Exception:
        return np.nan


ALL_METHODS = ["sa", "pt", "coral", "gfk", "tca", "rd", "art", "ot", "mida", "m3d"]
FAST_METHODS = ["sa", "pt", "coral", "gfk", "tca", "rd", "art", "ot"]  # drop slow by default

SCENARIOS = {
    "Different Means": ("N(0,1)", "N(0.8,1)"),
    "Different SD": ("N(0,1)", "N(0,2)"),
    "Covariance shift": ("I_d", "diag(0.5..2.0)"),
    "Heavy-tail (t)": ("t(5)", "t(3)+0.3"),
}


def _generate(scenario: str, n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    if scenario == "Different Means":
        return rng.normal(0, 1, (n, d)), rng.normal(0.8, 1, (n, d))
    if scenario == "Different SD":
        return rng.normal(0, 1, (n, d)), rng.normal(0, 2, (n, d))
    if scenario == "Covariance shift":
        return (rng.multivariate_normal(np.zeros(d), np.eye(d), n),
                rng.multivariate_normal(np.zeros(d), np.diag(np.linspace(0.5, 2.0, d)), n))
    if scenario == "Heavy-tail (t)":
        return rng.standard_t(5, (n, d)), rng.standard_t(3, (n, d)) + 0.3
    raise ValueError(scenario)


def _default_params(method: str, d: int) -> dict:
    if method == "sa":    return {"k": min(10, d)}
    if method == "pt":    return {}
    if method == "coral": return {"lam": 1e-5}
    if method == "gfk":   return {"d": min(10, d - 1) if d > 1 else 1}
    if method == "tca":   return {"k": min(10, d)}
    if method == "ot":    return {"reg": 0.1}
    if method == "mida":  return {"d": min(10, d)}
    if method == "m3d":   return {"d": min(10, d)}
    return {}


@st.cache_data(show_spinner=False)
def _run_sweep(scenario: str, n: int, d: int, seed: int, methods: tuple) -> pd.DataFrame:
    source, target = _generate(scenario, n, d, seed)
    base_mmd = _mmd(source, target)
    base_energy = compute_energy(source, target)
    base_wass = _wass_safe(source, target)
    base_maha = compute_mahalanobis(source, target)
    base_euclid = _euclid(source, target)

    rows = []
    for m in methods:
        try:
            t0 = time.perf_counter()
            res = domain_adaptation(source, target, method=m, control=_default_params(m, d))
            rt = (time.perf_counter() - t0) * 1000
            src_a = res.get("weighted_source_data", source)
            tgt_a = res.get("target_data", target)
            new_mmd = _mmd(src_a, tgt_a)
            new_energy = compute_energy(src_a, tgt_a)
            new_wass = _wass_safe(src_a, tgt_a)
            new_maha = compute_mahalanobis(src_a, tgt_a)
            new_euclid = _euclid(src_a, tgt_a)
            def _red(b, a):
                if b is None or np.isnan(b) or abs(b) < 1e-12 or np.isnan(a):
                    return np.nan
                return (b - a) / b * 100
            rows.append({
                "method": m, "runtime_ms": rt,
                "MMD_before": base_mmd, "MMD_after": new_mmd, "MMD_red%": _red(base_mmd, new_mmd),
                "Energy_red%": _red(base_energy, new_energy),
                "Wasserstein_red%": _red(base_wass, new_wass),
                "Maha_red%": _red(base_maha, new_maha),
                "Euclid_red%": _red(base_euclid, new_euclid),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"method": m, "runtime_ms": np.nan,
                          "MMD_before": base_mmd, "MMD_after": np.nan,
                          "MMD_red%": np.nan, "Energy_red%": np.nan,
                          "Wasserstein_red%": np.nan, "Maha_red%": np.nan,
                          "Euclid_red%": np.nan, "status": f"error: {exc}"})
    return pd.DataFrame(rows)


def render(store, dataset):
    st.header("DA Method Sweep")
    st.caption(
        "For a given shift scenario, run every DA method in DA4BCI and plot "
        "the accuracy–runtime Pareto frontier. Useful for picking a method "
        "that balances quality and latency for real-time BCI."
    )

    if not HAS_DA4BCI:
        st.error(f"DA4BCI not importable: {_DA4BCI_ERR}")
        return

    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox("Scenario", list(SCENARIOS.keys()))
    n = c2.number_input("n per domain", 100, 2000, 500, 100)
    d = c3.number_input("feature dim", 2, 50, 20, 1)
    seed = c4.number_input("seed", 0, 9999, 42, 1)

    include_slow = st.checkbox("Include slow methods (MIDA, M3D)", value=False)
    methods = ALL_METHODS if include_slow else FAST_METHODS

    st.caption(f"Source vs target: **{SCENARIOS[scenario][0]}** vs **{SCENARIOS[scenario][1]}**")

    with st.spinner("Running sweep..."):
        df = _run_sweep(scenario, int(n), int(d), int(seed), tuple(methods))

    if df.empty:
        st.warning("No methods returned results.")
        return

    # ── Table ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Results")
        st.dataframe(
            df.style.format({
                "runtime_ms": "{:.1f}",
                "MMD_before": "{:.4f}", "MMD_after": "{:.4f}",
                "MMD_red%": "{:+.1f}", "Energy_red%": "{:+.1f}",
                "Wasserstein_red%": "{:+.1f}", "Maha_red%": "{:+.1f}",
                "Euclid_red%": "{:+.1f}",
            }).background_gradient(subset=["MMD_red%"], cmap="Greens", vmin=0, vmax=100),
            use_container_width=True, hide_index=True,
        )

    # ── Pareto plot ─────────────────────────────────────────────────────────
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        with st.container(border=True):
            st.subheader("Pareto: MMD reduction vs runtime")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ok["runtime_ms"], y=ok["MMD_red%"],
                mode="markers+text", text=ok["method"], textposition="top center",
                marker=dict(size=14, color=ok["MMD_red%"], colorscale="Viridis",
                            colorbar=dict(title="MMD red %"), line=dict(width=1, color="#1E293B")),
                hovertemplate="%{text}<br>runtime=%{x:.1f} ms<br>MMD red=%{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="runtime (ms, log)", xaxis_type="log",
                yaxis_title="MMD reduction %",
            )
            style_figure(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

    # ── Scenario-specific commentary ────────────────────────────────────────
    tips = {
        "Different Means": "Mean shifts → CORAL / PT excel (fast + effective).",
        "Different SD":    "Scale shifts → SA, ART, PT typically reach 100% Mahalanobis reduction.",
        "Covariance shift": "SPD methods (PT, ART, Riemannian) designed for this.",
        "Heavy-tail (t)":   "Outliers break many methods; TCA is usually most robust.",
    }
    st.info(tips.get(scenario, ""))
