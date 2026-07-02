"""Page 10: DA Method Sweep — all DA methods from DA4BCI on a controlled shift."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    DA_LABEL, DA_METHODS, DA_LAB_UNSUPPORTED,
    SHIFT_SCENARIOS, SCENARIO_PAIR_LABEL, SCENARIO_TIPS,
    about_page, da_colors, da_control_params, download_bar, empty_state,
    euclid_distance, generate_shift, mmd_distance, style_figure,
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


# m3d needs class labels → unusable on the synthetic sweep.
ALL_METHODS = [m for m in DA_METHODS if m not in DA_LAB_UNSUPPORTED]
DEFAULT_METHODS = [m for m in ALL_METHODS if m != "mida"]  # mida slow on small dims


@st.cache_data(show_spinner=False)
def _run_sweep(scenario: str, n: int, d: int, seed: int,
               methods: tuple) -> pd.DataFrame:
    from da4bci import (
        domain_adaptation, compute_energy, compute_wasserstein,
        compute_mahalanobis,
    )

    def _wass(x, y):
        try:
            return compute_wasserstein(x, y)
        except Exception:
            return np.nan

    source, target = generate_shift(scenario, n, d, seed)
    base_mmd = mmd_distance(source, target)
    base_energy = compute_energy(source, target)
    base_wass = _wass(source, target)
    base_maha = compute_mahalanobis(source, target)
    base_euclid = euclid_distance(source, target)

    def _red(b, a):
        if b is None or np.isnan(b) or abs(b) < 1e-12 or np.isnan(a):
            return np.nan
        return (b - a) / b * 100

    rows = []
    for m in methods:
        try:
            t0 = time.perf_counter()
            res = domain_adaptation(source, target, method=m,
                                    control=da_control_params(m, d))
            rt = (time.perf_counter() - t0) * 1000
            src_a = res.get("weighted_source_data", source)
            tgt_a = res.get("target_data", target)
            rows.append({
                "method": m, "runtime_ms": rt,
                "MMD_before": base_mmd, "MMD_after": mmd_distance(src_a, tgt_a),
                "MMD_red%":         _red(base_mmd, mmd_distance(src_a, tgt_a)),
                "Energy_red%":      _red(base_energy, compute_energy(src_a, tgt_a)),
                "Wasserstein_red%": _red(base_wass, _wass(src_a, tgt_a)),
                "Maha_red%":        _red(base_maha, compute_mahalanobis(src_a, tgt_a)),
                "Euclid_red%":      _red(base_euclid, euclid_distance(src_a, tgt_a)),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({
                "method": m, "runtime_ms": np.nan,
                "MMD_before": base_mmd, "MMD_after": np.nan,
                "MMD_red%": np.nan, "Energy_red%": np.nan,
                "Wasserstein_red%": np.nan, "Maha_red%": np.nan,
                "Euclid_red%": np.nan,
                "status": f"error: {exc}",
            })
    return pd.DataFrame(rows)


def render(store):
    st.header("DA Method Sweep")
    st.caption(
        "For a shift scenario, run every selected DA method and compare on the "
        "accuracy–runtime Pareto frontier. Useful for picking a method that "
        "balances quality and latency for real-time BCI."
    )

    about_page(
        what_you_see=[
            "Table of per-method runtime + reduction % on 5 distance metrics.",
            "Pareto scatter: MMD reduction vs runtime.",
            "Tips for this scenario at the bottom.",
        ],
        how_to_read=[
            "Upper-left of the Pareto plot = fast + effective.",
            "Reduction % < 0 means the method made the distance *worse*.",
        ],
        paper_ref="Claim 2 / §5.3 intuition",
    )

    if not _ensure_da4bci():
        empty_state("DA4BCI not importable", _DA4BCI_ERR,
                    cmd="pip install -e /path/to/DA4BCI-Python")
        return

    dcolors = da_colors()

    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox("Scenario", list(SHIFT_SCENARIOS.keys()))
    n = c2.number_input("n per domain", 100, 2000, 500, 100)
    d = c3.number_input("feature dim", 2, 50, 20, 1)
    seed = c4.number_input("seed", 0, 9999, 42, 1)

    methods_sel = st.multiselect(
        "Methods to include", ALL_METHODS,
        default=DEFAULT_METHODS,
        format_func=lambda m: DA_LABEL.get(m, m),
        help="MIDA is off by default because it can be slow on small feature dims. "
             "M3D is unavailable (needs class labels).",
    )
    if not methods_sel:
        st.info("Select at least one DA method.")
        return

    pair = SCENARIO_PAIR_LABEL.get(scenario, ("source", "target"))
    st.caption(f"Source vs target: **{pair[0]}** vs **{pair[1]}**")

    with st.spinner("Running sweep…"):
        df = _run_sweep(scenario, int(n), int(d), int(seed), tuple(methods_sel))

    if df.empty:
        empty_state("No results", "The sweep returned no rows.")
        return

    # ── Table ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Results")
        df_sorted = df.sort_values("MMD_red%", ascending=False)
        st.dataframe(
            df_sorted.style.format({
                "runtime_ms": "{:.1f}",
                "MMD_before": "{:.4f}", "MMD_after": "{:.4f}",
                "MMD_red%": "{:+.1f}", "Energy_red%": "{:+.1f}",
                "Wasserstein_red%": "{:+.1f}", "Maha_red%": "{:+.1f}",
                "Euclid_red%": "{:+.1f}",
            }).background_gradient(subset=["MMD_red%"], cmap="Greens",
                                    vmin=0, vmax=100),
            use_container_width=True, hide_index=True,
        )
        download_bar("sweep_results", df_sorted,
                     f"sweep_{scenario.replace(' ', '_')}")

    # ── Pareto plot ─────────────────────────────────────────────────────────
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        with st.container(border=True):
            st.subheader("Pareto: MMD reduction vs runtime")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ok["runtime_ms"], y=ok["MMD_red%"],
                mode="markers+text",
                text=ok["method"], textposition="top center",
                marker=dict(
                    size=14, color=ok["MMD_red%"], colorscale="Viridis",
                    colorbar=dict(title="MMD red %"),
                    line=dict(width=1, color="#1E293B"),
                ),
                hovertemplate=(
                    "%{text}<br>runtime=%{x:.1f} ms<br>"
                    "MMD red=%{y:.1f}%<extra></extra>"
                ),
            ))
            fig.update_layout(
                xaxis_title="runtime (ms, log)", xaxis_type="log",
                yaxis_title="MMD reduction %",
            )
            style_figure(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

    errored = df[df["status"] != "ok"]
    if not errored.empty:
        with st.expander(f"{len(errored)} method(s) errored"):
            for _, r in errored.iterrows():
                st.caption(f"**{r['method']}** — {r['status']}")

    # ── Scenario tips as cards ──────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Scenario tips")
        tips = {k: v for k, v in SCENARIO_TIPS.items() if k in SHIFT_SCENARIOS}
        cards = st.columns(len(tips))
        for i, (name, tip) in enumerate(tips.items()):
            with cards[i]:
                badge = "► " if name == scenario else ""
                st.markdown(f"**{badge}{name}**")
                st.caption(tip)
