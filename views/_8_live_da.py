"""Page 8: Live DA Sandbox — powered by DA4BCI-Python."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from utils import (
    DA_LABEL, DA_METHODS, DA_LAB_UNSUPPORTED,
    SHIFT_SCENARIOS, about_page, da_control_params, download_bar,
    empty_state, euclid_distance, generate_shift, mmd_distance, style_figure,
)

# da4bci is imported lazily (see _ensure_da4bci) so the 8 non-DA-Lab pages don't
# pay its import cost on cold start.
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


# DA methods usable on the unlabeled synthetic sandbox (drop m3d — it needs
# source labels that synthetic shifts don't have).
_SANDBOX_METHODS = [m for m in DA_METHODS if m not in DA_LAB_UNSUPPORTED]


def render(store):
    st.header("Live DA Sandbox")
    st.caption(
        "Generate a synthetic source/target shift, apply any DA method, and "
        "inspect the alignment. Useful to build intuition for each method's behaviour."
    )

    about_page(
        what_you_see=[
            "Before / after headline card comparing the distance metrics.",
            "Details table with per-metric reduction %, plus Proxy A-Distance.",
            "2-D PCA projection of source (before + after) and target.",
        ],
        how_to_read=[
            "Positive reduction % ⇒ the DA method successfully aligned that metric.",
            "On the PCA scatter, blue (source after) should land on top of orange (target).",
            "Proxy A-Distance near 0 ⇒ a classifier can no longer tell the domains apart.",
        ],
        paper_ref="Claim 2 intuition",
    )

    if not _ensure_da4bci():
        empty_state(
            "DA4BCI not importable",
            f"{_DA4BCI_ERR}",
            cmd="pip install -e /path/to/DA4BCI-Python",
            cmd_label="Install DA4BCI",
        )
        return

    from da4bci import (
        domain_adaptation, compute_energy, compute_wasserstein,
        compute_mahalanobis,
    )
    try:
        from da4bci import proxy_a_distance
    except Exception:
        proxy_a_distance = None

    # ── Controls ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox("Shift scenario", list(SHIFT_SCENARIOS.keys()))
    n = c2.number_input("Samples per domain", 100, 2000, 500, 100)
    d = c3.number_input("Feature dim", 2, 50, 10, 1)
    seed = c4.number_input("Seed", 0, 9999, 42, 1)

    method = st.selectbox(
        "DA method", _SANDBOX_METHODS,
        index=_SANDBOX_METHODS.index("coral"),
        format_func=lambda m: DA_LABEL.get(m, m),
        help="M3D is excluded here because it needs class labels, which the "
             "synthetic shift does not provide.",
    )

    # Correct DA4BCI dispatcher control keys (e.g. coral→lambda, gfk→dim_subspace,
    # ot→eps). Editing these now actually changes the result.
    params = da_control_params(method, int(d))
    with st.expander("DA parameters"):
        if not params:
            st.caption("This method takes no tunable parameters.")
        for k, v in list(params.items()):
            if isinstance(v, bool):
                params[k] = st.checkbox(f"{method}.{k}", value=v, key=f"p_{method}_{k}")
            elif isinstance(v, int):
                params[k] = st.number_input(
                    f"{method}.{k}", 1, 100, int(v), 1, key=f"p_{method}_{k}")
            elif isinstance(v, float):
                params[k] = st.number_input(
                    f"{method}.{k}", 1e-8, 10.0, float(v),
                    format="%.6g", key=f"p_{method}_{k}")

    run = st.button("▶ Run DA", type="primary", use_container_width=False,
                    help="Runs once with the current controls. Avoids rerunning on every tweak.")

    if not run and "sandbox_last_result" not in st.session_state:
        st.info("Set your controls and click **Run DA** to see the alignment.")
        return

    if run:
        source, target = generate_shift(scenario, int(n), int(d), int(seed))

        t0 = time.perf_counter()
        try:
            result = domain_adaptation(source, target, method=method, control=params)
        except Exception as exc:
            st.error(f"DA failed: {exc}")
            return
        runtime_ms = (time.perf_counter() - t0) * 1000

        src_adapted = result.get("weighted_source_data", source)
        tgt_adapted = result.get("target_data", target)
        st.session_state["sandbox_last_result"] = {
            "source": source, "target": target,
            "src_adapted": src_adapted, "tgt_adapted": tgt_adapted,
            "runtime_ms": runtime_ms, "method": method,
            "scenario": scenario, "params": dict(params),
        }

    cache = st.session_state["sandbox_last_result"]
    source = cache["source"]; target = cache["target"]
    src_adapted = cache["src_adapted"]; tgt_adapted = cache["tgt_adapted"]

    # ── Distances before / after ────────────────────────────────────────────
    def _dist_row(xs, xt):
        row = {"MMD": mmd_distance(xs, xt), "Energy": compute_energy(xs, xt)}
        try:
            row["Wasserstein"] = compute_wasserstein(xs, xt)
        except Exception:
            row["Wasserstein"] = np.nan
        row["Mahalanobis"] = compute_mahalanobis(xs, xt)
        row["Euclidean"] = euclid_distance(xs, xt)
        if proxy_a_distance is not None:
            try:
                row["PAD"] = proxy_a_distance(xs, xt, seed=0)["pad"]
            except Exception:
                row["PAD"] = np.nan
        return row

    before = _dist_row(source, target)
    after = _dist_row(src_adapted, tgt_adapted)
    rows = []
    for k in before:
        b, a = before[k], after[k]
        rows.append({
            "metric": k, "before": b, "after": a,
            "Δ": a - b,
            "reduction %": (b - a) / b * 100 if b else np.nan,
        })
    dist_tbl = pd.DataFrame(rows)

    # ── Headline card ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Before vs After · {cache['method'].upper()} on {cache['scenario']}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Runtime", f"{cache['runtime_ms']:.1f} ms")
        k2.metric("MMD before", f"{before['MMD']:.4f}")
        k3.metric("MMD after", f"{after['MMD']:.4f}",
                  delta=f"{rows[0]['reduction %']:+.1f}%",
                  delta_color="inverse",
                  help="Larger reduction (more negative delta of distance) is better.")

    # ── Table ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Distance metrics")
        st.dataframe(
            dist_tbl.style.format({
                "before": "{:.4f}", "after": "{:.4f}",
                "Δ": "{:+.4f}", "reduction %": "{:+.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
        download_bar("sandbox_metrics", dist_tbl,
                     f"sandbox_{cache['method']}_{cache['scenario'].replace(' ', '_')}")

    # ── PCA scatter ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("PCA scatter (2-D)")
        fig = go.Figure()
        pca_before = PCA(n_components=2).fit(np.vstack([source, target]))
        s_b = pca_before.transform(source); t_b = pca_before.transform(target)

        fig.add_trace(go.Scatter(
            x=s_b[:, 0], y=s_b[:, 1], mode="markers",
            marker=dict(color="#94A3B8", size=5, opacity=0.5),
            name="source (before)",
        ))
        fig.add_trace(go.Scatter(
            x=t_b[:, 0], y=t_b[:, 1], mode="markers",
            marker=dict(color="#F59E0B", size=5, opacity=0.5, symbol="x"),
            name="target",
        ))
        # Subspace methods (SA/TCA/MIDA) return a k-D adapted source (k < d), which
        # cannot be projected through the original d-D PCA. Only overlay when dims match.
        if np.asarray(src_adapted).shape[1] == source.shape[1]:
            s_a = pca_before.transform(src_adapted)
            fig.add_trace(go.Scatter(
                x=s_a[:, 0], y=s_a[:, 1], mode="markers",
                marker=dict(color="#4F46E5", size=5, opacity=0.7),
                name="source (after)",
            ))
        else:
            st.caption(
                f"{cache['method'].upper()} maps the source into a "
                f"{np.asarray(src_adapted).shape[1]}-D subspace, so the adapted points "
                "are not overlaid in the original feature space."
            )
        fig.update_layout(
            xaxis_title="PC1", yaxis_title="PC2",
            legend=dict(orientation="h", y=-0.15),
        )
        style_figure(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Reference: paper reports SA provides ≈ +3.5 pp level shift without "
        "altering the drift-response slope (Claim 2). Try PT or ART on rotation "
        "/ covariance shifts to see Riemannian transport behaviour."
    )
