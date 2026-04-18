"""Page 8: Live DA Sandbox — powered by DA4BCI-Python.

Uses synthetic source/target data (paper's benchmark scenarios) to let the
user experiment with every DA method in DA4BCI and see before/after
distribution alignment. Since ShiftDx does not yet ship raw per-trial EEG
features, this page demonstrates DA4BCI's capability on controlled shifts;
the same code path will work on real features once a backing store is added.
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

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
    """Mean pairwise Euclidean distance between source and target sets."""
    try:
        D = compute_distance_matrix(x, y)
        return float(np.mean(D))
    except Exception:
        # Fallback: np
        x_ = np.asarray(x); y_ = np.asarray(y)
        return float(np.mean(np.linalg.norm(
            x_[:, None, :] - y_[None, :, :], axis=-1
        )))


SCENARIOS = {
    "Different Means": lambda n, d, rng: (rng.normal(0, 1, (n, d)),
                                           rng.normal(0.8, 1, (n, d))),
    "Different SD": lambda n, d, rng: (rng.normal(0, 1, (n, d)),
                                         rng.normal(0, 2, (n, d))),
    "Rotation": lambda n, d, rng: (_rotate(rng.normal(0, 1, (n, d)), 0, rng),
                                    _rotate(rng.normal(0, 1, (n, d)), np.pi / 6, rng)),
    "Heavy-tail (t)": lambda n, d, rng: (rng.standard_t(5, (n, d)),
                                          rng.standard_t(3, (n, d)) + 0.3),
    "Covariance shift": lambda n, d, rng: (
        rng.multivariate_normal(np.zeros(d), np.eye(d), n),
        rng.multivariate_normal(np.zeros(d), np.diag(np.linspace(0.5, 2.0, d)), n),
    ),
}


def _rotate(X, angle, rng):
    d = X.shape[1]
    R = np.eye(d)
    c, s = np.cos(angle), np.sin(angle)
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return X @ R


DA_METHODS = ["sa", "pt", "coral", "gfk", "tca", "rd", "art", "ot", "mida", "m3d"]

DA_DEFAULTS = {
    "sa": {"k": 10},
    "pt": {},
    "coral": {"lam": 1e-5},
    "gfk": {"d": 10},
    "tca": {"k": 10},
    "rd": {},
    "art": {},
    "ot": {"reg": 0.1},
    "mida": {"d": 10},
    "m3d": {"d": 10},
}


def render(store, dataset):
    st.header("Live DA Sandbox")
    st.caption(
        "Generate a synthetic source/target shift, apply any of 10 DA methods "
        "from DA4BCI, and inspect the distribution alignment. Real per-trial "
        "feature data can be wired in later."
    )

    if not HAS_DA4BCI:
        st.error(f"DA4BCI not importable: {_DA4BCI_ERR}")
        st.info("Install with: `pip install -e /path/to/DA4BCI-Python`")
        return

    # ── Controls ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox("Shift scenario", list(SCENARIOS.keys()))
    n = c2.number_input("Samples per domain", 100, 2000, 500, 100)
    d = c3.number_input("Feature dim", 2, 50, 10, 1)
    seed = c4.number_input("Seed", 0, 9999, 42, 1)

    METHOD_LABELS = {
        "sa": "SA (Subspace Alignment)",
        "pt": "PT (Parallel Transport)",
        "coral": "CORAL",
        "gfk": "GFK (Geodesic Flow Kernel)",
        "tca": "TCA (Transfer Component Analysis)",
        "rd": "RD (Riemannian Distance)",
        "art": "ART (Aligned Riemannian Transport)",
        "ot": "OT (Optimal Transport)",
        "mida": "MIDA",
        "m3d": "M3D",
    }
    method = st.selectbox("DA method", DA_METHODS, format_func=lambda m: METHOD_LABELS[m])

    # Per-method parameter override
    params = DA_DEFAULTS[method].copy()
    with st.expander("DA parameters"):
        for k, v in list(params.items()):
            if isinstance(v, int):
                params[k] = st.number_input(f"{method}.{k}", 1, 100, int(v), 1,
                                             key=f"p_{method}_{k}")
            elif isinstance(v, float):
                params[k] = st.number_input(f"{method}.{k}", 1e-8, 10.0, float(v),
                                             format="%.6g", key=f"p_{method}_{k}")

    # ── Generate data ───────────────────────────────────────────────────────
    rng = np.random.default_rng(int(seed))
    source, target = SCENARIOS[scenario](int(n), int(d), rng)

    # ── Run DA ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = domain_adaptation(source, target, method=method, control=params)
    except Exception as exc:
        st.error(f"DA failed: {exc}")
        return
    runtime_ms = (time.perf_counter() - t0) * 1000

    src_adapted = result.get("weighted_source_data", source)
    tgt_adapted = result.get("target_data", target)

    # ── Compute distances before/after ──────────────────────────────────────
    def _dist_row(xs, xt):
        row = {"MMD": _mmd(xs, xt), "Energy": compute_energy(xs, xt)}
        try:
            row["Wasserstein"] = compute_wasserstein(xs, xt)
        except Exception:
            row["Wasserstein"] = np.nan
        row["Mahalanobis"] = compute_mahalanobis(xs, xt)
        row["Euclidean"] = _euclid(xs, xt)
        return row

    before = _dist_row(source, target)
    after = _dist_row(src_adapted, tgt_adapted)
    rows = []
    for k in before:
        b, a = before[k], after[k]
        rows.append({"metric": k, "before": b, "after": a,
                     "Δ": a - b, "reduction %": (b - a) / b * 100 if b else np.nan})
    dist_tbl = pd.DataFrame(rows)

    k1, k2 = st.columns(2)
    k1.metric("Runtime", f"{runtime_ms:.1f} ms")
    k2.metric("MMD reduction", f"{rows[0]['reduction %']:+.1f}%")

    with st.container(border=True):
        st.subheader("Distance before vs after")
        st.dataframe(
            dist_tbl.style.format({"before": "{:.4f}", "after": "{:.4f}",
                                    "Δ": "{:+.4f}", "reduction %": "{:+.1f}"}),
            use_container_width=True, hide_index=True,
        )

    # ── PCA scatter ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("PCA scatter (2D)")
        fig = go.Figure()
        pca_before = PCA(n_components=2).fit(np.vstack([source, target]))
        s_b = pca_before.transform(source); t_b = pca_before.transform(target)
        s_a = pca_before.transform(src_adapted); t_a = pca_before.transform(tgt_adapted)

        fig.add_trace(go.Scatter(x=s_b[:, 0], y=s_b[:, 1], mode="markers",
                                  marker=dict(color="#94A3B8", size=5, opacity=0.5),
                                  name="source (before)"))
        fig.add_trace(go.Scatter(x=t_b[:, 0], y=t_b[:, 1], mode="markers",
                                  marker=dict(color="#F59E0B", size=5, opacity=0.5,
                                              symbol="x"),
                                  name="target"))
        fig.add_trace(go.Scatter(x=s_a[:, 0], y=s_a[:, 1], mode="markers",
                                  marker=dict(color="#4F46E5", size=5, opacity=0.7),
                                  name="source (after)"))

        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2",
                          legend=dict(orientation="h", y=-0.15))
        from utils import style_figure as _sf
        _sf(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Reference: paper reports that SA provides a +3.5 pp level shift "
        "without altering the drift-response slope (Claim 2). "
        "Try PT or ART to see Riemannian transport behavior on rotation / covariance shifts."
    )
