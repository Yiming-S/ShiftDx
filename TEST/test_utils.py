"""Unit tests for the shared helpers in utils.py."""

import numpy as np
import pandas as pd

import utils


def test_da_control_params_uses_dispatcher_keys():
    # These keys must match da4bci.methods.domain_adaptation, otherwise the
    # user's parameters are silently ignored.
    assert utils.da_control_params("coral", 10) == {"lambda": 1e-5}
    assert "dim_subspace" in utils.da_control_params("gfk", 10)
    assert "eps" in utils.da_control_params("ot", 10)
    assert "k" in utils.da_control_params("sa", 10)
    assert "max_dim" in utils.da_control_params("m3d", 10)
    assert utils.da_control_params("pt", 10) == {}


def test_is_retrain_filter():
    df = pd.DataFrame({
        "strategy": ["retrain", "retrain", "train_once"],
        "ref_session": [2, 0, 0],
        "target_session": [2, 2, 1],
    })
    out = utils.is_retrain(df)
    assert len(out) == 1
    assert out.iloc[0]["ref_session"] == 2


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.5, np.nan, 0.02])
    adj = utils.benjamini_hochberg(p)
    valid = adj[~np.isnan(adj)]
    assert np.all(valid >= 0) and np.all(valid <= 1)
    # adjusted >= raw for the smallest p
    assert adj[0] >= p[0]


def test_pvalue_badge_and_format():
    assert utils.pvalue_badge(0.0005) == "●●●"
    assert utils.pvalue_badge(0.005) == "●●"
    assert utils.pvalue_badge(0.03) == "●"
    assert utils.pvalue_badge(0.2) == "—"
    assert utils.format_pvalue(0.0001) == "<0.001"
    assert utils.ci_excludes_zero(0.1, 0.3) is True
    assert utils.ci_excludes_zero(-0.1, 0.3) is False


def test_generate_shift_shapes():
    s, t = utils.generate_shift("Different Means", 50, 4, seed=1)
    assert s.shape == (50, 4) and t.shape == (50, 4)
    # reproducible
    s2, _ = utils.generate_shift("Different Means", 50, 4, seed=1)
    assert np.allclose(s, s2)


def test_cluster_bootstrap_ci_recovers_slope():
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(8):
        x = rng.normal(0, 1, 12)
        y = -0.5 * x + rng.normal(0, 0.1, 12)
        for xi, yi in zip(x, y):
            rows.append({"uid": uid, "x": xi, "y": yi})
    df = pd.DataFrame(rows)
    point, lo, hi = utils.cluster_bootstrap_ci(
        df, lambda d: utils.ols_slope(d, "x", "y"),
        cluster_col="uid", n_boot=300, seed=1)
    assert lo <= point <= hi
    assert -0.65 < point < -0.35     # recovers the true slope (~ -0.5)
    assert lo <= -0.5 <= hi          # CI brackets the truth


def test_colorblind_palette_accessors():
    # Default (off) returns the brand palette.
    assert utils.feature_colors() is utils.FEATURE_COLORS
