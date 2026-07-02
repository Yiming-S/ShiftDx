"""The synthetic demo generator produces a schema-valid dataset the dashboard
can load."""

import gen_synthetic_demo as gen
from data_loader import DataStore, REQUIRED_COLUMNS


def test_generate_shapes():
    data = gen.generate(n_subjects=2, n_sessions=4, seed=0)
    assert not data["drift"].empty
    assert not data["merged"].empty
    # 2 subjects × 3 features × (3 baseline rows + 3 sessions × 3 clf ×
    # (1 No-DA + 10 DA + 1 retrain)) — just assert it's substantial.
    assert len(data["merged"]) > 100


def test_written_dataset_loads(tmp_path):
    data = gen.generate(n_subjects=2, n_sessions=4, seed=0)
    gen.write(str(tmp_path), data)
    store = DataStore(str(tmp_path))
    assert gen.DATASET in store.datasets
    _ = store.drift_df, store.eval_df, store.merged_df
    assert store.schema_issues == {}
    for col in REQUIRED_COLUMNS["merged_drift_accuracy"]:
        assert col in store.merged_df.columns
    assert "demo_synthetic" in store.manifests


def test_synthetic_has_expected_signal():
    """No-DA accuracy should fall with drift for CSP (the drift-sensitive
    feature) — a sanity check that the generator encodes the paper's story."""
    import numpy as np
    data = gen.generate(n_subjects=3, n_sessions=4, seed=0)
    m = data["merged"]
    csp = m[(m["feature"] == "CSP") & (m["strategy"] == "train_once") &
            (m["da"] == "none")]
    slope = np.polyfit(csp["drift_z"], csp["acc_centered"], 1)[0]
    assert slope < 0
