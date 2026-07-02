"""DataStore loads the bundled CSVs and they pass schema validation."""

import pandas as pd

from data_loader import REQUIRED_COLUMNS


def test_datasets_discovered(store):
    # zhou2016 and bnci004 ship in data/.
    assert "bnci004" in store.datasets
    assert "zhou2016" in store.datasets


def test_frames_nonempty(store):
    assert not store.drift_df.empty
    assert not store.eval_df.empty
    assert not store.merged_df.empty


def test_required_columns_present(store):
    for col in REQUIRED_COLUMNS["drift_trajectories"]:
        assert col in store.drift_df.columns
    for col in REQUIRED_COLUMNS["sequential_eval"]:
        assert col in store.eval_df.columns
    for col in REQUIRED_COLUMNS["merged_drift_accuracy"]:
        assert col in store.merged_df.columns


def test_schema_validation_clean(store):
    # Touch the lazy frames so schema_issues is populated, then assert clean.
    _ = store.drift_df, store.eval_df, store.merged_df
    assert store.schema_issues == {}


def test_no_dead_geometric_property(store):
    assert not hasattr(store, "geometric_df")


def test_subjects_helper(store):
    subs = store.subjects("bnci004")
    assert len(subs) > 0
    assert all(isinstance(int(s), int) for s in subs)
