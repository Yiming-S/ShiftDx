"""
Data loading module for the ShiftDx dashboard.

Reads the three CSV families produced by any ShiftDx builder:

  - drift_trajectories_{dataset}.csv   per-(subject, session, feature) with 5 distance metrics
  - sequential_eval_{dataset}.csv      per-(subject, session, feature, DA, strategy) accuracy
  - merged_drift_accuracy_combined.csv pooled drift + accuracy with drift_z

Datasets are auto-discovered by scanning `data/` for `drift_trajectories_*.csv`.
"""

import os
import logging

import streamlit as st
import pandas as pd

from utils import discover_datasets

logger = logging.getLogger(__name__)


# Minimal required columns per CSV family. Validation is advisory (warn, don't
# hard-fail) so a partially-built dataset still loads; views additionally guard
# with utils.require_columns before touching specific columns.
REQUIRED_COLUMNS = {
    "drift_trajectories": ["session_k", "feature", "subject", "dist_mmd"],
    "sequential_eval": ["target_session", "feature", "classifier", "da",
                        "strategy", "accuracy", "subject"],
    "merged_drift_accuracy": ["target_session", "feature", "classifier", "da",
                              "strategy", "accuracy", "subject", "drift_z"],
}


def _validate_schema(df: pd.DataFrame, prefix: str, ds: str) -> list[str]:
    """Return (and log) the list of required columns missing from `df`."""
    required = REQUIRED_COLUMNS.get(prefix, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("%s_%s.csv is missing columns: %s", prefix, ds, missing)
    return missing


@st.cache_resource(ttl=600)
def get_data_store(data_dir: str) -> "DataStore":
    return DataStore(data_dir)


class DataStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets = discover_datasets(data_dir)
        self._drift_df: pd.DataFrame | None = None
        self._eval_df: pd.DataFrame | None = None
        self._merged_df: pd.DataFrame | None = None
        self._manifests: dict | None = None
        self.schema_issues: dict[str, list[str]] = {}

    @property
    def drift_df(self) -> pd.DataFrame:
        if self._drift_df is None:
            self._drift_df = self._load_group("drift_trajectories")
        return self._drift_df

    @property
    def eval_df(self) -> pd.DataFrame:
        if self._eval_df is None:
            self._eval_df = self._load_group("sequential_eval")
        return self._eval_df

    @property
    def merged_df(self) -> pd.DataFrame:
        if self._merged_df is None:
            self._merged_df = self._load_merged()
        return self._merged_df

    @property
    def manifests(self) -> dict:
        """Per-dataset build provenance read from build_manifest_<ds>.json
        (written by scripts/build_moabb.py). Empty when none are present."""
        if self._manifests is None:
            self._manifests = self._load_manifests()
        return self._manifests

    # ------------------------------------------------------------------
    def _load_group(self, prefix: str, required: bool = True) -> pd.DataFrame:
        frames = []
        for ds in self.datasets:
            fpath = os.path.join(self.data_dir, f"{prefix}_{ds}.csv")
            if not os.path.isfile(fpath):
                if required:
                    logger.warning("Missing %s file: %s", prefix, fpath)
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", fpath, exc)
                continue
            missing = _validate_schema(df, prefix, ds)
            if missing:
                self.schema_issues[f"{prefix}_{ds}"] = missing
            if "dataset" not in df.columns:
                df["dataset"] = ds
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _load_manifests(self) -> dict:
        import json
        out: dict = {}
        for ds in self.datasets:
            p = os.path.join(self.data_dir, f"build_manifest_{ds}.json")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        out[ds] = json.load(f)
                except Exception as exc:
                    logger.warning("Failed to read manifest %s: %s", p, exc)
        return out

    def _load_merged(self) -> pd.DataFrame:
        """Concatenate the legacy `merged_drift_accuracy_combined.csv` (if present)
        with all per-dataset `merged_drift_accuracy_<ds>.csv` files, so datasets
        built after the combined file (e.g. zhou2016) are still picked up."""
        frames = []
        datasets_covered: set[str] = set()

        combined = os.path.join(self.data_dir, "merged_drift_accuracy_combined.csv")
        if os.path.isfile(combined):
            try:
                df = pd.read_csv(combined)
                frames.append(df)
                if "dataset" in df.columns:
                    datasets_covered.update(df["dataset"].dropna().astype(str).unique())
            except Exception as exc:
                logger.warning("Failed to read %s: %s", combined, exc)

        for ds in self.datasets:
            if ds in datasets_covered:
                continue
            p = os.path.join(self.data_dir, f"merged_drift_accuracy_{ds}.csv")
            if not os.path.isfile(p):
                continue
            try:
                df = pd.read_csv(p)
                missing = _validate_schema(df, "merged_drift_accuracy", ds)
                if missing:
                    self.schema_issues[f"merged_drift_accuracy_{ds}"] = missing
                if "dataset" not in df.columns:
                    df["dataset"] = ds
                frames.append(df)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", p, exc)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        # Legacy CSVs only have `drift_z` (MMD-based). Mirror it into
        # `drift_z_mmd` so the metric selector works on them too.
        if "drift_z" in out.columns and "drift_z_mmd" not in out.columns:
            out["drift_z_mmd"] = out["drift_z"]
        elif "drift_z" in out.columns and "drift_z_mmd" in out.columns:
            out["drift_z_mmd"] = out["drift_z_mmd"].fillna(out["drift_z"])
        return out

    def subjects(self, dataset: str) -> list[int]:
        df = self.drift_df
        if df.empty:
            return []
        if dataset == "All":
            return sorted(df["subject"].unique().tolist())
        return sorted(df.loc[df["dataset"] == dataset, "subject"].unique().tolist())
