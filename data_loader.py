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
        self._geometric_df: pd.DataFrame | None = None

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
    def geometric_df(self) -> pd.DataFrame:
        if self._geometric_df is None:
            self._geometric_df = self._load_group("geometric", required=False)
        return self._geometric_df

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
            if "dataset" not in df.columns:
                df["dataset"] = ds
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

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
