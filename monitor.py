"""Online drift monitor for deploy-and-monitor MI-EEG BCI (ShiftDx L3).

Turns the retrospective fixed-reference diagnostics into an *online controller*.
As each new session (or trial window) arrives, the monitor:

  1. computes pre-adaptation drift versus the fixed reference session
     (squared-MMD, the same coordinate the paper conditions on),
  2. standardizes it and feeds it to a Page-Hinkley change detector,
  3. recommends an action — KEEP / ADAPT (unsupervised DA) / RECALIBRATE
     (collect target-session labels and retrain) — using response functions
     S(z), B(z), R(z) fitted on historical data.

No target-session labels are needed online: the trigger runs on drift, which is
observed *before* any adaptation or labeling. Labels are only requested when the
policy decides recalibration is worth its cost.

This module is dependency-light (numpy + da4bci for MMD / Page-Hinkley) so it can
run inside a real deployment loop, in a CLI daemon, or be replayed offline for
back-testing. It is the seed of the extractable `shiftdx` engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ACTIONS = ("keep", "adapt", "recalibrate")


# ---------------------------------------------------------------------------
# Response-function policy (fitted on historical data)
# ---------------------------------------------------------------------------

@dataclass
class ResponsePolicy:
    """Linear response functions of standardized drift z, plus decision tolerances.

    S(z) = predicted No-DA accuracy loss vs the session-0 baseline
    B(z) = predicted DA benefit (DA - No-DA accuracy)
    R(z) = predicted residual gap (Retrain - DA accuracy)
    """
    s0: float
    s1: float
    b0: float
    b1: float
    r0: float
    r1: float
    loss_tol: float = 0.05   # acceptable No-DA loss (proportion) before acting
    gap_tol: float = 0.03    # residual retrain gap that justifies collecting labels

    def predict(self, z: float) -> dict:
        return {
            "loss": self.s0 + self.s1 * z,
            "da_benefit": self.b0 + self.b1 * z,
            "retrain_gap": self.r0 + self.r1 * z,
        }


@dataclass
class Decision:
    session: int
    drift_raw: float
    drift_z: float
    ph_stat: float
    ph_triggered: bool
    action: str
    predicted: dict
    rationale: str

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("session", "drift_raw", "drift_z", "ph_stat", "ph_triggered",
              "action", "rationale")}
        d.update({f"pred_{k}": v for k, v in self.predicted.items()})
        return d


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Stateful online drift monitor. Feed it one session at a time via update()."""

    def __init__(self, reference_features: Optional[np.ndarray] = None, *,
                 ref_drift_mean: float = 0.0, ref_drift_std: float = 1.0,
                 sigma: Optional[float] = None,
                 policy: Optional[ResponsePolicy] = None,
                 delta: float = 0.005, lam: float = 0.3, alpha: float = 0.9):
        from da4bci import ph_init
        self._ref = (np.asarray(reference_features, dtype=float)
                     if reference_features is not None else None)
        self._sigma = sigma
        self._mu = float(ref_drift_mean)
        self._sd = float(ref_drift_std) if ref_drift_std else 1.0
        self.policy = policy
        self._ph = ph_init(delta=delta, lambda_=lam, alpha=alpha)
        self.history: list[Decision] = []

    # -- drift from raw features (real deployment path) ----------------------
    def _drift_from_features(self, X_k: np.ndarray) -> float:
        from da4bci import compute_mmd, sigma_med
        if self._ref is None:
            raise ValueError("No reference features set; pass reference_features=…")
        sigma = self._sigma if self._sigma is not None else sigma_med(self._ref, X_k)
        return float(compute_mmd(self._ref, X_k, sigma))

    def _standardize(self, d: float) -> float:
        return (d - self._mu) / (self._sd + 1e-12)

    def update(self, *, features: Optional[np.ndarray] = None,
               drift: Optional[float] = None, drift_z: Optional[float] = None,
               session: Optional[int] = None) -> Decision:
        """Process one session. Provide exactly one of: raw `features`,
        precomputed raw `drift` (MMD²), or already-standardized `drift_z`."""
        if drift_z is not None:
            d, z = float("nan"), float(drift_z)
        elif drift is not None:
            d, z = float(drift), self._standardize(float(drift))
        elif features is not None:
            d = self._drift_from_features(features)
            z = self._standardize(d)
        else:
            raise ValueError("Provide features=, drift=, or drift_z=.")

        from da4bci import ph_update
        out = ph_update(self._ph, z)
        self._ph = out["state"]
        fired = bool(out["change"])
        ph_stat = self._ph["cum"] - self._ph["min_cum"]

        action, why, pred = self._decide(z, fired)
        dec = Decision(
            session=session if session is not None else len(self.history),
            drift_raw=d, drift_z=z, ph_stat=ph_stat, ph_triggered=fired,
            action=action, predicted=pred, rationale=why,
        )
        self.history.append(dec)
        return dec

    def _decide(self, z: float, fired: bool):
        if self.policy is None:
            if fired:
                return "recalibrate", "Page-Hinkley trigger (no response policy set)", {}
            return "keep", "no trigger; no policy", {}
        p = self.policy.predict(z)
        loss, benefit, gap = p["loss"], p["da_benefit"], p["retrain_gap"]
        if not fired and loss < self.policy.loss_tol:
            return "keep", f"predicted No-DA loss {loss:.3f} < tol {self.policy.loss_tol}", p
        if fired and gap > self.policy.gap_tol:
            return "recalibrate", (f"PH fired and residual gap {gap:.3f} > tol "
                                   f"{self.policy.gap_tol} → collect labels & retrain"), p
        if benefit > 0:
            return "adapt", (f"unsupervised DA recovers ~{benefit:.3f}; residual "
                             f"gap {gap:.3f} not worth labels yet"), p
        return ("recalibrate" if fired else "keep"), "fallback", p

    def reset_detector(self):
        from da4bci import ph_init
        st = self._ph
        self._ph = ph_init(delta=st["delta"], lambda_=st["lambda"], alpha=st["alpha"])

    # -- build a monitor whose policy is fitted on historical data -----------
    @classmethod
    def from_history(cls, merged_df, dataset: str, feature: str, da: str = "sa",
                     metric: str = "dist_mmd", **kw) -> "DriftMonitor":
        """Fit the S/B/R response functions and the drift standardizer on a
        historical `merged_drift_accuracy` frame, then return a ready monitor."""
        policy, mu, sd = _fit_policy(merged_df, dataset, feature, da, metric)
        return cls(ref_drift_mean=mu, ref_drift_std=sd, policy=policy, **kw)


# ---------------------------------------------------------------------------
# Policy fitting (reuses the retrospective diagnostics)
# ---------------------------------------------------------------------------

def _ols(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(intercept), float(slope)


def _fit_policy(merged, dataset, feature, da, metric):
    import pandas as pd  # noqa: F401
    sub = merged[(merged["dataset"] == dataset) & (merged["feature"] == feature)].copy()
    zcol = "drift_z_" + metric.replace("dist_", "")
    sub["z"] = sub[zcol] if zcol in sub.columns else sub.get("drift_z")

    noda = sub[(sub["strategy"] == "train_once") & (sub["da"] == "none")]
    dai = sub[(sub["strategy"] == "train_once_da") & (sub["da"] == da)]
    retr = sub[(sub["strategy"] == "retrain") &
               (sub["ref_session"] == sub["target_session"])]
    key = ["subject", "target_session"]

    # S(z): predicted No-DA loss = -(acc_centered)
    c0, c1 = _ols(noda["z"], noda["acc_centered"])
    s0, s1 = -c0, -c1

    # B(z): DA benefit
    pair = (noda[key + ["z", "accuracy"]].rename(columns={"accuracy": "an"})
            .merge(dai[key + ["accuracy"]].rename(columns={"accuracy": "ad"}), on=key))
    b0, b1 = _ols(pair["z"], pair["ad"] - pair["an"]) if len(pair) else (0.0, 0.0)

    # R(z): residual retrain gap
    pair2 = (dai[key + ["z", "accuracy"]].rename(columns={"accuracy": "ad"})
             .merge(retr[key + ["accuracy"]].rename(columns={"accuracy": "ar"}), on=key))
    r0, r1 = _ols(pair2["z"], pair2["ar"] - pair2["ad"]) if len(pair2) else (0.0, 0.0)

    mu = float(sub[metric].mean()) if metric in sub.columns else 0.0
    sd = float(sub[metric].std(ddof=1)) if metric in sub.columns else 1.0
    return ResponsePolicy(s0, s1, b0, b1, r0, r1), mu, sd


# ---------------------------------------------------------------------------
# Offline replay / back-test
# ---------------------------------------------------------------------------

def replay(monitor: DriftMonitor, drift_series, *, standardized: bool = False):
    """Feed a per-session drift series through a monitor; return the decisions."""
    out = []
    for k, d in enumerate(drift_series, start=1):
        if standardized:
            out.append(monitor.update(drift_z=float(d), session=k))
        else:
            out.append(monitor.update(drift=float(d), session=k))
    return out
