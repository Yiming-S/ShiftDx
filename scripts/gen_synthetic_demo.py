"""Generate a tiny synthetic ShiftDx dataset so the full dashboard runs without
MOABB / raw EEG. Useful for screenshots, demos, and the TEST/ suite.

Writes the three CSV families (+ a build manifest) for a `demo_synthetic`
dataset whose structure mirrors a real build:
  * drift grows with session_k,
  * No-DA accuracy falls with drift (steeply for CSP, flat for logvar),
  * DA partially recovers it (level shift + reduced slope),
  * Retrain is the drift-independent ceiling.

Usage:
    python scripts/gen_synthetic_demo.py                 # writes into data/
    python scripts/gen_synthetic_demo.py --out-dir /tmp  # elsewhere
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

DATASET = "demo_synthetic"
FEATURES = ["CSP", "logvar", "TS"]
CLASSIFIERS = ["lda", "svm_linear", "svm_radial"]
DA_METHODS = ["sa", "pt", "coral", "tca", "gfk", "rd", "art", "ot", "mida", "m3d"]

# Per-feature No-DA drift sensitivity (slope of accuracy on z-scored drift).
FEATURE_SLOPE = {"CSP": -0.08, "TS": -0.055, "logvar": -0.012}
FEATURE_BASELINE = {"CSP": 0.82, "TS": 0.80, "logvar": 0.71}
# Per-DA-method level shift (constant gain) on top of No-DA.
DA_LEVEL = {"sa": 0.040, "pt": 0.035, "coral": 0.010, "tca": 0.020,
            "gfk": 0.005, "rd": 0.022, "art": 0.045, "ot": 0.018,
            "mida": -0.010, "m3d": 0.030}
METRIC_SCALE = {"dist_mmd": 1.0, "dist_energy": 1.5, "dist_wasserstein": 0.8,
                "dist_mahalanobis": 2.0, "dist_euclidean": 3.0}


def generate(n_subjects: int = 3, n_sessions: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    merged_rows = []
    drift_rows = []

    for subj in range(1, n_subjects + 1):
        for feat in FEATURES:
            base_by_clf = {c: FEATURE_BASELINE[feat] + rng.normal(0, 0.01)
                           for c in CLASSIFIERS}
            # Session-0 baseline (5-fold CV) rows
            for clf in CLASSIFIERS:
                merged_rows.append(_row(subj, feat, clf, "none", "retrain",
                                        0, 0, base_by_clf[clf], base_by_clf[clf],
                                        {m: 0.0 for m in METRIC_SCALE}, n_sessions))
            for k in range(1, n_sessions):
                mmd = max(0.01, 0.18 * k + rng.normal(0, 0.04))
                dists = {m: mmd * s for m, s in METRIC_SCALE.items()}
                drift_rows.append({
                    "session_k": k, "feature": feat, **dists,
                    "subject": subj, "dataset": DATASET, "n_sessions": n_sessions,
                })
                for clf in CLASSIFIERS:
                    base = base_by_clf[clf]
                    # placeholder z (recomputed properly after assembling merged)
                    z = mmd  # monotone proxy; real drift_z computed below
                    noda = np.clip(base + FEATURE_SLOPE[feat] * z * 3
                                   + rng.normal(0, 0.02), 0.4, 0.98)
                    merged_rows.append(_row(subj, feat, clf, "none", "train_once",
                                            0, k, noda, base, dists, n_sessions))
                    for m in DA_METHODS:
                        acc = np.clip(base + FEATURE_SLOPE[feat] * z * 3 * 0.6
                                      + DA_LEVEL[m] + rng.normal(0, 0.02), 0.4, 0.98)
                        merged_rows.append(_row(subj, feat, clf, m, "train_once_da",
                                                0, k, acc, base, dists, n_sessions))
                    retr = np.clip(base + rng.normal(0, 0.02), 0.4, 0.99)
                    merged_rows.append(_row(subj, feat, clf, "none", "retrain",
                                            k, k, retr, base, dists, n_sessions))

    merged = pd.DataFrame(merged_rows)
    # Proper drift_z within (dataset × feature × classifier) blocks.
    for mcol, zcol in [("dist_mmd", "drift_z_mmd"), ("dist_energy", "drift_z_energy"),
                       ("dist_wasserstein", "drift_z_wasserstein"),
                       ("dist_mahalanobis", "drift_z_mahalanobis"),
                       ("dist_euclidean", "drift_z_euclidean")]:
        merged[zcol] = (merged.groupby(["dataset", "feature", "classifier"])[mcol]
                        .transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12)))
    merged["drift_z"] = merged["drift_z_mmd"]

    drift = pd.DataFrame(drift_rows)
    eval_cols = ["ref_session", "target_session", "feature", "classifier", "da",
                 "strategy", "accuracy", "subject", "dataset", "n_sessions"]
    eval_df = merged[eval_cols].copy()
    return {"drift": drift, "eval": eval_df, "merged": merged,
            "n_subjects": n_subjects, "n_sessions": n_sessions}


def _row(subj, feat, clf, da, strategy, ref, tgt, acc, base, dists, n_sessions):
    return {
        "ref_session": ref, "target_session": tgt, "feature": feat,
        "classifier": clf, "da": da, "strategy": strategy, "accuracy": acc,
        "subject": subj, "dataset": DATASET, "n_sessions": n_sessions,
        **dists, "baseline_acc": base, "acc_centered": acc - base,
        "uid": f"{DATASET}_{subj}",
    }


def write(out_dir: str, data: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data["drift"].to_csv(os.path.join(out_dir, f"drift_trajectories_{DATASET}.csv"), index=False)
    data["eval"].to_csv(os.path.join(out_dir, f"sequential_eval_{DATASET}.csv"), index=False)
    data["merged"].to_csv(os.path.join(out_dir, f"merged_drift_accuracy_{DATASET}.csv"), index=False)
    manifest = {
        "dataset": DATASET, "synthetic": True,
        "build_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_subjects": data["n_subjects"], "n_sessions": data["n_sessions"],
        "rows": {"drift": len(data["drift"]), "eval": len(data["eval"]),
                 "merged": len(data["merged"])},
        "da4bci_version": "n/a (synthetic)",
    }
    with open(os.path.join(out_dir, f"build_manifest_{DATASET}.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "data"))
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = generate(args.subjects, args.sessions, args.seed)
    write(args.out_dir, data)
    print(f"Wrote {DATASET} into {args.out_dir} "
          f"(drift={len(data['drift'])}, eval={len(data['eval'])}, "
          f"merged={len(data['merged'])} rows).")


if __name__ == "__main__":
    main()
