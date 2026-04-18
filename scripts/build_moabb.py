"""Build a ShiftDx dataset from MOABB, reusing CrossPython's feature/classifier code.

Default: Zhou2016 (4 subjects × 3 sessions × 14 channels × binary L vs R hand).

Feature extraction, classifier dispatch, and distance computation are delegated
to CrossPython to keep the scientific pipeline identical to the upstream paper.
ShiftDx only glues the per-(subject, ref=0, target=k) evaluation loop on top.

For each (subject, ref=0, target_k, feature, classifier) cell we record:
  * 5 distance metrics between ref and target feature distributions
  * No-DA accuracy (train ref → predict target)
  * DA accuracy for each of 10 DA4BCI methods
  * Retrain accuracy (5-fold CV within the target session)

Usage:
    python scripts/build_moabb.py --dataset zhou2016
    python scripts/build_moabb.py --dataset zhou2016 --subjects 1 2 --no-slow
    python scripts/build_moabb.py --dataset zhou2016 --classifiers lda svm_linear
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

CROSSPYTHON_ROOT = os.environ.get(
    "CROSSPYTHON_ROOT",
    "/Users/yiming/Documents/GitHub/CrossPython",
)
if CROSSPYTHON_ROOT not in sys.path:
    sys.path.insert(0, CROSSPYTHON_ROOT)

# ---------------------------------------------------------------------------
# CrossPython — reused unchanged
# ---------------------------------------------------------------------------

try:
    from CrossPython.pipelines.pipeline_utils import (
        extract_features_train, extract_features_test, get_classifier,
    )
    from CrossPython.core.workers import make_params
    # CrossPython's session loader handles the paper datasets (ma2020 via custom
    # CNT reader; bnci004/stieger2021 via MOABB LeftRightImagery(fmin=8, fmax=35))
    from CrossPython.core.workers import _load_subject_sessions as _cp_load
except ImportError as exc:
    sys.exit(f"ERROR: CrossPython not importable (set $CROSSPYTHON_ROOT).\n{exc}")

# ---------------------------------------------------------------------------
# DA4BCI — DA methods + distance metrics
# ---------------------------------------------------------------------------

try:
    from da4bci import (
        domain_adaptation, compute_mmd, compute_energy, compute_wasserstein,
        compute_mahalanobis, compute_distance_matrix, sigma_med,
    )
except ImportError as exc:
    sys.exit(f"ERROR: DA4BCI not installed.\n{exc}")

DA_METHODS = ["sa", "pt", "coral", "tca", "gfk", "rd", "art", "ot", "mida", "m3d"]
DEFAULT_CLASSIFIERS = ["lda", "svm_linear", "svm_radial"]

# Per-method default hyperparameters for DA4BCI's dispatcher.
# Keys MUST match the dispatcher control-dict lookup (not the underlying
# function kwargs — e.g. GFK wants `dim_subspace`, OT wants `eps`).
def _da_params(method: str, d: int) -> dict:
    k = min(8, max(2, d - 1))
    if method == "sa":    return {"k": k}
    if method == "pt":    return {}
    if method == "coral": return {"lambda": 1e-5}
    if method == "gfk":   return {"dim_subspace": k}
    if method == "tca":   return {"k": k}
    if method == "ot":    return {"eps": 0.1}
    if method == "mida":  return {"k": k}
    if method == "m3d":   return {"max_dim": max(5, k)}
    return {}


# ---------------------------------------------------------------------------
# MOABB dataset registry
# ---------------------------------------------------------------------------

MOABB_REGISTRY: dict[str, dict] = {
    # Handled directly in this script via MOABB paradigm
    "zhou2016":     {"cls": "Zhou2016",     "keep_classes": ["left_hand", "right_hand"]},
    "bnci2015_001": {"cls": "BNCI2015_001", "keep_classes": None},
    "bnci2014_001": {"cls": "BNCI2014_001", "keep_classes": ["left_hand", "right_hand"]},
    # Delegated to CrossPython._load_subject_sessions
    # (bnci004 / stieger2021 via MOABB LeftRightImagery(fmin=8, fmax=35);
    #  ma2020 via custom CNT reader under MNE-ma2020-data/)
    "bnci004":      {"via_crosspython": True},
    "stieger2021":  {"via_crosspython": True},
    "ma2020":       {"via_crosspython": True},
}


def _load_subject(dataset_key: str, subject: int,
                   data_dir: str | None = None) -> list[dict]:
    cfg = MOABB_REGISTRY[dataset_key]

    # ── CrossPython path (paper datasets) ──────────────────────────────
    if cfg.get("via_crosspython"):
        root = data_dir or os.environ.get("MNE_DATA", "")
        sessions = _cp_load(subject, dataset_key, data_dir=root)
        if sessions is None:
            raise RuntimeError(f"CrossPython loader returned None for {dataset_key}/S{subject}")
        # Normalize key names to this script's convention
        out = []
        for s in sessions:
            out.append({
                "session_id": s.get("id", s.get("label", "?")),
                "epochs":     s["x"],
                "labels":     np.asarray(s["y"]),
            })
        return out

    # ── Direct MOABB path (Zhou2016, BNCI2015_001, BNCI2014_001) ───────
    import moabb.datasets as mds
    from moabb.paradigms import LeftRightImagery, MotorImagery

    ds_cls = getattr(mds, cfg["cls"])()

    if cfg["keep_classes"] == ["left_hand", "right_hand"]:
        paradigm = LeftRightImagery()
    elif "bnci2015_001" in dataset_key.lower():
        paradigm = MotorImagery(n_classes=2)
    else:
        paradigm = MotorImagery(n_classes=2)

    X, y, meta = paradigm.get_data(dataset=ds_cls, subjects=[subject])
    if isinstance(y[0], str):
        classes = sorted(np.unique(y))
        y = np.array([classes.index(v) for v in y])

    sessions = []
    for sess_label, grp in meta.groupby("session"):
        idx = grp.index.to_numpy()
        sessions.append({
            "session_id": sess_label,
            "epochs": X[idx],
            "labels": np.asarray(y[idx]),
        })
    sessions.sort(key=lambda s: str(s["session_id"]))
    return sessions


def _subjects_for(dataset_key: str) -> list[int]:
    cfg = MOABB_REGISTRY[dataset_key]
    if cfg.get("via_crosspython"):
        # Map ShiftDx short names to MOABB classes where applicable
        import moabb.datasets as mds
        cls_map = {
            "bnci004":     "BNCI2014_004",
            "stieger2021": "Stieger2021",
        }
        if dataset_key in cls_map:
            ds = getattr(mds, cls_map[dataset_key])()
            return list(ds.subject_list)
        # Ma2020: scan MNE-ma2020-data/sub-NNN directories
        root = os.environ.get("MNE_DATA", "")
        ma_dir = os.path.join(root, "MNE-ma2020-data")
        if os.path.isdir(ma_dir):
            subs = []
            for d in sorted(os.listdir(ma_dir)):
                if d.startswith("sub-"):
                    try:
                        subs.append(int(d[4:]))
                    except ValueError:
                        pass
            return subs
        return []
    import moabb.datasets as mds
    ds_cls = getattr(mds, cfg["cls"])()
    return list(ds_cls.subject_list)


# ---------------------------------------------------------------------------
# Distance helpers — 5 metrics at once
# ---------------------------------------------------------------------------

def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return np.nan


def _five_distances(a: np.ndarray, b: np.ndarray) -> dict:
    try:
        sigma = sigma_med(a, b)
    except Exception:
        sigma = 1.0
    D = _safe_call(compute_distance_matrix, a, b)
    euclid = float(np.mean(D)) if isinstance(D, np.ndarray) else np.nan
    return {
        "dist_mmd":         _safe_call(compute_mmd, a, b, sigma),
        "dist_energy":      _safe_call(compute_energy, a, b),
        "dist_wasserstein": _safe_call(compute_wasserstein, a, b),
        "dist_mahalanobis": _safe_call(compute_mahalanobis, a, b),
        "dist_euclidean":   euclid,
    }


# ---------------------------------------------------------------------------
# Classifier wrappers (reuse CrossPython.get_classifier)
# ---------------------------------------------------------------------------

def _build_clf(clf_name: str, clf_params: dict):
    from sklearn.base import clone
    return clone(get_classifier(clf_name, clf_params))


def _fit_predict(clf_name: str, clf_params: dict,
                  X_tr: np.ndarray, y_tr: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray) -> float:
    try:
        clf = _build_clf(clf_name, clf_params)
        clf.fit(X_tr, y_tr)
        return float(clf.score(X_te, y_te))
    except Exception:
        return np.nan


def _retrain_cv(clf_name: str, clf_params: dict,
                 X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X, y):
            scores.append(_fit_predict(clf_name, clf_params, X[tr], y[tr], X[te], y[te]))
        return float(np.nanmean(scores))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# DA adapter
# ---------------------------------------------------------------------------

def _apply_da(source: np.ndarray, target: np.ndarray, method: str,
               source_labels: np.ndarray | None = None):
    ctrl = _da_params(method, source.shape[1])
    if method == "m3d":
        if source_labels is None:
            raise ValueError("M3D requires source_labels")
        ctrl["source_labels"] = np.asarray(source_labels)
    t0 = time.perf_counter()
    res = domain_adaptation(source, target, method=method, control=ctrl)
    rt = (time.perf_counter() - t0) * 1000
    return res.get("weighted_source_data", source), res.get("target_data", target), rt


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def build(dataset_key: str, subjects: list[int] | None = None,
          out_dir: str = DATA_DIR, slow_methods: bool = True,
          classifiers: list[str] | None = None) -> None:
    if dataset_key not in MOABB_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {list(MOABB_REGISTRY)}")

    os.makedirs(out_dir, exist_ok=True)
    all_subjects = _subjects_for(dataset_key)
    subjects = subjects or all_subjects
    classifiers = classifiers or DEFAULT_CLASSIFIERS
    print(f"[{dataset_key}] subjects={subjects}/{all_subjects}  classifiers={classifiers}")

    methods = list(DA_METHODS) if slow_methods else [m for m in DA_METHODS if m not in ("mida", "m3d")]

    params = make_params(dataset_key)
    clf_params_map = {c: params["classifier"].get(c, {}) for c in classifiers}
    feat_params_map = params["feature"]

    drift_rows, eval_rows, merged_rows = [], [], []

    t_global = time.perf_counter()
    for subj in subjects:
        print(f"\n== subject {subj} ==")
        t0 = time.perf_counter()
        try:
            sessions = _load_subject(
                dataset_key, subj,
                data_dir=os.environ.get("MNE_DATA", ""),
            )
        except Exception as exc:
            print(f"  skip subject {subj}: {exc}")
            continue
        if len(sessions) < 2:
            print(f"  skip subject {subj}: only {len(sessions)} session(s)")
            continue

        n_sessions = len(sessions)
        ref_ep, ref_lb = sessions[0]["epochs"], sessions[0]["labels"]

        # ── Fit feature extractors ONCE per (subject, feature) ─────────────
        ref_feats = {}
        ref_objs = {}
        for feat_name in ("CSP", "logvar", "TS"):
            try:
                feats, obj = extract_features_train(
                    ref_ep, ref_lb, feat_name, feat_params_map.get(feat_name, {}),
                )
                ref_feats[feat_name] = feats
                ref_objs[feat_name] = obj
            except Exception as exc:
                print(f"  [WARN] {feat_name} extractor failed on S{subj}: {exc}")

        # ── Baseline = 5-fold CV on session 0 per (feature, classifier) ────
        baseline_by_fc: dict[tuple, float] = {}
        for feat_name, ref_X in ref_feats.items():
            for clf_name in classifiers:
                acc0 = _retrain_cv(clf_name, clf_params_map[clf_name], ref_X, ref_lb)
                baseline_by_fc[(feat_name, clf_name)] = acc0
                eval_rows.append({
                    "ref_session": 0, "target_session": 0, "feature": feat_name,
                    "classifier": clf_name, "da": "none", "strategy": "retrain",
                    "accuracy": acc0, "subject": subj, "dataset": dataset_key,
                    "n_sessions": n_sessions,
                })

        # ── Per-target-session loop ────────────────────────────────────────
        for k in range(1, n_sessions):
            tgt_ep, tgt_lb = sessions[k]["epochs"], sessions[k]["labels"]

            for feat_name, ref_X in ref_feats.items():
                try:
                    tgt_X = extract_features_test(tgt_ep, ref_objs[feat_name], feat_name)
                except Exception as exc:
                    print(f"  [WARN] extract_test {feat_name} failed k={k}: {exc}")
                    continue

                # -- Distances (classifier-agnostic) ------------------------
                dists = _five_distances(ref_X, tgt_X)
                drift_rows.append({
                    "session_k": k, "feature": feat_name,
                    **dists,
                    "subject": subj, "dataset": dataset_key, "n_sessions": n_sessions,
                })

                # -- Pre-compute adapted features once per DA method --------
                adapted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for m in methods:
                    try:
                        src_a, tgt_a, _rt = _apply_da(ref_X, tgt_X, m, source_labels=ref_lb)
                        adapted[m] = (src_a, tgt_a)
                    except Exception as exc:
                        print(f"  DA {m} failed S{subj}/k={k}/{feat_name}: {exc}")
                        adapted[m] = (None, None)

                # -- Loop over classifiers ----------------------------------
                for clf_name in classifiers:
                    clf_params = clf_params_map[clf_name]
                    base_acc = baseline_by_fc[(feat_name, clf_name)]

                    # No DA
                    acc_noda = _fit_predict(clf_name, clf_params, ref_X, ref_lb, tgt_X, tgt_lb)
                    eval_rows.append({
                        "ref_session": 0, "target_session": k, "feature": feat_name,
                        "classifier": clf_name, "da": "none", "strategy": "train_once",
                        "accuracy": acc_noda, "subject": subj, "dataset": dataset_key,
                        "n_sessions": n_sessions,
                    })
                    merged_rows.append({
                        "ref_session": 0, "target_session": k, "feature": feat_name,
                        "classifier": clf_name, "da": "none", "strategy": "train_once",
                        "accuracy": acc_noda, "subject": subj, "dataset": dataset_key,
                        "n_sessions": n_sessions, **dists,
                        "baseline_acc": base_acc,
                        "acc_centered": acc_noda - base_acc if not np.isnan(base_acc) else np.nan,
                        "uid": f"{dataset_key}_{subj}",
                    })

                    # DA (10 methods)
                    for m in methods:
                        src_a, tgt_a = adapted[m]
                        if src_a is None:
                            acc_da = np.nan
                        else:
                            acc_da = _fit_predict(clf_name, clf_params, src_a, ref_lb, tgt_a, tgt_lb)
                        eval_rows.append({
                            "ref_session": 0, "target_session": k, "feature": feat_name,
                            "classifier": clf_name, "da": m, "strategy": "train_once_da",
                            "accuracy": acc_da, "subject": subj, "dataset": dataset_key,
                            "n_sessions": n_sessions,
                        })
                        merged_rows.append({
                            "ref_session": 0, "target_session": k, "feature": feat_name,
                            "classifier": clf_name, "da": m, "strategy": "train_once_da",
                            "accuracy": acc_da, "subject": subj, "dataset": dataset_key,
                            "n_sessions": n_sessions, **dists,
                            "baseline_acc": base_acc,
                            "acc_centered": acc_da - base_acc if not np.isnan(base_acc) else np.nan,
                            "uid": f"{dataset_key}_{subj}",
                        })

                    # Retrain — refit extractor on target, then CV with clf.
                    # To stay apples-to-apples with No-DA (same feature geometry),
                    # we CV on the tgt features computed with ref's extractor.
                    acc_rt = _retrain_cv(clf_name, clf_params, tgt_X, tgt_lb)
                    eval_rows.append({
                        "ref_session": k, "target_session": k, "feature": feat_name,
                        "classifier": clf_name, "da": "none", "strategy": "retrain",
                        "accuracy": acc_rt, "subject": subj, "dataset": dataset_key,
                        "n_sessions": n_sessions,
                    })
                    merged_rows.append({
                        "ref_session": k, "target_session": k, "feature": feat_name,
                        "classifier": clf_name, "da": "none", "strategy": "retrain",
                        "accuracy": acc_rt, "subject": subj, "dataset": dataset_key,
                        "n_sessions": n_sessions, **dists,
                        "baseline_acc": base_acc,
                        "acc_centered": acc_rt - base_acc if not np.isnan(base_acc) else np.nan,
                        "uid": f"{dataset_key}_{subj}",
                    })

        print(f"  subject {subj} done in {time.perf_counter() - t0:.1f}s")

    # ── Attach drift_z to merged rows (within dataset × feature × classifier) ──
    merged_df = pd.DataFrame(merged_rows)
    drift_df = pd.DataFrame(drift_rows)
    eval_df = pd.DataFrame(eval_rows)

    if not merged_df.empty:
        # Compute drift_z for each of the 5 distance metrics, within
        # (dataset × feature × classifier) blocks.
        METRIC_COLS = ["dist_mmd", "dist_energy", "dist_wasserstein",
                        "dist_mahalanobis", "dist_euclidean"]
        for mcol in METRIC_COLS:
            zcol = mcol.replace("dist_", "drift_z_")   # e.g. drift_z_mmd
            merged_df[zcol] = np.nan
            for _, idx in merged_df.groupby(["dataset", "feature", "classifier"]).groups.items():
                vals = merged_df.loc[idx, mcol]
                mu, sd = vals.mean(), vals.std(ddof=1)
                merged_df.loc[idx, zcol] = (vals - mu) / (sd + 1e-12)
        # Backwards-compatible alias (paper default)
        merged_df["drift_z"] = merged_df["drift_z_mmd"]

    # ── Persist ────────────────────────────────────────────────────────────
    drift_path  = os.path.join(out_dir, f"drift_trajectories_{dataset_key}.csv")
    eval_path   = os.path.join(out_dir, f"sequential_eval_{dataset_key}.csv")
    merged_path = os.path.join(out_dir, f"merged_drift_accuracy_{dataset_key}.csv")
    drift_df.to_csv(drift_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    merged_df.to_csv(merged_path, index=False)

    dt_all = time.perf_counter() - t_global
    print(f"\nDone in {dt_all / 60:.1f} min. Wrote:")
    print(f"  {drift_path}   ({len(drift_df):,} rows)")
    print(f"  {eval_path}    ({len(eval_df):,} rows)")
    print(f"  {merged_path}  ({len(merged_df):,} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default="zhou2016", choices=list(MOABB_REGISTRY))
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--classifiers", nargs="*", default=DEFAULT_CLASSIFIERS,
                    choices=["lda", "svm_linear", "svm_radial", "mdm", "lr",
                             "el", "lgbm", "elm", "catboost"])
    ap.add_argument("--mne-data-dir", default=None)
    ap.add_argument("--no-slow", action="store_true",
                    help="Skip MIDA and M3D (much faster).")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mne_data_dir:
        import mne
        mne.set_config("MNE_DATA", os.path.abspath(args.mne_data_dir))
        print(f"MNE_DATA set to: {os.path.abspath(args.mne_data_dir)}")
    build(args.dataset, subjects=args.subjects, out_dir=args.out_dir,
          slow_methods=not args.no_slow, classifiers=args.classifiers)
