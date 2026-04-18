# ShiftDx

Drift Diagnostics for Multi-Session MI-EEG BCI.

An interactive diagnostic dashboard for the paper *Drift Diagnostics, Adaptation,
and Recalibration in Multi-Session Motor-Imagery EEG* (Shen & Degras, 2026),
powered by [DA4BCI-Python](https://github.com/Yiming-S/DA4BCI-Python) and
[MOABB](https://moabb.neurotechx.com/).

## DA methods (10, via DA4BCI)

TCA · SA · MIDA · RD · CORAL · GFK · ART · PT · M3D · OT

## Distance metrics (5, via DA4BCI)

MMD · Energy · Wasserstein · Mahalanobis · Euclidean (mean pairwise)

## Feature families (3)

CSP (8 components) · log-variance · Tangent Space (AIRM)

## Quick Start

```bash
git clone <this repo>
cd ShiftDx
pip install -r requirements.txt
pip install -e /path/to/DA4BCI-Python   # enables DA Lab pages + build script
streamlit run app.py
```

## Build a dataset from MOABB

```bash
# Default: Zhou2016 (4 subjects × 3 sessions × 14 channels, binary L vs R hand)
python scripts/build_moabb.py --dataset zhou2016

# Fast smoke test: two subjects only, skip MIDA/M3D
python scripts/build_moabb.py --dataset zhou2016 --subjects 1 2 --no-slow

# Other supported datasets:
python scripts/build_moabb.py --dataset bnci2015_001
python scripts/build_moabb.py --dataset bnci2014_001
```

The script writes three CSVs into `data/`:

- `drift_trajectories_<ds>.csv`         — 5 distance metrics per (subject, session_k, feature)
- `sequential_eval_<ds>.csv`            — accuracy per (subject, session, feature, DA, strategy)
- `merged_drift_accuracy_<ds>.csv`      — pooled view with `drift_z`, `acc_centered`

The dashboard auto-detects any `drift_trajectories_*.csv` in `data/` and exposes it in the sidebar.

## Page Map

### Overview

- **Dataset Overview** — summary cards, MMD distribution, strategy coverage
- **Drift Trajectory** — per-subject MMD trajectory + feature-mean envelope

### Claim Explorer (paper §5)

- **Claim 1 — Drift Predicts Loss** — mixed-effects regression visualizer
- **Claim 2 — DA Decomposition** — level shift vs slope change (pick any of 10 DA methods)
- **Claim 3 — Retraining Gap** — `R_g(z)` fit + "DA closes X%" per dataset × feature
- **Claim 4 — Feature Robustness** — joint 4-condition table with ceiling anchor

### Deep Dive

- **Subject Explorer** — single-subject drift + strategy-wise accuracy evolution

### DA Lab (DA4BCI)

- **Live DA Sandbox** — synthetic shift scenario, pick a DA method, see before/after (5 distances)
- **Multi-Metric Drift Panel** — all 5 distances side by side; re-fit Claim 1 under each metric
- **DA Method Sweep** — run every DA method on a scenario, Pareto plot runtime vs alignment
- **Drift Detection Demo** — Page-Hinkley trigger on per-subject accuracy series

## Datasets supported by the builder

| Short name | MOABB class | Subjects | Sessions | Channels | Classes |
|---|---|---:|---:|---:|---|
| `zhou2016` | Zhou2016 | 4 | 3 | 14 | L/R hand (binarized) |
| `bnci2015_001` | BNCI2015_001 | 12 | 2–3 | 13 | right hand vs feet |
| `bnci2014_004` | BNCI2014_004 | 9 | 5 | 3 | L/R hand |
| `bnci2014_001` | BNCI2014_001 | 9 | 2 | 22 | L/R hand (binarized) |

## Authors

**Yiming Shen** and **David Degras**
Department of Mathematics, University of Massachusetts Boston

## License

MIT
