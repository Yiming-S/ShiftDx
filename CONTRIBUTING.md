# Contributing to ShiftDx

ShiftDx is the interactive companion to *Drift Diagnostics, Adaptation, and
Recalibration in Multi-Session Motor-Imagery EEG* (Shen & Degras, 2026). The
dashboard reads pre-built CSVs; the heavy science (feature extraction, DA,
distances) lives in [DA4BCI-Python](https://github.com/Yiming-S/DA4BCI-Python)
and [CrossPython](https://github.com/Yiming-S/CrossPython).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional, only needed to rebuild datasets:
export CROSSPYTHON_ROOT=/path/to/CrossPython
streamlit run app.py
```

No MOABB? Generate a synthetic dataset so the whole UI runs in seconds:

```bash
python scripts/gen_synthetic_demo.py
```

## Running tests

Tests live under `TEST/` (never beside source) and are plain `pytest`:

```bash
pytest TEST/
```

They cover CSV-schema validation, that every view imports cleanly, and the shared
helpers in `utils.py`. Run them before opening a PR.

## Project layout

| Path | Role |
|---|---|
| `app.py` | entry point: navigation, global CSS, sidebar |
| `utils.py` | constants, palettes, shared UI + statistical helpers |
| `data_loader.py` | `DataStore` — cached CSV loading + schema validation |
| `views/_N_*.py` | one `render(store)` per page |
| `scripts/build_moabb.py` | dataset builder (MOABB / CrossPython) |
| `scripts/gen_synthetic_demo.py` | synthetic demo-data generator |
| `data/` | pre-built CSVs |
| `TEST/` | pytest suite |

## Code style

- **English only** in all code, comments, and config.
- Match the surrounding style (the views share helpers from `utils.py` — prefer
  reusing `empty_state`, `about_page`, `download_bar`, `style_figure`,
  `is_retrain`, `cluster_bootstrap_ci`, the `format_*`/`pvalue_badge` helpers).
- New user-facing page titles / labels should be discussed before merging.

## Paper integrity

This is a research artifact. The headline statistical models on the Claim pages
(`_fit_claim1`, `_fit_claim2`, the `R_g` OLS fits, the Claim-4 conditions)
**replicate the published tables** and must not be silently changed. Alternative
inference (random-slope MixedLM, GEE, permutation tests) belongs in clearly
labeled, opt-in *supplementary* panels — never as a replacement for the published
fit. When in doubt, add a panel; don't mutate the canonical model.
