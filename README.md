# CES stress forecasting

Code to reproduce the results of the paper (RQ1-RQ4): next-week stress forecasting from
passive phone sensing and sparse self-reports on the College Experience Study (CES), with a
personalized LSTM regressor and a GBM peak classifier that corrects the forecast on the
weeks flagged as likely peaks.

## Data

The datasets are not included and must be requested from their owners.

- **CES** (RQ1-RQ3). Files used: `EMA/general_ema.csv` and `Sensing/sensing.csv`.
- **SSAQS** (RQ4). One sub-folder per user with the raw exports (`daily_questions.csv`,
  `hrv.csv`, `sleep.csv`, `oxygen.csv`, `steps.csv`, `activity_level.csv`, and
  `stress.csv` for the Fitbit Stress Score used in the evaluation).

Expected layout (paths are passed on the command line, so any location works):

```
<CES_DIR>/EMA/general_ema.csv
<CES_DIR>/Sensing/sensing.csv
<SSAQS_DIR>/<user_id>/daily_questions.csv, hrv.csv, sleep.csv, ...
```

`comparisons/ihope/ihope_rows.csv` (rows evaluated by I-HOPE) and
`comparisons/gala/phq4_users.txt` (the 120 users of the PHQ-4 comparison) contain the
pseudonymous CES user ids, so the same rows and users can be selected.

## Installation

Python 3.12.

```
python -m venv venv && source venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running everything

```
CES_DIR=/path/to/CES SSAQS_DIR=/path/to/SSAQS bash run_all.sh
```

A single step can be run with `bash run_all.sh <step>`. The steps, in order, are:
`data`, `ces`, `ces_eval`, `ihope`, `phq4` and `ssaqs`. All outputs go to `results/`.

| Step | What it does | Hardware |
|---|---|---|
| `data` | builds `ces_stress_forecast.npz` and `ces_phq4_forecast.npz` | CPU |
| `ces` | system, walk-forward (WF) and leave-users-out (LUO), seeds 42-44 | GPU |
| `ces_eval` | Tables 1-2, tests, figures, seed stability | CPU |
| `ihope` | I-HOPE reimplementation and comparison | CPU, several hours |
| `phq4` | PHQ-4 severity, comparison with Gala et al. | GPU, about 30 min |
| `ssaqs` | transfer to SSAQS | CPU |

## Correspondence with the paper

| Paper | Script | Output |
|---|---|---|
| Feature engineering (LSTM sequences, GBM features) | `features/build_forecast_dataset.py` | `ces_*_forecast.npz` |
| RQ1, Table 1 | `evaluation/rq1_table.py` | `results/tables/rq1_table1.txt` |
| RQ1, paired tests and signed error | `evaluation/rq1_tests.py` | `results/tables/rq1_tests.txt` |
| RQ2, Table 2 (AUC, tolerance F1, permutation tests) | `evaluation/rq2_table.py` | `results/tables/rq2_table2.txt` |
| RQ1 maturation figure, RQ2 peak-user figure | `evaluation/figures.py` | `results/figures/fig_rq1_maturation.*`, `fig_rq2_peak_users.*` |
| Stability across seeds 42-44 | `evaluation/seed_stability.py` | `results/tables/seed_stability.txt` |
| RQ3, comparison with I-HOPE | `comparisons/ihope/ihope_regressor.py`, `evaluation/rq3_ihope.py` | `results/tables/rq3_ihope.txt` |
| RQ3, PHQ-4 severity against Gala et al. | `pipeline/run_leave_users_out.py --target phq4`, `comparisons/gala/severity_stacking.py`, `comparisons/gala/evaluate_gala.py` | `results/tables/rq3_gala.txt` |
| RQ4, SSAQS table | `ssaqs/build_dataset.py`, `ssaqs/run_ensemble.py`, `ssaqs/evaluate.py`, `ssaqs/rq4_table.py` | `results/tables/rq4_table.txt` |
| RQ4, SSAQS figure | `ssaqs/rq4_figure.py` | `results/figures/` |

The tables of the paper report seed 42. Seeds 43 and 44 are used only for the stability
analysis and, in RQ4, by the evaluation script.

## Repository structure

- `features/`: builds the forecasting dataset from the raw CES files. The LSTM input is 14
  days × 15 sensing features; the GBM features are 4-epoch means with a robust
  per-user z-score, trends, recent values and the academic period.
- `pipeline/`: the system.
  - `lstm_model.py`: global LSTM and per-user online fine-tuning.
  - `peak_classifier.py`: GBM peak classifier.
  - `correction.py`: adaptive flag and peak-only correction.
  - `policy.py`: self-report policy (two onboarding reports, then one every 28 days).
  - `run_walk_forward.py`, `run_leave_users_out.py`: the two evaluation schemes.
- `evaluation/`: tables, tests and figures of RQ1-RQ2 and the I-HOPE comparison.
- `comparisons/`: the I-HOPE reimplementation and the PHQ-4 comparison with Gala et al.
- `ssaqs/`: the RQ4 pipeline on SSAQS.

The scripts import their sibling modules, so they are run as `python <folder>/<script>.py`
from any directory. Each script documents its arguments in its header.

## Determinism

- **`PYTHONHASHSEED=0`** must be set (`run_all.sh` does it). The walk-forward iterates a set
  of user ids, and its order depends on the hash seed.
- **GPU.** The CES runs (`ces`, `phq4`) were made on an NVIDIA TITAN X (Pascal) with the
  CUDA 12.1 build of PyTorch. On other GPUs, or on the CPU (`--device cpu`), the numbers
  can differ in the last digits.
- **CPU-only steps.** The SSAQS ensemble and I-HOPE run on the CPU. In SSAQS the LUO scheme
  depends on the random state left by the WF, so both run in the same process.
- **I-HOPE restarts.** The I-HOPE restarts do not change the results, because each fold
  sets its own seed.
