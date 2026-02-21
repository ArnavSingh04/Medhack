# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedHack Frontiers 2026 hackathon — patient deterioration prediction. Multiclass classification of patient status at each timestep within hospital encounters:
- **Labels**: 0=Normal, 1=Warning, 2=Crisis, 3=Death
- **Metric**: Macro F1 score
- **Key insight**: Labels are monotonically non-decreasing within each encounter (patients don't recover)

## Commands

```bash
# Python interpreter (miniconda)
/Users/Admin/miniconda3/bin/python

# Pipeline steps (run in order):
/Users/Admin/miniconda3/bin/python eda.py              # EDA - prints stats, no output files
/Users/Admin/miniconda3/bin/python features.py          # Feature engineering → {train,test,holdout}_features.parquet
/Users/Admin/miniconda3/bin/python train_model.py       # Train LightGBM → lgb_model.pkl, feature_cols.pkl
```

## Pipeline Architecture

Three-stage pipeline, each script is standalone:

1. **eda.py** — Exploratory analysis. Prints statistics about data distributions, missing values, encounter structure, label transitions, patient metadata, ECG signals. Read-only, no files produced.

2. **features.py** — Feature engineering. Reads raw CSVs + ECG `.npz` files → outputs `.parquet` files. Features include:
   - Rolling window stats (windows: 6, 12, 24, 60 timesteps) over 5 vitals
   - Derived vitals: pulse pressure, MAP, shock index, HR/RR ratio
   - Lags and diffs (1-step, 6-step)
   - Timestep position (absolute + percentage through encounter)
   - Patient metadata: age, gender, BMI, pain score, condition count, medication flag
   - ECG signal features: stats, skew, kurtosis, peak count, HR from ECG
   - HR discrepancy between vitals and ECG

3. **train_model.py** — LightGBM training. Combines train+test (both labeled) for more data. Uses inverse-frequency class weights, 5-fold stratified CV, early stopping. Final model trained on all data with 800 rounds.

## Data Layout

All data lives in the project root (`/Users/Admin/MedHack2026`):
- **Raw**: `train_data.csv`, `test_data.csv`, `holdout_data.csv` (vitals time series with `encounter_id`, `timestamp`, 5 vitals, `label`)
- **Patient metadata**: `patients.csv` (demographics, medical history, medications)
- **ECG signals**: `train_ecg.npz`, `test_ecg.npz`, `holdout_ecg.npz` (per-encounter ECG arrays + HR)
- **Engineered features**: `{train,test,holdout}_features.parquet` (output of features.py)
- **Submission template**: `sample_submission.csv`
- Each encounter has exactly 720 timesteps

## Key Constants

- `VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]`
- `WINDOWS = [6, 12, 24, 60]` for rolling features
- `DATA_DIR = Path("/Users/Admin/MedHack2026")` — hardcoded in all scripts
- Train+test are both labeled; holdout has no labels (submission target)

## Dependencies

Python packages: pandas, numpy, scipy, lightgbm, scikit-learn, pyarrow (for parquet)
