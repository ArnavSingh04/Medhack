# MedHack Frontiers — Dataset Summary

## Overview

This competition involves predicting real-time patient deterioration from multimodal ICU/emergency encounter data. Each encounter covers **1 hour of monitoring** (720 readings at 5-second intervals) paired with a **10-second ECG recording** and static patient demographics.

---

## File Types

### .npz Files (NumPy Compressed Archives)

A `.npz` file is a zipped archive of multiple named NumPy arrays, created with `np.savez_compressed()`. Each file holds the raw ECG signal data for one split.

| File | Encounters | Shape of `ecg_data` |
|---|---|---|
| `train_ecg.npz` | 2,930 | (2930, 2500) |
| `test_ecg.npz` | 627 | (627, 2500) |
| `holdout_ecg.npz` | 629 | (629, 2500) |

**Arrays inside each .npz:**

| Key | Shape | dtype | Description |
|---|---|---|---|
| `encounter_ids` | (n,) | str | UUID strings matching `encounter_id` in the CSV files |
| `ecg_data` | (n, 2500) | float64 | Raw ECG waveform — 2500 samples = 10 sec × 250 Hz |
| `hr_bpm` | (n,) | float64 | Heart rate in BPM derived from the ECG |
| `sample_rate` | scalar | int64 | 250 Hz |
| `duration_sec` | scalar | int64 | 10 seconds |

Each patient has a single-lead ECG recording captured at **250 Hz for 10 seconds**, yielding 2500 amplitude samples (millivolts). This is the clinically standard duration for a rhythm strip.

**Load example:**
```python
import numpy as np
data = np.load("data/train_ecg.npz", allow_pickle=True)
ecg = data["ecg_data"]   # shape (2930, 2500)
ids = data["encounter_ids"]
```

---

## CSV Files

All CSV files share the same time-series vital signs schema. Each row is one reading for one encounter at a 5-second resolution over 1 hour (720 rows per encounter).

### Schema

| Column | dtype | Description |
|---|---|---|
| `timestamp` | str (datetime) | `YYYY-MM-DD HH:MM:SS`, one reading every 5 seconds |
| `encounter_id` | str (UUID) | Links to `.npz` arrays and `patients.csv` |
| `heart_rate` | float64 | Beats per minute (BPM) |
| `systolic_bp` | float64 | Systolic blood pressure (mmHg) |
| `diastolic_bp` | float64 | Diastolic blood pressure (mmHg) |
| `respiratory_rate` | float64 | Breaths per minute |
| `oxygen_saturation` | float64 | Peripheral SpO2 (%) |
| `label` | int64 | Patient state (0–3); **absent from holdout_data.csv** |

### Row and Encounter Counts

| File | Rows | Encounters | Rows/Encounter | Has Label |
|---|---|---|---|---|
| `train_data.csv` | 2,109,600 | 2,930 | 720 | Yes |
| `test_data.csv` | 451,440 | 627 | 720 | Yes |
| `holdout_data.csv` | 452,880 | 629 | 720 | No (predict these) |

Each encounter = 720 readings × 5 seconds = **3,600 seconds = 1 hour** of monitoring.

### Missing Values

All vital sign columns have approximately **~2% missing values** uniformly across all splits and all label classes. Missingness appears to be random sensor dropout rather than clinically informative.

### Descriptive Statistics (train_data.csv, all labels)

| Column | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|---|
| heart_rate | 85.9 | 47.6 | 0.0 | 58.0 | 84.0 | 114.0 | 220.0 |
| systolic_bp | 110.3 | 33.5 | 0.0 | 100.0 | 114.0 | 128.0 | 280.0 |
| diastolic_bp | 69.3 | 22.2 | 0.0 | 64.0 | 72.0 | 80.0 | 280.0 |
| respiratory_rate | 16.5 | 5.8 | 0.0 | 14.0 | 17.0 | 20.0 | 40.0 |
| oxygen_saturation | 89.7 | 22.3 | 0.0 | 93.0 | 96.0 | 98.0 | 100.0 |

---

## The Target: `label` (Patient State)

The label is a **per-timestep multi-class classification** (0–3). A single encounter can oscillate between states, reflecting real clinical dynamics (e.g., deterioration followed by resuscitation).

| Label | Name | Train Rows | % of Train | Mean HR | Mean SBP | Mean RR | Mean SpO2 |
|---|---|---|---|---|---|---|---|
| 0 | Stable / Normal | 1,548,998 | 73.4% | 83.7 | 111.9 | 16.1 | 92.5% |
| 1 | Mild Deterioration | 359,553 | 17.0% | 104.3 | 116.4 | 20.1 | 93.1% |
| 2 | Moderate Deterioration | 139,107 | 6.6% | 99.4 | 124.9 | 18.3 | 88.5% |
| 3 | Critical / Cardiac Arrest | 61,942 | 2.9% | ~3 | ~4 | ~0.6 | ~3% |

**Key observations:**
- **Label 3** has near-zero vital signs — consistent with **cardiac arrest / code blue** (no effective circulation or respiration). The median heart rate and BP are both 0.
- **1,781 of 2,930 encounters** have at least one timestep with label > 0, meaning deterioration is common.
- **1,149 encounters** remain at label 0 throughout — fully stable patients.
- The dataset is **heavily class-imbalanced**: label 0 is 25× more common than label 3.

---

## patients.csv — Static Demographics

Contains one row per encounter (4,186 total = train + test + holdout combined).

| Column | dtype | Notes |
|---|---|---|
| `encounter_id` | str | Join key to CSV/NPZ files |
| `patient_name` | str | Synthetic name |
| `age` | int64 | Patient age in years |
| `date_of_birth` | str | DOB |
| `gender` | str | M / F (1,981 M, 2,205 F) |
| `race` | str | white (82.8%), black, asian, other |
| `ethnicity` | str | nonhispanic (89.6%) / hispanic |
| `marital_status` | str | M / S / D / W |
| `encounter_class` | str | All "emergency" |
| `encounter_description` | str | 9 categories; mostly "Emergency room admission" |
| `reason_for_visit` | str | Free-text clinical indication |
| `previous_medical_history` | str | Pipe-separated SNOMED conditions |
| `current_medications` | str | Pipe-separated drug names |
| `previous_medications` | str | Historical medications |
| `known_allergies` | str | Allergy list |
| `bmi` | float64 | Body mass index |
| `pain_score` | float64 | 0–9 scale; many nulls |

---

## Predictive Value of Each Feature

### Vital Signs (time-series)

| Feature | Why It Predicts State |
|---|---|
| **heart_rate** | Tachycardia (>100 bpm) is a cardinal sign of sepsis, haemorrhage, and pain (label 1). Bradycardia (<60 bpm) signals cardiac conduction failure. HR = 0 confirms cardiac arrest (label 3). |
| **systolic_bp** | Hypotension (<90 mmHg) indicates cardiogenic or septic shock — a key transition to labels 2–3. Hypertension may reflect pain or hypertensive crisis. |
| **diastolic_bp** | Low diastolic pressure with widened pulse pressure suggests vasodilation (sepsis/anaphylaxis). Narrow pulse pressure indicates poor cardiac output. |
| **respiratory_rate** | RR >20 breaths/min is one of the four SIRS criteria and a reliable early warning sign of sepsis or respiratory failure (label 1→2). RR = 0 confirms apnoea (label 3). |
| **oxygen_saturation** | SpO2 <88% defines severe hypoxaemia and respiratory failure (label 2). SpO2 near 0 in label 3 reflects absence of perfusion. The sharp drop from ~93% (label 1) to ~88% (label 2) is clinically meaningful. |

### ECG Signal (npz)

| Feature | Why It Predicts State |
|---|---|
| **ecg_data waveform** | Rhythm analysis can detect arrhythmias (AF, VT, VF), ST-segment changes (myocardial infarction), and flatline (asystole = label 3). These are often the **earliest indicators** of deterioration before vital signs change. |
| **hr_bpm** (ECG-derived) | A second, independent measurement of heart rate from waveform morphology — can be compared against the CSV `heart_rate` to detect sensor artefact or true rate changes. |

### Patient Demographics (patients.csv)

| Feature | Why It Predicts State |
|---|---|
| **age** | Older patients have reduced physiological reserve and higher baseline risk of deterioration. |
| **bmi** | Obesity increases risk of respiratory failure and complicates resuscitation. Low BMI may signal malnutrition. |
| **pain_score** | Elevated pain drives tachycardia and raised BP (label 1 pattern). |
| **previous_medical_history** | Pre-existing conditions (heart failure, COPD, diabetes) determine how quickly a patient deteriorates and what states are reachable. |
| **current_medications** | Beta-blockers blunt tachycardia; vasopressors affect BP. Medications modulate what "normal" vital signs look like for a given patient. |
| **reason_for_visit** | Trauma, sepsis, and cardiac presentations have very different deterioration trajectories. |
| **encounter_description** | Obstetric emergencies vs. trauma vs. general emergency have distinct physiology profiles. |

---

## Data Relationships

```
patients.csv  ──── encounter_id ────>  train/test/holdout_data.csv  (1 hour vitals, 5s resolution)
                                                │
                   train/test/holdout_ecg.npz ──┘  (10s ECG waveform per encounter)
```

The prediction target is the `label` column in `holdout_data.csv` (which is absent and must be submitted via `sample_submission.csv`).

---

## Key Modelling Considerations

1. **Multimodal input**: combine time-series vitals (CSV) + ECG waveform (NPZ) + static patient features (patients.csv).
2. **Class imbalance**: labels 2 and 3 are rare; use weighted loss, oversampling, or focal loss.
3. **Temporal structure**: the label can change within an encounter — sequential models (LSTM, Transformer) or sliding-window approaches are well-suited.
4. **ECG–CSV linkage**: join on `encounter_id` to align the 10s ECG snapshot with the longitudinal vitals.
5. **Missing data**: ~2% random sensor dropout in all vital columns; impute or use models robust to NaN.
