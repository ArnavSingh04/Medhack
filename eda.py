#!/usr/bin/env python3
"""MedHack Frontiers - Data Exploration"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/Admin/MedHack2026")
PYTHON = "/Users/Admin/miniconda3/bin/python"

VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    # ── Load data ────────────────────────────────────────────────────────
    section("1. FILE OVERVIEW")
    train = pd.read_csv(DATA_DIR / "train_data.csv")
    test = pd.read_csv(DATA_DIR / "test_data.csv")
    holdout = pd.read_csv(DATA_DIR / "holdout_data.csv")
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    for name, df in [("train_data", train), ("test_data", test), ("holdout_data", holdout),
                     ("patients", patients), ("sample_submission", submission)]:
        print(f"{name:20s}  shape={str(df.shape):16s}  columns={df.columns.tolist()}")

    # ── Label distribution ───────────────────────────────────────────────
    section("2. LABEL DISTRIBUTION")
    label_names = {0: "Normal", 1: "Warning", 2: "Crisis", 3: "Death"}
    for name, df in [("Train", train), ("Test", test)]:
        print(f"\n{name} ({len(df):,} rows, {df['encounter_id'].nunique()} encounters):")
        counts = df["label"].value_counts().sort_index()
        for label, count in counts.items():
            pct = count / len(df) * 100
            bar = "#" * int(pct)
            print(f"  {label} ({label_names.get(label, '?'):8s}): {count:>10,}  ({pct:5.1f}%)  {bar}")

    print(f"\nHoldout: {len(holdout):,} rows, {holdout['encounter_id'].nunique()} encounters, NO labels")

    # ── Vital sign statistics ────────────────────────────────────────────
    section("3. VITAL SIGN STATISTICS (Train)")
    print(train[VITALS].describe().round(2).to_string())

    section("4. VITAL SIGN STATISTICS BY LABEL (Train)")
    for label in sorted(train["label"].unique()):
        subset = train[train["label"] == label]
        print(f"\n--- Label {label} ({label_names.get(label, '?')}) - {len(subset):,} rows ---")
        print(subset[VITALS].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

    # ── Missing values ───────────────────────────────────────────────────
    section("5. MISSING VALUES")
    for name, df in [("Train", train), ("Test", test), ("Holdout", holdout)]:
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        has_missing = missing[missing > 0]
        if len(has_missing) > 0:
            print(f"\n{name}:")
            for col in has_missing.index:
                print(f"  {col:25s}: {missing[col]:>10,} ({missing_pct[col]:5.2f}%)")
        else:
            print(f"\n{name}: No missing values")

    # ── Encounter structure ──────────────────────────────────────────────
    section("6. ENCOUNTER STRUCTURE")
    for name, df in [("Train", train), ("Test", test), ("Holdout", holdout)]:
        rows_per_enc = df.groupby("encounter_id").size()
        print(f"{name}: {df['encounter_id'].nunique()} encounters, "
              f"rows/encounter: min={rows_per_enc.min()}, max={rows_per_enc.max()}, "
              f"mean={rows_per_enc.mean():.0f}")

    # Per-encounter label transitions
    section("7. LABEL TRANSITIONS PER ENCOUNTER (Train)")
    transitions = []
    for eid, grp in train.groupby("encounter_id"):
        labels = grp["label"].values
        changes = np.sum(labels[1:] != labels[:-1])
        final_label = labels[-1]
        has_crisis = int(2 in labels)
        has_death = int(3 in labels)
        transitions.append({"encounter_id": eid, "num_transitions": changes,
                            "final_label": final_label, "has_crisis": has_crisis,
                            "has_death": has_death})
    trans_df = pd.DataFrame(transitions)
    print("Label transitions per encounter:")
    print(trans_df["num_transitions"].describe().round(1).to_string())
    print(f"\nEncounters ending in death: {(trans_df['final_label'] == 3).sum()}")
    print(f"Encounters with any crisis: {trans_df['has_crisis'].sum()}")
    print(f"Encounters with any death:  {trans_df['has_death'].sum()}")

    # Typical label sequence pattern
    print("\nSample encounter label sequence (first encounter with death):")
    death_enc = trans_df[trans_df["has_death"] == 1].iloc[0]["encounter_id"]
    seq = train[train["encounter_id"] == death_enc]["label"].values
    # Show compressed sequence
    compressed = []
    prev = None
    for l in seq:
        if l != prev:
            compressed.append(str(int(l)))
            prev = l
    print(f"  {' -> '.join(compressed)}")

    # ── Patient metadata ─────────────────────────────────────────────────
    section("8. PATIENT METADATA")
    print(f"Total patients: {len(patients)}")
    print(f"\nAge: mean={patients['age'].mean():.1f}, min={patients['age'].min()}, max={patients['age'].max()}")
    print(f"\nGender:\n{patients['gender'].value_counts().to_string()}")
    print(f"\nEncounter class:\n{patients['encounter_class'].value_counts().to_string()}")
    print(f"\nBMI (non-null): {patients['bmi'].notna().sum()} values, "
          f"mean={patients['bmi'].mean():.1f}, std={patients['bmi'].std():.1f}")
    print(f"\nPain score (non-null): {patients['pain_score'].notna().sum()} values")
    print(f"  Distribution:\n{patients['pain_score'].value_counts().sort_index().to_string()}")

    # Medical history richness
    patients["num_conditions"] = patients["previous_medical_history"].fillna("").apply(
        lambda x: len([c for c in x.split(";") if c.strip()]) if x else 0)
    patients["has_medications"] = patients["current_medications"].notna().astype(int)
    print(f"\nMedical history conditions per patient: mean={patients['num_conditions'].mean():.1f}, "
          f"max={patients['num_conditions'].max()}")

    # ── ECG data ─────────────────────────────────────────────────────────
    section("9. ECG DATA")
    for name, path in [("Train", "train_ecg.npz"), ("Test", "test_ecg.npz"), ("Holdout", "holdout_ecg.npz")]:
        ecg = np.load(DATA_DIR / path, allow_pickle=True)
        print(f"\n{name} ECG:")
        for key in ecg.keys():
            arr = ecg[key]
            if arr.ndim == 0:
                print(f"  {key}: scalar = {arr.item()}")
            else:
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        # ECG signal stats
        ecg_data = ecg["ecg_data"]
        print(f"  Signal: mean={ecg_data.mean():.4f}, std={ecg_data.std():.4f}, "
              f"min={ecg_data.min():.4f}, max={ecg_data.max():.4f}")
        hr = ecg["hr_bpm"]
        print(f"  HR: mean={hr.mean():.1f}, std={hr.std():.1f}, min={hr.min():.1f}, max={hr.max():.1f}")

    # ── Correlations ─────────────────────────────────────────────────────
    section("10. VITAL-LABEL CORRELATIONS (Train)")
    corr = train[VITALS + ["label"]].corr()["label"].drop("label").sort_values()
    for col, val in corr.items():
        bar = "+" * int(abs(val) * 50) if val > 0 else "-" * int(abs(val) * 50)
        print(f"  {col:25s}: {val:+.4f}  {bar}")

    # ── Quick sanity: are labels temporally ordered? ─────────────────────
    section("11. LABEL TEMPORAL PATTERNS")
    # Check if labels always go 0->1->2->3 (monotonic) or can go backwards
    backward_count = 0
    for eid, grp in train.groupby("encounter_id"):
        labels = grp["label"].values
        diffs = np.diff(labels)
        if np.any(diffs < 0):
            backward_count += 1
    print(f"Encounters with backward label transitions (e.g., 2->1): {backward_count} / {train['encounter_id'].nunique()}")
    if backward_count == 0:
        print("Labels are MONOTONICALLY non-decreasing within each encounter!")
    else:
        print("Labels can go backwards - patients can recover from warning/crisis states.")

    print("\n" + "=" * 70)
    print("  EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
