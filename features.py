#!/usr/bin/env python3
"""MedHack Frontiers - Feature Engineering Pipeline"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_DIR = Path("/Users/Admin/MedHack2026")

VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]
WINDOWS = [6, 12, 24, 60]  # 30s, 1m, 2m, 5m


def load_patients():
    """Load and preprocess patient metadata."""
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    patients["gender_enc"] = (patients["gender"] == "M").astype(int)
    patients["num_conditions"] = patients["previous_medical_history"].fillna("").apply(
        lambda x: len([c for c in x.split(";") if c.strip()]) if x else 0)
    patients["has_medications"] = patients["current_medications"].notna().astype(int)
    patients["bmi"] = patients["bmi"].fillna(patients["bmi"].median())
    patients["pain_score"] = patients["pain_score"].fillna(0)
    patient_feats = patients[["encounter_id", "age", "gender_enc", "bmi", "pain_score",
                              "num_conditions", "has_medications"]].copy()
    return patient_feats


def load_ecg_features(npz_path):
    """Extract features from ECG signals."""
    ecg = np.load(npz_path, allow_pickle=True)
    encounter_ids = ecg["encounter_ids"]
    ecg_data = ecg["ecg_data"]
    hr_bpm = ecg["hr_bpm"]

    records = []
    for i, eid in enumerate(encounter_ids):
        signal = ecg_data[i]
        records.append({
            "encounter_id": eid,
            "ecg_hr_bpm": hr_bpm[i],
            "ecg_mean": signal.mean(),
            "ecg_std": signal.std(),
            "ecg_min": signal.min(),
            "ecg_max": signal.max(),
            "ecg_skew": stats.skew(signal),
            "ecg_kurtosis": stats.kurtosis(signal),
            "ecg_range": signal.max() - signal.min(),
            # Simple peak count as HRV proxy
            "ecg_peaks": np.sum((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])),
        })
    return pd.DataFrame(records)


def build_features(df, patient_feats, ecg_feats):
    """Build all features for a dataframe. Processes per-encounter for temporal features."""
    print(f"  Building features for {len(df):,} rows, {df['encounter_id'].nunique()} encounters...")

    # Forward-fill missing vitals within each encounter, then fill remaining with median
    df = df.copy()
    df[VITALS] = df.groupby("encounter_id")[VITALS].ffill()
    for col in VITALS:
        df[col] = df[col].fillna(df[col].median())

    # ── Derived vitals ───────────────────────────────────────────────────
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3
    df["shock_index"] = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"].fillna(0)
    df["hr_rr_ratio"] = df["heart_rate"] / df["respiratory_rate"].replace(0, np.nan)
    df["hr_rr_ratio"] = df["hr_rr_ratio"].fillna(0)

    # ── Position feature ─────────────────────────────────────────────────
    df["timestep"] = df.groupby("encounter_id").cumcount()
    df["timestep_pct"] = df["timestep"] / 719.0  # 0 to 1

    # ── Rolling / temporal features (per encounter) ──────────────────────
    print("  Computing rolling features...")
    all_rolling = []

    for eid, grp in df.groupby("encounter_id"):
        grp = grp.sort_values("timestep")
        row_feats = {}

        for vital in VITALS:
            vals = grp[vital]
            for w in WINDOWS:
                roll = vals.rolling(window=w, min_periods=1)
                row_feats[f"{vital}_roll_mean_{w}"] = roll.mean().values
                row_feats[f"{vital}_roll_std_{w}"] = roll.std().fillna(0).values
                row_feats[f"{vital}_roll_min_{w}"] = roll.min().values
                row_feats[f"{vital}_roll_max_{w}"] = roll.max().values

            # Lags and rate of change
            row_feats[f"{vital}_lag1"] = vals.shift(1).bfill().values
            row_feats[f"{vital}_lag6"] = vals.shift(6).bfill().values
            row_feats[f"{vital}_diff1"] = vals.diff().fillna(0).values
            row_feats[f"{vital}_diff6"] = (vals - vals.shift(6)).fillna(0).values

        roll_df = pd.DataFrame(row_feats, index=grp.index)
        all_rolling.append(roll_df)

    rolling_df = pd.concat(all_rolling)
    df = pd.concat([df, rolling_df], axis=1)

    # ── Merge patient metadata ───────────────────────────────────────────
    print("  Merging patient metadata...")
    df = df.merge(patient_feats, on="encounter_id", how="left")

    # Fill any missing patient features
    for col in patient_feats.columns:
        if col != "encounter_id" and df[col].isna().any():
            df[col] = df[col].fillna(0)

    # ── Merge ECG features ───────────────────────────────────────────────
    print("  Merging ECG features...")
    df = df.merge(ecg_feats, on="encounter_id", how="left")

    # HR discrepancy between vitals and ECG
    df["hr_ecg_diff"] = df["heart_rate"] - df["ecg_hr_bpm"]

    # Fill any missing ECG features
    for col in ecg_feats.columns:
        if col != "encounter_id" and df[col].isna().any():
            df[col] = df[col].fillna(0)

    # ── Drop non-feature columns ─────────────────────────────────────────
    drop_cols = ["timestamp", "encounter_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"  Final feature count: {df.shape[1]} columns")
    return df


def get_feature_columns(df):
    """Get list of feature columns (everything except label)."""
    return [c for c in df.columns if c != "label"]


def main():
    """Run feature engineering pipeline and save results."""
    print("Loading patient metadata...")
    patient_feats = load_patients()

    print("Loading ECG features...")
    train_ecg = load_ecg_features(DATA_DIR / "train_ecg.npz")
    test_ecg = load_ecg_features(DATA_DIR / "test_ecg.npz")
    holdout_ecg = load_ecg_features(DATA_DIR / "holdout_ecg.npz")

    print("\n--- TRAIN ---")
    train = pd.read_csv(DATA_DIR / "train_data.csv")
    train_feat = build_features(train, patient_feats, train_ecg)
    train_feat.to_parquet(DATA_DIR / "train_features.parquet", index=False)
    print(f"  Saved train_features.parquet: {train_feat.shape}")

    print("\n--- TEST ---")
    test = pd.read_csv(DATA_DIR / "test_data.csv")
    test_feat = build_features(test, patient_feats, test_ecg)
    test_feat.to_parquet(DATA_DIR / "test_features.parquet", index=False)
    print(f"  Saved test_features.parquet: {test_feat.shape}")

    print("\n--- HOLDOUT ---")
    holdout = pd.read_csv(DATA_DIR / "holdout_data.csv")
    holdout_feat = build_features(holdout, patient_feats, holdout_ecg)
    holdout_feat.to_parquet(DATA_DIR / "holdout_features.parquet", index=False)
    print(f"  Saved holdout_features.parquet: {holdout_feat.shape}")

    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()
