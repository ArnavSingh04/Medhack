#!/usr/bin/env python3
"""MedHack Frontiers - Generate Holdout Submission with Monotonicity Enforcement"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path("/Users/Admin/MedHack2026")


def main():
    # Load model and feature columns
    print("Loading model...")
    with open(DATA_DIR / "lgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(DATA_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # Load holdout features and raw data (for encounter_id)
    print("Loading holdout features...")
    holdout = pd.read_parquet(DATA_DIR / "holdout_features.parquet")
    holdout_raw = pd.read_csv(DATA_DIR / "holdout_data.csv")
    X_holdout = holdout[feature_cols]
    print(f"Holdout shape: {X_holdout.shape}")

    # Get raw probabilities
    print("Predicting...")
    probs = model.predict(X_holdout)  # shape: (n_rows, 4)

    # ── Bias toward Crisis/Death (lower threshold) ────────────────────────
    # Missed crisis = -10 penalty, so we boost crisis/death probabilities
    probs[:, 2] *= 1.5   # Crisis boost
    probs[:, 3] *= 1.3   # Death boost
    # Re-normalize
    probs = probs / probs.sum(axis=1, keepdims=True)

    preds = probs.argmax(axis=1)

    print(f"\nBefore monotonicity enforcement:")
    for label, name in enumerate(["Normal", "Warning", "Crisis", "Death"]):
        count = (preds == label).sum()
        print(f"  {label} ({name}): {count:,} ({count / len(preds) * 100:.1f}%)")

    # ── Enforce monotonicity per encounter ────────────────────────────────
    # Labels never decrease within an encounter (confirmed by EDA)
    # Take cumulative max within each encounter
    print("\nEnforcing monotonicity (labels never decrease within encounter)...")
    encounter_ids = holdout_raw["encounter_id"].values

    result = preds.copy()
    for eid in np.unique(encounter_ids):
        mask = encounter_ids == eid
        result[mask] = np.maximum.accumulate(preds[mask])

    preds = result

    print(f"\nAfter monotonicity enforcement:")
    for label, name in enumerate(["Normal", "Warning", "Crisis", "Death"]):
        count = (preds == label).sum()
        print(f"  {label} ({name}): {count:,} ({count / len(preds) * 100:.1f}%)")

    # Create submission
    submission = pd.DataFrame({
        "ID": range(1, len(preds) + 1),
        "predicted_label": preds,
    })
    out_path = DATA_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
