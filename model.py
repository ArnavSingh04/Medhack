"""
Logistic regression model to predict state label from vital signs.
Output: predictions.csv with only patient_id and label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "holdout_data.csv"
OUTPUT_PATH = DATA_DIR / "predictions.csv"

FEATURE_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
]
TARGET_COL = "label"
# Test CSV has encounter_id; submission must have column "ID" (1-based index)
ID_COL_CSV = "encounter_id"
OUTPUT_ID_COL = "ID"
OUTPUT_LABEL_COL = "predicted_label"


def remove_fully_null_feature_rows(df):
    """Remove rows where ALL feature columns are null."""
    mask = ~df[FEATURE_COLS].isnull().all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def drop_empty_test_rows(df):
    """Drop rows with missing encounter_id or all features null (no empty rows in predictions)."""
    # Drop rows with null/empty encounter_id
    if ID_COL_CSV in df.columns:
        df = df.dropna(subset=[ID_COL_CSV])
        df = df.loc[df[ID_COL_CSV].astype(str).str.strip() != ""]
    # Drop rows where all feature columns are null
    df = remove_fully_null_feature_rows(df)
    return df.reset_index(drop=True)


def load_and_prepare_train(path: Path):
    df = pd.read_csv(path)

    df = remove_fully_null_feature_rows(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, imputer, scaler


def load_and_prepare_test(path: Path, imputer, scaler):
    df = pd.read_csv(path)

    # Ensure we have an ID column from test data (1-based row index if not present)
    if "ID" not in df.columns:
        df["ID"] = np.arange(1, len(df) + 1, dtype=int)

    # Drop empty rows (missing id or all-null features) before creating predictions
    df = drop_empty_test_rows(df)

    X = df[FEATURE_COLS]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled, df


def main():
    print("Loading training data...")
    X_train, y_train, imputer, scaler = load_and_prepare_train(TRAIN_PATH)

    print("Training logistic regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("Loading test data...")
    X_test, test_df = load_and_prepare_test(TEST_PATH, imputer, scaler)

    print(f"Clean test samples: {len(test_df)}")

    predicted_label_indices = model.predict(X_test).astype(int)

    # Same format as requested: ID from test data, predicted_label
    predictions_df = pd.DataFrame({
        "ID": test_df["ID"],
        "predicted_label": predicted_label_indices
    })

    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions to {OUTPUT_PATH}")
    print(f"Prediction distribution: {dict(zip(*np.unique(predicted_label_indices, return_counts=True)))}")


if __name__ == "__main__":
    main()