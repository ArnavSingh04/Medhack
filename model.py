"""
LightGBM model to predict state label from vital signs.
Output: predictions.csv with columns: ID, predicted_label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

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
ID_COL_CSV = "encounter_id"
OUTPUT_ID_COL = "ID"
OUTPUT_LABEL_COL = "predicted_label"


def remove_fully_null_feature_rows(df):
    """Remove rows where ALL feature columns are null."""
    mask = ~df[FEATURE_COLS].isnull().all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def drop_empty_test_rows(df):
    """Drop rows with missing encounter_id or all features null."""
    if ID_COL_CSV in df.columns:
        df = df.dropna(subset=[ID_COL_CSV])
        df = df.loc[df[ID_COL_CSV].astype(str).str.strip() != ""]
    df = remove_fully_null_feature_rows(df)
    return df.reset_index(drop=True)


def load_and_prepare_train(path: Path):
    df = pd.read_csv(path)
    df = remove_fully_null_feature_rows(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # LightGBM can handle NaNs, but imputation keeps consistency
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y, imputer


def load_and_prepare_test(path: Path, imputer):
    df = pd.read_csv(path)

    # Ensure ID exists
    if "ID" not in df.columns:
        df["ID"] = np.arange(1, len(df) + 1, dtype=int)

    df = drop_empty_test_rows(df)

    X = df[FEATURE_COLS]
    X_imputed = imputer.transform(X)

    return X_imputed, df


def main():
    print("Loading training data...")
    X_train, y_train, imputer = load_and_prepare_train(TRAIN_PATH)

    print("Training LightGBM (multiclass)...")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Loading test data...")
    X_test, test_df = load_and_prepare_test(TEST_PATH, imputer)

    print(f"Clean test samples: {len(test_df)}")

    predicted_label_indices = model.predict(X_test).astype(int)

    predictions_df = pd.DataFrame({
        OUTPUT_ID_COL: test_df["ID"],
        OUTPUT_LABEL_COL: predicted_label_indices
    })

    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions to {OUTPUT_PATH}")
    print(f"Prediction distribution: {dict(zip(*np.unique(predicted_label_indices, return_counts=True)))}")


if __name__ == "__main__":
    main()