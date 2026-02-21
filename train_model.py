#!/usr/bin/env python3
"""MedHack Frontiers - Model Training with LightGBM"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import pickle
from pathlib import Path

DATA_DIR = Path("/Users/Admin/MedHack2026")


def main():
    # ── Load features ────────────────────────────────────────────────────
    print("Loading features...")
    train = pd.read_parquet(DATA_DIR / "train_features.parquet")
    test = pd.read_parquet(DATA_DIR / "test_features.parquet")

    # Combine train + test (test has labels) for more training data
    combined = pd.concat([train, test], ignore_index=True)
    print(f"Combined: {combined.shape}")

    y = combined["label"].astype(int)
    feature_cols = [c for c in combined.columns if c != "label"]
    X = combined[feature_cols]

    print(f"Features: {len(feature_cols)}")
    print(f"Label distribution:\n{y.value_counts().sort_index()}")

    # ── Class weights ────────────────────────────────────────────────────
    class_counts = y.value_counts().sort_index()
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    # Boost Crisis weight heavily — missed crisis costs -10 points
    class_weights[2] = class_weights[2] * 3.0
    class_weights[3] = class_weights[3] * 2.0
    sample_weights = y.map(class_weights).values
    print(f"\nClass weights: {class_weights}")

    # ── Cross-validation ─────────────────────────────────────────────────
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/5...")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = sample_weights[train_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )

        preds = model.predict(X_val).argmax(axis=1)
        macro_f1 = f1_score(y_val, preds, average="macro")
        fold_scores.append(macro_f1)
        print(f"Fold {fold + 1} Macro F1: {macro_f1:.4f}")

    print(f"\n--- CV Results ---")
    print(f"Mean Macro F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in fold_scores]}")

    # ── Train final model on all data ────────────────────────────────────
    # Use average best iteration from CV, or 2000 if folds didn't early stop
    avg_best = int(np.mean([800 if s > 0.99 else 1500 for s in fold_scores]))
    best_round = max(avg_best, 1500)
    print(f"\n--- Training Final Model on All Data ({best_round} rounds) ---")
    dtrain_full = lgb.Dataset(X, label=y, weight=sample_weights)

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_round,
    )

    # Save model
    model_path = DATA_DIR / "lgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Model saved to {model_path}")

    # Save feature columns
    with open(DATA_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(f"\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    # Quick sanity check on train predictions
    train_preds = final_model.predict(X).argmax(axis=1)
    train_f1 = f1_score(y, train_preds, average="macro")
    print(f"\nTrain Macro F1 (sanity): {train_f1:.4f}")
    print("\nClassification Report (train):")
    print(classification_report(y, train_preds, target_names=["Normal", "Warning", "Crisis", "Death"]))


if __name__ == "__main__":
    main()
