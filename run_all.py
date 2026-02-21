import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

print("Loading feature files ...")
train_df = pd.read_parquet("/Users/Admin/MedHack2026/train_features.parquet")
test_df  = pd.read_parquet("/Users/Admin/MedHack2026/test_features.parquet")
hold_df  = pd.read_parquet("/Users/Admin/MedHack2026/holdout_features.parquet")

print(f"  train  : {train_df.shape}")
print(f"  test   : {test_df.shape}")
print(f"  holdout: {hold_df.shape}")

label_col = "label"
train_labels = train_df[label_col].values
test_labels  = test_df[label_col].values
feature_cols = [c for c in train_df.columns if c != label_col]

X_train = train_df[feature_cols].values
y_train = train_labels
X_test  = test_df[feature_cols].values
y_test  = test_labels
X_hold  = hold_df[feature_cols].values if label_col not in hold_df.columns else hold_df[feature_cols].values

X_full = np.concatenate([X_train, X_test], axis=0)
y_full = np.concatenate([y_train, y_test], axis=0)

print(f"\nCombined training set : {X_full.shape}")
print("Label distribution (combined):")
unique, counts = np.unique(y_full, return_counts=True)
for u, c in zip(unique, counts):
    pct = c / len(y_full) * 100
    print(f"  class {int(u)}: {c}  ({pct:.1f}%)")

classes = np.array([0, 1, 2, 3])
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_full)
sample_weights_full = np.array([cw[int(label)] for label in y_full])
print(f"\nClass weights: {dict(zip(classes.tolist(), cw.tolist()))}")

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
num_boost_round = 800

print("\n--- Training on TRAIN split (for evaluation on TEST) ---")
dtrain = lgb.Dataset(X_train, label=y_train, weight=np.array([cw[int(l)] for l in y_train]))
model_eval = lgb.train(params, dtrain, num_boost_round=num_boost_round)

y_test_prob = model_eval.predict(X_test)
y_test_pred = np.argmax(y_test_prob, axis=1)
macro_f1 = f1_score(y_test, y_test_pred, average="macro")
print(f"\n=== Test-set Macro F1: {macro_f1:.5f} ===")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

print("--- Retraining on FULL (train + test) for holdout predictions ---")
dfull = lgb.Dataset(X_full, label=y_full, weight=sample_weights_full)
model_full = lgb.train(params, dfull, num_boost_round=num_boost_round)

y_hold_prob = model_full.predict(X_hold)
y_hold_pred = np.argmax(y_hold_prob, axis=1)

submission = pd.DataFrame({
    "ID": np.arange(1, len(y_hold_pred) + 1),
    "predicted_label": y_hold_pred.astype(int),
})
submission.to_csv("/Users/Admin/MedHack2026/submission.csv", index=False)
print(f"\nSubmission saved to /Users/Admin/MedHack2026/submission.csv  ({len(submission)} rows)")

print("\nHoldout prediction distribution:")
unique_h, counts_h = np.unique(y_hold_pred, return_counts=True)
for u, c in zip(unique_h, counts_h):
    pct = c / len(y_hold_pred) * 100
    print(f"  class {int(u)}: {c}  ({pct:.1f}%)")

print("\nFirst 10 rows of submission.csv:")
print(submission.head(10).to_string(index=False))
print("\nDone.")
