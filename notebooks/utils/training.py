"""Shared training utilities for MedHack model notebooks.

Centralises class weighting, scoring, threshold optimization, two-phase Optuna
tuning, and submission generation so each notebook stays DRY.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

from .scoring import SCORE_MATRIX, evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_COLS = ["timestamp", "encounter_id"]
TARGET = "label"
PRIOR_LABEL_COLS = [
    "prior_label",
    "max_label_last_60s",
    "max_label_encounter",
    "ever_deteriorated",
]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def get_feature_cols(
    df: pd.DataFrame,
    exclude_prior_labels: bool = True,
) -> list[str]:
    """Return numeric feature columns, excluding meta/target/prior-label cols."""
    exclude = set(META_COLS + [TARGET])
    if exclude_prior_labels:
        exclude.update(PRIOR_LABEL_COLS)
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert datetime cols to int64, drop non-numeric cols."""
    out = df.copy()
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(out[col].dtype):
            out[col] = out[col].astype("int64")
    return out


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

def compute_class_weights(
    y: np.ndarray,
    boost_map: dict[int, float] | None = None,
    strategy: str = "balanced",
) -> dict[int, float]:
    """Compute class weights with optional per-class boost multipliers.

    Args:
        y: Target labels.
        boost_map: Optional ``{class_id: multiplier}`` applied on top of
            balanced weights.  E.g. ``{2: 3.0, 3: 2.0}`` to further boost
            Crisis and Death.
        strategy: ``'balanced'`` (sklearn) or ``'inverse_freq'``.
    """
    classes = np.sort(np.unique(y))
    if strategy == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y)
    else:  # inverse_freq
        counts = np.bincount(y, minlength=int(classes.max()) + 1)
        weights = np.array([len(y) / (len(classes) * counts[c]) for c in classes])
    cw = dict(zip(classes.tolist(), weights.tolist()))
    if boost_map:
        for cls, mult in boost_map.items():
            if cls in cw:
                cw[cls] *= mult
    return cw


def make_sample_weights(
    y: np.ndarray, class_weight_dict: dict[int, float]
) -> np.ndarray:
    """Per-sample weight array from per-class weights."""
    return np.array([class_weight_dict[int(yi)] for yi in y])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def competition_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    alpha: float = 0.5,
) -> float:
    """Combined competition score (higher is better).

    ``alpha * per_sample_matrix_score + (1 - alpha) * AUC_alive_dead``
    """
    death_prob = y_prob[:, 3] if y_prob is not None else None
    result = evaluate(y_true, y_pred, death_prob)
    ms = result["matrix_score"] / max(result["n_matrix"], 1)
    auc = result["auc_alive_dead"] or 0.5
    return alpha * ms + (1 - alpha) * auc


def macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean macro average-precision (AUPRC) across 4 classes."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    return float(average_precision_score(y_bin, y_prob, average="macro"))


# ---------------------------------------------------------------------------
# False alarm rate
# ---------------------------------------------------------------------------

def false_alarm_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    encounter_ids: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of all-normal encounters with at least one alarm."""
    prob_pos = y_prob[:, 1:].sum(axis=1)
    df = pd.DataFrame(
        {"encounter_id": encounter_ids, "label": y_true, "prob_pos": prob_pos}
    )
    healthy = (
        df.groupby("encounter_id")
        .agg({"label": "max", "prob_pos": "max"})
        .reset_index()
    )
    healthy_enc = healthy[healthy["label"] == 0]
    n = len(healthy_enc)
    if n == 0:
        return 0.0
    return float((healthy_enc["prob_pos"] >= threshold).sum()) / n


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable | None = None,
    n_classes: int = 4,
) -> dict:
    """Grid-search per-class probability scales to maximise *metric_fn*.

    Instead of plain ``argmax``, compute ``adjusted[c] = prob[c] * scale[c]``
    then take the argmax.  Class-0 scale is fixed at 1.0.

    Returns dict with ``scales``, ``best_score``, ``y_pred_optimized``.
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp, yprob: competition_score(yt, yp, yprob)

    def _predict(scales: list[float]) -> np.ndarray:
        adjusted = y_prob * np.array(scales)
        return adjusted.argmax(axis=1)

    best_score = -np.inf
    best_scales = [1.0] * n_classes

    for s1 in np.arange(0.5, 5.1, 0.25):
        for s2 in np.arange(0.5, 8.1, 0.5):
            for s3 in np.arange(0.5, 5.1, 0.5):
                scales = [1.0, s1, s2, s3]
                preds = _predict(scales)
                score = metric_fn(y_true, preds, y_prob)
                if score > best_score:
                    best_score = score
                    best_scales = scales

    return {
        "scales": best_scales,
        "best_score": best_score,
        "y_pred_optimized": _predict(best_scales),
    }


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def generate_submission(
    y_pred: np.ndarray,
    output_path: str | Path,
    sample_submission_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate and save a competition submission CSV."""
    submission = pd.DataFrame({
        "ID": np.arange(1, len(y_pred) + 1),
        "predicted_label": y_pred.astype(int),
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    if sample_submission_path and Path(sample_submission_path).exists():
        sample = pd.read_csv(sample_submission_path)
        assert len(submission) == len(sample), (
            f"Row mismatch: {len(submission)} vs {len(sample)}"
        )
        assert list(submission.columns) == list(sample.columns), "Column mismatch"
        print("Format verified against sample_submission.csv")
    print(f"Saved submission to {output_path}")
    return submission


# ---------------------------------------------------------------------------
# Two-phase Optuna helpers
# ---------------------------------------------------------------------------

def stratified_encounter_sample(
    df: pd.DataFrame,
    frac: float,
    seed: int = 42,
    encounter_col: str = "encounter_id",
    label_col: str = "label",
) -> pd.Index:
    """Return boolean mask for encounter-level stratified sample."""
    enc_labels = df.groupby(encounter_col)[label_col].max()
    sampled = enc_labels.groupby(enc_labels, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed)
    )
    return df[encounter_col].isin(sampled.index)
