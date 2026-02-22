"""Utility modules for MedHack."""

from .features import FeatureEngineer
from .scoring import SCORE_MATRIX, evaluate, score_from_confusion_matrix
from .training import (
    META_COLS,
    PRIOR_LABEL_COLS,
    TARGET,
    competition_score,
    compute_class_weights,
    ensure_numeric,
    false_alarm_rate,
    generate_submission,
    get_feature_cols,
    macro_auprc,
    make_sample_weights,
    optimize_thresholds,
    stratified_encounter_sample,
)

__all__ = [
    "FeatureEngineer",
    "evaluate",
    "SCORE_MATRIX",
    "score_from_confusion_matrix",
    "META_COLS",
    "PRIOR_LABEL_COLS",
    "TARGET",
    "competition_score",
    "compute_class_weights",
    "ensure_numeric",
    "false_alarm_rate",
    "generate_submission",
    "get_feature_cols",
    "macro_auprc",
    "make_sample_weights",
    "optimize_thresholds",
    "stratified_encounter_sample",
]
