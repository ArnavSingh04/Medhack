"""Evaluation metrics for MedHack patient label predictions.

Labels: 0=Normal, 1=Warning, 2=Crisis, 3=Death
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# Scoring matrix: SCORE_MATRIX[true_label, pred_label] for labels 0, 1, 2
# Covers Normal, Warning, Crisis only (death/label 3 is excluded from matrix score)
SCORE_MATRIX = np.array([
    [0, -2, -2],   # True Normal (0): guess Normal=0, Warning=-2, Crisis=-2
    [-3, 2, -1],   # True Warning (1): guess Normal=-3, Warning=+2, Crisis=-1
    [-10, -3, 3],  # True Crisis (2): guess Normal=-10, Warning=-3, Crisis=+3
])


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba_death: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Evaluate model predictions using AUC (alive vs dead) and the crisis matrix score.

    Args:
        y_true: True labels (0, 1, 2, 3).
        y_pred: Predicted labels (0, 1, 2, 3).
        y_pred_proba_death: Optional probability of class 3 (death). Required for AUC
            calculation. If None, AUC will be None.

    Returns:
        Dict with keys:
        - 'auc_alive_dead': ROC AUC for alive (0,1,2) vs dead (3), or None if
          y_pred_proba_death not provided.
        - 'matrix_score': Sum of points from the 3x3 Normal/Warning/Crisis matrix.
          Only samples where both true and predicted labels are 0, 1, or 2 are included.
        - 'n_matrix': Number of samples included in matrix_score.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # AUC: alive (0,1,2) vs dead (3)
    auc_alive_dead = None
    if y_pred_proba_death is not None:
        y_pred_proba_death = np.asarray(y_pred_proba_death).ravel()
        y_binary = (y_true == 3).astype(np.int32)
        if np.unique(y_binary).size > 1:
            auc_alive_dead = float(roc_auc_score(y_binary, y_pred_proba_death))
        else:
            auc_alive_dead = 0.5  # single class, undefined AUC

    # Matrix score: only for samples where both true and pred are 0, 1, or 2
    mask = (y_true >= 0) & (y_true <= 2) & (y_pred >= 0) & (y_pred <= 2)
    t = y_true[mask].astype(int)
    p = y_pred[mask].astype(int)
    matrix_score = float(np.sum(SCORE_MATRIX[t, p]))
    n_matrix = int(np.sum(mask))

    return {
        "auc_alive_dead": auc_alive_dead,
        "matrix_score": matrix_score,
        "n_matrix": n_matrix,
    }


def score_from_confusion_matrix(cm: np.ndarray) -> float:
    """Compute the matrix score from a confusion matrix.

    Uses the 3x3 Normal/Warning/Crisis block. cm[i, j] = count of samples with
    true label i and predicted label j (sklearn convention).

    Args:
        cm: Confusion matrix, shape (n_classes, n_classes). Can be 3x3 or 4x4;
            if 4x4, only the upper-left 3x3 block (labels 0, 1, 2) is used.

    Returns:
        Total matrix score (sum of count * score for each cell in 0–2 × 0–2).
    """
    cm = np.asarray(cm)
    # Use 3x3 block for Normal, Warning, Crisis
    cm_3x3 = cm[:3, :3]
    return float(np.sum(cm_3x3 * SCORE_MATRIX))
