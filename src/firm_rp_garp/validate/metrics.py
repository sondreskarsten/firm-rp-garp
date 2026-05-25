"""Detection and separation metrics for the validation rig.

These metrics convert the rig's per-firm test results into the
five contract-level statistics used in
:func:`firm_rp_garp.validate.rig.run_validation_contract`.
"""
from __future__ import annotations

import math
import statistics


def auc_roc(scores: list[float], labels: list[bool]) -> float:
    """Compute AUC-ROC by counting concordant pairs.

    Parameters
    ----------
    scores : list[float]
        Predicted scores. Higher values should correspond to True
        labels for a good classifier.
    labels : list[bool]
        Ground-truth binary labels.

    Returns
    -------
    float
        Area under the ROC curve. 1.0 = perfect ranking, 0.5 = chance,
        0.0 = perfectly anti-ranked.

    Notes
    -----
    Uses the Mann-Whitney U statistic with mid-rank ties. For tiny
    samples this is robust without scikit-learn.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = [s for s, lab in zip(scores, labels) if lab]
    negatives = [s for s, lab in zip(scores, labels) if not lab]
    if not positives or not negatives:
        return float("nan")
    concordant = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                concordant += 1.0
            elif p == n:
                concordant += 0.5
    return concordant / (len(positives) * len(negatives))


def welch_t_statistic(group_a: list[float], group_b: list[float]) -> float:
    """Compute the Welch t-statistic for unequal-variance two-sample tests.

    Parameters
    ----------
    group_a : list[float]
        First group of observations.
    group_b : list[float]
        Second group of observations.

    Returns
    -------
    float
        Welch t-statistic; large absolute values indicate distinct
        group means relative to within-group variance.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)
    denominator = math.sqrt(var_a / len(group_a) + var_b / len(group_b))
    if denominator == 0.0:
        return float("nan")
    return (mean_a - mean_b) / denominator
