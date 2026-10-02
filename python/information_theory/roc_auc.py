"""ROC / AUC neurometric analysis of spike-rate distributions.

Port of ``COMP3.m`` from ``src/information-theory/``: two spike-rate
distributions (e.g. one neuron's responses under two stimulus conditions)
are histogrammed over shared integer bins, an ROC curve is traced by
sweeping a cumulative detection threshold, and the area under the curve
(AUC) quantifies how well an ideal observer could tell the two conditions
apart from a single trial.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["rate_distributions", "roc_curve", "auc", "roc_auc_score"]


def rate_distributions(
    condition1: ArrayLike, condition2: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Normalized spike-rate histograms over shared integer bins.

    Port of part II of ``COMP3.m``: bins are the integers
    ``0 .. max(rate)`` and each histogram is divided by its number of
    trials, giving per-bin response probabilities.

    Args:
        condition1: Non-negative integer spike rates under condition 1,
            shape ``(n_trials_1,)``.
        condition2: Non-negative integer spike rates under condition 2,
            shape ``(n_trials_2,)``.

    Returns:
        A tuple ``(dist1, dist2, bins)`` with the two normalized
        histograms and the integer bin values they refer to.

    Raises:
        ValueError: If the rates are negative or non-integer.
    """
    c1 = np.asarray(condition1, dtype=float).ravel()
    c2 = np.asarray(condition2, dtype=float).ravel()
    if c1.size == 0 or c2.size == 0:
        raise ValueError("Both conditions need at least one trial.")
    for name, c in (("condition1", c1), ("condition2", c2)):
        if np.any(c < 0) or not np.all(c == np.floor(c)):
            raise ValueError(f"{name} must contain non-negative integer rates.")
    n_bins = int(max(c1.max(), c2.max())) + 1
    bins = np.arange(n_bins, dtype=np.intp)
    dist1 = np.bincount(c1.astype(np.intp), minlength=n_bins) / c1.size
    dist2 = np.bincount(c2.astype(np.intp), minlength=n_bins) / c2.size
    return dist1, dist2, bins


def roc_curve(
    condition1: ArrayLike, condition2: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cumulative-threshold ROC curve between two rate distributions.

    Port of part III of ``COMP3.m``: for each integer threshold ``k`` the
    true/false positive rates are the tail masses
    ``P(rate >= k | condition)``. Condition 1 plays the role of the
    "signal" (hits) and condition 2 of the "noise" (false alarms), so
    place the higher-rate distribution first for an AUC near 1.

    Args:
        condition1: Non-negative integer spike rates under the signal
            condition, shape ``(n_trials_1,)``.
        condition2: Non-negative integer spike rates under the noise
            condition, shape ``(n_trials_2,)``.

    Returns:
        A tuple ``(fpr, tpr)`` of monotone arrays running from ``(0, 0)``
        to ``(1, 1)``.
    """
    dist1, dist2, _ = rate_distributions(condition1, condition2)
    # Tail masses P(rate >= k) for k = 0 .. n_bins - 1, ordered from the
    # strictest threshold (k = n_bins, both rates 0) down to k = 0 (both 1).
    tail1 = np.cumsum(dist1[::-1])  # P(rate >= k) for k = n_bins-1 .. 0
    tail2 = np.cumsum(dist2[::-1])
    tpr = np.concatenate(([0.0], tail1))
    fpr = np.concatenate(([0.0], tail2))
    return fpr, tpr


def auc(fpr: ArrayLike, tpr: ArrayLike) -> float:
    """Area under an ROC curve via the trapezoidal rule.

    Port of part IV of ``COMP3.m`` (``abs(trapz(...))``): the absolute
    value makes the result independent of the direction in which the
    curve is traversed.

    Args:
        fpr: False-positive rates, shape ``(n_thresholds,)``.
        tpr: True-positive rates, shape ``(n_thresholds,)``.

    Returns:
        The area under the curve between 0 and 1.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    fpr = np.asarray(fpr, dtype=float).ravel()
    tpr = np.asarray(tpr, dtype=float).ravel()
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same length.")
    return float(abs(np.trapezoid(tpr, fpr)))


def roc_auc_score(condition1: ArrayLike, condition2: ArrayLike) -> float:
    """AUC between two spike-rate distributions in a single call.

    Args:
        condition1: Non-negative integer spike rates under the signal
            condition, shape ``(n_trials_1,)``.
        condition2: Non-negative integer spike rates under the noise
            condition, shape ``(n_trials_2,)``.

    Returns:
        The area under the cumulative-threshold ROC curve, between 0 and 1
        (near 1 for well-separated distributions with condition 1 higher,
        near 0.5 for identical distributions).
    """
    fpr, tpr = roc_curve(condition1, condition2)
    return auc(fpr, tpr)
