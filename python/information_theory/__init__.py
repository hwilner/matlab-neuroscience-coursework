"""Information-theory tools for neural data, ported from MATLAB coursework.

Ports ``src/information-theory/`` (``MINF.m``, ``joint.m``,
``calculate_spike_direction_MI.m``, ``COMP3.m``) and the synergy part of
``src/signal-processing/ass6.m``, and adds bias-corrected
mutual-information estimation for small-sample neuron ranking.
"""

from .mi import (
    entropy,
    joint_histogram,
    mutual_information_miller_madow,
    mutual_information_plugin,
    mutual_information_shuffle_corrected,
    neuron_ranking,
)
from .roc_auc import auc, rate_distributions, roc_auc_score, roc_curve
from .synergy import synergy_matrix

__all__ = [
    "entropy",
    "joint_histogram",
    "mutual_information_plugin",
    "mutual_information_miller_madow",
    "mutual_information_shuffle_corrected",
    "neuron_ranking",
    "rate_distributions",
    "roc_curve",
    "roc_auc_score",
    "auc",
    "synergy_matrix",
]
