"""Machine learning ports and research extensions.

This package ports the MATLAB coursework kNN classifier and perceptron from
``src/machine-learning/`` and adds GeoKNN, a geodesic-kernel-weighted
Riemannian kNN classifier operating on symmetric positive definite (SPD)
matrix embeddings of multivariate windows.
"""

from .perceptron import Perceptron
from .knn import KNNClassifier, cross_validate, get_knn_classification
from .geoknn import (
    GeoKNNClassifier,
    MDMClassifier,
    karcher_mean,
    log_euclidean_distance,
    log_euclidean_distance_matrix,
    spd_embed,
    tangent_space_map,
)

__all__ = [
    "Perceptron",
    "KNNClassifier",
    "cross_validate",
    "get_knn_classification",
    "GeoKNNClassifier",
    "MDMClassifier",
    "karcher_mean",
    "log_euclidean_distance",
    "log_euclidean_distance_matrix",
    "spd_embed",
    "tangent_space_map",
]
