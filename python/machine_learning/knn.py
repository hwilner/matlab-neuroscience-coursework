"""Multiclass Euclidean k-nearest-neighbors classifier with 7-fold CV.

This module ports the MATLAB coursework functions
``src/machine-learning/get_kNN_classification.m`` and
``src/machine-learning/predict_kNN_multiclass_classification.m``. The
original classifier compared multivariate windows by the Euclidean distance
of their time-summed features (``compute_distance``); that distance is
preserved here as :func:`summed_feature_distance`, while
:class:`KNNClassifier` generalizes to ordinary feature matrices. Voting is
majority over the k nearest neighbors with ties broken by the label of the
single nearest neighbor, exactly as in the MATLAB implementation.
"""

from __future__ import annotations

import numpy as np


def summed_feature_distance(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """Computes the original coursework distance between two windows.

    Direct port of the MATLAB ``compute_distance``: each ``T x d`` window is
    collapsed to its time-summed feature vector ``sum(sample, axis=0)`` and
    the Euclidean distance between those vectors is returned.

    Args:
        sample1: Window of shape ``(T, d)`` or an already-summed vector.
        sample2: Window of the same shape as ``sample1``.

    Returns:
        Euclidean distance between the time-summed feature vectors.
    """
    a = np.asarray(sample1, dtype=float)
    b = np.asarray(sample2, dtype=float)
    if a.ndim == 2:
        a = a.sum(axis=0)
    if b.ndim == 2:
        b = b.sum(axis=0)
    return float(np.linalg.norm(a - b))


def _prepare_features(X: np.ndarray, summed_features: bool) -> np.ndarray:
    """Converts input windows/features into a 2D feature matrix.

    Args:
        X: Either a feature matrix ``(n_samples, n_features)`` or a stack of
            windows ``(n_samples, T, d)``.
        summed_features: If True and ``X`` is 3D, collapse each window by
            summing over the time axis (the original coursework features).

    Returns:
        Feature matrix of shape ``(n_samples, n_features)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        if summed_features:
            return X.sum(axis=1)
        return X.reshape(X.shape[0], -1)
    return X


def _vote(neighbor_labels: np.ndarray, nearest_label: int) -> int:
    """Majority vote with nearest-neighbor tie-breaking (MATLAB semantics).

    Args:
        neighbor_labels: Labels of the k nearest neighbors.
        nearest_label: Label of the single closest neighbor.

    Returns:
        The winning class label.
    """
    labels, counts = np.unique(neighbor_labels, return_counts=True)
    best = labels[counts == counts.max()]
    if len(best) > 1:
        return int(nearest_label)
    return int(best[0])


def get_knn_classification(
    k: int,
    test_X: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    summed_features: bool = False,
) -> np.ndarray:
    """Classifies a test set with k-nearest neighbors (MATLAB port).

    Functional port of ``get_kNN_classification.m``: for every test sample,
    distances to all training samples are computed and the label is the
    majority vote among the k nearest neighbors, ties broken by the nearest
    neighbor's label.

    Args:
        k: Number of neighbors.
        test_X: Test windows ``(n_test, T, d)`` or features ``(n_test, d)``.
        train_X: Training windows or features, matching ``test_X``.
        train_y: Training labels of shape ``(n_train,)``.
        summed_features: If True, use the original time-summed feature
            distance for 3D window inputs.

    Returns:
        Predicted labels of shape ``(n_test,)``.
    """
    Xtr = _prepare_features(train_X, summed_features)
    Xte = _prepare_features(test_X, summed_features)
    train_y = np.asarray(train_y).ravel()
    diff = Xte[:, None, :] - Xtr[None, :, :]
    distances = np.sqrt(np.einsum("ntd,ntd->nt", diff, diff))
    order = np.argsort(distances, axis=1)
    predictions = np.empty(Xte.shape[0], dtype=train_y.dtype)
    for i in range(Xte.shape[0]):
        idx = order[i, :k]
        predictions[i] = _vote(train_y[idx], train_y[order[i, 0]])
    return predictions


class KNNClassifier:
    """Multiclass Euclidean k-nearest-neighbors classifier.

    Attributes:
        n_neighbors: Number of neighbors used for voting.
        summed_features: Whether 3D window inputs are collapsed to
            time-summed features (the original coursework behavior).
        classes_: Sorted unique class labels seen during ``fit``.
    """

    def __init__(self, n_neighbors: int = 1, summed_features: bool = False) -> None:
        """Initializes the classifier.

        Args:
            n_neighbors: Number of nearest neighbors ``k``.
            summed_features: If True, use the time-summed window features of
                the original MATLAB implementation.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = int(n_neighbors)
        self.summed_features = bool(summed_features)
        self.classes_: np.ndarray | None = None
        self._train_X: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Stores the training set (kNN is a lazy learner).

        Args:
            X: Training windows ``(n_samples, T, d)`` or features
                ``(n_samples, d)``.
            y: Training labels of shape ``(n_samples,)``.

        Returns:
            The fitted classifier (``self``).
        """
        self._train_X = _prepare_features(X, self.summed_features)
        self._train_y = np.asarray(y).ravel()
        self.classes_ = np.unique(self._train_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for the given samples.

        Args:
            X: Test windows or features, matching the training layout.

        Returns:
            Predicted labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._train_X is None:
            raise RuntimeError("KNNClassifier must be fitted before predict.")
        return get_knn_classification(
            self.n_neighbors, X, self._train_X, self._train_y,
            summed_features=self.summed_features,
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes classification accuracy on a labeled dataset.

        Args:
            X: Test windows or features.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_folds: int = 7,
    summed_features: bool = False,
) -> tuple[float, np.ndarray]:
    """Estimates kNN accuracy with n-fold cross-validation (MATLAB port).

    Port of the cross-validation loop in
    ``predict_kNN_multiclass_classification.m``: the data is split into
    ``n_folds`` contiguous segments; each segment serves once as validation
    set while the remainder is used for training.

    Args:
        X: Samples, windows ``(n_samples, T, d)`` or features
            ``(n_samples, d)``.
        y: Labels of shape ``(n_samples,)``.
        k: Number of neighbors.
        n_folds: Number of cross-validation folds (7 in the coursework).
        summed_features: If True, use time-summed window features.

    Returns:
        Tuple ``(mean_accuracy, fold_accuracies)`` where
        ``fold_accuracies`` has shape ``(n_folds,)``.
    """
    n = len(y)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])
    fold_accuracies = np.empty(n_folds)
    for i in range(n_folds):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        clf = KNNClassifier(n_neighbors=k, summed_features=summed_features)
        clf.fit(np.asarray(X)[mask], np.asarray(y)[mask])
        fold_accuracies[i] = clf.score(
            np.asarray(X)[lo:hi], np.asarray(y)[lo:hi]
        )
    return float(fold_accuracies.mean()), fold_accuracies
