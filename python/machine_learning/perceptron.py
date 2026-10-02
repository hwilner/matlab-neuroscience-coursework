"""Perceptron classifier with the classic online update rule.

This is a Python port of ``src/machine-learning/ex1_perceptron.m``. The
perceptron scans the training set one sample at a time ("online"), applies
the update ``w <- w + lr * (target - output) * x`` (and likewise for the
bias) whenever the prediction differs from the target, and repeats full
passes ("turns") over the data until no errors occur or ``max_turns`` is
reached. On linearly separable data the perceptron convergence theorem
guarantees termination with zero training error.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Binary perceptron trained with the online (per-sample) update rule.

    Attributes:
        learning_rate: Step size of the weight/bias update.
        max_turns: Maximum number of full passes over the training set.
        weights_: Learned weight vector of shape ``(n_features,)``.
        bias_: Learned bias (output neuron threshold).
        n_turns_: Number of full passes actually performed during ``fit``.
        errors_per_turn_: Number of misclassified samples in each turn.
    """

    def __init__(self, learning_rate: float = 0.2, max_turns: int = 1000) -> None:
        """Initializes the perceptron hyperparameters.

        Args:
            learning_rate: Learning rate for the online update rule.
            max_turns: Maximum number of epochs (full passes over the data).
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.learning_rate = float(learning_rate)
        self.max_turns = int(max_turns)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.n_turns_: int = 0
        self.errors_per_turn_: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Trains the perceptron with the online learning rule.

        Each turn iterates over all samples in order; whenever the predicted
        label differs from the target, weights and bias are updated as
        ``w += lr * (t - o) * x`` and ``b += lr * (t - o)``. Training stops
        when a full turn produces no errors or ``max_turns`` is reached.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Binary targets of shape ``(n_samples,)`` with values in
                ``{0, 1}``.

        Returns:
            The fitted perceptron (``self``).

        Raises:
            ValueError: If ``y`` contains labels other than 0 and 1.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if not np.isin(np.unique(y), [0.0, 1.0]).all():
            raise ValueError("Perceptron targets must be in {0, 1}.")

        n_features = X.shape[1]
        rng = np.random.default_rng(0)
        # Deterministic small random init, mirroring the MATLAB exercise
        # which started from fixed non-zero weights.
        self.weights_ = rng.uniform(-0.5, 0.5, size=n_features)
        self.bias_ = 0.9
        self.errors_per_turn_ = []
        self.n_turns_ = 0

        errors_exist = True
        while errors_exist and self.n_turns_ < self.max_turns:
            self.n_turns_ += 1
            errors_exist = False
            errors_num = 0
            for xi, ti in zip(X, y):
                output = float(xi @ self.weights_ + self.bias_ > 0.0)
                if output != ti:
                    self.weights_ += self.learning_rate * (ti - output) * xi
                    self.bias_ += self.learning_rate * (ti - output)
                    errors_exist = True
                    errors_num += 1
            self.errors_per_turn_.append(errors_num)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Computes the net input ``X @ w + b`` for each sample.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Net inputs of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the perceptron has not been fitted.
        """
        if self.weights_ is None:
            raise RuntimeError("Perceptron must be fitted before prediction.")
        return np.asarray(X, dtype=float) @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary labels (1 if net input > 0, else 0).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels of shape ``(n_samples,)`` with values in
            ``{0, 1}``.
        """
        return (self.decision_function(X) > 0.0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes classification accuracy on a labeled dataset.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
