"""SpikeSPD: spike-waveform clustering on the SPD manifold.

Novel research component. Each spike waveform ``x`` (length L) is
embedded as a symmetric positive-definite (SPD) covariance descriptor

    C = x x^T + eps * I,

which lives on the SPD manifold. Instead of treating waveforms as
Euclidean vectors (the PCA + k-means baseline), SpikeSPD clusters the
descriptors with log-Euclidean geometry: matrices are mapped to the
tangent space at the identity via the matrix logarithm, vectorized with
the ``vech`` half-vectorization (off-diagonal entries weighted by
sqrt(2) so that Euclidean norm equals Frobenius norm), and clustered
with a k-means variant whose centroid updates are exact log-Euclidean
(Frechet) means ``expm(mean(log C_i))`` computed on the manifold.

References:
    Arsigny, Fillard, Pennec & Ayache (2006). "Log-Euclidean metrics for
    fast and simple calculus on diffusion tensors." Magnetic Resonance
    in Medicine 56(2).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

__all__ = [
    "spd_descriptor",
    "spd_descriptors",
    "logm_spd",
    "expm_spd",
    "vech",
    "tangent_features",
    "log_euclidean_mean",
    "log_euclidean_kmeans",
    "sort_spikes_spd",
]

_MIN_EIG = 1e-12


# ---------------------------------------------------------------------------
# SPD descriptors and matrix maps
# ---------------------------------------------------------------------------


def spd_descriptor(waveform: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Map a spike waveform to its regularized SPD covariance descriptor.

    Computes ``C = x x^T + eps * (tr(x x^T) / L) * I``. The regularizer
    is scaled to the average signal power so that ``eps`` is
    dimensionless and the descriptor stays well conditioned even for
    rank-1 outer products.

    Args:
        waveform: 1-D spike waveform of length L.
        eps: Relative regularization strength (added to each eigenvalue
            in units of mean signal power).

    Returns:
        Symmetric positive-definite matrix of shape ``(L, L)``.

    Raises:
        ValueError: If ``waveform`` is empty or all zeros.
    """
    x = np.asarray(waveform, dtype=float).ravel()
    if x.size == 0 or not np.any(x):
        raise ValueError("waveform must be non-empty and not all zeros.")
    c = np.outer(x, x)
    ridge = eps * np.trace(c) / x.size
    return c + ridge * np.eye(x.size)


def spd_descriptors(waveforms: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Map a set of waveforms to SPD covariance descriptors.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        eps: Relative regularization strength (see :func:`spd_descriptor`).

    Returns:
        Array of SPD matrices of shape ``(n_spikes, L, L)``.
    """
    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_spikes, L).")
    return np.stack([spd_descriptor(x, eps) for x in w])


def logm_spd(c: np.ndarray) -> np.ndarray:
    """Matrix logarithm of an SPD matrix via symmetric eigendecomposition.

    Computes ``U diag(log w) U^T`` from the eigendecomposition
    ``C = U diag(w) U^T``. Eigenvalues are floored at a tiny positive
    value for numerical safety.

    Args:
        c: Symmetric positive-definite matrix.

    Returns:
        The (symmetric) matrix logarithm of ``c``.
    """
    w, u = np.linalg.eigh(np.asarray(c, dtype=float))
    w = np.maximum(w, _MIN_EIG)
    return (u * np.log(w)) @ u.T


def expm_spd(m: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix via eigendecomposition.

    Computes ``U diag(exp(w)) U^T``; the result is SPD.

    Args:
        m: Symmetric matrix.

    Returns:
        The SPD matrix exponential of ``m``.
    """
    w, u = np.linalg.eigh(np.asarray(m, dtype=float))
    return (u * np.exp(w)) @ u.T


def vech(m: np.ndarray) -> np.ndarray:
    """Half-vectorize a symmetric matrix with sqrt(2) off-diagonal weighting.

    Stacks the upper triangle of ``m`` (row by row), multiplying
    off-diagonal entries by ``sqrt(2)`` so that ``||vech(A) - vech(B)||_2
    = ||A - B||_F``: Euclidean distances in tangent-feature space are
    exactly log-Euclidean distances on the SPD manifold.

    Args:
        m: Square symmetric matrix of shape ``(L, L)``.

    Returns:
        Vector of length ``L * (L + 1) / 2``.
    """
    m = np.asarray(m, dtype=float)
    iu = np.triu_indices(m.shape[0])
    v = m[iu]
    off_diag = iu[0] != iu[1]
    v = v.copy()
    v[off_diag] *= np.sqrt(2.0)
    return v


def tangent_features(descriptors: np.ndarray) -> np.ndarray:
    """Compute log-Euclidean tangent features of SPD descriptors.

    Applies the matrix logarithm to each descriptor and half-vectorizes
    the result with :func:`vech`.

    Args:
        descriptors: Array of SPD matrices, shape ``(n, L, L)``.

    Returns:
        Tangent-feature matrix of shape ``(n, L * (L + 1) / 2)``.
    """
    cs = np.asarray(descriptors, dtype=float)
    return np.stack([vech(logm_spd(c)) for c in cs])


def log_euclidean_mean(descriptors: np.ndarray) -> np.ndarray:
    """Log-Euclidean (Frechet) mean of SPD matrices.

    Computes ``expm( (1/n) sum_i logm(C_i) )`` -- the closed-form
    barycenter under the log-Euclidean metric.

    Args:
        descriptors: Array of SPD matrices, shape ``(n, L, L)``.

    Returns:
        SPD mean matrix of shape ``(L, L)``.

    Raises:
        ValueError: If ``descriptors`` is empty.
    """
    cs = np.asarray(descriptors, dtype=float)
    if cs.shape[0] == 0:
        raise ValueError("Cannot take the mean of an empty set of descriptors.")
    mean_log = np.mean([logm_spd(c) for c in cs], axis=0)
    return expm_spd(mean_log)


# ---------------------------------------------------------------------------
# Log-Euclidean k-means
# ---------------------------------------------------------------------------


def log_euclidean_kmeans(
    descriptors: np.ndarray,
    n_clusters: int,
    max_iter: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """K-means on SPD matrices with exact log-Euclidean mean updates.

    Assignment uses the log-Euclidean distance
    ``d(C, M) = ||logm(C) - logm(M)||_F`` (evaluated in the ``vech``
    tangent representation), and each centroid update is the exact
    log-Euclidean mean :func:`log_euclidean_mean` of the assigned
    descriptors. Initialization labels come from a Euclidean k-means on
    the tangent features. Empty clusters are re-seeded to the descriptor
    farthest from its centroid.

    Args:
        descriptors: Array of SPD matrices, shape ``(n, L, L)``.
        n_clusters: Number of clusters.
        max_iter: Maximum number of assignment/update iterations.
        random_state: Seed for the initializing k-means (deterministic).

    Returns:
        Integer cluster labels of shape ``(n,)``.

    Raises:
        ValueError: If there are fewer descriptors than clusters.
    """
    cs = np.asarray(descriptors, dtype=float)
    n = cs.shape[0]
    if n < n_clusters:
        raise ValueError("Need at least as many descriptors as clusters.")

    logs = np.stack([logm_spd(c) for c in cs])
    feats = np.stack([vech(m) for m in logs])

    init = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = init.fit_predict(feats)

    for _ in range(max_iter):
        # Update: exact log-Euclidean centroid per cluster.
        centroid_feats = np.empty((n_clusters, feats.shape[1]))
        nonempty = np.array([np.any(labels == j) for j in range(n_clusters)])
        for j in np.flatnonzero(nonempty):
            mean_log = logs[labels == j].mean(axis=0)
            centroid_feats[j] = vech(mean_log)
        if not np.all(nonempty):
            # Re-seed each empty cluster on the descriptor that is farthest
            # (in log-Euclidean distance) from all current centroids.
            d2_avail = (
                (feats**2).sum(axis=1, keepdims=True)
                - 2.0 * feats @ centroid_feats[nonempty].T
                + (centroid_feats[nonempty] ** 2).sum(axis=1)
            )
            min_d = d2_avail.min(axis=1)
            for j in np.flatnonzero(~nonempty):
                farthest = int(np.argmax(min_d))
                centroid_feats[j] = feats[farthest]
                min_d[farthest] = 0.0  # do not pick the same point twice

        # Assignment: nearest centroid in log-Euclidean distance.
        d2 = (
            (feats**2).sum(axis=1, keepdims=True)
            - 2.0 * feats @ centroid_feats.T
            + (centroid_feats**2).sum(axis=1)
        )
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


# ---------------------------------------------------------------------------
# Full SpikeSPD pipeline
# ---------------------------------------------------------------------------


def sort_spikes_spd(
    waveforms: np.ndarray,
    n_clusters: int,
    eps: float = 1e-3,
    max_iter: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster spike waveforms on the SPD manifold (SpikeSPD).

    Pipeline: (1) each waveform is mapped to the SPD covariance
    descriptor ``C = x x^T + eps * I``; (2) descriptors are clustered
    with :func:`log_euclidean_kmeans`, i.e. log-Euclidean distances and
    exact log-Euclidean centroid updates.

    Args:
        waveforms: Array of shape ``(n_spikes, L)``.
        n_clusters: Number of clusters (units) to find.
        eps: Relative regularization strength of the descriptors.
        max_iter: Maximum k-means iterations.
        random_state: Seed for the k-means initialization.

    Returns:
        Integer cluster labels of shape ``(n_spikes,)``.

    Raises:
        ValueError: If there are fewer waveforms than clusters.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.linspace(-1, 2, 91)
        >>> a = np.exp(-t**2 / 0.1) + 0.05 * rng.standard_normal((40, 91))
        >>> b = -np.exp(-(t - 0.5) ** 2 / 0.2) + 0.05 * rng.standard_normal((40, 91))
        >>> labels = sort_spikes_spd(np.vstack([a, b]), 2, random_state=0)
        >>> len(np.unique(labels))
        2
    """
    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2:
        raise ValueError("waveforms must be a 2-D array (n_spikes, L).")
    if w.shape[0] < n_clusters:
        raise ValueError("Need at least as many waveforms as clusters.")
    descriptors = spd_descriptors(w, eps=eps)
    return log_euclidean_kmeans(
        descriptors, n_clusters, max_iter=max_iter, random_state=random_state
    )
