"""GeoKNN: Geodesic-Kernel-Weighted Riemannian kNN on SPD embeddings.

Each multivariate window ``X`` of shape ``(T, d)`` (T time steps, d
channels) is embedded as a symmetric positive definite (SPD) matrix

    C = X^T X / T + eps * I,   eps = 1e-6 * trace(X^T X / T) / d,

i.e. a regularized channel-covariance (second moment) matrix. Distances on
the SPD manifold use the log-Euclidean metric

    d(A, B) = || logm(A) - logm(B) ||_F,

where the matrix logarithm is computed via the symmetric eigendecomposition
``logm(A) = V diag(log w) V^T``. GeoKNN classifies a query by its k nearest
SPD neighbors, voting with Riemannian Gaussian kernel weights
``w = exp(-d^2 / (2 sigma^2))`` where ``sigma`` is the median of the query's
k neighbor distances (self-tuned per query). A minimum-distance-to-mean
(MDM) classifier against class-conditional log-Euclidean means is provided
as a geometric baseline, and optional tangent-space mapping at the Karcher
(Frechet) mean under the affine-invariant metric is exposed via
:func:`karcher_mean` and :func:`tangent_space_map`.
"""

from __future__ import annotations

import numpy as np

_EIGEN_FLOOR = 1e-12


def spd_embed(X: np.ndarray, eps_factor: float = 1e-6) -> np.ndarray:
    """Embeds multivariate windows as regularized SPD covariance matrices.

    For a window ``X`` of shape ``(T, d)`` the embedding is
    ``C = X^T X / T + eps * I`` with ``eps = eps_factor * trace(X^T X/T) / d``
    (a trace-proportional ridge, so the matrix stays well conditioned even
    for rank-deficient windows).

    Args:
        X: Window of shape ``(T, d)`` or stack of windows ``(n, T, d)``.
        eps_factor: Relative regularization strength.

    Returns:
        SPD matrix ``(d, d)`` or stack of SPD matrices ``(n, d, d)``.
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 2
    if single:
        X = X[None]
    if X.ndim != 3:
        raise ValueError("spd_embed expects windows of shape (n, T, d).")
    n, t, d = X.shape
    C = np.einsum("nti,ntj->nij", X, X) / t
    trace = np.trace(C, axis1=1, axis2=2)
    eps = eps_factor * trace / d
    C = C + eps[:, None, None] * np.eye(d)
    return C[0] if single else C


def logm_spd(A: np.ndarray) -> np.ndarray:
    """Computes the matrix logarithm of SPD matrices via ``eigh``.

    Args:
        A: SPD matrix ``(d, d)`` or stack ``(n, d, d)``.

    Returns:
        Symmetric matrix logarithm(s) of the same shape.
    """
    A = np.asarray(A, dtype=float)
    single = A.ndim == 2
    if single:
        A = A[None]
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, _EIGEN_FLOOR)
    L = (V * np.log(w)[:, None, :]) @ np.swapaxes(V, 1, 2)
    return L[0] if single else L


def expm_sym(S: np.ndarray) -> np.ndarray:
    """Computes the matrix exponential of symmetric matrices via ``eigh``.

    Args:
        S: Symmetric matrix ``(d, d)`` or stack ``(n, d, d)``.

    Returns:
        SPD matrix exponential(s) of the same shape.
    """
    S = np.asarray(S, dtype=float)
    single = S.ndim == 2
    if single:
        S = S[None]
    w, V = np.linalg.eigh(S)
    E = (V * np.exp(w)[:, None, :]) @ np.swapaxes(V, 1, 2)
    return E[0] if single else E


def _as_stack(A: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalizes input to an ``(n, d, d)`` stack.

    Args:
        A: Matrix ``(d, d)`` or stack ``(n, d, d)``.

    Returns:
        Tuple of the stack and whether the input was a single matrix.
    """
    A = np.asarray(A, dtype=float)
    single = A.ndim == 2
    return (A[None] if single else A), single


def log_euclidean_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Computes the log-Euclidean distance between two SPD matrices.

    ``d(A, B) = || logm(A) - logm(B) ||_F``. The metric is symmetric and
    ``d(A, A) = 0``.

    Args:
        A: SPD matrix of shape ``(d, d)``.
        B: SPD matrix of shape ``(d, d)``.

    Returns:
        Non-negative geodesic distance.
    """
    return float(np.linalg.norm(logm_spd(A) - logm_spd(B)))


def log_euclidean_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the pairwise log-Euclidean distance matrix.

    Uses precomputed matrix logarithms and the expansion
    ``||L_a - L_b||_F^2 = ||L_a||_F^2 + ||L_b||_F^2 - 2 <L_a, L_b>`` so the
    cost is a single matrix product after the eigendecompositions.

    Args:
        A: SPD matrix ``(d, d)`` or stack ``(n_a, d, d)``.
        B: SPD matrix ``(d, d)`` or stack ``(n_b, d, d)``.

    Returns:
        Distance matrix of shape ``(n_a, n_b)``.
    """
    A, single_a = _as_stack(A)
    B, single_b = _as_stack(B)
    La = logm_spd(A).reshape(len(A), -1)
    Lb = logm_spd(B).reshape(len(B), -1)
    na = np.einsum("ij,ij->i", La, La)
    nb = np.einsum("ij,ij->i", Lb, Lb)
    d2 = na[:, None] + nb[None, :] - 2.0 * (La @ Lb.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def airm_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices.

    Computes ``d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F = sqrt(sum(log^2
    lam_i))`` where ``lam_i`` are the generalized eigenvalues of
    ``B v = lam A v`` (equivalently, the eigenvalues of ``A^{-1} B``).
    Unlike the log-Euclidean metric, AIRM is invariant under congruence
    transformations ``C -> G C G^T`` (e.g., channel re-scaling or mixing).

    Args:
        A: SPD matrix of shape ``(d, d)``.
        B: SPD matrix of shape ``(d, d)``.

    Returns:
        The affine-invariant geodesic distance.
    """
    from scipy.linalg import eigh as scipy_eigh

    lam = scipy_eigh(np.asarray(B, float), np.asarray(A, float), eigvals_only=True)
    lam = np.maximum(lam, 1e-12)
    return float(np.sqrt(np.sum(np.log(lam) ** 2)))


def airm_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise affine-invariant distances between two stacks of SPD matrices.

    Args:
        A: Stack of SPD matrices, shape ``(n, d, d)``.
        B: Stack of SPD matrices, shape ``(m, d, d)``.

    Returns:
        Distance matrix of shape ``(n, m)``.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    out = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            out[i, j] = airm_distance(A[i], B[j])
    return out


def log_euclidean_mean(mats: np.ndarray) -> np.ndarray:
    """Computes the log-Euclidean (Frechet) mean of SPD matrices.

    The log-Euclidean mean is ``expm(mean_i logm(A_i))`` in closed form.

    Args:
        mats: Stack of SPD matrices ``(n, d, d)``.

    Returns:
        SPD mean matrix of shape ``(d, d)``.
    """
    mats, _ = _as_stack(mats)
    return expm_sym(logm_spd(mats).mean(axis=0))


def _sqrtm_spd(A: np.ndarray, power: float) -> np.ndarray:
    """Computes a symmetric matrix power ``A**power`` via ``eigh``.

    Args:
        A: SPD matrix of shape ``(d, d)``.
        power: Exponent (e.g. 0.5 for the square root, -0.5 for its inverse).

    Returns:
        Symmetric matrix power of shape ``(d, d)``.
    """
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, _EIGEN_FLOOR)
    return (V * (w ** power)[None, :]) @ V.T


def karcher_mean(
    mats: np.ndarray, max_iter: int = 15, tol: float = 1e-9
) -> np.ndarray:
    """Computes the affine-invariant Karcher mean by fixed-point iteration.

    Iterates ``M <- M^{1/2} expm(mean_i logm(M^{-1/2} A_i M^{-1/2})) M^{1/2}``
    starting from the log-Euclidean mean, for at most ``max_iter`` steps
    (~15 suffices in practice).

    Args:
        mats: Stack of SPD matrices ``(n, d, d)``.
        max_iter: Maximum number of fixed-point iterations.
        tol: Convergence threshold on the Frobenius norm of the mean
            tangent increment.

    Returns:
        SPD Karcher mean of shape ``(d, d)``.
    """
    mats, _ = _as_stack(mats)
    M = log_euclidean_mean(mats)
    for _ in range(max_iter):
        M_inv_sqrt = _sqrtm_spd(M, -0.5)
        whitened = M_inv_sqrt @ mats @ M_inv_sqrt
        S = logm_spd(whitened).mean(axis=0)
        if np.linalg.norm(S) < tol:
            break
        M_sqrt = _sqrtm_spd(M, 0.5)
        M = M_sqrt @ expm_sym(S) @ M_sqrt
    return M


def tangent_space_map(mats: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Maps SPD matrices to the tangent space at a reference SPD mean.

    Computes ``logm(mean^{-1/2} A mean^{-1/2})`` and vectorizes it using the
    upper triangle with off-diagonal entries scaled by ``sqrt(2)`` (so the
    Euclidean norm of the vector equals the affine-invariant geodesic norm).

    Args:
        mats: Stack of SPD matrices ``(n, d, d)``.
        mean: Reference SPD matrix of shape ``(d, d)`` (e.g. the Karcher
            mean of the training set).

    Returns:
        Tangent vectors of shape ``(n, d * (d + 1) / 2)``.
    """
    mats, _ = _as_stack(mats)
    mean_inv_sqrt = _sqrtm_spd(mean, -0.5)
    tangents = logm_spd(mean_inv_sqrt @ mats @ mean_inv_sqrt)
    d = mats.shape[-1]
    iu = np.triu_indices(d)
    vecs = tangents[:, iu[0], iu[1]]
    off_diag = iu[0] != iu[1]
    vecs[:, off_diag] *= np.sqrt(2.0)
    return vecs


class GeoKNNClassifier:
    """Geodesic-kernel-weighted Riemannian kNN on SPD embeddings.

    Windows are embedded as SPD covariance matrices (or supplied directly as
    SPD matrices). A query is classified by its k nearest neighbors under
    the chosen Riemannian metric, voting with Gaussian kernel weights
    ``w = exp(-d^2 / (2 sigma^2))`` where ``sigma`` is the median of the k
    neighbor distances of that query (self-tuned bandwidth). Voting ties are
    broken by the nearest neighbor's label.

    Attributes:
        n_neighbors: Number of Riemannian neighbors used for voting.
        metric: ``"logeuclid"`` (default) or ``"airm"`` (affine-invariant).
        classes_: Sorted unique class labels seen during ``fit``.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "logeuclid") -> None:
        """Initializes the classifier.

        Args:
            n_neighbors: Number of nearest SPD neighbors ``k``.
            metric: Riemannian distance: ``"logeuclid"`` (fast, precomputed
                matrix logarithms) or ``"airm"`` (affine-invariant; slower,
                pairwise generalized eigenvalues).
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if metric not in ("logeuclid", "airm"):
            raise ValueError(f"unknown metric: {metric!r}")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.classes_: np.ndarray | None = None
        self._train_logm: np.ndarray | None = None
        self._train_spd: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Converts windows or SPD matrices into a stack of SPD matrices.

        Args:
            X: Windows ``(n, T, d)`` or SPD matrices ``(n, d, d)``.

        Returns:
            Stack of SPD matrices ``(n, d, d)``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None]
        if X.ndim != 3:
            raise ValueError("Expected windows (n, T, d) or SPDs (n, d, d).")
        if X.shape[-1] == X.shape[-2]:
            return X  # already SPD matrices
        return spd_embed(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeoKNNClassifier":
        """Embeds the training set and precomputes matrix logarithms.

        Args:
            X: Training windows ``(n_samples, T, d)`` or SPD matrices
                ``(n_samples, d, d)``.
            y: Training labels of shape ``(n_samples,)``.

        Returns:
            The fitted classifier (``self``).
        """
        C = self._embed(X)
        self._train_logm = logm_spd(C).reshape(len(C), -1)
        self._train_spd = C if self.metric == "airm" else None
        self._train_y = np.asarray(y).ravel()
        self.classes_ = np.unique(self._train_y)
        return self

    def _distance_matrix_from_logm(self, Lq: np.ndarray) -> np.ndarray:
        """Pairwise log-Euclidean distances from precomputed logarithms.

        Args:
            Lq: Flattened query matrix logarithms ``(n_query, d*d)``.

        Returns:
            Distance matrix of shape ``(n_query, n_train)``.
        """
        Lt = self._train_logm
        nq = np.einsum("ij,ij->i", Lq, Lq)
        nt = np.einsum("ij,ij->i", Lt, Lt)
        d2 = nq[:, None] + nt[None, :] - 2.0 * (Lq @ Lt.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels with kernel-weighted Riemannian voting.

        Args:
            X: Query windows ``(n, T, d)`` or SPD matrices ``(n, d, d)``.

        Returns:
            Predicted labels of shape ``(n,)``.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._train_logm is None:
            raise RuntimeError("GeoKNNClassifier must be fitted before predict.")
        C = self._embed(X)
        if self.metric == "airm":
            D = airm_distance_matrix(C, self._train_spd)
        else:
            Lq = logm_spd(C).reshape(len(C), -1)
            D = self._distance_matrix_from_logm(Lq)
        k = min(self.n_neighbors, D.shape[1])
        order = np.argsort(D, axis=1)
        predictions = np.empty(len(C), dtype=self._train_y.dtype)
        for i in range(len(C)):
            idx = order[i, :k]
            d_k = D[i, idx]
            sigma = max(float(np.median(d_k)), 1e-12)
            weights = np.exp(-(d_k ** 2) / (2.0 * sigma ** 2))
            totals = np.zeros(len(self.classes_))
            for c, cls in enumerate(self.classes_):
                totals[c] = weights[self._train_y[idx] == cls].sum()
            best = np.flatnonzero(totals == totals.max())
            if len(best) > 1:
                predictions[i] = self._train_y[order[i, 0]]
            else:
                predictions[i] = self.classes_[best[0]]
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes classification accuracy on a labeled dataset.

        Args:
            X: Query windows or SPD matrices.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


class MDMClassifier:
    """Minimum-distance-to-mean classifier on the SPD manifold.

    Each class is represented by its class-conditional log-Euclidean mean
    ``expm(mean logm(C_i))`` and a query is assigned to the class whose mean
    is closest under the log-Euclidean metric. MDM is the geometric analog
    of nearest-class-mean and the ``k -> n`` limit of Riemannian kNN.

    Attributes:
        classes_: Sorted unique class labels seen during ``fit``.
    """

    def __init__(self) -> None:
        """Initializes the classifier (no hyperparameters)."""
        self.classes_: np.ndarray | None = None
        self._mean_logm: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MDMClassifier":
        """Computes the class-conditional log-Euclidean means.

        Args:
            X: Training windows ``(n_samples, T, d)`` or SPD matrices
                ``(n_samples, d, d)``.
            y: Training labels of shape ``(n_samples,)``.

        Returns:
            The fitted classifier (``self``).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None]
        C = X if X.shape[-1] == X.shape[-2] else spd_embed(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        logms = logm_spd(C)
        means = [expm_sym(logms[y == cls].mean(axis=0)) for cls in self.classes_]
        self._mean_logm = logm_spd(np.stack(means)).reshape(len(means), -1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels by nearest class-conditional geometric mean.

        Args:
            X: Query windows ``(n, T, d)`` or SPD matrices ``(n, d, d)``.

        Returns:
            Predicted labels of shape ``(n,)``.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if self._mean_logm is None:
            raise RuntimeError("MDMClassifier must be fitted before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None]
        C = X if X.shape[-1] == X.shape[-2] else spd_embed(X)
        Lq = logm_spd(C).reshape(len(C), -1)
        Lm = self._mean_logm
        nq = np.einsum("ij,ij->i", Lq, Lq)
        nm = np.einsum("ij,ij->i", Lm, Lm)
        d2 = nq[:, None] + nm[None, :] - 2.0 * (Lq @ Lm.T)
        np.maximum(d2, 0.0, out=d2)
        return self.classes_[np.argmin(d2, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes classification accuracy on a labeled dataset.

        Args:
            X: Query windows or SPD matrices.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
