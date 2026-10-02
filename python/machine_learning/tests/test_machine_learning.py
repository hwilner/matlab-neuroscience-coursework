"""Tests for the machine_learning package (perceptron, kNN, GeoKNN)."""

import numpy as np
import pytest

from machine_learning.geoknn import (
    GeoKNNClassifier,
    MDMClassifier,
    karcher_mean,
    log_euclidean_distance,
    log_euclidean_distance_matrix,
    logm_spd,
    spd_embed,
    tangent_space_map,
)
from machine_learning.knn import KNNClassifier, cross_validate
from machine_learning.perceptron import Perceptron


def _make_covariance_split(
    n_train: int, n_test: int, n_classes: int = 7, t: int = 64, d: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic covariance-discriminative windows: zero means, class-
    specific covariance rotations/anisotropies shared by train and test."""
    rng = np.random.default_rng(seed)
    covariances = []
    for _ in range(n_classes):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        log_eigs = rng.normal(0.0, 0.8, size=d)
        cov = (q * np.exp(log_eigs)[None, :]) @ q.T
        # Normalize trace so summed features cannot leak per-class scale.
        covariances.append(cov * (d / np.trace(cov)))

    def sample(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for c, cov in enumerate(covariances):
            w, v = np.linalg.eigh(cov)
            z = rng.standard_normal((n_per_class, t, d))
            xs.append(z @ ((v * np.sqrt(w)[None, :]) @ v.T).T)
            ys.append(np.full(n_per_class, c))
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = sample(n_train)
    X_test, y_test = sample(n_test)
    return X_train, y_train, X_test, y_test


def test_perceptron_converges_on_separable_data() -> None:
    """Perceptron must reach zero training error on separable data."""
    rng = np.random.default_rng(42)
    n = 200
    X_pos = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n // 2, 2))
    X_neg = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    clf = Perceptron(learning_rate=0.2, max_turns=1000).fit(X, y)
    assert clf.errors_per_turn_[-1] == 0
    assert clf.n_turns_ < clf.max_turns
    assert clf.score(X, y) == 1.0


def test_knn_perfect_on_well_separated_blobs() -> None:
    """Baseline multiclass Euclidean kNN is perfect on separated blobs."""
    rng = np.random.default_rng(7)
    centers = 10.0 * np.eye(5)
    X = np.vstack([rng.normal(centers[c], 0.3, size=(30, 5)) for c in range(5)])
    y = np.concatenate([np.full(30, c) for c in range(5)])
    clf = KNNClassifier(n_neighbors=3).fit(X, y)
    assert clf.score(X, y) == 1.0
    # Shuffle so contiguous CV folds contain all classes.
    perm = rng.permutation(len(y))
    mean_acc, fold_accs = cross_validate(X[perm], y[perm], k=3, n_folds=5)
    assert fold_accs.shape == (5,)
    assert mean_acc == pytest.approx(1.0)


def test_spd_embed_is_spd() -> None:
    """SPD embeddings are symmetric with strictly positive eigenvalues."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 64, 8))
    C = spd_embed(X)
    assert C.shape == (20, 8, 8)
    assert np.allclose(C, np.swapaxes(C, 1, 2), atol=1e-10)
    eigvals = np.linalg.eigvalsh(C)
    assert (eigvals > 0).all()


def test_log_euclidean_distance_properties() -> None:
    """Log-Euclidean metric: symmetry, zero diagonal, diagonal sanity."""
    rng = np.random.default_rng(2)
    A = spd_embed(rng.standard_normal((64, 8)))
    B = spd_embed(rng.standard_normal((64, 8)))
    assert log_euclidean_distance(A, A) == pytest.approx(0.0, abs=1e-10)
    assert log_euclidean_distance(A, B) == pytest.approx(
        log_euclidean_distance(B, A), abs=1e-10
    )
    # For diagonal SPDs, logm is diag(log lambda) and the metric reduces to
    # the Euclidean distance between log-eigenvalue vectors.
    a = np.array([1.0, 2.0, 5.0])
    b = np.array([4.0, 0.5, 3.0])
    L = logm_spd(np.diag(a))
    assert np.allclose(L, np.diag(np.log(a)))
    expected = float(np.linalg.norm(np.log(a) - np.log(b)))
    assert log_euclidean_distance(np.diag(a), np.diag(b)) == pytest.approx(expected)
    D = log_euclidean_distance_matrix(np.diag(a), np.diag(b))
    assert D.shape == (1, 1)
    assert D[0, 0] == pytest.approx(expected)
    stack = np.stack([np.diag(a), np.diag(b)])
    D2 = log_euclidean_distance_matrix(stack, stack)
    assert np.allclose(np.diag(D2), 0.0, atol=1e-10)
    assert D2[0, 1] == pytest.approx(expected)


def test_geoknn_beats_chance_on_covariance_regime() -> None:
    """GeoKNN >> chance on covariance-discriminative data where the
    Euclidean-on-summed-features baseline is at chance."""
    X_train, y_train, X_test, y_test = _make_covariance_split(80, 30, seed=3)
    geo = GeoKNNClassifier(n_neighbors=5).fit(X_train, y_train)
    geo_acc = geo.score(X_test, y_test)
    euc = KNNClassifier(n_neighbors=5, summed_features=True).fit(X_train, y_train)
    euc_acc = euc.score(X_test, y_test)
    chance = 1.0 / 7
    assert geo_acc > 0.8
    assert euc_acc < 0.3
    assert geo_acc > euc_acc + 0.5


def test_mdm_matches_geoknn_at_larger_k() -> None:
    """MDM (class-mean limit) should track GeoKNN at large-ish k."""
    X_train, y_train, X_test, y_test = _make_covariance_split(80, 30, seed=5)
    geo = GeoKNNClassifier(n_neighbors=15).fit(X_train, y_train)
    mdm = MDMClassifier().fit(X_train, y_train)
    geo_acc = geo.score(X_test, y_test)
    mdm_acc = mdm.score(X_test, y_test)
    assert mdm_acc > 0.8
    assert abs(geo_acc - mdm_acc) < 0.15


def test_karcher_mean_and_tangent_space() -> None:
    """Karcher mean is SPD and tangent mapping has the right dimensions."""
    X, _, _, _ = _make_covariance_split(20, 5, n_classes=2, seed=8)
    C = spd_embed(X)
    mean = karcher_mean(C, max_iter=15)
    assert (np.linalg.eigvalsh(mean) > 0).all()
    assert np.allclose(mean, mean.T, atol=1e-10)
    vecs = tangent_space_map(C, mean)
    assert vecs.shape == (len(C), 8 * 9 // 2)
    assert np.isfinite(vecs).all()


def test_airm_metric_properties() -> None:
    """AIRM is zero on the diagonal, symmetric, and congruence-invariant."""
    from machine_learning.geoknn import airm_distance, airm_distance_matrix

    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    A = A @ A.T + np.eye(5)
    B = rng.standard_normal((5, 5))
    B = B @ B.T + 2 * np.eye(5)
    assert airm_distance(A, A) < 1e-10
    assert abs(airm_distance(A, B) - airm_distance(B, A)) < 1e-8
    # Congruence invariance: d(GAG^T, GBG^T) = d(A, B) for invertible G.
    G = np.diag(np.array([1.0, 2.0, 0.5, 1.5, 3.0]))
    assert abs(airm_distance(G @ A @ G.T, G @ B @ G.T) - airm_distance(A, B)) < 1e-6
    D = airm_distance_matrix(np.stack([A, B]), np.stack([A, B]))
    assert D.shape == (2, 2) and np.allclose(np.diag(D), 0.0)


def test_geoknn_airm_matches_logeuclid_on_covariance_data() -> None:
    """AIRM GeoKNN is competitive with log-Euclidean GeoKNN."""
    from machine_learning.geoknn import GeoKNNClassifier as GK

    X_train, y_train, X_test, y_test = _make_covariance_split(80, 30, seed=11)
    le = GK(n_neighbors=5, metric="logeuclid").fit(X_train, y_train)
    ai = GK(n_neighbors=5, metric="airm").fit(X_train, y_train)
    acc_le = le.score(X_test, y_test)
    acc_ai = ai.score(X_test, y_test)
    assert acc_ai > 0.8
    assert abs(acc_ai - acc_le) < 0.15


def test_geoknn_invalid_metric_raises() -> None:
    """Unknown metrics are rejected."""
    from machine_learning.geoknn import GeoKNNClassifier as GK

    with pytest.raises(ValueError):
        GK(n_neighbors=3, metric="mahalanobis")
