"""GeoKNN experiment: covariance-discriminative vs. mean-discriminative control.

Compares four classifiers on two synthetic 7-class regimes of multivariate
windows (T=64 time steps, d=8 channels):

* ``euc_summed``: baseline Euclidean kNN on raw time-summed features (the
  repo's original MATLAB coursework classifier).
* ``euc_cov``: Euclidean kNN on vectorized covariance matrices.
* ``geoknn``: geodesic-kernel-weighted Riemannian kNN on SPD embeddings.
* ``mdm``: minimum distance to class-conditional log-Euclidean means.

Regime (a) COVARIANCE-DISCRIMINATIVE: all class means are zero and classes
differ only by covariance rotation/anisotropy. Regime (b) CONTROL: all
classes share one covariance and differ only in mean location.

For each regime and training size in {100, 400} samples/class, k is swept
over 1..15 with 7-fold cross-validation; the best k is evaluated on a
held-out test set. Accuracy mean +/- std over seeds, a k-sensitivity table
(k = 1, 3, 5, 11, 15), and per-query runtimes are written to RESULTS.md.

Run: ``python3 python/machine_learning/experiment_geoknn.py``
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.geoknn import (
    MDMClassifier,
    log_euclidean_distance_matrix,
    logm_spd,
    spd_embed,
)

N_CLASSES = 7
N_CHANNELS = 8
WINDOW = 64
N_TEST_PER_CLASS = 50
TRAIN_SIZES = (100, 400)
K_SWEEP = list(range(1, 16))
K_TABLE = (1, 3, 5, 11, 15)
N_FOLDS = 7
SEEDS = (11, 23, 37)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    """Draws a uniformly random orthogonal matrix via QR decomposition."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def make_covariance_regime(
    n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates the covariance-discriminative regime (a).

    All class means are zero; class ``c`` windows are drawn from
    ``N(0, Q_c diag(lambda_c) Q_c^T)`` with a class-specific random rotation
    ``Q_c`` and eigenvalue (anisotropy) profile ``lambda_c``.

    Args:
        n_train: Training windows per class.
        n_test: Test windows per class.
        seed: Deterministic seed.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)`` with windows of shape
        ``(n, T, d)``.
    """
    rng = np.random.default_rng(seed)
    d, t = N_CHANNELS, WINDOW
    covariances = []
    for _ in range(N_CLASSES):
        q = _random_orthogonal(rng, d)
        log_eigs = rng.normal(0.0, 0.3, size=d)
        cov = (q * np.exp(log_eigs)[None, :]) @ q.T
        # Equalize total variance across classes so the time-summed-feature
        # baseline cannot exploit per-class scale differences.
        covariances.append(cov * (d / np.trace(cov)))

    def sample(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for c, cov in enumerate(covariances):
            z = rng.standard_normal((n_per_class, t, d))
            w, v = np.linalg.eigh(cov)
            xs.append(z @ ((v * np.sqrt(w)[None, :]) @ v.T).T)
            ys.append(np.full(n_per_class, c))
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = sample(n_train)
    X_test, y_test = sample(n_test)
    return X_train, y_train, X_test, y_test


def make_control_regime(
    n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates the mean-discriminative control regime (b).

    All classes share one fixed covariance; class ``c`` windows are drawn
    from ``N(mu_c, Sigma_shared)`` with well-separated class means.

    Args:
        n_train: Training windows per class.
        n_test: Test windows per class.
        seed: Deterministic seed.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)``.
    """
    rng = np.random.default_rng(seed)
    d, t = N_CHANNELS, WINDOW
    q = _random_orthogonal(rng, d)
    shared_cov = (q * np.exp(rng.normal(0.0, 0.5, size=d))[None, :]) @ q.T
    mean_dirs = rng.standard_normal((N_CLASSES, d))
    mean_dirs /= np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    means = 1.2 * mean_dirs

    def sample(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        w, v = np.linalg.eigh(shared_cov)
        chol = (v * np.sqrt(w)[None, :]) @ v.T
        for c in range(N_CLASSES):
            z = rng.standard_normal((n_per_class, t, d))
            xs.append(means[c] + z @ chol.T)
            ys.append(np.full(n_per_class, c))
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = sample(n_train)
    X_test, y_test = sample(n_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Distance-matrix-driven kNN machinery (rank neighbors once, sweep k free)
# ---------------------------------------------------------------------------

def _euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix between rows of A and B."""
    na = np.einsum("ij,ij->i", A, A)
    nb = np.einsum("ij,ij->i", B, B)
    d2 = na[:, None] + nb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _vote_euclidean(
    sorted_labels: np.ndarray, classes: np.ndarray, k: int
) -> np.ndarray:
    """Unweighted majority vote over the first k sorted neighbors.

    Ties are broken by the nearest neighbor's label (MATLAB semantics).
    """
    first_k = sorted_labels[:, :k]
    counts = (first_k[:, :, None] == classes[None, None, :]).sum(axis=1)
    winners = counts == counts.max(axis=1, keepdims=True)
    tie = winners.sum(axis=1) > 1
    pred = classes[np.argmax(counts, axis=1)]
    pred[tie] = sorted_labels[tie, 0]
    return pred


def _vote_geoknn(
    sorted_dist: np.ndarray, sorted_labels: np.ndarray,
    classes: np.ndarray, k: int,
) -> np.ndarray:
    """Gaussian-kernel-weighted vote with per-query self-tuned bandwidth.

    Weights are ``exp(-d^2 / (2 sigma^2))`` with ``sigma`` the median of the
    query's k neighbor distances; ties break to the nearest neighbor.
    """
    d_k = sorted_dist[:, :k]
    sigma = np.maximum(np.median(d_k, axis=1), 1e-12)
    weights = np.exp(-(d_k ** 2) / (2.0 * sigma[:, None] ** 2))
    mask = sorted_labels[:, :k, None] == classes[None, None, :]
    totals = (weights[:, :, None] * mask).sum(axis=1)
    winners = totals == totals.max(axis=1, keepdims=True)
    tie = winners.sum(axis=1) > 1
    pred = classes[np.argmax(totals, axis=1)]
    pred[tie] = sorted_labels[tie, 0]
    return pred


class _DistanceMethod:
    """kNN-style method defined by a feature/distance pair."""

    def __init__(self, name: str, transform, geometric: bool = False) -> None:
        """Bundles a feature transform with a voting rule.

        Args:
            name: Method identifier.
            transform: Callable mapping windows ``(n, T, d)`` to either flat
                features ``(n, m)`` (Euclidean) or SPD matrices ``(n, d, d)``.
            geometric: If True, use log-Euclidean distances and
                kernel-weighted GeoKNN voting; otherwise Euclidean distances
                and unweighted majority voting.
        """
        self.name = name
        self.transform = transform
        self.geometric = geometric

    def distances(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Pairwise distance matrix between transformed samples."""
        fa, fb = self.transform(A), self.transform(B)
        if self.geometric:
            return log_euclidean_distance_matrix(fa, fb)
        return _euclidean_distances(fa, fb)

    def vote(self, sorted_dist, sorted_labels, classes, k):
        """Votes over sorted neighbors with the method's rule."""
        if self.geometric:
            return _vote_geoknn(sorted_dist, sorted_labels, classes, k)
        return _vote_euclidean(sorted_labels, classes, k)


def _summed_features(X: np.ndarray) -> np.ndarray:
    """Time-summed window features (original MATLAB coursework)."""
    return X.sum(axis=1)


def _vectorized_covariances(X: np.ndarray) -> np.ndarray:
    """Flattened covariance matrices (d x d -> d^2 features)."""
    return spd_embed(X).reshape(len(X), -1)


def _cv_accuracy_curve(
    method: _DistanceMethod, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Computes 7-fold CV accuracy for every k in K_SWEEP.

    The data is shuffled with a fixed seed before taking contiguous folds
    (so each fold sees all classes), and neighbors are ranked once per fold
    and reused for all k values.
    """
    perm = np.random.default_rng(0).permutation(len(y))
    X, y = X[perm], y[perm]
    n = len(y)
    fold_sizes = np.full(N_FOLDS, n // N_FOLDS)
    fold_sizes[: n % N_FOLDS] += 1
    bounds = np.concatenate([[0], np.cumsum(fold_sizes)])
    feats = method.transform(X)
    full = (
        log_euclidean_distance_matrix(feats, feats)
        if method.geometric else _euclidean_distances(feats, feats)
    )
    classes = np.unique(y)
    curve = np.zeros(len(K_SWEEP))
    for f in range(N_FOLDS):
        val = np.arange(bounds[f], bounds[f + 1])
        tr = np.concatenate([np.arange(0, bounds[f]),
                             np.arange(bounds[f + 1], n)])
        sub = full[np.ix_(val, tr)]
        order = np.argsort(sub, axis=1)
        sorted_dist = np.take_along_axis(sub, order, axis=1)
        sorted_labels = y[tr][order]
        for j, k in enumerate(K_SWEEP):
            pred = method.vote(sorted_dist, sorted_labels, classes, k)
            curve[j] += np.mean(pred == y[val])
    return curve / N_FOLDS


def _mdm_fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fits MDM on SPD embeddings and predicts the test set."""
    clf = MDMClassifier().fit(X_train, y_train)
    return clf.predict(X_test)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_single(
    regime_fn, n_train: int, seed: int
) -> dict[str, dict[str, object]]:
    """Runs one (regime, train size, seed) configuration for all methods.

    Returns:
        Mapping method name -> {"best_k", "test_acc", "cv_curve",
        "ms_per_query"}. MDM has no k; its entries are k-independent.
    """
    X_tr, y_tr, X_te, y_te = regime_fn(n_train, N_TEST_PER_CLASS, seed)
    classes = np.unique(y_tr)
    methods = [
        _DistanceMethod("euc_summed", _summed_features),
        _DistanceMethod("euc_cov", _vectorized_covariances),
        _DistanceMethod("geoknn", spd_embed, geometric=True),
    ]
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        cv_curve = _cv_accuracy_curve(method, X_tr, y_tr)
        best_k = K_SWEEP[int(np.argmax(cv_curve))]
        feats_tr, feats_te = method.transform(X_tr), method.transform(X_te)
        start = time.perf_counter()
        if method.geometric:
            dist = log_euclidean_distance_matrix(feats_te, feats_tr)
        else:
            dist = _euclidean_distances(feats_te, feats_tr)
        order = np.argsort(dist, axis=1)
        sorted_dist = np.take_along_axis(dist, order, axis=1)
        sorted_labels = y_tr[order]
        pred = method.vote(sorted_dist, sorted_labels, classes, best_k)
        elapsed = time.perf_counter() - start
        results[method.name] = {
            "best_k": best_k,
            "test_acc": float(np.mean(pred == y_te)),
            "cv_curve": cv_curve,
            "ms_per_query": 1e3 * elapsed / len(y_te),
        }
    start = time.perf_counter()
    pred = _mdm_fit_predict(X_tr, y_tr, X_te)
    elapsed = time.perf_counter() - start
    results["mdm"] = {
        "best_k": None,
        "test_acc": float(np.mean(pred == y_te)),
        "cv_curve": np.full(len(K_SWEEP), np.nan),
        "ms_per_query": 1e3 * elapsed / len(y_te),
    }
    return results


def fmt_pm(values: list[float]) -> str:
    """Formats a list of accuracies as 'mean +/- std' percentages."""
    arr = np.asarray(values)
    return f"{100 * arr.mean():.1f} +/- {100 * arr.std():.1f}"


def main() -> str:
    """Runs the full experiment and writes captioned tables to RESULTS.md.

    Returns:
        The RESULTS.md markdown content.
    """
    regimes = {
        "covariance": make_covariance_regime,
        "control": make_control_regime,
    }
    method_names = ["euc_summed", "euc_cov", "geoknn", "mdm"]
    method_titles = {
        "euc_summed": "Euclidean kNN (summed features, original)",
        "euc_cov": "Euclidean kNN (vectorized covariance)",
        "geoknn": "GeoKNN (Riemannian, kernel-weighted)",
        "mdm": "MDM (log-Euclidean class means)",
    }
    # all[regime][n_train][method] -> list over seeds of per-run dicts
    all_results: dict = {r: {n: {m: [] for m in method_names}
                             for n in TRAIN_SIZES} for r in regimes}
    total_start = time.perf_counter()
    for regime, fn in regimes.items():
        for n_train in TRAIN_SIZES:
            for rep, seed in enumerate(SEEDS):
                cfg_seed = seed + 10_000 * TRAIN_SIZES.index(n_train)
                res = run_single(fn, n_train, cfg_seed)
                for m in method_names:
                    all_results[regime][n_train][m].append(res[m])
    total_elapsed = time.perf_counter() - total_start

    lines: list[str] = []
    lines.append("# GeoKNN Experiment Results\n")
    lines.append(
        "Synthetic 7-class classification of multivariate windows "
        f"(T={WINDOW}, d={N_CHANNELS}). Two regimes: **(a) "
        "covariance-discriminative** (all class means zero, classes differ "
        "only by covariance rotation/anisotropy) and **(b) control** "
        "(shared covariance, classes differ only in mean location). k swept "
        f"over 1..15 with {N_FOLDS}-fold CV; best-k test accuracy reported "
        f"as mean +/- std over {len(SEEDS)} seeds (test set: "
        f"{N_TEST_PER_CLASS} windows/class). Runtime: total experiment "
        f"{total_elapsed:.1f} s.\n")

    for regime, title in (("covariance", "Regime (a): covariance-discriminative"),
                          ("control", "Regime (b): control (mean-discriminative)")):
        lines.append(f"\n## Table: {title} — test accuracy (%) at CV-best k\n")
        lines.append("| Method | 100/class | 400/class |")
        lines.append("|---|---|---|")
        for m in method_names:
            cells = [fmt_pm([r["test_acc"]
                             for r in all_results[regime][n][m]])
                     for n in TRAIN_SIZES]
            lines.append(f"| {method_titles[m]} | {cells[0]} | {cells[1]} |")

    lines.append(
        "\n## Table: k-sensitivity — 7-fold CV accuracy (%) "
        "at 400 train/class\n")
    lines.append("| Regime | Method | " + " | ".join(f"k={k}" for k in K_TABLE) + " |")
    lines.append("|---|---|" + "---|" * len(K_TABLE))
    for regime in regimes:
        for m in method_names:
            runs = all_results[regime][400][m]
            cells = []
            for k in K_TABLE:
                j = K_SWEEP.index(k)
                vals = [r["cv_curve"][j] for r in runs]
                if np.isnan(vals[0]):
                    cells.append("n/a")
                else:
                    cells.append(f"{100 * np.mean(vals):.1f} +/- {100 * np.std(vals):.1f}")
            lines.append(f"| {regime} | {method_titles[m]} | " + " | ".join(cells) + " |")

    lines.append(
        "\n## Table: runtime per query (ms) at 400 train/class "
        "(mean over seeds)\n")
    lines.append("| Method | covariance | control |")
    lines.append("|---|---|---|")
    for m in method_names:
        cells = [f"{np.mean([r['ms_per_query'] for r in all_results[r_][400][m]]):.3f}"
                 for r_ in regimes]
        lines.append(f"| {method_titles[m]} | {cells[0]} | {cells[1]} |")

    cov_geo = np.mean([r["test_acc"] for r in all_results["covariance"][400]["geoknn"]])
    cov_euc = np.mean([r["test_acc"] for r in all_results["covariance"][400]["euc_summed"]])
    ctl_geo = np.mean([r["test_acc"] for r in all_results["control"][400]["geoknn"]])
    ctl_euc = np.mean([r["test_acc"] for r in all_results["control"][400]["euc_summed"]])
    lines.append("\n## Where Euclidean kNN wins, and why\n")
    lines.append(
        f"In the covariance-discriminative regime (a), GeoKNN reaches "
        f"{100 * cov_geo:.1f}% at 400/class while the original Euclidean kNN "
        f"on time-summed features is at {100 * cov_euc:.1f}% — far below, "
        f"and only weakly above the {100 / N_CLASSES:.1f}% chance level. "
        "Every class has zero mean, so the window sums contain no first-order "
        "class signal; the residual above-chance accuracy comes from "
        "second-order leakage (the distribution of a window sum still depends "
        "on the class covariance orientation). The SPD embedding captures "
        "the class-specific covariance rotations directly. In the control "
        "regime "
        f"(b), the roles reverse: Euclidean kNN on summed features achieves "
        f"{100 * ctl_euc:.1f}% vs. {100 * ctl_geo:.1f}% for GeoKNN. The "
        "summed features are a sufficient statistic for the class mean "
        "(signal-to-noise grows linearly with window length), while the SPD "
        "embedding only sees the means through the second moment "
        "mu mu^T, which is a much weaker and partially ambiguous signal — "
        "classes whose means are negatives of each other share nearly the "
        "same second moment. Euclidean distance in the right feature space "
        "is unbeatable when that space aligns with the class signal; "
        "Riemannian geometry pays off precisely when the discriminative "
        "information lives in covariance structure rather than in means. "
        "MDM tracks GeoKNN closely in regime (a), consistent with MDM being "
        "the large-k (class-mean) limit of Riemannian kNN.\n")

    content = "\n".join(lines) + "\n"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
    with open(out_path, "w") as fh:
        fh.write(content)
    print(content)
    print(f"Total experiment runtime: {total_elapsed:.1f}s -> {out_path}")
    return content


if __name__ == "__main__":
    main()
